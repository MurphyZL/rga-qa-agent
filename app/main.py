import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.audit import append_audit_event, list_hot_questions
from app.auth import verify_api_token
from app.config import Settings, get_settings
from app.embedding import EmbeddingService
from app.feedback_store import (
    append_dislike_feedback,
    append_like_feedback,
    approve_pending_like,
    list_pending_likes,
    reject_pending_like,
)
from app.llm import DeepSeekClient
from app.rag_service import RagService
from app.reranker import RerankerService
from app.schemas import (
    AskRequest,
    AskResponse,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackReviewRequest,
    PendingFeedbackItem,
    PendingFeedbackResponse,
    ReindexResponse,
    SearchResponse,
)
from app.vector_store import QdrantStore


BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理 FastAPI 生命周期：启动时组装 RAG 依赖，关闭时交给框架释放资源。"""
    # 这里统一创建各个服务实例，避免在每次请求里重复加载模型或连接向量库。
    # Embedding 和 Reranker 模型加载成本较高，放在应用启动阶段更适合教学和生产场景。
    settings = get_settings()
    embedding_service = EmbeddingService(settings.embedding_model)
    reranker_service = RerankerService(settings.reranker_model)
    vector_store = QdrantStore(settings.qdrant_url, settings.qdrant_collection)
    llm_client = DeepSeekClient(settings)
    rag_service = RagService(settings, embedding_service, reranker_service, vector_store, llm_client)

    # 如果开启自动重建索引，服务启动时会把本地知识文件重新向量化并写入 Qdrant。
    # 试运行默认关闭；知识变更后请调用 /api/reindex 或 scripts/reindex.py。
    if settings.auto_reindex_on_startup:
        rag_service.reindex()

    # app.state 是 FastAPI 提供的全局状态容器，路由函数可以从这里复用同一个 RAG 服务。
    app.state.rag_service = rag_service
    yield


# 创建 FastAPI 应用，并把 lifespan 交给框架执行，确保初始化逻辑只运行一次。
settings = get_settings()
app = FastAPI(title="系统问答智能体", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """提供健康检查接口，用于确认服务进程是否正常响应。"""
    return {"status": "ok"}


@app.get("/api/client-config")
def client_config():
    """给问答页返回当前 .env 中的试运行 Token，避免中台缓存旧值。"""
    return {"apiToken": (Settings().ai_agent_api_token or "").strip()}


@app.get("/api/hot-questions")
def hot_questions(limit: int = Query(default=4, ge=1, le=8)):
    """返回最近热门问题，供底部快捷入口动态展示。"""
    return {"questions": list_hot_questions(limit)}


@app.post("/api/ask", response_model=AskResponse, dependencies=[Depends(verify_api_token)])
def ask(request: AskRequest):
    """系统问答接口：接收问题和用户上下文，返回大模型生成的答案。"""
    started = time.perf_counter()
    result = app.state.rag_service.ask(request)
    duration_ms = int((time.perf_counter() - started) * 1000)
    append_audit_event(
        "ask",
        user_id=request.userId,
        route_name=request.routeName,
        route_path=request.routePath,
        question=request.question,
        answer=result.get("answer"),
        source_ids=[item.id for item in result.get("sources") or []],
        rejected=bool(result.get("rejected")),
        duration_ms=duration_ms,
    )
    return result


@app.post("/api/search", response_model=SearchResponse, dependencies=[Depends(verify_api_token)])
def search(request: AskRequest):
    """知识检索调试接口：只返回召回和重排结果，不调用大模型生成答案。"""
    started = time.perf_counter()
    results = app.state.rag_service.search(request)
    duration_ms = int((time.perf_counter() - started) * 1000)
    append_audit_event(
        "search",
        user_id=request.userId,
        route_name=request.routeName,
        route_path=request.routePath,
        question=request.question,
        source_ids=[(item.get("payload") or {}).get("id", "") for item in results],
        duration_ms=duration_ms,
    )
    return {"results": results}


@app.post("/api/reindex", response_model=ReindexResponse, dependencies=[Depends(verify_api_token)])
def reindex():
    """手动重建知识索引：重新读取本地知识文件，并把向量写入 Qdrant。"""
    started = time.perf_counter()
    total = app.state.rag_service.reindex()
    duration_ms = int((time.perf_counter() - started) * 1000)
    current_settings = get_settings()
    append_audit_event(
        "reindex",
        duration_ms=duration_ms,
        extra={"total": total, "collection": current_settings.qdrant_collection},
    )
    return ReindexResponse(total=total, collection=current_settings.qdrant_collection)


@app.post("/api/feedback/like", response_model=FeedbackResponse, dependencies=[Depends(verify_api_token)])
def like_feedback(request: FeedbackRequest):
    """点赞接口：写入待审核队列，审核通过后才会进入知识库。"""
    request.feedbackType = "like"
    record = append_like_feedback(request)
    append_audit_event(
        "feedback_like",
        user_id=request.userId,
        route_name=request.routeName,
        route_path=request.routePath,
        question=request.question,
        answer=request.answer,
        feedback_type="like",
        pending_id=record.get("pendingId"),
    )
    return FeedbackResponse(status="pending", pendingId=record.get("pendingId"))


@app.post("/api/feedback/dislike", response_model=FeedbackResponse, dependencies=[Depends(verify_api_token)])
def dislike_feedback(request: FeedbackRequest):
    """倒赞接口：记录低质量回答，后续检索时降低该答案出现概率。"""
    request.feedbackType = "dislike"
    append_dislike_feedback(request)
    append_audit_event(
        "feedback_dislike",
        user_id=request.userId,
        route_name=request.routeName,
        route_path=request.routePath,
        question=request.question,
        answer=request.answer,
        feedback_type="dislike",
    )
    return FeedbackResponse(status="ok")


@app.get(
    "/api/feedback/pending",
    response_model=PendingFeedbackResponse,
    dependencies=[Depends(verify_api_token)],
)
def pending_feedback(status: Optional[str] = Query(default="pending")):
    """查询点赞待审核列表。"""
    items = list_pending_likes(status=status)
    return PendingFeedbackResponse(
        items=[
            PendingFeedbackItem(
                pendingId=item.get("pendingId", ""),
                status=item.get("status", ""),
                question=item.get("question", ""),
                answer=item.get("answer", ""),
                routeName=item.get("routeName"),
                routePath=item.get("routePath"),
                userId=item.get("userId"),
                createdAt=item.get("createdAt", ""),
                knowledgeId=item.get("knowledgeId"),
            )
            for item in items
        ]
    )


@app.post("/api/feedback/approve", response_model=FeedbackResponse, dependencies=[Depends(verify_api_token)])
def approve_feedback(request: FeedbackReviewRequest):
    """审核通过点赞：写入知识库并全量重建索引（试运行阶段成本可接受）。"""
    started = time.perf_counter()
    current_settings = get_settings()
    record, item = approve_pending_like(request.pendingId, current_settings.resolved_knowledge_path)
    # 当前向量库以全量 recreate 为主，审核通过后重建索引以让新知识可被检索。
    total = app.state.rag_service.reindex()
    duration_ms = int((time.perf_counter() - started) * 1000)
    append_audit_event(
        "feedback_approve",
        user_id=record.get("userId"),
        route_name=record.get("routeName"),
        route_path=record.get("routePath"),
        question=record.get("question"),
        answer=record.get("answer"),
        feedback_type="approve",
        knowledge_id=item.id,
        pending_id=request.pendingId,
        duration_ms=duration_ms,
        extra={"reindexTotal": total},
    )
    return FeedbackResponse(status="approved", knowledgeId=item.id, pendingId=request.pendingId)


@app.post("/api/feedback/reject", response_model=FeedbackResponse, dependencies=[Depends(verify_api_token)])
def reject_feedback(request: FeedbackReviewRequest):
    """拒绝点赞待办，不写入知识库。"""
    record = reject_pending_like(request.pendingId)
    append_audit_event(
        "feedback_reject",
        user_id=record.get("userId"),
        route_name=record.get("routeName"),
        route_path=record.get("routePath"),
        question=record.get("question"),
        answer=record.get("answer"),
        feedback_type="reject",
        pending_id=request.pendingId,
    )
    return FeedbackResponse(status="rejected", pendingId=request.pendingId)


INDEX_TOKEN_PLACEHOLDER = "__AI_AGENT_API_TOKEN__"


def render_index_html() -> str:
    """读取问答页，并把当前 .env 中的试运行 Token 注入页面。"""
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    token = (Settings().ai_agent_api_token or "").strip()
    return html.replace(INDEX_TOKEN_PLACEHOLDER, token)


@app.get("/", include_in_schema=False)
def serve_index():
    """返回注入 Token 后的问答页，避免中台 localStorage 里的旧 Token 导致 401。"""
    return HTMLResponse(
        render_index_html(),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/index.html", include_in_schema=False)
def serve_index_html():
    """兼容直接访问 /index.html。"""
    return HTMLResponse(
        render_index_html(),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


# 挂载前端静态页面；放在最后，避免覆盖前面的 API 路由。
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
