from typing import Any, Dict, List

from fastapi import HTTPException

from app.config import Settings
from app.embedding import EmbeddingService
from app.feedback_store import load_dislike_records
from app.knowledge_loader import build_knowledge_text, load_knowledge_items
from app.llm import DeepSeekClient
from app.reranker import RerankerService
from app.schemas import AskRequest, KnowledgeItem, SourceItem
from app.vector_store import QdrantStore

# 低置信或无命中时的固定拒答文案。
REJECT_ANSWER = "当前知识库没有找到明确答案，建议补充 Q&A 或联系对应业务负责人确认。"


class RagService:
    """系统问答 RAG 编排服务，负责把检索、过滤、重排和生成串成完整链路。"""

    def __init__(
        self,
        settings: Settings,
        embedding_service: EmbeddingService,
        reranker_service: RerankerService,
        vector_store: QdrantStore,
        llm_client: DeepSeekClient,
    ):
        """保存各层依赖，便于问答流程中复用同一批模型和外部连接。"""
        self.settings = settings
        self.embedding_service = embedding_service
        self.reranker_service = reranker_service
        self.vector_store = vector_store
        self.llm_client = llm_client

    def reindex(self) -> int:
        """重新加载本地知识文件，生成向量后写入 Qdrant。"""
        # RAG 的知识准备阶段：先把结构化 Q&A 转成适合向量化的文本，再批量生成向量。
        items = load_knowledge_items(self.settings.resolved_knowledge_path)
        texts = [build_knowledge_text(item) for item in items]
        vectors = self.embedding_service.encode(texts)

        # 重建集合可以让向量库内容和本地知识文件保持一致，适合当前本地知识源场景。
        self.vector_store.recreate_collection(self.embedding_service.vector_size)
        self.vector_store.upsert_items(items, vectors)
        return len(items)

    def ask(self, request: AskRequest) -> Dict[str, Any]:
        """执行完整问答流程：检索候选知识，再交给大模型生成最终答案。"""
        if not self.settings.deepseek_api_key:
            raise HTTPException(status_code=500, detail="请先配置 DEEPSEEK_API_KEY")

        # search 会完成向量召回、权限过滤和重排，返回最适合回答当前问题的知识片段。
        results = self.search(request)
        if self._should_reject(results):
            return {
                "answer": REJECT_ANSWER,
                "sources": [],
                "rejected": True,
            }

        contexts = [KnowledgeItem(**item["payload"]) for item in results]
        # 大模型只接收经过筛选的上下文，避免把整库内容塞进提示词导致成本和噪声上升。
        answer = self.llm_client.answer(request, contexts)
        return {
            "answer": answer,
            "sources": [self._to_source_item(item) for item in results],
            "rejected": False,
        }

    def search(self, request: AskRequest) -> List[Dict[str, Any]]:
        """执行检索链路：问题向量化、向量召回、权限过滤、CrossEncoder 重排。"""
        # 先把用户问题转成向量，用于在 Qdrant 中查找语义相近的知识条目。
        query_vector = self.embedding_service.encode_one(request.question)
        results = self.vector_store.search(query_vector, self.settings.top_k)
        filtered_results = self._filter_by_permission(request, results)
        filtered_results = self._filter_disliked(filtered_results)
        return self.reranker_service.rerank(request.question, filtered_results, self.settings.rerank_top_n)

    def _should_reject(self, results: List[Dict[str, Any]]) -> bool:
        """判断检索结果是否应拒答：无命中或重排分过低。"""
        if not results:
            return True
        top_score = results[0].get("rerankScore")
        if top_score is None:
            return False
        return float(top_score) < float(self.settings.min_rerank_score)

    def _filter_by_permission(self, request: AskRequest, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """根据当前用户权限过滤候选知识，避免返回用户无权限看到的操作说明。"""
        if not self.settings.enforce_permission_filter:
            return results

        # 菜单权限和按钮权限都可能决定用户是否能操作某个功能，这里统一合并后判断。
        user_permissions = set(request.menuPermissions + request.buttonPermissions)
        if not user_permissions:
            # 用户没有任何权限时，只保留不要求权限编码的公共知识。
            return [item for item in results if not item["payload"].get("permissionCodes")]

        filtered_results = []
        for item in results:
            permission_codes = set(item["payload"].get("permissionCodes") or [])
            # permission_codes 为空表示公共知识；有交集表示当前用户具备相关权限。
            if not permission_codes or permission_codes & user_permissions:
                filtered_results.append(item)

        return filtered_results

    def _filter_disliked(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """过滤已被倒赞的问题和答案，降低低质量样本再次被召回的概率。"""
        dislike_records = load_dislike_records()
        if not dislike_records:
            return results

        disliked_questions = {item.get("question", "").strip() for item in dislike_records}
        disliked_answers = {item.get("answer", "").strip() for item in dislike_records}
        filtered_results = []
        for item in results:
            payload = item.get("payload") or {}
            question = str(payload.get("question") or "").strip()
            answer = str(payload.get("answer") or "").strip()
            if question in disliked_questions or answer in disliked_answers:
                continue
            filtered_results.append(item)
        return filtered_results

    def _to_source_item(self, item: Dict[str, Any]) -> SourceItem:
        """把向量库检索结果转换为接口返回的来源信息。"""
        payload = item["payload"]
        return SourceItem(
            id=payload.get("id", ""),
            score=float(item.get("score") or 0),
            rerankScore=float(item.get("rerankScore")) if item.get("rerankScore") is not None else None,
            question=payload.get("question", ""),
            module=payload.get("module", ""),
            menuPath=payload.get("menuPath", ""),
            source=payload.get("source", ""),
            updatedAt=payload.get("updatedAt", ""),
            images=payload.get("images") or [],
        )
