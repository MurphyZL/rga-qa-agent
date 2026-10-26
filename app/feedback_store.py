import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException

from app.knowledge_loader import load_knowledge_items, save_knowledge_items
from app.schemas import FeedbackRequest, KnowledgeItem


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LIKES_PATH = DATA_DIR / "feedback_likes.json"
DISLIKES_PATH = DATA_DIR / "feedback_dislikes.json"
PENDING_LIKES_PATH = DATA_DIR / "feedback_likes_pending.json"


def _read_json_list(path: Path) -> List[Dict[str, Any]]:
    """读取 JSON 数组文件。"""
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json_list(path: Path, items: List[Dict[str, Any]]) -> None:
    """写入 JSON 数组文件。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(items, file, ensure_ascii=False, indent=4)
        file.write("\n")


def append_like_feedback(request: FeedbackRequest) -> Dict[str, Any]:
    """将点赞问答写入待审核队列，不直接进入正式知识库。"""
    items = _read_json_list(PENDING_LIKES_PATH)
    pending_id = _next_pending_id(items)
    record = {
        "pendingId": pending_id,
        "status": "pending",
        "feedbackType": "like",
        "question": request.question.strip(),
        "answer": request.answer.strip(),
        "routeName": request.routeName,
        "routePath": request.routePath,
        "userId": request.userId,
        "sources": [source.model_dump() for source in request.sources],
        "history": [message.model_dump() for message in request.history],
        "createdAt": datetime.now().isoformat(timespec="seconds"),
        "knowledgeId": None,
        "reviewedAt": None,
    }
    items.append(record)
    _write_json_list(PENDING_LIKES_PATH, items)
    _append_feedback_log(LIKES_PATH, request, pending_id=pending_id)
    return record


def append_dislike_feedback(request: FeedbackRequest) -> Dict[str, Any]:
    """记录倒赞反馈，供后续检索时过滤低质量答案。"""
    record = {
        "feedbackType": request.feedbackType,
        "question": request.question,
        "answer": request.answer,
        "routeName": request.routeName,
        "routePath": request.routePath,
        "userId": request.userId,
        "sources": [source.model_dump() for source in request.sources],
        "history": [message.model_dump() for message in request.history],
        "createdAt": datetime.now().isoformat(timespec="seconds"),
    }
    items = _read_json_list(DISLIKES_PATH)
    items.append(record)
    _write_json_list(DISLIKES_PATH, items)
    return record


def load_dislike_records() -> List[Dict[str, Any]]:
    """读取已记录的倒赞样本。"""
    return _read_json_list(DISLIKES_PATH)


def list_pending_likes(status: Optional[str] = "pending") -> List[Dict[str, Any]]:
    """列出点赞待审核记录，默认只返回 pending。"""
    items = _read_json_list(PENDING_LIKES_PATH)
    if status is None:
        return items
    return [item for item in items if item.get("status") == status]


def approve_pending_like(pending_id: str, knowledge_path: Path) -> Tuple[Dict[str, Any], KnowledgeItem]:
    """审核通过点赞：写入知识库并标记待办为 approved。"""
    items = _read_json_list(PENDING_LIKES_PATH)
    record = _find_pending(items, pending_id)
    if record.get("status") != "pending":
        raise HTTPException(status_code=400, detail="该反馈已处理，无法重复审核")

    knowledge_items = load_knowledge_items(knowledge_path)
    next_id = _next_qa_id(knowledge_items)
    sources = record.get("sources") or []
    first_source = sources[0] if sources else {}
    module = (first_source.get("module") if isinstance(first_source, dict) else "") or record.get("routeName") or "问答反馈"
    menu_path = (
        (first_source.get("menuPath") if isinstance(first_source, dict) else "")
        or record.get("routePath")
        or record.get("routeName")
        or "问答反馈"
    )
    item = KnowledgeItem(
        id=next_id,
        question=str(record.get("question") or "").strip(),
        answer=str(record.get("answer") or "").strip(),
        module=module,
        menuPath=menu_path,
        roles=["IT支持", "业务用户"],
        permissionCodes=[],
        source="点赞反馈审核入库",
        updatedAt=datetime.now().strftime("%Y-%m-%d"),
        trustLevel="verified",
        keywords=_build_keywords_from_record(record),
        similarQuestions=[],
    )
    knowledge_items.append(item)
    save_knowledge_items(knowledge_path, knowledge_items)

    record["status"] = "approved"
    record["knowledgeId"] = item.id
    record["reviewedAt"] = datetime.now().isoformat(timespec="seconds")
    _write_json_list(PENDING_LIKES_PATH, items)
    return record, item


def reject_pending_like(pending_id: str) -> Dict[str, Any]:
    """拒绝点赞待办，不写入知识库。"""
    items = _read_json_list(PENDING_LIKES_PATH)
    record = _find_pending(items, pending_id)
    if record.get("status") != "pending":
        raise HTTPException(status_code=400, detail="该反馈已处理，无法重复审核")
    record["status"] = "rejected"
    record["reviewedAt"] = datetime.now().isoformat(timespec="seconds")
    _write_json_list(PENDING_LIKES_PATH, items)
    return record


def _find_pending(items: List[Dict[str, Any]], pending_id: str) -> Dict[str, Any]:
    """按 pendingId 查找待审核记录。"""
    for item in items:
        if item.get("pendingId") == pending_id:
            return item
    raise HTTPException(status_code=404, detail="未找到待审核反馈")


def _next_qa_id(items: List[KnowledgeItem]) -> str:
    """生成下一个 qa 编号。"""
    max_num = 0
    for item in items:
        if item.id.startswith("qa-"):
            try:
                max_num = max(max_num, int(item.id.split("-", 1)[1]))
            except (IndexError, ValueError):
                continue
    return f"qa-{max_num + 1:03d}"


def _next_pending_id(items: List[Dict[str, Any]]) -> str:
    """生成下一个待审核编号。"""
    max_num = 0
    for item in items:
        pending_id = str(item.get("pendingId") or "")
        if pending_id.startswith("pending-"):
            try:
                max_num = max(max_num, int(pending_id.split("-", 1)[1]))
            except (IndexError, ValueError):
                continue
    return f"pending-{max_num + 1:03d}"


def _build_keywords_from_record(record: Dict[str, Any]) -> List[str]:
    """从待审核记录中提取基础关键词。"""
    keywords = []
    for value in [record.get("routeName"), record.get("routePath")]:
        if value and value not in keywords:
            keywords.append(str(value))
    for source in (record.get("sources") or [])[:2]:
        if isinstance(source, dict):
            module = source.get("module")
            if module and module not in keywords:
                keywords.append(str(module))
    return keywords


def _append_feedback_log(path: Path, request: FeedbackRequest, pending_id: str = "") -> None:
    """追加点赞反馈日志，便于后续人工核查。"""
    items = _read_json_list(path)
    items.append(
        {
            "pendingId": pending_id,
            "question": request.question,
            "answer": request.answer,
            "routeName": request.routeName,
            "routePath": request.routePath,
            "userId": request.userId,
            "createdAt": datetime.now().isoformat(timespec="seconds"),
        }
    )
    _write_json_list(path, items)
