import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
AUDIT_PATH = DATA_DIR / "audit.jsonl"
# 问题摘要最大长度，避免审计日志膨胀。
QUESTION_SUMMARY_LIMIT = 120
ANSWER_SUMMARY_LIMIT = 80
# 最近热门统计时最多回看的审计行数。
HOT_QUESTION_SCAN_LIMIT = 500
# 没有足够提问记录时，用门店常用业务问题补齐。
FALLBACK_HOT_QUESTIONS = [
    "售后单可以发起借样吗",
    "如何操作【复制下单】",
    "门店盖章流程是什么",
    "怎么删除/修改支付流水",
]


def append_audit_event(
    event_type: str,
    *,
    user_id: Optional[str] = None,
    route_name: Optional[str] = None,
    route_path: Optional[str] = None,
    question: Optional[str] = None,
    answer: Optional[str] = None,
    source_ids: Optional[list] = None,
    rejected: Optional[bool] = None,
    duration_ms: Optional[int] = None,
    feedback_type: Optional[str] = None,
    knowledge_id: Optional[str] = None,
    pending_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """追加一条 JSONL 审计记录。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "eventType": event_type,
        "userId": user_id or "",
        "routeName": route_name or "",
        "routePath": route_path or "",
    }
    if question is not None:
        payload["questionSummary"] = _truncate(question, QUESTION_SUMMARY_LIMIT)
    if answer is not None:
        payload["answerSummary"] = _truncate(answer, ANSWER_SUMMARY_LIMIT)
    if source_ids is not None:
        payload["sourceIds"] = source_ids
    if rejected is not None:
        payload["rejected"] = rejected
    if duration_ms is not None:
        payload["durationMs"] = duration_ms
    if feedback_type is not None:
        payload["feedbackType"] = feedback_type
    if knowledge_id is not None:
        payload["knowledgeId"] = knowledge_id
    if pending_id is not None:
        payload["pendingId"] = pending_id
    if extra:
        payload.update(extra)

    with AUDIT_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def list_hot_questions(limit: int = 4) -> List[str]:
    """根据最近提问次数统计热门问题，不足时用业务常用问题补齐。"""
    size = max(1, min(int(limit), 8))
    counts, last_seen = _collect_ask_counts()
    ranked = sorted(counts.items(), key=lambda item: (item[1], last_seen.get(item[0], "")), reverse=True)
    questions = [question for question, _count in ranked]
    for fallback in FALLBACK_HOT_QUESTIONS:
        if fallback not in questions:
            questions.append(fallback)
        if len(questions) >= size:
            break
    return questions[:size]


def _collect_ask_counts() -> Tuple[Dict[str, int], Dict[str, str]]:
    """从最近审计记录中统计有效提问次数和最后出现时间。"""
    counts: Dict[str, int] = {}
    last_seen: Dict[str, str] = {}
    for event in _read_recent_audit_events(HOT_QUESTION_SCAN_LIMIT):
        if event.get("eventType") != "ask":
            continue
        question = _normalize_hot_question(str(event.get("questionSummary") or ""))
        if not question:
            continue
        counts[question] = counts.get(question, 0) + 1
        last_seen[question] = str(event.get("time") or "")
    return counts, last_seen


def _normalize_hot_question(text: str) -> str:
    """清洗提问文本，过滤调试和无效问题。"""
    value = (text or "").replace("\n", " ").strip()
    if len(value) < 4:
        return ""
    if value.endswith("..."):
        value = value[:-3].strip()
    skip_prefixes = ("ping", "请先输入", "请从中台", "正在检索", "请求失败")
    if any(value.lower().startswith(prefix) or value.startswith(prefix) for prefix in skip_prefixes):
        return ""
    return value


def _read_recent_audit_events(limit: int) -> List[Dict[str, Any]]:
    """读取最近若干条审计记录。"""
    if not AUDIT_PATH.exists():
        return []
    lines = AUDIT_PATH.read_text(encoding="utf-8").splitlines()
    events = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _truncate(text: str, limit: int) -> str:
    """截断长文本，保留审计可读摘要。"""
    value = (text or "").replace("\n", " ").strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "..."
