"""试运行黄金评测脚本：基于检索命中率与应拒答误答率验收。"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import urllib.error
import urllib.request

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings


def main() -> None:
    """执行评测并输出简表与报告文件。"""
    parser = argparse.ArgumentParser(description="运行智能问答试运行评测")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010", help="问答服务地址")
    parser.add_argument("--token", default="", help="试运行 API Token，默认读 .env")
    parser.add_argument(
        "--eval-file",
        default=str(ROOT_DIR / "data" / "eval_set.json"),
        help="评测集路径",
    )
    parser.add_argument(
        "--report-file",
        default=str(ROOT_DIR / "data" / "eval_report.json"),
        help="评测报告输出路径",
    )
    args = parser.parse_args()

    settings = get_settings()
    token = (args.token or settings.ai_agent_api_token or "").strip()
    if not token:
        raise SystemExit("请通过 --token 或 AI_AGENT_API_TOKEN 提供试运行 Token")

    eval_items = json.loads(Path(args.eval_file).read_text(encoding="utf-8"))
    details: List[Dict[str, Any]] = []
    core_total = 0
    core_hit = 0
    reject_total = 0
    reject_false = 0

    for item in eval_items:
        question = item["question"]
        expect_reject = bool(item.get("expectReject"))
        expected_ids = set(item.get("expectedKnowledgeIds") or [])
        result = call_search(args.base_url, token, question)
        top_ids = [((row.get("payload") or {}).get("id") or "") for row in result]
        top_score = None
        if result:
            raw_score = result[0].get("rerankScore")
            top_score = float(raw_score) if raw_score is not None else None

        rejected = (not result) or (
            top_score is not None and top_score < float(settings.min_rerank_score)
        )
        hit = False
        if not expect_reject and expected_ids:
            hit = bool(expected_ids.intersection(top_ids))
            core_total += 1
            if hit:
                core_hit += 1
        if expect_reject:
            reject_total += 1
            if not rejected:
                reject_false += 1

        details.append(
            {
                "id": item.get("id"),
                "category": item.get("category"),
                "question": question,
                "expectReject": expect_reject,
                "expectedKnowledgeIds": list(expected_ids),
                "topIds": top_ids[:4],
                "topRerankScore": top_score,
                "rejected": rejected,
                "hit": hit,
            }
        )
        status = "HIT" if (expect_reject and rejected) or hit else "MISS"
        print(f"[{status}] {item.get('id')} reject={rejected} score={top_score} q={question[:40]}")

    core_rate = (core_hit / core_total) if core_total else 0.0
    false_answer_rate = (reject_false / reject_total) if reject_total else 0.0
    report = {
        "coreTotal": core_total,
        "coreHit": core_hit,
        "coreHitRate": round(core_rate, 4),
        "rejectTotal": reject_total,
        "rejectFalseAnswer": reject_false,
        "rejectFalseAnswerRate": round(false_answer_rate, 4),
        "passCoreGate": core_rate >= 0.7,
        "passRejectGate": false_answer_rate <= 0.2,
        "details": details,
    }
    Path(args.report_file).write_text(json.dumps(report, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
    print("")
    print(f"核心题命中率: {core_hit}/{core_total} = {core_rate:.2%} （门槛 >= 70%）")
    print(f"应拒答误答率: {reject_false}/{reject_total} = {false_answer_rate:.2%} （门槛 <= 20%）")
    print(f"报告已写入: {args.report_file}")
    if not report["passCoreGate"] or not report["passRejectGate"]:
        raise SystemExit(1)


def call_search(base_url: str, token: str, question: str) -> List[Dict[str, Any]]:
    """调用检索调试接口，避免评测时消耗大模型额度。"""
    payload = json.dumps(
        {
            "question": question,
            "userId": "eval-runner",
            "roleIds": ["eval"],
            # 评测默认带上样例权限，便于命中带 permissionCodes 的核心题。
            "menuPermissions": [
                "salesOrder",
                "systemMaintenance",
                "helpWord",
                "knowledgeReview",
            ],
            "buttonPermissions": ["salesOrderCreate", "editRole"],
            "history": [],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/search",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "X-AI-Agent-Token": token,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"search 失败 HTTP {error.code}: {detail}") from error
    return body.get("results") or []


if __name__ == "__main__":
    main()
