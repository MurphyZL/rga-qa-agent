import json
from pathlib import Path
from typing import List

from app.schemas import KnowledgeItem


def load_knowledge_items(path: Path) -> List[KnowledgeItem]:
    """从 JSON 文件加载知识条目。"""
    if not path.exists():
        raise FileNotFoundError(f"知识文件不存在：{path}")

    with path.open("r", encoding="utf-8") as file:
        raw_items = json.load(file)

    return [KnowledgeItem(**item) for item in raw_items]


def build_knowledge_text(item: KnowledgeItem) -> str:
    """把结构化知识拼接为适合向量化的文本。"""
    return "\n".join(
        [
            f"问题：{item.question}",
            f"答案：{item.answer}",
            f"模块：{item.module}",
            f"菜单路径：{item.menuPath}",
            f"适用角色：{'、'.join(item.roles)}",
            f"权限编码：{'、'.join(item.permissionCodes)}",
            f"关键词：{'、'.join(item.keywords)}",
            f"相似问法：{'、'.join(item.similarQuestions)}",
        ]
    )


def save_knowledge_items(path: Path, items: List[KnowledgeItem]) -> None:
    """将知识条目写回 JSON 文件，保持 UTF-8 和缩进格式。"""
    with path.open("w", encoding="utf-8") as file:
        json.dump([item.model_dump() for item in items], file, ensure_ascii=False, indent=4)
        file.write("\n")
