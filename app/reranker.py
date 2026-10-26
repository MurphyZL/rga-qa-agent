from typing import Any, Dict, List

from sentence_transformers import CrossEncoder

from app.knowledge_loader import build_knowledge_text
from app.schemas import KnowledgeItem


class RerankerService:
    """重排服务：使用 CrossEncoder 对向量召回结果做二次排序。"""

    def __init__(self, model_name: str):
        """加载重排模型，CrossEncoder 会同时阅读问题和候选知识并输出相关性分数。"""
        self.model = CrossEncoder(model_name)

    def rerank(self, question: str, results: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
        """根据用户问题和候选知识计算重排分，并返回最相关的前 top_n 条。"""
        if not results:
            return []

        pairs = []
        for result in results:
            item = KnowledgeItem(**result["payload"])
            # CrossEncoder 的输入是一组“问题 + 候选文本”配对，比单纯向量相似度更精细。
            pairs.append((question, build_knowledge_text(item)))

        scores = self.model.predict(pairs)

        for index, score in enumerate(scores):
            # 把重排分写回结果，接口返回时可以用于调试和解释排序原因。
            results[index]["rerankScore"] = float(score)

        # 分数越高表示问题和候选知识越匹配，因此按倒序截取前 top_n 条。
        results.sort(key=lambda item: item["rerankScore"], reverse=True)
        return results[:top_n]
