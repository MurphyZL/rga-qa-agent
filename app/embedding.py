from typing import List

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """文本向量服务：使用 bge-m3 把问题或知识内容转换为可检索的数字向量。"""

    def __init__(self, model_name: str):
        """加载 SentenceTransformer 模型，后续所有向量化请求都复用这个模型实例。"""
        self.model_name = model_name
        # 模型会在初始化时加载到本地运行环境，因此不要在每次请求中重复创建。
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> List[List[float]]:
        """批量生成归一化向量，用于知识入库或批量检索场景。"""
        vectors = self.model.encode(
            texts,
            # 归一化后可以直接配合余弦距离计算相似度，分数更稳定。
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # SentenceTransformer 返回 numpy 数组，转成普通 list 后便于 Qdrant 客户端序列化。
        return [vector.tolist() for vector in vectors]

    def encode_one(self, text: str) -> List[float]:
        """生成单条文本向量，主要用于把用户问题转换成检索向量。"""
        return self.encode([text])[0]

    @property
    def vector_size(self) -> int:
        """返回当前模型的向量维度，创建 Qdrant 集合时必须使用相同维度。"""
        return self.model.get_sentence_embedding_dimension()
