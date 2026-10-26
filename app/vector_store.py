from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.schemas import KnowledgeItem


class QdrantStore:
    """Qdrant 向量库访问层，集中封装集合管理、写入和相似度检索。"""

    def __init__(self, url: str, collection_name: str):
        """创建 Qdrant 客户端，并记录当前业务使用的集合名称。"""
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name

    def ensure_collection(self, vector_size: int) -> None:
        """确保 Qdrant 集合存在，不存在时自动创建。"""
        collections = self.client.get_collections().collections
        exists = any(collection.name == self.collection_name for collection in collections)

        if exists:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            # size 必须和 Embedding 模型输出维度一致，distance 决定向量相似度计算方式。
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    def recreate_collection(self, vector_size: int) -> None:
        """删除并重建 Qdrant 集合，确保索引内容和本地知识文件完全一致。"""
        collections = self.client.get_collections().collections
        exists = any(collection.name == self.collection_name for collection in collections)

        if exists:
            # 当前知识源来自本地 JSON，重建集合比增量比对更直观，也更适合教学演示。
            self.client.delete_collection(collection_name=self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    def upsert_items(self, items: List[KnowledgeItem], vectors: List[List[float]]) -> None:
        """把知识条目和对应向量写入 Qdrant。"""
        points = []

        for index, item in enumerate(items):
            # 一个 point 由 id、vector、payload 组成；payload 保存原始知识，回答时会再取出来。
            points.append(
                models.PointStruct(
                    id=index + 1,
                    vector=vectors[index],
                    payload=item.model_dump(),
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: List[float], limit: int) -> List[Dict[str, Any]]:
        """按向量相似度检索知识，并统一返回 score 与 payload。"""
        try:
            # 旧版 qdrant-client 使用 search 接口进行向量查询。
            result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
            )
        except AttributeError:
            # 新版 qdrant-client 将查询能力收敛到 query_points，这里保持现有兼容逻辑。
            query_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True,
            )
            result = query_result.points

        return [
            {
                "score": point.score,
                "payload": point.payload,
            }
            for point in result
        ]
