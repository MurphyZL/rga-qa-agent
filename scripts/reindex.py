import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.embedding import EmbeddingService
from app.llm import DeepSeekClient
from app.rag_service import RagService
from app.reranker import RerankerService
from app.vector_store import QdrantStore


def main():
    """命令行重建 Qdrant 知识索引。"""
    settings = get_settings()
    embedding_service = EmbeddingService(settings.embedding_model)
    reranker_service = RerankerService(settings.reranker_model)
    vector_store = QdrantStore(settings.qdrant_url, settings.qdrant_collection)
    llm_client = DeepSeekClient(settings)
    rag_service = RagService(settings, embedding_service, reranker_service, vector_store, llm_client)
    total = rag_service.reindex()
    print(f"reindex completed, total={total}, collection={settings.qdrant_collection}")


if __name__ == "__main__":
    main()
