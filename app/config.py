from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置：从默认值、.env 和环境变量中读取运行参数。"""

    # FastAPI 服务基础信息。
    app_name: str = "rag-qa-agent"
    app_host: str = "127.0.0.1"
    app_port: int = 8010

    # 试运行共享 Token，调用 /api/* 时必须携带。
    ai_agent_api_token: str = ""
    # CORS 白名单，逗号分隔；为空时仅允许本地调试源。
    cors_allow_origins: str = "http://127.0.0.1:8080,http://localhost:8080,http://127.0.0.1:8010,http://localhost:8010"

    # DeepSeek 使用 OpenAI 兼容协议调用，api_key 通常放在 .env 中。
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"
    # 温度越低，回答越稳定；系统问答场景通常不需要太强随机性。
    llm_temperature: float = 0.2

    # Qdrant 负责保存知识向量和原始 payload。
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "rag_knowledge"

    # embedding_model 用于向量召回，reranker_model 用于对召回结果二次排序。
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # top_k 是向量库初步召回数量，rerank_top_n 是最终交给大模型的知识数量。
    top_k: int = 8
    rerank_top_n: int = 4
    # 开启后会按请求里的菜单权限和按钮权限过滤知识。
    enforce_permission_filter: bool = True
    # 开启后应用启动时自动重建索引；试运行默认关闭，避免启动过慢。
    auto_reindex_on_startup: bool = False
    # 重排最高分低于该阈值时直接拒答，不调用大模型。
    min_rerank_score: float = -2.0

    # 本地知识库文件路径，相对路径会基于当前运行目录解析。
    knowledge_path: str = "data/knowledge.json"

    # pydantic-settings 会自动读取 .env，并用 UTF-8 解析中文配置。
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def resolved_knowledge_path(self) -> Path:
        """返回知识文件的绝对路径，供知识加载器直接读取。"""
        return Path(self.knowledge_path).resolve()

    @property
    def cors_origins(self) -> List[str]:
        """解析 CORS 白名单列表。"""
        origins = [item.strip() for item in self.cors_allow_origins.split(",") if item.strip()]
        return origins or ["http://127.0.0.1:8010"]


@lru_cache
def get_settings() -> Settings:
    """获取全局配置单例，避免每次请求都重复解析 .env。"""
    return Settings()
