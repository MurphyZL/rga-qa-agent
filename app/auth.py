from typing import Optional

from fastapi import Header, HTTPException

from app.config import Settings


def verify_api_token(
    authorization: Optional[str] = Header(default=None),
    x_ai_agent_token: Optional[str] = Header(default=None, alias="X-AI-Agent-Token"),
) -> None:
    """校验试运行 API Token，支持 Bearer 或 X-AI-Agent-Token。"""
    # 每次读取 .env，避免改 Token 后仍使用启动时缓存的旧值。
    expected = (Settings().ai_agent_api_token or "").strip()
    if not expected:
        raise HTTPException(status_code=500, detail="请先配置 AI_AGENT_API_TOKEN")

    provided = ""
    if x_ai_agent_token:
        provided = x_ai_agent_token.strip()
    elif authorization:
        value = authorization.strip()
        if value.lower().startswith("bearer "):
            provided = value[7:].strip()
        else:
            provided = value

    if not provided or provided != expected:
        raise HTTPException(status_code=401, detail="无效的 API Token")
