from typing import List

from openai import OpenAI

from app.config import Settings
from app.schemas import AskRequest, KnowledgeItem


class DeepSeekClient:
    """大模型客户端：通过 OpenAI 兼容协议调用 DeepSeek 生成最终回答。"""

    def __init__(self, settings: Settings):
        """根据配置初始化 OpenAI 兼容客户端。"""
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )

    def answer(self, request: AskRequest, contexts: List[KnowledgeItem]) -> str:
        """基于检索到的知识生成系统问答。"""
        response = self.client.chat.completions.create(
            model=self.settings.deepseek_model,
            # 较低温度让回答更稳定，减少偏离知识库内容的自由发挥。
            temperature=self.settings.llm_temperature,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是中台系统的问答智能体。"
                        "你只能基于给定资料回答，资料不足时必须说明当前知识库没有明确答案。"
                        "回答必须使用中文，并按以下结构输出：结论、操作路径、操作步骤、权限说明、注意事项、参考来源。"
                        "资料配图由页面单独展示，不要编造、改写或输出图片链接。"
                    ),
                },
                {
                    "role": "user",
                    # 用户消息里同时放入问题、页面上下文、权限信息和检索资料。
                    "content": self._build_user_prompt(request, contexts),
                },
            ],
        )
        return response.choices[0].message.content or ""

    def _build_user_prompt(self, request: AskRequest, contexts: List[KnowledgeItem]) -> str:
        """构建包含页面上下文、历史对话、用户权限和检索资料的提示词。"""
        # 把多条知识和最近历史对话拼接，让模型在知识库范围内理解当前问题。
        context_text = "\n\n".join([self._format_context(index, item) for index, item in enumerate(contexts, start=1)])
        history_text = self._format_history(request)
        return "\n".join(
            [
                "最近对话历史：",
                history_text or "暂无历史对话",
                "",
                f"用户问题：{request.question}",
                f"当前路由：{request.routeName or ''}",
                f"当前路径：{request.routePath or ''}",
                f"当前用户：{request.userId or ''}",
                f"角色：{'、'.join(request.roleIds)}",
                f"菜单权限：{'、'.join(request.menuPermissions)}",
                f"按钮权限：{'、'.join(request.buttonPermissions)}",
                "",
                "检索资料：",
                context_text or "未检索到资料",
            ]
        )

    def _format_history(self, request: AskRequest) -> str:
        """格式化最近对话历史，帮助模型理解用户当前问题中的上下文指代。"""
        role_names = {
            "user": "用户",
            "assistant": "AI",
        }
        recent_history = request.history[-6:]
        return "\n".join(
            [f"{role_names.get(message.role, message.role)}：{message.content}" for message in recent_history if message.content]
        )

    def _format_context(self, index: int, item: KnowledgeItem) -> str:
        """格式化单条知识上下文，让大模型能稳定识别每个字段的含义。"""
        image_hint = f"配图数量：{len(item.images)}" if item.images else "配图数量：0"
        return "\n".join(
            [
                f"资料 {index}",
                f"问题：{item.question}",
                f"答案：{item.answer}",
                f"模块：{item.module}",
                f"操作路径：{item.menuPath}",
                f"适用角色：{'、'.join(item.roles)}",
                f"权限编码：{'、'.join(item.permissionCodes)}",
                f"来源：{item.source}",
                f"更新时间：{item.updatedAt}",
                image_hint,
            ]
        )
