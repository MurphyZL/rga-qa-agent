from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class KnowledgeImage(BaseModel):
    """知识配图，回答页按检索来源原样展示，不依赖大模型复述图片地址。"""

    # 本地静态路径，例如 /knowledge-images/qa-001-1.png。
    url: str
    # 配图说明，便于页面展示和后续人工校对。
    caption: str = ""


class KnowledgeItem(BaseModel):
    """知识库条目结构，对应本地 knowledge.json 中的一条 Q&A 知识。"""

    # 知识唯一标识，便于来源追踪和后续人工维护。
    id: str
    # 标准问题，通常是用户可能会问的业务问题。
    question: str
    # 标准答案，是大模型生成回答时最重要的依据。
    answer: str
    # 所属业务模块，用于帮助模型理解功能归属。
    module: str = ""
    # 菜单或页面路径，用于回答“在哪里操作”。
    menuPath: str = ""
    # 适用角色列表，用于说明哪些角色通常能使用该功能。
    roles: List[str] = Field(default_factory=list)
    # 权限编码列表，用于检索后按当前用户权限过滤知识。
    permissionCodes: List[str] = Field(default_factory=list)
    # 知识来源，接口返回 sources 时会展示给前端。
    source: str = ""
    # 知识更新时间，帮助判断答案是否可能过期。
    updatedAt: str = ""
    # 可信等级，当前默认 verified，便于后续扩展审核状态。
    trustLevel: str = "verified"
    # 关键词用于补充语义信息，提升向量化后的召回效果。
    keywords: List[str] = Field(default_factory=list)
    # 相似问法用于覆盖用户不同表达方式，提升检索命中率。
    similarQuestions: List[str] = Field(default_factory=list)
    # 配图列表，由知识导入时下载到本地静态目录。
    images: List[KnowledgeImage] = Field(default_factory=list)


class ChatMessage(BaseModel):
    """前端传入的最近对话消息，用于让模型理解多轮上下文。"""

    # 消息角色，当前只使用 user 和 assistant 两类。
    role: str
    # 消息内容，通常是用户问题或 AI 回答。
    content: str


class AskRequest(BaseModel):
    """问答接口请求，包含用户问题、历史对话和当前页面、角色、权限上下文。"""

    # 用户输入的自然语言问题。
    question: str
    # 当前前端路由名称，帮助模型理解用户所在页面。
    routeName: Optional[str] = None
    # 当前前端路由路径，回答操作路径时可作为补充上下文。
    routePath: Optional[str] = None
    # 当前用户标识，仅作为上下文传入，不在本服务内做用户查询。
    userId: Optional[str] = None
    # 当前用户角色列表，用于提示模型说明适用角色。
    roleIds: List[str] = Field(default_factory=list)
    # 当前用户拥有的菜单权限编码。
    menuPermissions: List[str] = Field(default_factory=list)
    # 当前用户拥有的按钮权限编码。
    buttonPermissions: List[str] = Field(default_factory=list)
    # 最近几轮历史对话，用于让模型理解“继续说明”“它”等上下文指代。
    history: List[ChatMessage] = Field(default_factory=list)


class SourceItem(BaseModel):
    """回答来源信息，用于告诉前端答案参考了哪些知识条目。"""

    # 来源知识的唯一标识。
    id: str
    # 向量检索分数，表示语义相似度。
    score: float
    # 重排模型分数，存在时表示 CrossEncoder 重新评估后的相关性。
    rerankScore: Optional[float] = None
    # 来源知识中的标准问题。
    question: str
    # 来源知识所属模块。
    module: str
    # 来源知识对应的操作路径。
    menuPath: str
    # 来源说明，例如文档、人工整理或系统页面。
    source: str
    # 来源知识最后更新时间。
    updatedAt: str
    # 该来源附带的配图，供前端在回答下方展示。
    images: List[KnowledgeImage] = Field(default_factory=list)


class AskResponse(BaseModel):
    """问答接口响应，包含最终答案和引用来源。"""

    # 大模型生成的中文答案。
    answer: str
    # 回答参考的知识来源列表，便于前端展示和人工核对。
    sources: List[SourceItem]
    # 是否因低置信或无命中而拒答。
    rejected: bool = False


class SearchResponse(BaseModel):
    """检索接口响应，用于调试向量召回、权限过滤和重排效果。"""

    # 原始检索结果列表，包含 score、payload 和可选 rerankScore。
    results: List[Dict[str, Any]]


class ReindexResponse(BaseModel):
    """重建索引接口响应，告诉调用方本次写入了多少条知识。"""

    # 本次重建后写入向量库的知识条数。
    total: int
    # 使用的 Qdrant 集合名称。
    collection: str


class FeedbackRequest(BaseModel):
    """问答反馈请求，包含当前问题、回答和页面上下文。"""

    # 反馈类型，like 表示点赞，dislike 表示倒赞。
    feedbackType: str
    # 当前用户问题。
    question: str
    # 当前 AI 回答。
    answer: str
    # 当前页面路由名称。
    routeName: Optional[str] = None
    # 当前页面路由路径。
    routePath: Optional[str] = None
    # 当前用户标识。
    userId: Optional[str] = None
    # 当前回答引用的知识来源。
    sources: List[SourceItem] = Field(default_factory=list)
    # 最近历史对话，便于后续核对入库上下文。
    history: List[ChatMessage] = Field(default_factory=list)


class FeedbackResponse(BaseModel):
    """问答反馈响应，告诉前端本次反馈是否已保存。"""

    # 处理结果状态。
    status: str
    # 点赞入库后生成的知识编号。
    knowledgeId: Optional[str] = None
    # 待审核记录编号。
    pendingId: Optional[str] = None


class FeedbackReviewRequest(BaseModel):
    """审核点赞待办请求。"""

    # 待审核记录编号。
    pendingId: str


class PendingFeedbackItem(BaseModel):
    """待审核点赞条目。"""

    pendingId: str
    status: str
    question: str
    answer: str
    routeName: Optional[str] = None
    routePath: Optional[str] = None
    userId: Optional[str] = None
    createdAt: str
    knowledgeId: Optional[str] = None


class PendingFeedbackResponse(BaseModel):
    """待审核列表响应。"""

    items: List[PendingFeedbackItem]
