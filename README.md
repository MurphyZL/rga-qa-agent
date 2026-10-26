# 系统问答智能体

这是面向中台系统的 RAG 问答服务公开示例。仓库只包含可运行的示例代码和 5 条虚构知识，不包含生产密钥、真实工单、用户审计和内部截图。使用前请自行填写 `.env`，不要把密钥提交到仓库。

技术选型：

```text
DeepSeek API
+ bge-m3
+ bge-reranker-v2-m3
+ Qdrant
+ FastAPI
+ iframe 问答页
```

## 目录结构

```text
rag-qa-agent/
├── app/
│   ├── main.py              # FastAPI 入口
│   ├── auth.py              # 试运行 Token 校验
│   ├── audit.py             # 审计日志
│   ├── rag_service.py       # RAG 编排（含拒答）
│   ├── embedding.py         # bge-m3 向量化
│   ├── reranker.py          # bge-reranker-v2-m3 重排
│   ├── vector_store.py      # Qdrant 封装
│   ├── llm.py               # DeepSeek 调用
│   ├── schemas.py           # 请求和响应结构
│   ├── feedback_store.py    # 点赞待审核 / 倒赞
│   └── knowledge_loader.py  # 本地知识加载
├── data/
│   ├── knowledge.json       # 公开示例知识（5 条）
│   ├── eval_set.json        # 对应示例知识的评测集
│   ├── feedback_*.json      # 运行后生成，已忽略
│   ├── audit.jsonl          # 审计日志（运行后生成，已忽略）
│   └── eval_report.json     # 评测报告（脚本生成，已忽略）
├── scripts/
│   ├── reindex.py           # 命令行重建索引
│   └── run_eval.py          # 试运行评测
├── web/
│   └── index.html           # iframe 页面
├── docker-compose.yml       # Qdrant
├── config.example           # 配置示例
└── requirements.txt
```

## 一、准备环境

建议使用 Python 3.10 或 3.11。

```powershell
cd rag-qa-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

首次安装 `sentence-transformers` 和模型依赖会比较慢，`bge-m3`、`bge-reranker-v2-m3` 首次运行时会自动下载模型。

## 二、启动 Qdrant

```powershell
cd rag-qa-agent
docker compose up -d
```

Qdrant 默认地址：

```text
http://127.0.0.1:6333
```

## 三、试运行配置

复制配置示例为 `.env`：

```powershell
Copy-Item config.example .env
```

必填项：

```text
DEEPSEEK_API_KEY=你的 DeepSeek API Key
AI_AGENT_API_TOKEN=请换成足够复杂的试运行 Token
CORS_ALLOW_ORIGINS=http://127.0.0.1:8080,http://localhost:8080,你的中台域名
ENFORCE_PERMISSION_FILTER=true
AUTO_REINDEX_ON_STARTUP=false
MIN_RERANK_SCORE=-2.0
```

说明：

- 除 `/health` 与静态页外，所有 `/api/*` 必须带 `Authorization: Bearer <token>` 或 `X-AI-Agent-Token`。
- 试运行默认**关闭**启动全量重建索引；首次或知识变更后请手动 reindex。
- 试运行默认**开启**权限过滤。

中台侧请配置与 `.env` 一致的 Token：

```text
VUE_APP_AI_AGENT_API_TOKEN=与 AI_AGENT_API_TOKEN 相同
```

本地联调也可在浏览器执行：

```js
localStorage.setItem('aiAgentApiToken', '与 AI_AGENT_API_TOKEN 相同')
```

## 四、启动服务

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8010
```

首次使用请先重建索引：

```powershell
python scripts\reindex.py
```

或：

```powershell
curl -X POST http://127.0.0.1:8010/api/reindex -H "Authorization: Bearer 你的Token"
```

访问页面：

```text
http://127.0.0.1:8010
```

注意：单独打开该页会提示「请从中台打开 AI 面板」。正式问答需要由父页面 `postMessage` 注入用户上下文。

健康检查：

```text
http://127.0.0.1:8010/health
```

## 五、接口说明

### 1. 系统问答

```text
POST /api/ask
```

请求头：

```text
Authorization: Bearer <AI_AGENT_API_TOKEN>
```

请求示例：

```json
{
    "question": "销售订单怎么创建？",
    "routeName": "salesOrder",
    "routePath": "/sales/salesOrder",
    "userId": "10001",
    "roleIds": ["sales"],
    "menuPermissions": ["salesOrder"],
    "buttonPermissions": ["salesOrderCreate"],
    "history": []
}
```

检索为空，或重排最高分低于 `MIN_RERANK_SCORE` 时，接口直接拒答（`rejected=true`），不调用 DeepSeek。

### 2. 检索调试

```text
POST /api/search
```

### 3. 重建索引

```text
POST /api/reindex
```

审核通过点赞后也会自动触发一次全量 reindex（试运行阶段可接受；知识量大时注意耗时）。

### 4. 反馈

- `POST /api/feedback/like`：写入待审核队列，不直接入库
- `POST /api/feedback/dislike`：记录低质量样本并参与检索过滤
- `GET /api/feedback/pending`：查看待审核列表
- `POST /api/feedback/approve`：审核通过并写入知识库 + reindex
- `POST /api/feedback/reject`：拒绝入库

## 六、接入中台 iframe

父页面在 iframe `load` 与路由变化时推送：

```text
type=QA_USER_CONTEXT
userId / roleIds / menuPermissions / buttonPermissions / routeName / routePath / apiToken
```

默认页面地址：

```text
http://127.0.0.1:8010
```

父页面来源需要写在 `web/index.html` 的 `ALLOWED_PARENT_ORIGINS` 中。单独打开该页时没有用户上下文，不能直接提问。

## 七、权限过滤验收说明

`permissionCodes` 为空表示公共知识，所有用户可见。

带权限样例（手工验收）：

| 知识 ID | 问题 | permissionCodes |
|---------|------|-----------------|
| qa-001 | 销售订单怎么创建？ | salesOrder, salesOrderCreate |
| qa-002 | 我看不到某个按钮？ | systemMaintenance, editRole |
| qa-003 | 系统帮助文档在哪里维护？ | helpWord |

验收步骤：

1. 用不包含 `salesOrder` 的账号从中台打开 AI 面板，问「销售订单怎么创建？」——`sources` 不应出现 qa-001。
2. 给该账号补上 `salesOrder` / `salesOrderCreate` 后再问——应能命中 qa-001。
3. `permissionCodes` 为空的条目属于公共知识，所有已登录用户都可以检索到。

## 八、黄金评测与放行门槛

```powershell
python scripts\run_eval.py --token 你的Token
```

门槛：

- 核心题检索命中率 ≥ 70%
- 应拒答题误答率 ≤ 20%

报告输出：`data/eval_report.json`。

## 九、试运行日常运维

1. 每日查看 `data/audit.jsonl`（问答、反馈、审核、reindex）。
2. 审核 `GET /api/feedback/pending`，通过后知识才会进入检索。
3. 每周跑一次 `scripts/run_eval.py`。
4. 页面倒赞会隐藏当前回答，并参与后续检索过滤。

## 十、当前边界

- 鉴权为共享试运行 Token，尚未对接统一登录验票。
- 知识仍以本地 JSON 为主；点赞需人工审核。
- 审核通过后执行全量 reindex，知识量增大后需改为增量 upsert。
- 首次启动会下载 bge 模型，需要能访问 Hugging Face 或配置内部模型缓存。
