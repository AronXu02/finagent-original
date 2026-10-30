# FinAgent

基于动态记忆与多智能体协同的宏微观金融决策 Agent 服务。

## 主要依赖

- **Python 3.11+**，包管理使用 [uv](https://docs.astral.sh/uv/)
- **Ollama**（本地 LLM 推理），默认模型 `qwen3.5:4b`
- **Docker Desktop**（可选，生产模式需 PostgreSQL / Milvus / Neo4j / Temporal）

### 安装依赖

```powershell
# 开发/测试依赖
uv sync --extra dev

# 生产依赖（含 PostgreSQL、Milvus、Neo4j、Temporal 客户端）
uv sync --extra production
```

## 快速启动（本地预览）

本地预览模式使用 SQLite + 内存向量/图存储，无需外部数据库，直接对接本地 Ollama。

### 1. 确认 Ollama 运行并拉取模型

```powershell
ollama serve
ollama pull qwen3.5:4b
ollama pull nomic-embed-text
```

### 2. 启动 API

```powershell
uv run python -m finagent.scripts.run_local_preview_api --port 8001
```

Ollama 在非默认端口时指定 `--llm-base-url`：

```powershell
uv run python -m finagent.scripts.run_local_preview_api --port 8001 --llm-base-url http://127.0.0.1:8000
```

### 3. 验证

```powershell
# 健康检查
curl http://127.0.0.1:8001/health

# 流式对话测试
curl -N -X POST http://127.0.0.1:8001/v1/chat/stream `
  -H "Content-Type: application/json" `
  -H "X-Tenant-ID: tenant_default" `
  -H "X-User-ID: user_default" `
  -d '{"session_id":"session-001","message":"什么是ETF？","account_scope":"default"}'
```

## 启动参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | `8001` | API 端口 |
| `--llm` | `ollama` | LLM 后端：`ollama` / `openai-compatible` / `local` |
| `--llm-base-url` | `http://127.0.0.1:11434` | LLM 服务地址 |
| `--llm-model` | `qwen3.5:4b` | 模型名称 |
| `--db` | `finagent/data/local-preview.sqlite3` | SQLite 数据库路径 |

## API 端点

- `GET  /health` — 健康检查
- `GET  /ready` — 浅层就绪检查
- `GET  /ready/deep` — 深层依赖探测
- `POST /v1/chat/stream` — 流式金融对话（SSE）
- `GET  /v1/chat/history` — 对话历史
- `POST /v1/tasks/{task_id}/resume` — 恢复中断任务
- `GET  /v1/memory/profile` — 用户记忆画像
- `POST /v1/memory/preferences` — 更新偏好

## 生产部署（Docker Compose）

启动基础设施（PostgreSQL、Milvus、Neo4j、Temporal）：

```powershell
docker compose -f finagent\infra\docker\docker-compose.yml up -d
```

初始化数据库并启动生产 API：

```powershell
uv run python -m finagent.scripts.init_db
uv run python -m finagent.scripts.run_api
```

## 运行测试

```powershell
uv run python -m pytest finagent/tests -q
```
