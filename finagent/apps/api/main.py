from __future__ import annotations

import atexit
from collections.abc import Iterator
from contextlib import asynccontextmanager
import json
import socket
import time
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from finagent.packages.common.config import Settings, get_settings
from finagent.packages.common.context import AuthContext
from finagent.packages.common.errors import FinAgentError
from finagent.packages.common.store import Store, build_store
from finagent.apps.api.frontend import FRONTEND_HTML
from finagent.packages.memory.graph_store import build_market_graph_store
from finagent.packages.memory.vector_store import build_memory_vector_store
from finagent.packages.ops.readiness import run_deep_readiness_checks
from finagent.packages.orchestrator.pipeline import FinAgentOrchestrator
from finagent.packages.orchestrator.temporal import build_temporal_chat_gateway
from finagent.packages.rag.gateway import build_research_retriever


class ChatStreamRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=8000)
    account_scope: str | None = Field(default=None, max_length=64)
    risk_level: str | None = Field(default=None, max_length=32)


class ResumeRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    last_event_id: str | None = None
    last_sequence: int = Field(default=0, ge=0)


class PreferenceRequest(BaseModel):
    content: str = Field(min_length=1, max_length=1000)
    session_id: str = Field(default="memory_management", min_length=1, max_length=128)


def get_auth_context(
    settings: Settings = Depends(get_settings),
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
    x_user_id: str | None = Header(default=None, alias="X-User-ID"),
) -> AuthContext:
    return AuthContext(
        tenant_id=x_tenant_id or settings.default_tenant_id,
        user_id=x_user_id or settings.default_user_id,
    )


_UNSET = object()


def create_app(
    settings: Settings | None = None,
    store: Store | None = None,
    *,
    close_store_on_shutdown: bool | None = None,
    memory_vector_store: object = _UNSET,
    graph_store: object = _UNSET,
    research_retriever: object = _UNSET,
    temporal_gateway: object = _UNSET,
    market_data_provider: object = _UNSET,
    llm_client: object = _UNSET,
) -> FastAPI:
    settings = settings or get_settings()
    settings.validate_production_dependencies()
    owns_store = store is None if close_store_on_shutdown is None else close_store_on_shutdown
    store = store or build_store(settings)
    if memory_vector_store is _UNSET:
        memory_vector_store = build_memory_vector_store(settings)
    if graph_store is _UNSET:
        graph_store = build_market_graph_store(settings)
    if research_retriever is _UNSET:
        research_retriever = build_research_retriever(settings)
    if temporal_gateway is _UNSET:
        temporal_gateway = (
            build_temporal_chat_gateway(settings)
            if settings.orchestrator_backend == "temporal"
            else None
        )
    elif (
        temporal_gateway is None
        and settings.runtime_mode == "production"
        and settings.orchestrator_backend == "temporal"
    ):
        raise ValueError("生产运行使用 Temporal 后端时必须配置 Temporal 网关")
    orchestrator = FinAgentOrchestrator(
        store,
        settings=settings,
        market_data_provider=None if market_data_provider is _UNSET else market_data_provider,
        memory_vector_store=memory_vector_store,
        graph_store=graph_store,
        research_retriever=research_retriever,
        llm_client=None if llm_client is _UNSET else llm_client,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            if owns_store:
                store.close()

    if owns_store:
        atexit.register(store.close)

    app = FastAPI(title="FinAgent API", version="0.1.0", lifespan=lifespan)
    app.state.store = store
    app.state.orchestrator = orchestrator
    app.state.temporal_gateway = temporal_gateway
    app.dependency_overrides[get_settings] = lambda: settings

    @app.exception_handler(FinAgentError)
    async def finagent_error_handler(_, exc: FinAgentError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": {"code": exc.code, "message": exc.message}},
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def frontend() -> HTMLResponse:
        return HTMLResponse(FRONTEND_HTML)

    @app.get("/ready")
    def ready() -> dict:
        database_ready = bool(settings.database_url)
        vector_ready = _tcp_ready(settings.milvus_uri)
        embedding_ready = _tcp_ready(settings.embedding_base_url)
        graph_ready = (
            bool(settings.neo4j_user)
            and bool(settings.neo4j_password)
            and _tcp_ready(settings.neo4j_uri)
        )
        intent_classifier_ready = (
            settings.intent_classifier_backend == "rules"
            or _tcp_ready(settings.intent_classifier_base_url)
        )
        rag_ready = _tcp_ready(settings.rag_base_url)
        market_data_ready = _tcp_ready(settings.market_data_base_url)
        sec_ready = bool(settings.sec_user_agent)
        llm_target = (
            settings.llm_gateway_base_url
            if settings.llm_provider == "openai_compatible"
            else settings.ollama_base_url
        )
        llm_ready = _tcp_ready(llm_target)
        dependencies = {
            "database_configured": database_ready,
            "milvus_reachable": vector_ready,
            "embedding_reachable": embedding_ready,
            "neo4j_reachable": graph_ready,
            "intent_classifier_reachable": intent_classifier_ready,
            "rag_reachable": rag_ready,
            "market_data_reachable": market_data_ready,
            "sec_configured": sec_ready,
            "llm_reachable": llm_ready,
        }
        if settings.orchestrator_backend == "temporal":
            dependencies["temporal_reachable"] = _tcp_ready(settings.temporal_target_host)
        unready_dependencies = [
            name
            for name, ready in dependencies.items()
            if not ready
        ]
        return {
            "status": "ready" if not unready_dependencies else "not_ready",
            "runtime_mode": settings.runtime_mode,
            "database_backend": settings.database_backend,
            "vector_store_backend": settings.vector_store_backend,
            "graph_store_backend": settings.graph_store_backend,
            "orchestrator_backend": settings.orchestrator_backend,
            "rag_backend": settings.rag_backend,
            "market_data_provider": settings.market_data_provider,
            "market_data_sources": list(settings.market_data_sources),
            "llm_provider": settings.llm_provider,
            "dependencies": dependencies,
            "unready_dependencies": unready_dependencies,
        }

    @app.get("/ready/deep")
    def ready_deep(auth: AuthContext = Depends(get_auth_context)) -> dict:
        result = run_deep_readiness_checks(orchestrator=orchestrator, auth=auth)
        return {
            **result,
            "runtime_mode": settings.runtime_mode,
            "orchestrator_backend": settings.orchestrator_backend,
        }

    @app.post("/v1/chat/stream")
    def chat_stream(body: ChatStreamRequest, auth: AuthContext = Depends(get_auth_context)) -> StreamingResponse:
        rate_limit_key = f"chat:{auth.tenant_id}:{auth.user_id}"
        if not _check_rate_limit(rate_limit_key, settings.rate_limit_max_requests):
            return JSONResponse(
                status_code=429,
                content={"detail": {"code": "rate_limited", "message": "请求过于频繁，请稍后重试。"}},
            )

        def event_stream() -> Iterator[str]:
            try:
                if temporal_gateway:
                    events = temporal_gateway.run_chat(
                        auth=auth,
                        session_id=body.session_id,
                        content=body.message,
                        account_scope=body.account_scope,
                    )
                else:
                    events = orchestrator.run_chat(
                        auth=auth,
                        session_id=body.session_id,
                        content=body.message,
                        account_scope=body.account_scope,
                        risk_level=body.risk_level,
                    )
                for event in events:
                    yield f"event: {event.event_type}\n"
                    yield f"data: {event.model_dump_json()}\n\n"
            except FinAgentError as exc:
                error_event = {
                    "event_type": "error",
                    "payload": {"code": exc.code, "message": exc.message},
                }
                yield "event: error\n"
                yield f"data: {json.dumps(error_event, ensure_ascii=True)}\n\n"
            except Exception as exc:
                error_event = {
                    "event_type": "error",
                    "payload": {"code": "runtime_error", "message": f"服务运行异常：{exc}"},
                }
                yield "event: error\n"
                yield f"data: {json.dumps(error_event, ensure_ascii=True)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/v1/chat/history")
    def chat_history(
        session_id: str = Query(min_length=1, max_length=128),
        limit: int = Query(default=50, ge=1, le=200),
        auth: AuthContext = Depends(get_auth_context),
    ) -> dict:
        store.ensure_conversation(auth.tenant_id, auth.user_id, session_id)
        return {
            "session_id": session_id,
            "messages": store.get_recent_messages(auth.tenant_id, session_id, auth.user_id, limit),
        }

    @app.post("/v1/tasks/{task_id}/resume")
    def resume(task_id: str, body: ResumeRequest, auth: AuthContext = Depends(get_auth_context)) -> dict:
        events = orchestrator.resume_events(
            auth=auth,
            task_id=task_id,
            session_id=body.session_id,
            last_sequence=body.last_sequence,
            last_event_id=body.last_event_id,
        )
        return {"events": [event.model_dump(mode="json") for event in events]}

    @app.get("/v1/memory/profile")
    def memory_profile(auth: AuthContext = Depends(get_auth_context)) -> dict:
        return {"items": store.list_memories(auth.tenant_id, auth.user_id)}

    @app.post("/v1/memory/preferences")
    def add_preference(body: PreferenceRequest, auth: AuthContext = Depends(get_auth_context)) -> dict:
        store.ensure_conversation(auth.tenant_id, auth.user_id, body.session_id)
        created = orchestrator.memory.extract_and_store_preferences(
            tenant_id=auth.tenant_id,
            user_id=auth.user_id,
            session_id=body.session_id,
            text=body.content,
        )
        return {"created": [item.model_dump(mode="json") for item in created]}

    @app.delete("/v1/memory/{memory_id}")
    def delete_memory(memory_id: str, auth: AuthContext = Depends(get_auth_context)) -> dict:
        orchestrator.memory.delete_memory(auth.tenant_id, auth.user_id, memory_id)
        return {"deleted": True, "memory_id": memory_id}

    @app.get("/v1/memory/audit")
    def memory_audit(auth: AuthContext = Depends(get_auth_context)) -> dict:
        return {"items": store.list_memories(auth.tenant_id, auth.user_id, include_deleted=True)}

    @app.get("/v1/portfolio/accounts")
    def portfolio_accounts(auth: AuthContext = Depends(get_auth_context)) -> dict:
        return {"items": store.list_portfolio_accounts(auth.tenant_id, auth.user_id)}

    return app


def _tcp_ready(target: str | None, timeout: float = 0.5) -> bool:
    if not target:
        return False
    parsed = urlparse(target if "://" in target else f"tcp://{target}")
    host = parsed.hostname
    port = parsed.port
    if not host:
        return False
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


_rate_limit_store: dict[str, list[float]] = {}
_rate_limit_cleanup_counter: int = 0
_RATE_LIMIT_CLEANUP_INTERVAL = 500


def _check_rate_limit(key: str, max_requests: int, window_seconds: int = 60) -> bool:
    global _rate_limit_cleanup_counter
    _rate_limit_cleanup_counter += 1
    if _rate_limit_cleanup_counter >= _RATE_LIMIT_CLEANUP_INTERVAL:
        _rate_limit_cleanup_counter = 0
        now_cleanup = time.monotonic()
        stale = [k for k, v in _rate_limit_store.items() if not any(now_cleanup - ts < window_seconds for ts in v)]
        for k in stale:
            del _rate_limit_store[k]
    now = time.monotonic()
    timestamps = _rate_limit_store.get(key, [])
    timestamps = [ts for ts in timestamps if now - ts < window_seconds]
    if len(timestamps) >= max_requests:
        _rate_limit_store[key] = timestamps
        return False
    timestamps.append(now)
    _rate_limit_store[key] = timestamps
    return True
