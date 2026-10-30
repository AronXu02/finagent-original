from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

from finagent.packages.common.config import Settings
from finagent.packages.orchestrator.models import AgentState, StreamEvent


class Store(Protocol):
    def close(self) -> None: ...

    def init_schema(self) -> None: ...

    def ensure_user(self, tenant_id: str, user_id: str) -> None: ...

    def ensure_conversation(self, tenant_id: str, user_id: str, session_id: str) -> None: ...

    def get_user_risk_level(self, tenant_id: str, user_id: str) -> str: ...

    def add_message(
        self,
        tenant_id: str,
        session_id: str,
        user_id: str,
        message_id: str,
        role: str,
        content: str,
        token_count: int = 0,
    ) -> None: ...

    def get_recent_messages(self, tenant_id: str, session_id: str, user_id: str, limit: int) -> list[dict[str, Any]]: ...

    def upsert_portfolio_snapshot(
        self,
        tenant_id: str,
        user_id: str,
        *,
        account_id: str,
        currency: str,
        positions: list[dict[str, Any]],
        snapshot_at: str | None = None,
    ) -> None: ...

    def get_portfolio_snapshot(
        self,
        tenant_id: str,
        user_id: str,
        *,
        account_scope: str | None = None,
        sector: str | None = None,
    ) -> dict[str, Any] | None: ...

    def list_portfolio_accounts(self, tenant_id: str, user_id: str) -> list[dict[str, Any]]: ...

    def create_task(self, tenant_id: str, session_id: str, user_id: str, task_id: str, trace_id: str) -> None: ...

    def update_task(
        self,
        tenant_id: str,
        task_id: str,
        *,
        status: str,
        active_node: str | None = None,
        checkpoint_id: str | None = None,
    ) -> None: ...

    def validate_task_owner(
        self,
        tenant_id: str,
        user_id: str,
        task_id: str,
        session_id: str | None = None,
    ) -> dict[str, Any]: ...

    def save_checkpoint(self, tenant_id: str, task_id: str, state: AgentState) -> str: ...

    def get_checkpoint(self, tenant_id: str, checkpoint_id: str) -> AgentState: ...

    def get_latest_session_state(self, tenant_id: str, session_id: str, user_id: str) -> AgentState | None: ...

    def append_event(
        self,
        tenant_id: str,
        task_id: str,
        checkpoint_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> StreamEvent: ...

    def get_events_after(
        self,
        tenant_id: str,
        task_id: str,
        *,
        last_sequence: int = 0,
        last_event_id: str | None = None,
    ) -> list[StreamEvent]: ...

    def log_tool_call(
        self,
        tenant_id: str,
        task_id: str,
        tool_call_id: str,
        tool_name: str,
        args_hash: str,
        status: str,
        latency_ms: int,
        source_ts: str | None,
    ) -> None: ...

    def insert_risk_review(
        self,
        tenant_id: str,
        task_id: str,
        review_id: str,
        verdict: str,
        risk_flags: Iterable[dict[str, Any]],
        rewrite_suggestion: str | None,
    ) -> None: ...

    def insert_memory(
        self,
        tenant_id: str,
        user_id: str,
        *,
        memory_type: str,
        content: str,
        importance: float,
        decay_weight: float,
        source: str | None,
        is_core: bool,
        market_tag: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str: ...

    def list_memories(self, tenant_id: str, user_id: str, include_deleted: bool = False) -> list[dict[str, Any]]: ...

    def record_memory_access(self, tenant_id: str, user_id: str, memory_ids: Iterable[str]) -> None: ...

    def delete_memory(self, tenant_id: str, user_id: str, memory_id: str) -> None: ...


def build_store(settings: Settings) -> Store:
    settings.validate_storage_configuration()
    if settings.database_backend == "sqlite":
        from finagent.packages.common.sqlite_store import SQLiteStore
        db_path = settings.database_url or "data/finagent.db"
        return SQLiteStore(db_path)
    from finagent.packages.common.postgres_store import PostgreSQLStore

    if settings.database_url is None:
        raise ValueError("使用 PostgreSQL 数据库存储时必须配置 FINAGENT_DATABASE_URL")
    return PostgreSQLStore(
        settings.database_url,
        connect_timeout_seconds=settings.database_connect_timeout_seconds,
    )
