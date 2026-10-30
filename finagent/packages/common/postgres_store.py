from __future__ import annotations

import threading
from typing import Any

from finagent.packages.common.ids import new_id
from finagent.packages.common.sqlite_store import SCHEMA, SQLiteStore, dumps
from finagent.packages.common.time import utc_now_iso


POSTGRES_SCHEMA = "\n".join(
    line for line in SCHEMA.splitlines() if not line.strip().upper().startswith("PRAGMA ")
)


class _PostgreSQLCursor:
    def __init__(self, cursor: Any) -> None:
        self._cursor = cursor

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount

    def fetchone(self) -> Any:
        return self._cursor.fetchone()

    def fetchall(self) -> list[Any]:
        return self._cursor.fetchall()


class _PostgreSQLConnectionAdapter:
    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def execute(self, sql: str, parameters: tuple[Any, ...] = ()) -> _PostgreSQLCursor:
        return _PostgreSQLCursor(self._connection.execute(_translate_qmark_sql(sql), parameters))

    def commit(self) -> None:
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()


class PostgreSQLStore(SQLiteStore):
    """PostgreSQL implementation of the relational audit/state store.

    This store is selected by the runtime factory and requires FINAGENT_DATABASE_URL.
    """

    def __init__(self, database_url: str, *, connect_timeout_seconds: int = 10) -> None:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError(
                "PostgreSQL 存储需要安装 psycopg[binary]。请使用 production extra 安装 finagent。"
            ) from exc

        self.database_url = database_url
        self._lock = threading.RLock()
        self._conn = _PostgreSQLConnectionAdapter(
            psycopg.connect(
                database_url,
                row_factory=dict_row,
                connect_timeout=connect_timeout_seconds,
            )
        )
        self._closed = False
        self.init_schema()

    def init_schema(self) -> None:
        with self._lock:
            for statement in _split_sql_script(POSTGRES_SCHEMA):
                self._conn.execute(statement)
            self._conn.commit()

    def ensure_user(self, tenant_id: str, user_id: str) -> None:
        now = utc_now_iso()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO users (tenant_id, id, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT (tenant_id, id) DO NOTHING
                """,
                (tenant_id, user_id, now),
            )
            self._conn.commit()

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
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO tool_call_logs
                (tenant_id, tool_call_id, task_id, tool_name, args_hash, status, latency_ms, source_ts, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (tenant_id, tool_call_id) DO UPDATE SET
                    task_id = excluded.task_id,
                    tool_name = excluded.tool_name,
                    args_hash = excluded.args_hash,
                    status = excluded.status,
                    latency_ms = excluded.latency_ms,
                    source_ts = excluded.source_ts,
                    created_at = excluded.created_at
                """,
                (
                    tenant_id,
                    tool_call_id,
                    task_id,
                    tool_name,
                    args_hash,
                    status,
                    latency_ms,
                    source_ts,
                    utc_now_iso(),
                ),
            )
            self._conn.commit()

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
    ) -> str:
        memory_id = new_id("mem")
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO memory_items
                (tenant_id, memory_id, user_id, type, content, importance, decay_weight,
                 source, is_core, market_tag, metadata_json, valid_from)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tenant_id,
                    memory_id,
                    user_id,
                    memory_type,
                    content,
                    importance,
                    decay_weight,
                    source,
                    1 if is_core else 0,
                    market_tag or "",
                    dumps(metadata or {}),
                    utc_now_iso(),
                ),
            )
            self._conn.commit()
        return memory_id


_QMARK_PLACEHOLDER_RE = __import__("re").compile(r"\?")


def _translate_qmark_sql(sql: str) -> str:
    return _QMARK_PLACEHOLDER_RE.sub("%s", sql)


def _split_sql_script(script: str) -> list[str]:
    return [statement.strip() for statement in script.split(";") if statement.strip()]
