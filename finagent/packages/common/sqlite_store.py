from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any

from pydantic import BaseModel

from finagent.packages.common.errors import ForbiddenError, NotFoundError
from finagent.packages.common.ids import new_id
from finagent.packages.common.time import utc_now_iso
from finagent.packages.orchestrator.models import AgentState, StreamEvent


SCHEMA = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS users (
    tenant_id TEXT NOT NULL,
    id TEXT NOT NULL,
    risk_level TEXT NOT NULL DEFAULT 'balanced',
    locale TEXT NOT NULL DEFAULT 'zh-Hans',
    created_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, id)
);

CREATE TABLE IF NOT EXISTS conversations (
    tenant_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    title TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, session_id)
);

CREATE TABLE IF NOT EXISTS messages (
    tenant_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, message_id)
);

CREATE TABLE IF NOT EXISTS portfolios (
    tenant_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    account_id TEXT NOT NULL,
    currency TEXT NOT NULL,
    snapshot_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, user_id, account_id)
);

CREATE TABLE IF NOT EXISTS portfolio_positions (
    tenant_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    account_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    market TEXT NOT NULL,
    name TEXT,
    quantity REAL NOT NULL,
    price REAL NOT NULL,
    cost_price REAL NOT NULL,
    sector TEXT,
    source_ts TEXT NOT NULL,
    PRIMARY KEY (tenant_id, user_id, account_id, ticker, market)
);

CREATE INDEX IF NOT EXISTS idx_portfolio_positions_owner
ON portfolio_positions (tenant_id, user_id, account_id, sector);

CREATE TABLE IF NOT EXISTS agent_tasks (
    tenant_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    status TEXT NOT NULL,
    active_node TEXT,
    checkpoint_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, task_id)
);

CREATE INDEX IF NOT EXISTS idx_agent_tasks_session_status
ON agent_tasks (tenant_id, session_id, status);

CREATE TABLE IF NOT EXISTS task_checkpoints (
    tenant_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    state_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, checkpoint_id)
);

CREATE TABLE IF NOT EXISTS task_events (
    tenant_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    checkpoint_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, event_id),
    UNIQUE (tenant_id, task_id, sequence)
);

CREATE INDEX IF NOT EXISTS idx_task_events_replay
ON task_events (tenant_id, task_id, sequence);

CREATE TABLE IF NOT EXISTS tool_call_logs (
    tenant_id TEXT NOT NULL,
    tool_call_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    args_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    source_ts TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, tool_call_id)
);

CREATE TABLE IF NOT EXISTS risk_reviews (
    tenant_id TEXT NOT NULL,
    review_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    verdict TEXT NOT NULL,
    risk_flags TEXT NOT NULL,
    rewrite_suggestion TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, review_id)
);

CREATE TABLE IF NOT EXISTS memory_items (
    tenant_id TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    importance REAL NOT NULL,
    decay_weight REAL NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    source TEXT,
    is_core INTEGER NOT NULL DEFAULT 0,
    market_tag TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    valid_from TEXT NOT NULL,
    valid_to TEXT,
    deleted_at TEXT,
    PRIMARY KEY (tenant_id, memory_id)
);

CREATE INDEX IF NOT EXISTS idx_memory_lookup
ON memory_items (tenant_id, user_id, type, deleted_at);

CREATE TABLE IF NOT EXISTS outbox_events (
    tenant_id TEXT NOT NULL,
    id TEXT NOT NULL,
    aggregate_type TEXT NOT NULL,
    aggregate_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    retry_count INTEGER NOT NULL DEFAULT 0,
    next_retry_at TEXT,
    published_at TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, id)
);

CREATE INDEX IF NOT EXISTS idx_outbox_pending
ON outbox_events (status, next_retry_at);
"""


def _json_default(value: Any) -> str:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return str(value)


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, default=_json_default, sort_keys=True)


def loads(value: str) -> Any:
    return json.loads(value)


def _iso_after(candidate: str, previous: str) -> str:
    try:
        candidate_dt = datetime.fromisoformat(candidate)
        previous_dt = datetime.fromisoformat(previous)
    except ValueError:
        return candidate
    if candidate_dt <= previous_dt:
        candidate_dt = previous_dt + timedelta(microseconds=1)
    return candidate_dt.isoformat()


class SQLiteStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._closed = False
        self.init_schema()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._conn.close()
            self._closed = True

    def __enter__(self) -> "SQLiteStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def ensure_user(self, tenant_id: str, user_id: str) -> None:
        now = utc_now_iso()
        with self._lock:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO users (tenant_id, id, created_at)
                VALUES (?, ?, ?)
                """,
                (tenant_id, user_id, now),
            )
            self._conn.commit()

    def ensure_conversation(self, tenant_id: str, user_id: str, session_id: str) -> None:
        now = utc_now_iso()
        with self._lock:
            self.ensure_user(tenant_id, user_id)
            row = self._conn.execute(
                """
                SELECT user_id FROM conversations
                WHERE tenant_id = ? AND session_id = ?
                """,
                (tenant_id, session_id),
            ).fetchone()
            if row and row["user_id"] != user_id:
                raise ForbiddenError("该 session_id 不属于当前用户")
            if row is None:
                self._conn.execute(
                    """
                    INSERT INTO conversations
                    (tenant_id, session_id, user_id, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (tenant_id, session_id, user_id, now, now),
                )
            else:
                self._conn.execute(
                    """
                    UPDATE conversations SET updated_at = ?
                    WHERE tenant_id = ? AND session_id = ?
                    """,
                    (now, tenant_id, session_id),
                )
            self._conn.commit()

    def get_user_risk_level(self, tenant_id: str, user_id: str) -> str:
        with self._lock:
            row = self._conn.execute(
                "SELECT risk_level FROM users WHERE tenant_id = ? AND id = ?",
                (tenant_id, user_id),
            ).fetchone()
            return row["risk_level"] if row else "balanced"

    def add_message(
        self,
        tenant_id: str,
        session_id: str,
        user_id: str,
        message_id: str,
        role: str,
        content: str,
        token_count: int = 0,
    ) -> None:
        with self._lock:
            created_at = self._next_message_created_at(tenant_id, session_id, user_id)
            self._conn.execute(
                """
                INSERT INTO messages
                (tenant_id, message_id, session_id, user_id, role, content, token_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (tenant_id, message_id, session_id, user_id, role, content, token_count, created_at),
            )
            self._conn.commit()

    def _next_message_created_at(self, tenant_id: str, session_id: str, user_id: str) -> str:
        created_at = utc_now_iso()
        row = self._conn.execute(
            """
            SELECT created_at
            FROM messages
            WHERE tenant_id = ? AND session_id = ? AND user_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (tenant_id, session_id, user_id),
        ).fetchone()
        if row is None:
            return created_at
        return _iso_after(created_at, row["created_at"])

    def get_recent_messages(self, tenant_id: str, session_id: str, user_id: str, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT message_id, role, content, token_count, created_at
                FROM messages
                WHERE tenant_id = ? AND session_id = ? AND user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (tenant_id, session_id, user_id, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def upsert_portfolio_snapshot(
        self,
        tenant_id: str,
        user_id: str,
        *,
        account_id: str,
        currency: str,
        positions: list[dict[str, Any]],
        snapshot_at: str | None = None,
    ) -> None:
        snapshot_at = snapshot_at or utc_now_iso()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    """
                    INSERT INTO portfolios
                    (tenant_id, user_id, account_id, currency, snapshot_at, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(tenant_id, user_id, account_id) DO UPDATE SET
                        currency = excluded.currency,
                        snapshot_at = excluded.snapshot_at
                    """,
                    (tenant_id, user_id, account_id, currency, snapshot_at, utc_now_iso()),
                )
                self._conn.execute(
                    """
                    DELETE FROM portfolio_positions
                    WHERE tenant_id = ? AND user_id = ? AND account_id = ?
                    """,
                    (tenant_id, user_id, account_id),
                )
                for item in positions:
                    self._conn.execute(
                        """
                        INSERT INTO portfolio_positions
                        (tenant_id, user_id, account_id, ticker, market, name, quantity,
                         price, cost_price, sector, source_ts)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            tenant_id,
                            user_id,
                            account_id,
                            item["ticker"],
                            item.get("market", "US"),
                            item.get("name"),
                            item["quantity"],
                            item["price"],
                            item["cost_price"],
                            item.get("sector"),
                            item.get("source_ts", snapshot_at),
                        ),
                    )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def get_portfolio_snapshot(
        self,
        tenant_id: str,
        user_id: str,
        *,
        account_scope: str | None = None,
        sector: str | None = None,
    ) -> dict[str, Any] | None:
        account_id = account_scope or "default"
        with self._lock:
            portfolio = self._conn.execute(
                """
                SELECT account_id, currency, snapshot_at
                FROM portfolios
                WHERE tenant_id = ? AND user_id = ? AND account_id = ?
                """,
                (tenant_id, user_id, account_id),
            ).fetchone()
            if portfolio is None:
                return None
            if sector:
                rows = self._conn.execute(
                    """
                    SELECT ticker, market, name, quantity, price, cost_price, sector, source_ts
                    FROM portfolio_positions
                    WHERE tenant_id = ? AND user_id = ? AND account_id = ? AND lower(sector) = lower(?)
                    ORDER BY ticker
                    """,
                    (tenant_id, user_id, account_id, sector),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT ticker, market, name, quantity, price, cost_price, sector, source_ts
                    FROM portfolio_positions
                    WHERE tenant_id = ? AND user_id = ? AND account_id = ?
                    ORDER BY ticker
                    """,
                    (tenant_id, user_id, account_id),
                ).fetchall()
        holdings = [dict(row) for row in rows]
        total_value = round(sum(float(item["quantity"]) * float(item["price"]) for item in holdings), 2)
        return {
            "account_id": portfolio["account_id"],
            "currency": portfolio["currency"],
            "snapshot_at": portfolio["snapshot_at"],
            "holdings": holdings,
            "total_value": total_value,
        }

    def list_portfolio_accounts(self, tenant_id: str, user_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT account_id, currency, snapshot_at
                FROM portfolios
                WHERE tenant_id = ? AND user_id = ?
                ORDER BY account_id
                """,
                (tenant_id, user_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def create_task(self, tenant_id: str, session_id: str, user_id: str, task_id: str, trace_id: str) -> None:
        now = utc_now_iso()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO agent_tasks
                (tenant_id, task_id, session_id, user_id, trace_id, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'running', ?, ?)
                """,
                (tenant_id, task_id, session_id, user_id, trace_id, now, now),
            )
            self._conn.commit()

    def update_task(self, tenant_id: str, task_id: str, *, status: str, active_node: str | None = None, checkpoint_id: str | None = None) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE agent_tasks
                SET status = ?, active_node = COALESCE(?, active_node),
                    checkpoint_id = COALESCE(?, checkpoint_id), updated_at = ?
                WHERE tenant_id = ? AND task_id = ?
                """,
                (status, active_node, checkpoint_id, utc_now_iso(), tenant_id, task_id),
            )
            self._conn.commit()

    def validate_task_owner(self, tenant_id: str, user_id: str, task_id: str, session_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM agent_tasks
                WHERE tenant_id = ? AND task_id = ?
                """,
                (tenant_id, task_id),
            ).fetchone()
        if row is None:
            raise NotFoundError("未找到任务")
        task = dict(row)
        if task["user_id"] != user_id:
            raise ForbiddenError("该任务不属于当前用户")
        if session_id is not None and task["session_id"] != session_id:
            raise ForbiddenError("该任务不属于当前会话")
        return task

    def save_checkpoint(self, tenant_id: str, task_id: str, state: AgentState) -> str:
        checkpoint_id = new_id("ckpt")
        if state.active_task is not None and state.active_task.task_id == task_id:
            state.active_task.checkpoint_id = checkpoint_id
        state_json = state.model_dump(mode="json")
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO task_checkpoints
                (tenant_id, checkpoint_id, task_id, state_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (tenant_id, checkpoint_id, task_id, dumps(state_json), utc_now_iso()),
            )
            self._conn.execute(
                """
                UPDATE agent_tasks SET checkpoint_id = ?, updated_at = ?
                WHERE tenant_id = ? AND task_id = ?
                """,
                (checkpoint_id, utc_now_iso(), tenant_id, task_id),
            )
            self._conn.commit()
        return checkpoint_id

    def get_checkpoint(self, tenant_id: str, checkpoint_id: str) -> AgentState:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT state_json FROM task_checkpoints
                WHERE tenant_id = ? AND checkpoint_id = ?
                """,
                (tenant_id, checkpoint_id),
            ).fetchone()
        if row is None:
            raise NotFoundError("未找到 checkpoint")
        return AgentState.model_validate(loads(row["state_json"]))

    def get_latest_session_state(self, tenant_id: str, session_id: str, user_id: str) -> AgentState | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT checkpoint_id
                FROM agent_tasks
                WHERE tenant_id = ? AND session_id = ? AND user_id = ?
                  AND checkpoint_id IS NOT NULL
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (tenant_id, session_id, user_id),
            ).fetchone()
        if row is None:
            return None
        return self.get_checkpoint(tenant_id, row["checkpoint_id"])

    def append_event(
        self,
        tenant_id: str,
        task_id: str,
        checkpoint_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> StreamEvent:
        event_id = new_id("evt")
        payload_json = dumps(payload)
        with self._lock:
            row = self._conn.execute(
                """
                SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence
                FROM task_events
                WHERE tenant_id = ? AND task_id = ?
                """,
                (tenant_id, task_id),
            ).fetchone()
            sequence = int(row["next_sequence"])
            created_at = utc_now_iso()
            self._conn.execute(
                """
                INSERT INTO task_events
                (tenant_id, event_id, task_id, sequence, checkpoint_id, event_type, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (tenant_id, event_id, task_id, sequence, checkpoint_id, event_type, payload_json, created_at),
            )
            self._conn.execute(
                """
                INSERT INTO outbox_events
                (tenant_id, id, aggregate_type, aggregate_id, event_type, payload_json, next_retry_at, created_at)
                VALUES (?, ?, 'task', ?, ?, ?, ?, ?)
                """,
                (tenant_id, new_id("outbox"), task_id, event_type, payload_json, created_at, created_at),
            )
            self._conn.commit()
        return StreamEvent(
            task_id=task_id,
            event_id=event_id,
            sequence=sequence,
            checkpoint_id=checkpoint_id,
            event_type=event_type,
            payload=payload,
        )

    def get_events_after(
        self,
        tenant_id: str,
        task_id: str,
        *,
        last_sequence: int = 0,
        last_event_id: str | None = None,
    ) -> list[StreamEvent]:
        if last_event_id:
            with self._lock:
                row = self._conn.execute(
                    """
                    SELECT sequence FROM task_events
                    WHERE tenant_id = ? AND task_id = ? AND event_id = ?
                    """,
                    (tenant_id, task_id, last_event_id),
                ).fetchone()
            if row is not None:
                last_sequence = max(last_sequence, int(row["sequence"]))
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT event_id, sequence, checkpoint_id, event_type, payload_json, created_at
                FROM task_events
                WHERE tenant_id = ? AND task_id = ? AND sequence > ?
                ORDER BY sequence ASC
                """,
                (tenant_id, task_id, last_sequence),
            ).fetchall()
        return [
            StreamEvent(
                task_id=task_id,
                event_id=row["event_id"],
                sequence=int(row["sequence"]),
                checkpoint_id=row["checkpoint_id"],
                event_type=row["event_type"],
                payload=loads(row["payload_json"]),
            )
            for row in rows
        ]

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
                INSERT OR REPLACE INTO tool_call_logs
                (tenant_id, tool_call_id, task_id, tool_name, args_hash, status, latency_ms, source_ts, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (tenant_id, tool_call_id, task_id, tool_name, args_hash, status, latency_ms, source_ts, utc_now_iso()),
            )
            self._conn.commit()

    def insert_risk_review(
        self,
        tenant_id: str,
        task_id: str,
        review_id: str,
        verdict: str,
        risk_flags: Iterable[dict[str, Any]],
        rewrite_suggestion: str | None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO risk_reviews
                (tenant_id, review_id, task_id, verdict, risk_flags, rewrite_suggestion, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (tenant_id, review_id, task_id, verdict, dumps(list(risk_flags)), rewrite_suggestion, utc_now_iso()),
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

    def list_memories(self, tenant_id: str, user_id: str, include_deleted: bool = False) -> list[dict[str, Any]]:
        deleted_clause = "" if include_deleted else "AND deleted_at IS NULL"
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT * FROM memory_items
                WHERE tenant_id = ? AND user_id = ? {deleted_clause}
                ORDER BY is_core DESC, importance DESC, valid_from DESC
                """,
                (tenant_id, user_id),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["metadata"] = loads(item.pop("metadata_json"))
            item["is_core"] = bool(item["is_core"])
            result.append(item)
        return result

    def record_memory_access(self, tenant_id: str, user_id: str, memory_ids: Iterable[str]) -> None:
        ids = [memory_id for memory_id in memory_ids if memory_id]
        if not ids:
            return
        with self._lock:
            for memory_id in ids:
                self._conn.execute(
                    """
                    UPDATE memory_items
                    SET access_count = access_count + 1
                    WHERE tenant_id = ? AND user_id = ? AND memory_id = ? AND deleted_at IS NULL
                    """,
                    (tenant_id, user_id, memory_id),
                )
            self._conn.commit()

    def delete_memory(self, tenant_id: str, user_id: str, memory_id: str) -> None:
        with self._lock:
            cursor = self._conn.execute(
                """
                UPDATE memory_items
                SET deleted_at = ?
                WHERE tenant_id = ? AND user_id = ? AND memory_id = ? AND deleted_at IS NULL
                """,
                (utc_now_iso(), tenant_id, user_id, memory_id),
            )
            self._conn.commit()
        if cursor.rowcount == 0:
            raise NotFoundError("未找到记忆项")
