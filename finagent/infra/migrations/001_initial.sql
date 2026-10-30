-- Relational schema for tenant-scoped state, audit, events, and memory metadata.

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
