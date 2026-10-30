-- Source tables for the configured HTTP market-data and RAG gateway services.

CREATE TABLE IF NOT EXISTS market_quotes (
    ticker TEXT NOT NULL,
    market TEXT NOT NULL,
    price NUMERIC NOT NULL,
    currency TEXT NOT NULL,
    change_value NUMERIC,
    change_percent NUMERIC,
    source_name TEXT NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'market_data',
    source_url TEXT,
    source_ts TIMESTAMPTZ NOT NULL,
    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (ticker, market, source_ts)
);

CREATE INDEX IF NOT EXISTS idx_market_quotes_latest
ON market_quotes (ticker, market, source_ts DESC);

CREATE TABLE IF NOT EXISTS historical_prices (
    ticker TEXT NOT NULL,
    market TEXT NOT NULL,
    frequency TEXT NOT NULL,
    price_date DATE NOT NULL,
    open_price NUMERIC,
    high_price NUMERIC,
    low_price NUMERIC,
    close_price NUMERIC NOT NULL,
    adjusted_close NUMERIC,
    volume BIGINT,
    source_name TEXT NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'market_data',
    source_url TEXT,
    source_ts TIMESTAMPTZ NOT NULL,
    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (ticker, market, frequency, price_date)
);

CREATE INDEX IF NOT EXISTS idx_historical_prices_lookup
ON historical_prices (ticker, market, frequency, price_date DESC);

CREATE TABLE IF NOT EXISTS macro_indicators (
    indicator TEXT NOT NULL,
    region TEXT NOT NULL,
    as_of TIMESTAMPTZ NOT NULL,
    value NUMERIC,
    unit TEXT,
    source_name TEXT NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'market_data',
    source_url TEXT,
    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (indicator, region, as_of)
);

CREATE INDEX IF NOT EXISTS idx_macro_indicators_latest
ON macro_indicators (indicator, region, as_of DESC);

CREATE TABLE IF NOT EXISTS cn_announcements (
    ticker TEXT NOT NULL,
    doc_type TEXT NOT NULL,
    report_year INTEGER,
    title TEXT NOT NULL,
    summary TEXT,
    source_name TEXT NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'filing',
    source_url TEXT,
    published_at TIMESTAMPTZ NOT NULL,
    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (ticker, doc_type, published_at, title)
);

CREATE INDEX IF NOT EXISTS idx_cn_announcements_lookup
ON cn_announcements (ticker, doc_type, report_year, published_at DESC);

CREATE TABLE IF NOT EXISTS research_snippets (
    tenant_id TEXT NOT NULL DEFAULT 'global',
    document_id TEXT NOT NULL,
    ticker TEXT,
    title TEXT NOT NULL,
    snippet TEXT NOT NULL,
    score NUMERIC NOT NULL DEFAULT 1.0,
    checksum TEXT,
    source_name TEXT NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'research_report',
    source_url TEXT,
    published_at TIMESTAMPTZ NOT NULL,
    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (tenant_id, document_id)
);

CREATE INDEX IF NOT EXISTS idx_research_snippets_lookup
ON research_snippets (tenant_id, ticker, published_at DESC);
