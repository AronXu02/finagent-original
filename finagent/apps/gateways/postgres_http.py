from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
import json
from typing import Any

from fastapi import FastAPI, HTTPException

from finagent.packages.common.config import get_settings
from finagent.packages.tools.schemas import (
    CNAnnouncementsInput,
    HistoricalPricesInput,
    MacroIndicatorInput,
    ResearchSearchInput,
    TickerMarketInput,
)


def create_app() -> FastAPI:
    settings = get_settings()
    settings.validate_storage_configuration()
    app = FastAPI(title="FinAgent Postgres HTTP Gateway", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        with _connect(settings.database_url, connect_timeout_seconds=settings.database_connect_timeout_seconds) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return {"status": "ok"}

    @app.post("/v1/quotes/realtime")
    def realtime_quote(body: TickerMarketInput) -> dict[str, Any]:
        with _connect(settings.database_url, connect_timeout_seconds=settings.database_connect_timeout_seconds) as conn:
            row = _fetch_one(
                conn,
                """
                SELECT *
                FROM market_quotes
                WHERE ticker = %s AND market = %s
                ORDER BY source_ts DESC
                LIMIT 1
                """,
                (body.ticker, body.market),
            )
        if row is None:
            raise HTTPException(status_code=404, detail=f"未找到行情：{body.market}:{body.ticker}")
        source_ts = _iso(row["source_ts"])
        return {
            **_payload(row),
            "ticker": row["ticker"],
            "market": row["market"],
            "price": float(row["price"]),
            "currency": row["currency"],
            "change": _float_or_none(row.get("change_value")),
            "change_percent": _float_or_none(row.get("change_percent")),
            "source_ts": source_ts,
            "sources": [_source(row, f"{row['ticker']} {row['market']} quote", "realtime")],
        }

    @app.post("/v1/prices/history")
    def historical_prices(body: HistoricalPricesInput) -> dict[str, Any]:
        with _connect(settings.database_url, connect_timeout_seconds=settings.database_connect_timeout_seconds) as conn:
            rows = _fetch_all(
                conn,
                """
                SELECT *
                FROM historical_prices
                WHERE ticker = %s AND market = %s AND frequency = %s
                ORDER BY price_date DESC
                LIMIT %s
                """,
                (body.ticker, body.market, body.frequency, _period_limit(body.period)),
            )
        if not rows:
            raise HTTPException(status_code=404, detail=f"未找到历史价格：{body.market}:{body.ticker}")
        ordered = list(reversed(rows))
        latest = ordered[-1]
        return {
            "ticker": body.ticker,
            "market": body.market,
            "period": body.period,
            "frequency": body.frequency,
            "adjusted": body.adjusted,
            "prices": [
                {
                    "date": str(row["price_date"]),
                    "open": _float_or_none(row.get("open_price")),
                    "high": _float_or_none(row.get("high_price")),
                    "low": _float_or_none(row.get("low_price")),
                    "close": float(row["close_price"]),
                    "adjusted_close": _float_or_none(row.get("adjusted_close")),
                    "volume": int(row["volume"]) if row.get("volume") is not None else None,
                }
                for row in ordered
            ],
            "source_ts": _iso(latest["source_ts"]),
            "sources": [_source(latest, f"{body.ticker} historical prices", "snapshot")],
        }

    @app.post("/v1/macro/indicator")
    def macro_indicator(body: MacroIndicatorInput) -> dict[str, Any]:
        with _connect(settings.database_url, connect_timeout_seconds=settings.database_connect_timeout_seconds) as conn:
            row = _fetch_one(
                conn,
                """
                SELECT *
                FROM macro_indicators
                WHERE indicator = %s AND region = %s
                ORDER BY as_of DESC
                LIMIT 1
                """,
                (body.indicator, body.region),
            )
        if row is None:
            raise HTTPException(status_code=404, detail=f"未找到宏观指标：{body.region}:{body.indicator}")
        source_ts = _iso(row["as_of"])
        return {
            **_payload(row),
            "indicator": row["indicator"],
            "region": row["region"],
            "value": _float_or_none(row.get("value")),
            "unit": row.get("unit"),
            "source_ts": source_ts,
            "sources": [_source(row, f"{row['region']} {row['indicator']}", "snapshot")],
        }

    @app.post("/v1/announcements/cn")
    def cn_announcements(body: CNAnnouncementsInput) -> dict[str, Any]:
        with _connect(settings.database_url, connect_timeout_seconds=settings.database_connect_timeout_seconds) as conn:
            rows = _fetch_all(
                conn,
                """
                SELECT *
                FROM cn_announcements
                WHERE ticker = %s AND doc_type = %s
                  AND (%s::INTEGER IS NULL OR report_year = %s)
                ORDER BY published_at DESC
                LIMIT 5
                """,
                (body.ticker, body.doc_type, body.year, body.year),
            )
        if not rows:
            raise HTTPException(status_code=404, detail=f"未找到 A 股公告：{body.ticker}")
        latest = rows[0]
        return {
            "ticker": body.ticker,
            "doc_type": body.doc_type,
            "year": body.year,
            "announcements": [
                {
                    **_payload(row),
                    "title": row["title"],
                    "url": row.get("source_url"),
                    "published_at": _iso(row["published_at"]),
                    "summary": row.get("summary"),
                }
                for row in rows
            ],
            "source_ts": _iso(latest["published_at"]),
            "sources": [_source(row, str(row["title"]), "snapshot") for row in rows],
        }

    @app.post("/v1/research/search")
    def research_search(raw_body: dict[str, Any]) -> dict[str, Any]:
        body = ResearchSearchInput.model_validate(
            {key: raw_body.get(key) for key in ("query", "ticker", "top_k") if key in raw_body}
        )
        tenant_id = str(raw_body.get("tenant_id") or "global")
        tokens = [token for token in body.query.split() if token][:6]
        if not tokens:
            raise HTTPException(status_code=422, detail="研报查询必须包含可检索文本")
        token_clauses: list[str] = []
        params: list[Any] = [tenant_id]
        for token in tokens:
            like = f"%{token}%"
            token_clauses.append(
                "(title ILIKE %s OR snippet ILIKE %s OR coalesce(ticker, '') ILIKE %s OR payload_json::TEXT ILIKE %s)"
            )
            params.extend((like, like, like, like))
        ticker_clause = ""
        if body.ticker:
            ticker_clause = "AND (ticker IS NULL OR ticker = %s)"
            params.append(body.ticker.upper())
        params.append(body.top_k)
        with _connect(settings.database_url, connect_timeout_seconds=settings.database_connect_timeout_seconds) as conn:
            rows = _fetch_all(
                conn,
                f"""
                SELECT *
                FROM research_snippets
                WHERE tenant_id IN ('global', %s)
                  AND {" AND ".join(token_clauses)}
                  {ticker_clause}
                ORDER BY score DESC, published_at DESC
                LIMIT %s
                """,
                tuple(params),
            )
        if not rows:
            raise HTTPException(status_code=404, detail=f"未找到研报片段：{body.query}")
        snippets = [
            {
                **_payload(row),
                "document_id": row["document_id"],
                "ticker": row.get("ticker"),
                "title": row["title"],
                "url": row.get("source_url"),
                "published_at": _iso(row["published_at"]),
                "snippet": row["snippet"],
                "score": float(row["score"]),
                "checksum": row.get("checksum"),
            }
            for row in rows
        ]
        return {
            "status": "success",
            "query": body.query,
            "ticker": body.ticker,
            "snippets": snippets,
            "sources": [_source(row, str(row["title"]), "snapshot", checksum=row.get("checksum")) for row in rows],
        }

    @app.post("/v1/web/search")
    def web_search(raw_body: dict[str, Any]) -> dict[str, Any]:
        result = research_search(raw_body)
        result["search_type"] = "web"
        return result

    return app


def _connect(database_url: str | None, *, connect_timeout_seconds: int = 10):
    if not database_url:
        raise RuntimeError("必须配置 FINAGENT_DATABASE_URL")
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RuntimeError("Postgres HTTP 网关需要安装 psycopg。请使用 production extra 安装 finagent。") from exc
    return psycopg.connect(
        database_url,
        row_factory=dict_row,
        connect_timeout=connect_timeout_seconds,
    )


def _fetch_one(conn: Any, query: str, params: Sequence[Any]) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchone()


def _fetch_all(conn: Any, query: str, params: Sequence[Any]) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(query, params)
        return list(cur.fetchall())


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("payload_json") or {}
    if isinstance(value, str):
        return json.loads(value)
    return dict(value)


def _source(
    row: dict[str, Any],
    title: str,
    freshness: str,
    *,
    checksum: str | None = None,
) -> dict[str, Any]:
    published_at = row.get("source_ts") or row.get("published_at") or row.get("as_of")
    return {
        "source_type": row.get("source_type") or "market_data",
        "title": title,
        "url": row.get("source_url"),
        "published_at": _iso(published_at) if published_at else None,
        "fetched_at": datetime.now(UTC).isoformat(),
        "freshness": freshness,
        "checksum": checksum,
    }


def _float_or_none(value: Any) -> float | None:
    return None if value is None else float(value)


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.isoformat()
    return str(value)


def _period_limit(period: str) -> int:
    return {"1mo": 23, "3mo": 66, "6mo": 132, "1y": 252}[period]
