from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from finagent.packages.common.config import get_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="同步真实行情数据到 PostgreSQL 网关源表。")
    parser.add_argument("--ticker", action="append", required=True, help="股票代码，例如 AAPL。")
    parser.add_argument("--market", default="US")
    parser.add_argument("--range", default="1mo")
    parser.add_argument("--interval", default="1d")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    if not settings.database_url:
        raise RuntimeError("必须配置 FINAGENT_DATABASE_URL")
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("sync_market_data 需要 psycopg，请安装 production 依赖。") from exc

    with psycopg.connect(settings.database_url, connect_timeout=settings.database_connect_timeout_seconds) as conn:
        for ticker in args.ticker:
            chart = _fetch_yahoo_chart(ticker, range_=args.range, interval=args.interval)
            _upsert_chart(conn, ticker=ticker.upper(), market=args.market, chart=chart)
            print(f"已同步 Yahoo Finance 行情数据：{args.market}:{ticker.upper()}")


def _fetch_yahoo_chart(ticker: str, *, range_: str, interval: str) -> dict[str, Any]:
    encoded = quote(ticker.upper(), safe=".")
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}"
        f"?range={quote(range_)}&interval={quote(interval)}&includePrePost=false&events=div%2Csplits"
    )
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "FinAgent market-data sync contact@example.com",
        },
        method="GET",
    )
    with urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    result = (payload.get("chart", {}).get("result") or [None])[0]
    if not isinstance(result, dict):
        error = payload.get("chart", {}).get("error")
        raise RuntimeError(f"Yahoo Finance 未返回 {ticker} 的图表结果：{error}")
    result["_source_url"] = url
    return result


def _upsert_chart(conn: Any, *, ticker: str, market: str, chart: dict[str, Any]) -> None:
    meta = chart.get("meta") or {}
    timestamps = chart.get("timestamp") or []
    quote_data = ((chart.get("indicators") or {}).get("quote") or [{}])[0]
    adjclose = ((chart.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose") or []
    currency = str(meta.get("currency") or _default_currency(market))
    source_url = chart["_source_url"]

    latest_price = _number(meta.get("regularMarketPrice"))
    latest_ts = _timestamp(meta.get("regularMarketTime"))
    latest_change = _number(meta.get("regularMarketChange"))
    latest_change_percent = _number(meta.get("regularMarketChangePercent"))

    with conn.cursor() as cur:
        latest_close: float | None = None
        latest_close_ts: datetime | None = None
        for idx, ts in enumerate(timestamps):
            observed_at = _timestamp(ts)
            close_price = _number(_at(quote_data.get("close"), idx))
            if observed_at is None or close_price is None:
                continue
            latest_close = close_price
            latest_close_ts = observed_at
            cur.execute(
                """
                INSERT INTO historical_prices
                (ticker, market, frequency, price_date, open_price, high_price, low_price,
                 close_price, adjusted_close, volume, source_name, source_type, source_url, source_ts, payload_json)
                VALUES (%s, %s, 'daily', %s, %s, %s, %s, %s, %s, %s,
                        'yahoo_finance', 'market_data', %s, %s, %s)
                ON CONFLICT (ticker, market, frequency, price_date) DO UPDATE SET
                    open_price = excluded.open_price,
                    high_price = excluded.high_price,
                    low_price = excluded.low_price,
                    close_price = excluded.close_price,
                    adjusted_close = excluded.adjusted_close,
                    volume = excluded.volume,
                    source_name = excluded.source_name,
                    source_type = excluded.source_type,
                    source_url = excluded.source_url,
                    source_ts = excluded.source_ts,
                    payload_json = excluded.payload_json
                """,
                (
                    ticker,
                    market,
                    observed_at.date(),
                    _number(_at(quote_data.get("open"), idx)),
                    _number(_at(quote_data.get("high"), idx)),
                    _number(_at(quote_data.get("low"), idx)),
                    close_price,
                    _number(_at(adjclose, idx)),
                    _integer(_at(quote_data.get("volume"), idx)),
                    source_url,
                    observed_at,
                    json.dumps({"provider": "yahoo_finance", "meta": meta}, ensure_ascii=True),
                ),
            )

        price = latest_price if latest_price is not None else latest_close
        source_ts = latest_ts or latest_close_ts or datetime.now(UTC)
        if price is None:
            raise RuntimeError(f"Yahoo Finance 的 {ticker} 图表未包含可用价格")
        cur.execute(
            """
            INSERT INTO market_quotes
            (ticker, market, price, currency, change_value, change_percent,
             source_name, source_type, source_url, source_ts, payload_json)
            VALUES (%s, %s, %s, %s, %s, %s,
                    'yahoo_finance', 'market_data', %s, %s, %s)
            ON CONFLICT (ticker, market, source_ts) DO UPDATE SET
                price = excluded.price,
                currency = excluded.currency,
                change_value = excluded.change_value,
                change_percent = excluded.change_percent,
                source_name = excluded.source_name,
                source_type = excluded.source_type,
                source_url = excluded.source_url,
                payload_json = excluded.payload_json
            """,
            (
                ticker,
                market,
                price,
                currency,
                latest_change,
                latest_change_percent,
                source_url,
                source_ts,
                json.dumps({"provider": "yahoo_finance", "meta": meta}, ensure_ascii=True),
            ),
        )
    conn.commit()


def _at(values: Any, index: int) -> Any:
    if not isinstance(values, list) or index >= len(values):
        return None
    return values[index]


def _number(value: Any) -> float | None:
    return None if value is None else float(value)


def _integer(value: Any) -> int | None:
    return None if value is None else int(value)


def _timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    return datetime.fromtimestamp(int(value), tz=UTC)


def _default_currency(market: str) -> str:
    return {"CN": "CNY", "HK": "HKD", "FX": "JPY"}.get(market, "USD")


if __name__ == "__main__":
    main()
