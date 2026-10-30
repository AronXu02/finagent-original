from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from typing import Any

from finagent.packages.common.config import get_settings
from finagent.packages.market_data.adapters import SecEdgarProvider
from finagent.packages.tools.registry import ToolContext
from finagent.packages.tools.schemas import SecFilingsInput


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="同步 SEC 披露文件片段到 PostgreSQL RAG 源表。")
    parser.add_argument("--tenant-id", default="global")
    parser.add_argument("--ticker", action="append", required=True, help="股票代码，例如 AAPL。")
    parser.add_argument("--form-type", default="10-K", choices=("10-K", "10-Q", "8-K"))
    parser.add_argument("--section", default="risk_factors")
    parser.add_argument("--fiscal-year", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    if not settings.database_url:
        raise RuntimeError("必须配置 FINAGENT_DATABASE_URL")
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("sync_sec_rag 需要 psycopg，请安装 production 依赖。") from exc

    provider = SecEdgarProvider(settings)
    context = ToolContext(
        tenant_id=args.tenant_id,
        user_id="rag_sync",
        task_id="rag_sync",
        trace_id="rag_sync",
    )
    with psycopg.connect(settings.database_url, connect_timeout=settings.database_connect_timeout_seconds) as conn:
        for ticker in args.ticker:
            filing = provider.fetch_sec_filings(
                SecFilingsInput(
                    ticker=ticker,
                    form_type=args.form_type,
                    section=args.section,
                    fiscal_year=args.fiscal_year,
                ),
                context,
            )
            _upsert_filing_snippet(conn, tenant_id=args.tenant_id, filing=filing)
            print(f"已同步 SEC RAG 片段：{filing['ticker']} {filing['form_type']} {filing['section']}")


def _upsert_filing_snippet(conn: Any, *, tenant_id: str, filing: dict[str, Any]) -> None:
    excerpt = str(filing.get("section_excerpt") or "").strip()
    if not excerpt:
        raise RuntimeError("SEC 披露文件未包含 section_excerpt")
    snippet = _compact(excerpt[:1800])
    checksum = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
    accession = str(filing.get("accession_number") or checksum[:12])
    published_at = _published_at(filing.get("source_ts"))
    document_id = f"sec:{filing['ticker']}:{filing['form_type']}:{accession}:{filing['section']}"
    title = f"{filing['ticker']} {filing['form_type']} {filing['section']} from SEC EDGAR"
    payload = {
        "provider": "sec_edgar",
        "ticker": filing.get("ticker"),
        "company": filing.get("company"),
        "cik": filing.get("cik"),
        "form_type": filing.get("form_type"),
        "section": filing.get("section"),
        "filing_date": filing.get("filing_date"),
        "accession_number": filing.get("accession_number"),
        "highlights": filing.get("highlights") or [],
    }
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO research_snippets
            (tenant_id, document_id, ticker, title, snippet, score, checksum,
             source_name, source_type, source_url, published_at, payload_json)
            VALUES (%s, %s, %s, %s, %s, 1.0, %s,
                    'sec_edgar', 'filing', %s, %s, %s)
            ON CONFLICT (tenant_id, document_id) DO UPDATE SET
                ticker = excluded.ticker,
                title = excluded.title,
                snippet = excluded.snippet,
                score = excluded.score,
                checksum = excluded.checksum,
                source_name = excluded.source_name,
                source_type = excluded.source_type,
                source_url = excluded.source_url,
                published_at = excluded.published_at,
                payload_json = excluded.payload_json
            """,
            (
                tenant_id,
                document_id,
                filing.get("ticker"),
                title,
                snippet,
                checksum,
                filing.get("filing_url"),
                published_at,
                json.dumps(payload, ensure_ascii=True),
            ),
        )
    conn.commit()


def _published_at(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return datetime.now(UTC)


def _compact(value: str) -> str:
    return " ".join(value.split())


if __name__ == "__main__":
    main()
