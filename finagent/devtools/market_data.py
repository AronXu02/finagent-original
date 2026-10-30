from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from finagent.packages.common.errors import ExternalDependencyError, ForbiddenError, ToolTimeoutError
from finagent.packages.orchestrator.models import SourceRef
from finagent.packages.tools.registry import ToolContext
from finagent.packages.tools.schemas import (
    CNAnnouncementsInput,
    HistoricalPricesInput,
    MacroIndicatorInput,
    ResearchSearchInput,
    SecFilingsInput,
    TickerMarketInput,
    UserPortfolioInput,
)


class InMemoryMarketDataProvider:
    now = datetime(2026, 6, 1, 9, 30, tzinfo=UTC)

    quotes = {
        ("AAPL", "US"): {"price": 193.42, "currency": "USD", "exchange": "NASDAQ", "change_percent": 0.74},
        ("NVDA", "US"): {"price": 122.85, "currency": "USD", "exchange": "NASDAQ", "change_percent": -0.31},
        ("JPM", "US"): {"price": 204.1, "currency": "USD", "exchange": "NYSE", "change_percent": 0.21},
        ("BAC", "US"): {"price": 39.6, "currency": "USD", "exchange": "NYSE", "change_percent": -0.12},
        ("600519.SH", "CN"): {"price": 1512.4, "currency": "CNY", "exchange": "SSE", "change_percent": 0.42},
        ("USDJPY", "FX"): {"price": 156.82, "currency": "JPY", "exchange": "FX", "change_percent": 0.08},
    }

    def get_realtime_stock(self, payload: TickerMarketInput, context: ToolContext) -> dict[str, Any]:
        if payload.ticker == "TIMEOUT":
            raise ToolTimeoutError("主行情提供方请求超时")
        quote = self.quotes.get((payload.ticker, payload.market))
        if not quote:
            raise ExternalDependencyError(f"未配置 {payload.ticker}.{payload.market} 的行情数据")
        return {
            "ticker": payload.ticker,
            "market": payload.market,
            **quote,
            "source_ts": self.now,
            "sources": [self._source(f"{payload.ticker} realtime quote", "realtime")],
        }

    def get_delayed_quote(self, payload: TickerMarketInput, context: ToolContext) -> dict[str, Any]:
        history = self.get_historical_prices(
            HistoricalPricesInput(
                ticker=payload.ticker,
                market=payload.market,
                period="1mo",
                frequency="daily",
                adjusted=True,
            ),
            context,
        )
        prices = history["prices"]
        latest = prices[-1]
        if isinstance(latest, dict):
            price = latest.get("adjusted_close") or latest.get("close")
        else:
            price = latest
        return {
            "ticker": payload.ticker,
            "market": payload.market,
            "price": float(price),
            "currency": self.quotes.get((payload.ticker, payload.market), {"currency": "USD"})["currency"],
            "freshness": "delayed",
            "source_ts": self.now - timedelta(days=1),
            "sources": [self._source(f"{payload.ticker} delayed close", "delayed")],
        }

    def get_historical_prices(self, payload: HistoricalPricesInput, context: ToolContext) -> dict[str, Any]:
        base = self.quotes.get((payload.ticker, payload.market), {"price": 100.0})["price"]
        prices = [round(base * (1 + ((idx - 10) / 1000)), 2) for idx in range(20)]
        return {
            "ticker": payload.ticker,
            "market": payload.market,
            "period": payload.period,
            "frequency": payload.frequency,
            "adjusted": payload.adjusted,
            "prices": prices,
            "source_ts": self.now,
            "sources": [self._source(f"{payload.ticker} historical prices", "snapshot")],
        }

    def get_macro_indicator(self, payload: MacroIndicatorInput, context: ToolContext) -> dict[str, Any]:
        indicator = payload.indicator.lower()
        if "fx" in indicator or "usd" in indicator or "jpy" in indicator:
            data = {"pair": "USD/JPY", "rate": 156.82, "change_percent": 0.08}
            title = "USD/JPY FX snapshot"
        elif "cpi" in indicator:
            data = {"indicator": "US_CPI_YOY", "value": 3.2, "unit": "%", "period": "2026-05"}
            title = "US CPI snapshot"
        else:
            data = {
                "event": "fomc_rate_decision",
                "decision": "+25bps",
                "target_range": "5.25%-5.50%",
                "statement_summary": "Inflation remains above target; credit sensitivity should be monitored.",
            }
            title = "FOMC rate decision snapshot"
        return {
            **data,
            "region": payload.region,
            "source_ts": self.now,
            "sources": [self._source(title, "snapshot")],
        }

    def fetch_sec_filings(self, payload: SecFilingsInput, context: ToolContext) -> dict[str, Any]:
        return {
            "ticker": payload.ticker,
            "form_type": payload.form_type,
            "section": payload.section,
            "fiscal_year": payload.fiscal_year or 2025,
            "highlights": [
                "Demand concentration and supply constraints remain material business risks.",
                "Export controls and customer concentration may affect revenue visibility.",
                "Valuation sensitivity is elevated when rates and growth assumptions move together.",
            ],
            "source_ts": self.now - timedelta(days=3),
            "sources": [
                SourceRef(
                    source_type="filing",
                    title=f"{payload.ticker} {payload.form_type} {payload.section}",
                    url=f"memory://sec-edgar/{payload.ticker}/{payload.form_type}",
                    published_at=self.now - timedelta(days=3),
                    fetched_at=self.now,
                    freshness="snapshot",
                )
            ],
        }

    def fetch_cn_announcements(self, payload: CNAnnouncementsInput, context: ToolContext) -> dict[str, Any]:
        return {
            "ticker": payload.ticker,
            "doc_type": payload.doc_type,
            "year": payload.year or 2025,
            "highlights": ["Announcement parsed from controlled test data.", "No trading instruction is generated."],
            "source_ts": self.now - timedelta(days=1),
            "sources": [self._source(f"{payload.ticker} announcement snapshot", "snapshot")],
        }

    def search_research_reports(self, payload: ResearchSearchInput, context: ToolContext) -> dict[str, Any]:
        snippets = [
            {
                "title": "Bank net interest margin sensitivity",
                "snippet": "Rate hikes may support net interest margins but can pressure credit quality.",
            },
            {
                "title": "Portfolio risk note",
                "snippet": "Concentration and drawdown thresholds should be checked before allocation changes.",
            },
        ][: payload.top_k]
        return {
            "query": payload.query,
            "results": snippets,
            "source_ts": self.now - timedelta(hours=6),
            "sources": [self._source("research report index", "snapshot")],
        }

    def web_search(self, payload: ResearchSearchInput, context: ToolContext) -> dict[str, Any]:
        result = self.search_research_reports(payload, context)
        result["search_type"] = "web"
        result["sources"] = [self._source("cleaned web/news search", "snapshot")]
        return result

    def get_user_portfolio(self, payload: UserPortfolioInput, context: ToolContext) -> dict[str, Any]:
        if payload.account_scope and payload.account_scope not in {"default", "acct_default"}:
            raise ForbiddenError("account_scope 不属于当前用户")
        holdings = [
            {
                "ticker": "JPM",
                "market": "US",
                "name": "JPMorgan Chase",
                "quantity": 120,
                "price": 204.1,
                "cost_price": 172.4,
                "sector": "Financials",
            },
            {
                "ticker": "BAC",
                "market": "US",
                "name": "Bank of America",
                "quantity": 350,
                "price": 39.6,
                "cost_price": 33.2,
                "sector": "Financials",
            },
            {
                "ticker": "AAPL",
                "market": "US",
                "name": "Apple",
                "quantity": 40,
                "price": 193.42,
                "cost_price": 150.0,
                "sector": "Technology",
            },
        ]
        if payload.sector:
            holdings = [item for item in holdings if item["sector"].lower() == payload.sector.lower()]
        total_value = round(sum(item["quantity"] * item["price"] for item in holdings), 2)
        return {
            "account_id": "acct_default",
            "currency": "USD",
            "snapshot_at": self.now,
            "holdings": holdings,
            "total_value": total_value,
            "source_ts": self.now,
            "sources": [
                SourceRef(
                    source_type="portfolio",
                    title="User portfolio snapshot",
                    published_at=self.now,
                    fetched_at=self.now,
                    freshness="snapshot",
                )
            ],
        }

    def _source(self, title: str, freshness: str) -> SourceRef:
        return SourceRef(
            source_type="test_data",
            title=title,
            url="memory://market-data",
            published_at=self.now,
            fetched_at=self.now,
            freshness=freshness,  # type: ignore[arg-type]
        )
