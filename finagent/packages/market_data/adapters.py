from __future__ import annotations

from datetime import UTC, datetime
import html
from html.parser import HTMLParser
import json
import re
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from finagent.packages.common.config import Settings
from finagent.packages.common.errors import ExternalDependencyError, ForbiddenError, ToolTimeoutError
from finagent.packages.common.store import Store
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


class MarketDataProvider(Protocol):
    def get_realtime_stock(self, payload: TickerMarketInput, context: ToolContext) -> dict[str, Any]: ...

    def get_delayed_quote(self, payload: TickerMarketInput, context: ToolContext) -> dict[str, Any]: ...

    def get_historical_prices(self, payload: HistoricalPricesInput, context: ToolContext) -> dict[str, Any]: ...

    def get_macro_indicator(self, payload: MacroIndicatorInput, context: ToolContext) -> dict[str, Any]: ...

    def fetch_sec_filings(self, payload: SecFilingsInput, context: ToolContext) -> dict[str, Any]: ...

    def fetch_cn_announcements(self, payload: CNAnnouncementsInput, context: ToolContext) -> dict[str, Any]: ...

    def search_research_reports(self, payload: ResearchSearchInput, context: ToolContext) -> dict[str, Any]: ...

    def get_user_portfolio(self, payload: UserPortfolioInput, context: ToolContext) -> dict[str, Any]: ...


class ConfiguredHttpMarketDataProvider:
    """Production adapter for an internal licensed market-data gateway."""

    def __init__(self, settings: Settings, store: Store) -> None:
        self.settings = settings
        self.store = store

    def get_realtime_stock(self, payload: TickerMarketInput, context: ToolContext) -> dict[str, Any]:
        return self._post("/v1/quotes/realtime", payload.model_dump(mode="json"), "quote realtime")

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
        price, observed_at = _latest_history_close(history)
        return {
            "ticker": payload.ticker,
            "market": payload.market,
            "price": price,
            "currency": history.get("currency", _default_currency(payload.market)),
            "freshness": "delayed",
            "source_ts": observed_at or history.get("source_ts"),
            "sources": [
                SourceRef(
                    source_type="market_data",
                    title=f"{payload.ticker} delayed close",
                    published_at=observed_at,
                    fetched_at=_utc_now(),
                    freshness="delayed",
                )
            ],
        }

    def get_historical_prices(self, payload: HistoricalPricesInput, context: ToolContext) -> dict[str, Any]:
        return self._post("/v1/prices/history", payload.model_dump(mode="json"), "historical prices")

    def get_macro_indicator(self, payload: MacroIndicatorInput, context: ToolContext) -> dict[str, Any]:
        return self._post("/v1/macro/indicator", payload.model_dump(mode="json"), "macro indicator")

    def fetch_sec_filings(self, payload: SecFilingsInput, context: ToolContext) -> dict[str, Any]:
        return SecEdgarProvider(self.settings).fetch_sec_filings(payload, context)

    def fetch_cn_announcements(self, payload: CNAnnouncementsInput, context: ToolContext) -> dict[str, Any]:
        return self._post("/v1/announcements/cn", payload.model_dump(mode="json"), "CN announcement")

    def search_research_reports(self, payload: ResearchSearchInput, context: ToolContext) -> dict[str, Any]:
        return self._post("/v1/research/search", payload.model_dump(mode="json"), "research search")

    def get_user_portfolio(self, payload: UserPortfolioInput, context: ToolContext) -> dict[str, Any]:
        snapshot = self.store.get_portfolio_snapshot(
            context.tenant_id,
            context.user_id,
            account_scope=payload.account_scope,
            sector=payload.sector,
        )
        if snapshot is None:
            raise ForbiddenError("当前租户/用户/账户下没有可用的组合快照")
        return {
            **snapshot,
            "source_ts": _parse_datetime(snapshot["snapshot_at"]),
            "sources": [
                SourceRef(
                    source_type="portfolio",
                    title="User portfolio snapshot",
                    published_at=_parse_datetime(snapshot["snapshot_at"]),
                    fetched_at=_utc_now(),
                    freshness="snapshot",
                )
            ],
        }

    def _post(self, path: str, payload: dict[str, Any], label: str) -> dict[str, Any]:
        if not self.settings.market_data_base_url:
            raise ExternalDependencyError("生产行情数据需要配置 FINAGENT_MARKET_DATA_BASE_URL")
        url = self.settings.market_data_base_url.rstrip("/") + path
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.settings.market_data_api_key:
            headers["Authorization"] = f"Bearer {self.settings.market_data_api_key}"
        if self.settings.market_data_sources:
            headers["X-FinAgent-Market-Sources"] = ",".join(self.settings.market_data_sources)
        headers["X-FinAgent-Market-Primary-Source"] = self.settings.market_data_primary_source
        headers["X-FinAgent-Market-Fallback-Source"] = self.settings.market_data_fallback_source
        request = Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        response = _json_request(request, timeout=self.settings.request_timeout_seconds, label=label)
        return _normalize_provider_response(response, default_title=label, default_url=url)


class SecEdgarProvider:
    """SEC EDGAR adapter for auditable filing metadata and extracted filing text."""

    _ticker_cache: dict[str, dict[str, Any]] | None = None

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def fetch_sec_filings(self, payload: SecFilingsInput, context: ToolContext) -> dict[str, Any]:
        if not self.settings.sec_user_agent:
            raise ExternalDependencyError("访问 SEC EDGAR 需要配置 FINAGENT_SEC_USER_AGENT")
        ticker_info = self._lookup_ticker(payload.ticker)
        cik = f"{int(ticker_info['cik_str']):010d}"
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        data = self._get_json(url, "SEC submissions")
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accession_numbers = recent.get("accessionNumber", [])
        primary_documents = recent.get("primaryDocument", [])
        filing_dates = recent.get("filingDate", [])
        for idx, form in enumerate(forms):
            if form != payload.form_type:
                continue
            filing_date = filing_dates[idx]
            if payload.fiscal_year and not filing_date.startswith(str(payload.fiscal_year)):
                continue
            accession = accession_numbers[idx]
            accession_no_dash = accession.replace("-", "")
            primary_doc = primary_documents[idx]
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no_dash}/{primary_doc}"
            document_text = self._get_text(filing_url, "SEC filing document")
            section_text = _extract_filing_section(document_text, payload.section)
            if not section_text:
                raise ExternalDependencyError(
                    f"未找到 {payload.ticker} {payload.form_type} 的 SEC 财报章节：{payload.section}"
                )
            source_ts = _parse_date(filing_date)
            return {
                "ticker": payload.ticker,
                "company": ticker_info.get("title"),
                "cik": cik,
                "form_type": payload.form_type,
                "section": payload.section,
                "filing_date": filing_date,
                "accession_number": accession,
                "primary_document": primary_doc,
                "filing_url": filing_url,
                "section_excerpt": section_text[:4000],
                "highlights": _section_highlights(section_text),
                "source_ts": source_ts,
                "sources": [
                    SourceRef(
                        source_type="filing",
                        title=f"{payload.ticker} {payload.form_type} filed {filing_date}",
                        url=filing_url,
                        published_at=source_ts,
                        fetched_at=_utc_now(),
                        freshness="snapshot",
                    )
                ],
            }
        raise ExternalDependencyError(f"未找到 {payload.ticker} 的 SEC {payload.form_type} 财报")

    def _lookup_ticker(self, ticker: str) -> dict[str, Any]:
        if SecEdgarProvider._ticker_cache is None:
            data = self._get_json("https://www.sec.gov/files/company_tickers.json", "SEC company tickers")
            SecEdgarProvider._ticker_cache = {
                item["ticker"].upper(): item for item in data.values()
            }
        try:
            return SecEdgarProvider._ticker_cache[ticker.upper()]
        except KeyError as exc:
            raise ExternalDependencyError(f"SEC 未找到 ticker：{ticker}") from exc

    def _get_json(self, url: str, label: str) -> dict[str, Any]:
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": self.settings.sec_user_agent or "",
            },
            method="GET",
        )
        return _json_request(request, timeout=self.settings.request_timeout_seconds, label=label)

    def _get_text(self, url: str, label: str) -> str:
        request = Request(
            url,
            headers={
                "Accept": "text/html,application/xhtml+xml,text/plain",
                "User-Agent": self.settings.sec_user_agent or "",
            },
            method="GET",
        )
        return _text_request(request, timeout=self.settings.request_timeout_seconds, label=label)


def build_market_data_provider(settings: Settings, store: Store) -> MarketDataProvider:
    return ConfiguredHttpMarketDataProvider(settings, store)


def _json_request(request: Request, *, timeout: float, label: str) -> dict[str, Any]:
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except TimeoutError as exc:
        raise ToolTimeoutError(f"{label} 请求超时") from exc
    except HTTPError as exc:
        raise ExternalDependencyError(f"{label} 返回 HTTP {exc.code}") from exc
    except URLError as exc:
        raise ExternalDependencyError(f"{label} 不可用：{exc.reason}") from exc
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ExternalDependencyError(f"{label} 返回了无效 JSON") from exc
    if not isinstance(data, dict):
        raise ExternalDependencyError(f"{label} 返回的 JSON 不是对象")
    return data


def _text_request(request: Request, *, timeout: float, label: str, max_bytes: int = 5_000_000) -> str:
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read(max_bytes + 1)
    except TimeoutError as exc:
        raise ToolTimeoutError(f"{label} 请求超时") from exc
    except HTTPError as exc:
        raise ExternalDependencyError(f"{label} 返回 HTTP {exc.code}") from exc
    except URLError as exc:
        raise ExternalDependencyError(f"{label} 不可用：{exc.reason}") from exc
    if len(payload) > max_bytes:
        raise ExternalDependencyError(f"{label} 超出当前解析器支持的大小")
    text = payload.decode("utf-8", errors="replace")
    return _normalize_text(_strip_html(text))


def _normalize_provider_response(data: dict[str, Any], *, default_title: str, default_url: str) -> dict[str, Any]:
    normalized = dict(data)
    sources = normalized.get("sources")
    if not sources:
        normalized["sources"] = [
            SourceRef(
                source_type="market_data",
                title=default_title,
                url=default_url,
                fetched_at=_utc_now(),
                freshness="snapshot",
            )
        ]
    return normalized


def _latest_history_close(history: dict[str, Any]) -> tuple[float, datetime | None]:
    prices = history.get("prices")
    if not isinstance(prices, list) or not prices:
        raise ExternalDependencyError("延迟报价降级需要可用的历史价格")
    latest = prices[-1]
    if isinstance(latest, dict):
        value = latest.get("adjusted_close") or latest.get("close")
        observed = latest.get("date") or latest.get("source_ts") or history.get("source_ts")
    else:
        value = latest
        observed = history.get("source_ts")
    if not isinstance(value, int | float):
        raise ExternalDependencyError("最新历史收盘价不是数值")
    observed_at = _parse_datetime(str(observed)) if observed else None
    return float(value), observed_at


def _default_currency(market: str) -> str:
    return {"CN": "CNY", "HK": "HKD", "FX": "JPY"}.get(market, "USD")


def _parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _utc_now() -> datetime:
    return datetime.now(UTC)


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data)


def _strip_html(value: str) -> str:
    parser = _HTMLTextExtractor()
    try:
        parser.feed(value)
        return html.unescape(" ".join(parser.parts))
    except (ValueError, UnicodeError, html.parser.HTMLParseError, RuntimeError):
        return html.unescape(re.sub(r"<[^>]+>", " ", value))


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _extract_filing_section(document_text: str, section: str) -> str:
    patterns, end_patterns = _section_patterns(section)
    if not patterns:
        return ""
    lowered = document_text.lower()
    starts: list[re.Match[str]] = []
    for pattern in patterns:
        starts.extend(re.finditer(pattern, lowered, re.IGNORECASE))
    starts.sort(key=lambda match: match.start())
    for start in starts:
        end_index = len(document_text)
        for end_pattern in end_patterns:
            end = re.search(end_pattern, lowered[start.end() :], re.IGNORECASE)
            if end:
                end_index = min(end_index, start.end() + end.start())
        excerpt = document_text[start.start() : end_index].strip()
        if len(excerpt) >= 500:
            return excerpt[:20_000]
    if starts:
        return document_text[starts[-1].start() : starts[-1].start() + 20_000].strip()
    return ""


def _section_patterns(section: str) -> tuple[list[str], list[str]]:
    normalized = section.strip().lower().replace("-", "_")
    if normalized == "risk_factors":
        return (
            [r"\bitem\s+1a\.?\s+risk\s+factors\b", r"\bpart\s+ii\s+item\s+1a\.?\s+risk\s+factors\b"],
            [r"\bitem\s+1b\b", r"\bitem\s+2\b", r"\bpart\s+ii\s+item\s+2\b"],
        )
    if normalized in {"mda", "md_a", "management_discussion"}:
        return (
            [r"\bitem\s+7\.?\s+management", r"\bitem\s+2\.?\s+management"],
            [r"\bitem\s+7a\b", r"\bitem\s+8\b", r"\bitem\s+3\b"],
        )
    return ([re.escape(section.lower())], [r"\bitem\s+\d+[a-z]?\b"])


def _section_highlights(section_text: str, limit: int = 3) -> list[str]:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", section_text)
        if len(sentence.strip()) >= 80
    ]
    if not sentences:
        sentences = [section_text[idx : idx + 500].strip() for idx in range(0, min(len(section_text), 1500), 500)]
    return sentences[:limit]
