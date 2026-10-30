from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable


@dataclass(frozen=True)
class AssetTarget:
    ticker: str
    market: str
    canonical_name: str


@dataclass(frozen=True)
class AssetAlias:
    alias: str
    target: AssetTarget


_BUILTIN_ALIAS_ROWS: tuple[tuple[AssetTarget, tuple[str, ...]], ...] = (
    (AssetTarget("AAPL", "US", "Apple Inc."), ("apple", "apple inc", "\u82f9\u679c", "\u82f9\u679c\u516c\u53f8")),
    (AssetTarget("MSFT", "US", "Microsoft"), ("microsoft", "microsoft corp", "\u5fae\u8f6f")),
    (AssetTarget("NVDA", "US", "NVIDIA"), ("nvidia", "\u82f1\u4f1f\u8fbe", "\u8f89\u8fbe")),
    (AssetTarget("TSLA", "US", "Tesla"), ("tesla", "\u7279\u65af\u62c9")),
    (AssetTarget("JPM", "US", "JPMorgan Chase"), ("jpmorgan", "jpmorgan chase", "\u6469\u6839\u5927\u901a")),
    (AssetTarget("BAC", "US", "Bank of America"), ("bank of america", "\u7f8e\u56fd\u94f6\u884c")),
    (AssetTarget("GOOGL", "US", "Alphabet"), ("alphabet", "google", "\u8c37\u6b4c")),
    (AssetTarget("AMZN", "US", "Amazon"), ("amazon", "\u4e9a\u9a6c\u900a")),
    (AssetTarget("META", "US", "Meta Platforms"), ("meta", "facebook", "\u8138\u4e66")),
    (AssetTarget("BABA", "US", "Alibaba"), ("alibaba us", "baba", "\u963f\u91cc\u7f8e\u80a1")),
    (AssetTarget("600519.SH", "CN", "\u8d35\u5dde\u8305\u53f0"), ("\u8d35\u5dde\u8305\u53f0", "\u8305\u53f0")),
    (AssetTarget("000001.SZ", "CN", "\u5e73\u5b89\u94f6\u884c"), ("\u5e73\u5b89\u94f6\u884c",)),
    (AssetTarget("00700.HK", "HK", "Tencent Holdings"), ("tencent", "\u817e\u8baf", "\u817e\u8baf\u63a7\u80a1")),
    (AssetTarget("09988.HK", "HK", "Alibaba HK"), ("alibaba hk", "\u963f\u91cc\u6e2f\u80a1", "\u963f\u91cc\u5df4\u5df4\u6e2f\u80a1")),
)


def resolve_asset_alias(text: str, *, markets: Iterable[str] | None = None) -> tuple[str, str] | None:
    allowed_markets = {market.upper() for market in markets} if markets is not None else None
    normalized_text = _normalize_alias(text)
    if not normalized_text:
        return None
    for item in _alias_index():
        target = item.target
        if allowed_markets is not None and target.market not in allowed_markets:
            continue
        if item.alias and item.alias in normalized_text:
            return target.ticker, target.market
    return None


def clear_asset_alias_cache() -> None:
    _alias_index.cache_clear()


@lru_cache(maxsize=1)
def _alias_index() -> tuple[AssetAlias, ...]:
    aliases = [
        AssetAlias(alias=_normalize_alias(alias), target=target)
        for target, alias_values in _BUILTIN_ALIAS_ROWS
        for alias in alias_values
    ]
    aliases.extend(_load_external_aliases())
    return tuple(sorted(aliases, key=lambda item: len(item.alias), reverse=True))


def _load_external_aliases() -> list[AssetAlias]:
    path = os.getenv("FINAGENT_ASSET_ALIAS_FILE")
    if not path:
        return []
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"Unable to read FINAGENT_ASSET_ALIAS_FILE: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"FINAGENT_ASSET_ALIAS_FILE is not valid JSON: {exc}") from exc
    rows = payload if isinstance(payload, list) else _dict_payload_to_rows(payload)
    aliases: list[AssetAlias] = []
    for row in rows:
        parsed = _parse_alias_row(row)
        if parsed:
            aliases.extend(parsed)
    return aliases


def _dict_payload_to_rows(payload: Any) -> list[Any]:
    if not isinstance(payload, dict):
        return []
    rows = []
    for alias, target in payload.items():
        if isinstance(target, dict):
            rows.append({**target, "aliases": [alias, *target.get("aliases", [])]})
        elif isinstance(target, str):
            ticker, _, market = target.partition(":")
            rows.append({"ticker": ticker, "market": market or "US", "aliases": [alias]})
    return rows


def _parse_alias_row(row: Any) -> list[AssetAlias]:
    if not isinstance(row, dict):
        return []
    ticker = str(row.get("ticker", "")).strip().upper()
    market = str(row.get("market", "")).strip().upper()
    if not ticker or not market:
        return []
    canonical_name = str(row.get("canonical_name") or row.get("name") or ticker)
    aliases = row.get("aliases", [])
    if isinstance(aliases, str):
        aliases = [aliases]
    if not isinstance(aliases, list):
        return []
    target = AssetTarget(ticker=ticker, market=market, canonical_name=canonical_name)
    return [
        AssetAlias(alias=_normalize_alias(str(alias)), target=target)
        for alias in aliases
        if str(alias).strip()
    ]


def _normalize_alias(value: str) -> str:
    return re.sub(r"[\s\-_.,:;()\[\]{}]+", "", value.lower())
