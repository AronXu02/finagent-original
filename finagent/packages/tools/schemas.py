from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TickerMarketInput(ToolInput):
    ticker: str = Field(min_length=1, max_length=24)
    market: Literal["US", "CN", "HK", "FX", "ETF"] = "US"

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if any(char in cleaned for char in (";", "'", '"', "\\", "/", " ")):
            raise ValueError("ticker contains forbidden characters")
        return cleaned


class ForexRateInput(ToolInput):
    base: str = Field(default="USD", min_length=3, max_length=3)
    quote: str = Field(default="JPY", min_length=3, max_length=3)

    @field_validator("base", "quote")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if not cleaned.isalpha():
            raise ValueError("currency code must contain letters only")
        return cleaned


class HistoricalPricesInput(TickerMarketInput):
    period: Literal["1mo", "3mo", "6mo", "1y"] = "6mo"
    frequency: Literal["daily", "weekly", "monthly"] = "daily"
    adjusted: bool = True


class MacroIndicatorInput(ToolInput):
    indicator: str = Field(default="fomc_rate_decision", min_length=2, max_length=64)
    region: Literal["US", "CN", "HK", "GLOBAL"] = "US"
    latest: bool = True

    @model_validator(mode="before")
    @classmethod
    def accept_documented_event_alias(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        event = normalized.pop("event", None)
        if event is not None and "indicator" not in normalized:
            normalized["indicator"] = event
        return normalized


class SecFilingsInput(ToolInput):
    ticker: str = Field(min_length=1, max_length=24)
    form_type: Literal["10-K", "10-Q", "8-K"] = "10-K"
    section: str = Field(default="risk_factors", min_length=2, max_length=64)
    fiscal_year: int | None = Field(default=None, ge=2000, le=2100)

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return value.strip().upper()


class CNAnnouncementsInput(ToolInput):
    ticker: str = Field(min_length=1, max_length=24)
    doc_type: Literal["annual_report", "quarterly_report", "inquiry_letter", "announcement"] = "announcement"
    year: int | None = Field(default=None, ge=2000, le=2100)


class ResearchSearchInput(ToolInput):
    query: str = Field(min_length=2, max_length=256)
    ticker: str | None = Field(default=None, max_length=24)
    top_k: int = Field(default=3, ge=1, le=10)


class UserPortfolioInput(ToolInput):
    account_scope: str | None = Field(default=None, max_length=64)
    sector: str | None = Field(default=None, max_length=64)


class PositionValueInput(ToolInput):
    quantity: float = Field(gt=0)
    price: float = Field(gt=0)
    currency: str = Field(default="USD", min_length=3, max_length=8)


class PositionInput(ToolInput):
    ticker: str = Field(min_length=1, max_length=24)
    market: Literal["US", "CN", "HK", "ETF"] = "US"
    quantity: float = Field(gt=0)
    price: float = Field(gt=0)
    cost_price: float = Field(gt=0)
    sector: str | None = None
    price_history: list[float] = Field(default_factory=list)

    @field_validator("price_history")
    @classmethod
    def price_history_must_be_positive(cls, value: list[float]) -> list[float]:
        if any(item <= 0 for item in value):
            raise ValueError("price_history must contain positive prices")
        return value


class PortfolioMetricsInput(ToolInput):
    positions: list[PositionInput] = Field(min_length=1)
    risk_free_rate: float = Field(default=0.02, ge=0.0, le=0.2)


class ScenarioAnalysisInput(ToolInput):
    positions: list[PositionInput] = Field(min_length=1)
    scenario: Literal["rate_hike_25bps", "rate_cut_25bps", "usd_strength", "oil_shock", "sector_stress"]
    shock_bps: int = Field(default=25, ge=-500, le=500)


class ImpactAnalysisInput(ToolInput):
    macro_event: str = Field(min_length=2, max_length=128)
    sector: str | None = Field(default=None, max_length=64)
    holdings: list[str] = Field(default_factory=list, max_length=50)


class ComplianceCheckInput(ToolInput):
    text: str = Field(min_length=1, max_length=8000)
    user_input: str | None = Field(default=None, max_length=8000)
    source_count: int = Field(default=0, ge=0)
    user_risk_level: Literal["conservative", "balanced", "aggressive"] = "balanced"

    @model_validator(mode="after")
    def text_must_not_be_blank(self) -> "ComplianceCheckInput":
        if not self.text.strip():
            raise ValueError("text must not be blank")
        if self.user_input is not None and not self.user_input.strip():
            raise ValueError("user_input must not be blank when provided")
        return self
