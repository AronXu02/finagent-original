from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Any

from finagent.packages.orchestrator.models import SourceRef
from finagent.packages.tools.registry import ToolContext
from finagent.packages.tools.schemas import PortfolioMetricsInput, ScenarioAnalysisInput


def _returns(prices: list[float]) -> list[float]:
    return [(prices[idx] / prices[idx - 1]) - 1 for idx in range(1, len(prices)) if prices[idx - 1] != 0]


def _max_drawdown(values: list[float]) -> float:
    if not values:
        return 0.0
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        drawdown = (value / peak) - 1 if peak else 0
        worst = min(worst, drawdown)
    return abs(worst)


def calc_portfolio_metrics(payload: PortfolioMetricsInput, context: ToolContext) -> dict[str, Any]:
    positions = [item.model_dump() for item in payload.positions]
    total_value = sum(item["quantity"] * item["price"] for item in positions)
    histories = [item.get("price_history") or [] for item in positions]
    usable_histories = [history for history in histories if len(history) >= 2]
    if len(usable_histories) != len(positions):
        missing = [
            item["ticker"]
            for item, history in zip(positions, histories, strict=True)
            if len(history) < 2
        ]
        return {
            "status": "error",
            "message": "计算组合风险指标需要历史价格",
            "missing_tickers": missing,
            "sources": [
                SourceRef(
                    source_type="calculation",
                    title="Portfolio metrics input validation",
                    freshness="snapshot",
                )
            ],
        }

    unrealized_pl = sum(item["quantity"] * (item["price"] - item["cost_price"]) for item in positions)
    weights = {
        item["ticker"]: round((item["quantity"] * item["price"]) / total_value, 6)
        for item in positions
    }
    history_length = min(len(history) for history in usable_histories)
    portfolio_curve = []
    for idx in range(history_length):
        day_value = sum(
            item["quantity"] * item["price_history"][-history_length + idx]
            for item in positions
        )
        portfolio_curve.append(day_value)
    rets = _returns(portfolio_curve)
    volatility = pstdev(rets) * math.sqrt(252) if len(rets) > 1 else 0.0
    annual_return = mean(rets) * 252 if rets else 0.0
    sharpe = (annual_return - payload.risk_free_rate) / volatility if volatility else 0.0
    sorted_returns = sorted(rets)
    var_95 = abs(sorted_returns[max(0, int(len(sorted_returns) * 0.05) - 1)]) * total_value if sorted_returns else 0.0
    return {
        "total_value": round(total_value, 2),
        "unrealized_pl": round(unrealized_pl, 2),
        "weights": weights,
        "volatility": round(volatility, 6),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(_max_drawdown(portfolio_curve), 6),
        "var_95": round(var_95, 2),
        "history_points": history_length,
        "sources": [
            SourceRef(
                source_type="calculation",
                title="Historical-price portfolio metrics engine",
                freshness="snapshot",
            )
        ],
    }


def run_scenario_analysis(payload: ScenarioAnalysisInput, context: ToolContext) -> dict[str, Any]:
    shock_map = {
        "rate_hike_25bps": {"Financials": 0.012, "Technology": -0.018, "default": -0.005},
        "rate_cut_25bps": {"Financials": -0.006, "Technology": 0.015, "default": 0.004},
        "usd_strength": {"Financials": 0.002, "Technology": -0.006, "default": -0.003},
        "oil_shock": {"Energy": 0.025, "Financials": -0.004, "default": -0.007},
        "sector_stress": {"Financials": -0.035, "Technology": -0.02, "default": -0.015},
    }
    scenario = shock_map[payload.scenario]
    impacts = []
    total_impact = 0.0
    for position in payload.positions:
        sector = position.sector or "default"
        shock = scenario.get(sector, scenario["default"])
        market_value = position.quantity * position.price
        pnl = market_value * shock
        total_impact += pnl
        impacts.append(
            {
                "ticker": position.ticker,
                "sector": sector,
                "shock": round(shock, 6),
                "estimated_pnl": round(pnl, 2),
            }
        )
    return {
        "scenario": payload.scenario,
        "shock_bps": payload.shock_bps,
        "estimated_total_pnl": round(total_impact, 2),
        "position_impacts": impacts,
        "sources": [
            SourceRef(
                source_type="calculation",
                title="Deterministic scenario analysis engine",
                freshness="snapshot",
            )
        ],
    }
