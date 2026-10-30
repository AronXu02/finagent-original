from __future__ import annotations

import json
import re
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from finagent.packages.common.config import Settings
from finagent.packages.common.errors import ExternalDependencyError
from finagent.packages.orchestrator.models import (
    AgentState,
    IntentType,
    InterruptionInfo,
    RouteStep,
    TaskFrame,
)


INTERRUPT_RE = re.compile(r"(\u7b49\u7b49|\u5148\u522b\u7ba1|\u6362\u4e2a\u95ee\u9898|hold on|wait)", re.IGNORECASE)
HIGH_RISK_RE = re.compile(r"(\u5fc5\u6da8|\u80af\u5b9a\u6da8|\u6da8\u505c|\u6ee1\u4ed3|\u68ad\u54c8|\u7a33\u8d5a|all[- ]?in|guaranteed)", re.IGNORECASE)
PREFERENCE_RE = re.compile(
    r"(preference|risk tolerance|drawdown tolerance|"
    r"\u504f\u597d|\u4e0d\u4e70|\u4e0d\u78b0|\u907f\u514d|"
    r"\u6700\u5927\u56de\u64a4.{0,10}(\u4e0d\u8d85\u8fc7|\u4e0d\u80fd\u8d85\u8fc7|\u63a7\u5236|"
    r"\u5bb9\u5fcd|\u627f\u53d7))",
    re.IGNORECASE,
)
RELEVANCE_FEEDBACK_RE = re.compile(
    r"(\u4e0d\u592a?\u76f8\u5173|\u7b54\u975e\u6240\u95ee|\u6ca1\u56de\u7b54|"
    r"\u6ca1\u7b54\u5230|\u4e0d\u662f\u6211\u60f3\u95ee|irrelevant|not relevant|off[- ]?topic)",
    re.IGNORECASE,
)
CAPABILITY_QUERY_RE = re.compile(
    r"(\u4f60.*(\u80fd|\u4f1a|\u53ef\u4ee5).{0,8}(\u56de\u7b54|\u505a|\u5e2e)|"
    r"(\u80fd|\u4f1a|\u53ef\u4ee5).{0,8}(\u56de\u7b54|\u505a|\u5e2e).{0,8}\u4ec0\u4e48|"
    r"\u4f60.*(\u65e0\u6cd5|\u4e0d\u80fd|\u56de\u7b54\u4e0d\u4e86)|"
    r"\u4f60\u652f\u6301\u4ec0\u4e48|"
    r"what can you|capabilit(?:y|ies)|help me with)",
    re.IGNORECASE,
)
FOLLOWUP_PREFIXES = ("那", "那个", "这个", "这", "它", "继续", "然后")
ENGLISH_FOLLOWUP_RE = re.compile(r"^(what about|how about|then)\b", re.IGNORECASE)
PORTFOLIO_CONTEXT_RE = re.compile(
    r"(portfolio|holding|holdings|drawdown|var|exposure|"
    r"组合|持仓|回撤|波动|风险敞口|权重|组合风险)",
    re.IGNORECASE,
)
FOLLOWUP_ANALYSIS_RE = re.compile(r"(分析|评估|报告|analysis|analy[sz]e|assess)", re.IGNORECASE)


class IntentClassifier(Protocol):
    model_name: str

    def classify(self, text: str) -> tuple[IntentType, float]: ...


class LightweightIntentClassifier:
    model_name = "rule-bert-compatible-intent-classifier"

    def classify(self, text: str) -> tuple[IntentType, float]:
        lowered = text.lower()
        if HIGH_RISK_RE.search(text):
            return IntentType.HIGH_RISK_REQUEST, 0.99
        if RELEVANCE_FEEDBACK_RE.search(text):
            return IntentType.CLARIFICATION, 0.96
        if CAPABILITY_QUERY_RE.search(text):
            return IntentType.GENERAL, 0.96
        if any(token in lowered for token in ("10-k", "10-q", "filing", "sec")) or any(
            token in text for token in ("\u8d22\u62a5", "\u516c\u544a", "\u5e74\u62a5", "\u5b63\u62a5", "\u95ee\u8be2")
        ):
            return IntentType.REPORT_ANALYSIS, 0.92
        macro_tokens = (
            "fomc",
            "rate hike",
            "rate cut",
            "fed",
            "cpi",
            "inflation",
            "pmi",
            "gdp",
            "nonfarm",
            "unemployment",
            "crude",
            "brent",
            "oil shock",
            "strong dollar",
            "usd strength",
        )
        macro_zh_tokens = (
            "\u7f8e\u8054\u50a8",
            "\u52a0\u606f",
            "\u964d\u606f",
            "\u901a\u80c0",
            "\u7269\u4ef7",
            "\u91c7\u8d2d\u7ecf\u7406",
            "\u7ecf\u6d4e\u589e\u957f",
            "\u975e\u519c",
            "\u5931\u4e1a",
            "\u5c31\u4e1a",
            "\u539f\u6cb9",
            "\u6cb9\u4ef7",
            "\u7f8e\u5143\u8d70\u5f3a",
        )
        if any(token in lowered for token in macro_tokens) or any(token in text for token in macro_zh_tokens):
            if any(token in lowered for token in ("portfolio", "holding")) or "\u6301\u4ed3" in text:
                return IntentType.PORTFOLIO_IMPACT, 0.94
            return IntentType.MACRO_EVENT, 0.86
        if PREFERENCE_RE.search(text):
            return IntentType.PREFERENCE_UPDATE, 0.9
        if any(token in lowered for token in ("portfolio", "holding", "holdings", "drawdown", "var")) or any(token in text for token in ("\u7ec4\u5408", "\u6301\u4ed3", "\u56de\u64a4")):
            return IntentType.PORTFOLIO_RISK, 0.88
        if any(token in lowered for token in ("price", "quote", "usd/jpy", "jpy", "shares")) or any(
            token in text for token in ("\u80a1\u4ef7", "\u884c\u60c5", "\u6c47\u7387", "\u65e5\u5143", "\u5e02\u503c", "\u503c\u591a\u5c11")
        ):
            return IntentType.MARKET_QUOTE, 0.86
        if _looks_like_followup(text):
            return IntentType.CLARIFICATION, 0.52
        return IntentType.GENERAL, 0.55


class HttpBertIntentClassifier:
    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key: str | None = None,
        timeout_seconds: float = 3.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def classify(self, text: str) -> tuple[IntentType, float]:
        request = Request(
            f"{self.base_url}/v1/intent/classify",
            data=json.dumps({"text": text, "model": self.model_name}).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise ExternalDependencyError(f"BERT 意图分类网关返回 HTTP {exc.code}") from exc
        except (OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ExternalDependencyError(f"BERT 意图分类网关不可用：{exc}") from exc
        if not isinstance(payload, dict):
            raise ExternalDependencyError("BERT 意图分类网关返回的 JSON 不是对象")
        raw_intent = payload.get("intent") or payload.get("label")
        if not isinstance(raw_intent, str):
            raise ExternalDependencyError("BERT 意图分类网关缺少 intent/label 字段")
        try:
            intent = IntentType(raw_intent)
        except ValueError as exc:
            raise ExternalDependencyError(f"BERT 意图分类网关返回了未知意图：{raw_intent}") from exc
        confidence = payload.get("confidence", payload.get("score", 0.0))
        if not isinstance(confidence, int | float):
            confidence = 0.0
        return intent, max(0.0, min(float(confidence), 1.0))

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


class HybridIntentClassifier:
    _lightweight: LightweightIntentClassifier | None = None

    def __init__(self, primary: IntentClassifier, fallback: IntentClassifier | None = None) -> None:
        self.primary = primary
        self.fallback = fallback
        self.model_name = primary.model_name

    def classify(self, text: str) -> tuple[IntentType, float]:
        if HybridIntentClassifier._lightweight is None:
            HybridIntentClassifier._lightweight = LightweightIntentClassifier()
        rule_intent, rule_confidence = HybridIntentClassifier._lightweight.classify(text)
        if rule_confidence >= 0.9 or rule_intent in {
            IntentType.HIGH_RISK_REQUEST,
            IntentType.PREFERENCE_UPDATE,
        }:
            return rule_intent, rule_confidence
        try:
            intent, confidence = self.primary.classify(text)
        except ExternalDependencyError:
            if self.fallback is None:
                raise
            return self.fallback.classify(text)
        return intent, confidence


def build_intent_classifier(settings: Settings) -> IntentClassifier:
    fallback = LightweightIntentClassifier()
    if settings.runtime_mode == "test" or settings.intent_classifier_backend == "rules":
        return fallback
    if not settings.intent_classifier_base_url:
        raise ExternalDependencyError("BERT 意图分类需要配置 FINAGENT_INTENT_CLASSIFIER_BASE_URL")
    return HybridIntentClassifier(
        HttpBertIntentClassifier(
            base_url=settings.intent_classifier_base_url,
            api_key=settings.intent_classifier_api_key,
            model_name=settings.intent_classifier_model,
            timeout_seconds=settings.intent_classifier_timeout_seconds,
        ),
        fallback=fallback,
    )


class UserProxy:
    def __init__(self, intent_classifier: IntentClassifier | None = None) -> None:
        self.intent_classifier = intent_classifier or LightweightIntentClassifier()

    def plan(self, state: AgentState, user_input: str) -> AgentState:
        interrupt_signal = bool(INTERRUPT_RE.search(user_input) and state.active_task)
        if interrupt_signal:
            intent, confidence = LightweightIntentClassifier().classify(user_input)
        else:
            intent, confidence = self.classify(user_input)
        if _looks_like_followup(user_input):
            if state.active_task and intent == IntentType.CLARIFICATION:
                intent = state.active_task.intent
                confidence = max(confidence, 0.72)
            elif not state.active_task:
                inferred = _infer_followup_intent_from_context(state, user_input)
                if inferred is not None:
                    intent = inferred
                    confidence = max(confidence, 0.78)
                elif intent == IntentType.CLARIFICATION:
                    confidence = min(confidence, 0.58)
        if interrupt_signal and state.active_task:
            state.task_stack.append(state.active_task)
            state.interruption = InterruptionInfo(
                interrupted_task_id=state.active_task.task_id,
                new_intent=intent,
                reason="interrupt_signal_detected",
            )
        state.current_intent = intent
        state.intent_confidence = confidence
        state.active_task = TaskFrame(task_id=state.task_id, intent=intent, description=user_input[:160])
        state.route_plan = self._route_for(intent, user_input)
        return state

    def classify(self, text: str) -> tuple[IntentType, float]:
        return self.intent_classifier.classify(text)

    def _route_for(self, intent: IntentType, user_input: str = "") -> list[RouteStep]:
        if intent == IntentType.PORTFOLIO_IMPACT:
            return [
                RouteStep(node="macro_agent", reason="Assess macro event and transmission path", tools=["macro_indicator", "web_search"]),
                RouteStep(node="quant_agent", reason="Read holdings and run scenario impact", tools=["get_user_portfolio", "get_historical_prices", "calc_portfolio_metrics", "run_scenario_analysis", "analyze_impact"]),
                RouteStep(node="critic", reason="Mandatory pre-output risk review", tools=["compliance_rule_check"]),
                RouteStep(node="final_builder", reason="Format reviewed answer", tools=[]),
            ]
        if intent == IntentType.MACRO_EVENT:
            return [
                RouteStep(node="macro_agent", reason="Assess macro event and related research/news", tools=["macro_indicator", "web_search"]),
                RouteStep(node="critic", reason="Mandatory pre-output risk review", tools=["compliance_rule_check"]),
                RouteStep(node="final_builder", reason="Format reviewed answer", tools=[]),
            ]
        if intent == IntentType.REPORT_ANALYSIS:
            report_tools = ["fetch_cn_announcements"] if _looks_like_cn_report(user_input) else ["fetch_sec_filings"]
            return [
                RouteStep(node="report_agent", reason="Fetch filing/announcement section and summarize risks", tools=report_tools),
                RouteStep(node="critic", reason="Mandatory pre-output risk review", tools=["compliance_rule_check"]),
                RouteStep(node="final_builder", reason="Format reviewed answer", tools=[]),
            ]
        if intent == IntentType.PORTFOLIO_RISK:
            return [
                RouteStep(node="quant_agent", reason="Compute portfolio risk metrics", tools=["get_user_portfolio", "get_historical_prices", "calc_portfolio_metrics"]),
                RouteStep(node="critic", reason="Mandatory pre-output risk review", tools=["compliance_rule_check"]),
                RouteStep(node="final_builder", reason="Format reviewed answer", tools=[]),
            ]
        if intent == IntentType.MARKET_QUOTE:
            market_tools = _market_tools_for(user_input)
            return [
                RouteStep(node="market_agent", reason="Fetch market quote, FX rate, or position value", tools=market_tools),
                RouteStep(node="critic", reason="Mandatory pre-output risk review", tools=["compliance_rule_check"]),
                RouteStep(node="final_builder", reason="Format reviewed answer", tools=[]),
            ]
        if intent == IntentType.HIGH_RISK_REQUEST:
            return [
                RouteStep(node="router", reason="High-risk request should skip recommendation generation", tools=[]),
                RouteStep(node="critic", reason="Mandatory rejection or rewrite", tools=["compliance_rule_check"]),
                RouteStep(node="final_builder", reason="Format safe refusal", tools=[]),
            ]
        if intent == IntentType.PREFERENCE_UPDATE:
            return [
                RouteStep(node="memory_agent", reason="Store and confirm user preference constraints", tools=[]),
                RouteStep(node="critic", reason="Mandatory pre-output risk review", tools=["compliance_rule_check"]),
                RouteStep(node="final_builder", reason="Format memory update confirmation", tools=[]),
            ]
        if intent == IntentType.CLARIFICATION:
            return [
                RouteStep(node="router", reason="Ask for clarification before tool calls", tools=[]),
                RouteStep(node="critic", reason="Mandatory pre-output risk review", tools=["compliance_rule_check"]),
                RouteStep(node="final_builder", reason="Format clarification question", tools=[]),
            ]
        return [
            RouteStep(node="router", reason="General analysis with memory context", tools=[]),
            RouteStep(node="critic", reason="Mandatory pre-output risk review", tools=["compliance_rule_check"]),
            RouteStep(node="final_builder", reason="Format reviewed answer", tools=[]),
        ]


def _looks_like_followup(text: str) -> bool:
    stripped = text.strip()
    lowered = stripped.lower()
    return (
        stripped.startswith(FOLLOWUP_PREFIXES)
        or bool(ENGLISH_FOLLOWUP_RE.search(stripped))
        or lowered in {"that one", "this one", "continue"}
    )


def _infer_followup_intent_from_context(state: AgentState, text: str) -> IntentType | None:
    if not FOLLOWUP_ANALYSIS_RE.search(text):
        return None
    recent_context = " ".join(
        [
            state.short_term_summary or "",
            *[message.content for message in state.messages[-6:]],
        ]
    )
    if PORTFOLIO_CONTEXT_RE.search(recent_context):
        return IntentType.PORTFOLIO_RISK
    return None


def _looks_like_cn_report(text: str) -> bool:
    return bool(re.search(r"(?<!\d)\d{5,6}(?:\.(?:SH|SZ))?(?!\d)", text.upper())) or any(
        token in text for token in ("公告", "年报", "季报", "问询")
    )


def _market_tools_for(text: str) -> list[str]:
    lowered = text.lower()
    if "jpy" in lowered or "汇率" in text or "日元" in text:
        return ["get_forex_rate"]
    tools = ["get_realtime_stock"]
    if re.search(r"(?<![\d.])\d+(?:\.\d+)?\s*(?:shares?|股)", text, re.IGNORECASE):
        tools.append("calc_position_value")
    return tools
