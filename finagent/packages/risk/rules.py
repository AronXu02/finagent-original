from __future__ import annotations

from hashlib import sha256
import re
from typing import Any

from finagent.packages.orchestrator.models import RiskFlag, RiskReview, SourceRef


DETERMINISTIC_RETURN_PATTERNS = (
    r"\u5fc5\u6da8",
    r"\u80af\u5b9a\u6da8",
    r"\u7a33\u8d5a",
    r"\u4fdd\u672c",
    r"\u65e0\u98ce\u9669",
    r"\u6da8\u505c",
    r"guaranteed",
    r"risk[- ]?free",
    r"sure profit",
    r"must rise",
)

UNSUITABLE_TRADING_PATTERNS = (
    r"\u6ee1\u4ed3",
    r"\u68ad\u54c8",
    r"\u52a0\u6760\u6746",
    r"all[- ]?in",
    r"leverage up",
)

SOURCE_REQUIRED_PATTERNS = (
    r"portfolio",
    r"holding",
    r"position",
    r"quote",
    r"price",
    r"filing",
    r"10-k",
    r"10-q",
    r"fomc",
    r"\u6301\u4ed3",
    r"\u7ec4\u5408",
    r"\u80a1\u4ef7",
    r"\u884c\u60c5",
    r"\u8d22\u62a5",
    r"\u7814\u62a5",
    r"\u4f30\u503c",
    r"\u7f8e\u8054\u50a8",
    r"\u52a0\u606f",
    r"\u964d\u606f",
)

GUARDRAIL_CONTEXT_RE = re.compile(
    r"(不|别|勿|无意|不能|不会|不可|不得|不要|不应|不再|拒绝|避免|禁止|"
    r"无法|不提供|不保证|不承诺|不作为|不是|不等于|"
    r"do not|don't|does not|not|cannot|can't|won't|without|avoid|reject|refuse|no\s+)",
    re.IGNORECASE,
)


class RiskService:
    def __init__(self) -> None:
        self._deterministic = [re.compile(pattern, re.IGNORECASE) for pattern in DETERMINISTIC_RETURN_PATTERNS]
        self._unsuitable = [re.compile(pattern, re.IGNORECASE) for pattern in UNSUITABLE_TRADING_PATTERNS]
        self._source_required = [re.compile(pattern, re.IGNORECASE) for pattern in SOURCE_REQUIRED_PATTERNS]

    def review(
        self,
        *,
        candidate_text: str,
        user_input: str,
        sources: list[SourceRef],
        user_risk_level: str = "balanced",
    ) -> RiskReview:
        flags: list[RiskFlag] = []
        self_reflection = [
            "已检查候选回复和用户请求中是否包含确定性收益承诺。",
            "已检查是否包含满仓、杠杆或直接交易执行指令。",
        ]
        flags.extend(self._match_flags(candidate_text, target="candidate"))
        flags.extend(self._match_flags(user_input, target="user_input"))
        if (
            self._requires_auditable_sources(user_input)
            and not sources
            and not self._looks_like_guardrail(candidate_text)
        ):
            flags.append(
                RiskFlag(
                    code="missing_source",
                    severity="medium",
                    message="金融分析必须包含可审计来源。",
                )
            )
            self_reflection.append("面向数据的金融请求未满足可审计来源要求。")
        else:
            self_reflection.append("可审计来源要求已满足，或当前请求无需该要求。")
        if user_risk_level == "conservative" and any(flag.code == "unsuitable_trading" for flag in flags):
            flags.append(
                RiskFlag(
                    code="suitability_mismatch",
                    severity="high",
                    message="请求的交易风格与保守风险画像冲突。",
                )
            )
        verdict = "pass"
        if any(flag.severity == "high" for flag in flags):
            verdict = "reject" if any(flag.evidence == "user_input" for flag in flags) else "rewrite"
        elif flags:
            verdict = "rewrite"
        suggestion = None
        if verdict in {"rewrite", "reject"}:
            suggestion = "删除确定性收益承诺，避免仓位指令，展示数据来源，并改写为客观筛选或风险分析框架。"
            self_reflection.append("最终判定要求在展示给用户前进行阻断或改写。")
        else:
            self_reflection.append("最终判定可通过，无需合规改写。")
        return RiskReview(
            verdict=verdict,  # type: ignore[arg-type]
            flags=flags,
            rewrite_suggestion=suggestion,
            self_reflection=self_reflection,
            reviewed_text_hash=sha256(
                f"candidate:{candidate_text}\nuser_input:{user_input}".encode("utf-8")
            ).hexdigest(),
        )

    def safe_response(self, *, user_input: str, sources: list[SourceRef] | None = None) -> str:
        return (
            "\u6211\u4e0d\u80fd\u63d0\u4f9b\u786e\u5b9a\u6027\u6536\u76ca\u627f\u8bfa\u3001"
            "\u96c6\u4e2d\u4e0b\u5355\u3001\u6760\u6746\u52a0\u7801\u6216"
            "\u76f4\u63a5\u4ea4\u6613\u6267\u884c\u6307\u4ee4\u3002"
            "\u53ef\u4ee5\u6539\u4e3a\u53ef\u5ba1\u8ba1\u7684\u5206\u6790\u6846\u67b6\uff1a"
            "\u5148\u660e\u786e\u5e02\u573a\u8303\u56f4\u548c\u8d44\u4ea7\u6c60\uff0c"
            "\u6838\u5bf9\u884c\u60c5\u3001\u8d22\u62a5\u3001\u516c\u544a\u548c\u6210\u672c\u7a0e\u8d39\uff0c"
            "\u518d\u6bd4\u8f83\u6ce2\u52a8\u3001\u56de\u64a4\u3001\u6d41\u52a8\u6027\u548c"
            "\u4e2a\u4eba\u98ce\u9669\u627f\u53d7\u80fd\u529b\u3002"
        )

    def rewrite_response(self, *, review: RiskReview, candidate_text: str, user_input: str = "") -> str:
        flag_codes = {flag.code for flag in review.flags}
        if any(flag.severity == "high" for flag in review.flags):
            return (
                "\u539f\u5019\u9009\u7b54\u590d\u89e6\u53d1\u5408\u89c4\u6539\u5199\uff0c"
                "\u4e0d\u80fd\u4f5c\u4e3a\u6700\u7ec8\u8f93\u51fa\u3002"
                + self.safe_response(user_input=candidate_text)
            )
        if "missing_source" in flag_codes:
            topic = self._topic_label(user_input)
            hints = "\u3001".join(self._missing_data_hints(user_input))
            return (
                f"\u8fd9\u4e2a\u95ee\u9898\u6d89\u53ca{topic}\uff0c"
                "\u9700\u8981\u5148\u8865\u9f50\u53ef\u5ba1\u8ba1\u6570\u636e\u540e\u518d\u4e0b\u7ed3\u8bba\u3002"
                f"\u5efa\u8bae\u4f18\u5148\u63d0\u4f9b\u6216\u914d\u7f6e\uff1a{hints}\u3002"
                "\u5728\u6570\u636e\u8865\u9f50\u524d\uff0c\u6211\u53ea\u80fd\u8bf4\u660e\u5206\u6790\u601d\u8def\uff0c"
                "\u4e0d\u628a\u5019\u9009\u5185\u5bb9\u5f53\u4f5c\u6295\u8d44\u7ed3\u8bba\u3002"
            )
        return (
            "\u539f\u5019\u9009\u7b54\u590d\u9700\u8981\u5408\u89c4\u6539\u5199\u540e\u624d\u80fd\u8f93\u51fa\u3002"
            "\u8bf7\u8865\u5145\u53ef\u6838\u9a8c\u6570\u636e\u548c\u660e\u786e\u7684\u98ce\u9669\u8fb9\u754c\u3002"
        )

    def tool_check(self, payload: Any, context: Any) -> dict[str, Any]:
        user_input = payload.user_input or payload.text
        review = self.review(
            candidate_text=payload.text,
            user_input=user_input,
            sources=[] if payload.source_count == 0 else [SourceRef(source_type="external", title="provided source")],
            user_risk_level=payload.user_risk_level,
        )
        return {
            "verdict": review.verdict,
            "flags": [flag.model_dump(mode="json") for flag in review.flags],
            "rewrite_suggestion": review.rewrite_suggestion,
            "self_reflection": review.self_reflection,
            "reviewed_text_hash": review.reviewed_text_hash,
            "user_input_checked": payload.user_input is not None,
        }

    def _match_flags(self, text: str, *, target: str) -> list[RiskFlag]:
        flags: list[RiskFlag] = []
        for pattern in self._deterministic:
            match = pattern.search(text)
            if match and not self._is_guarded_match(text, match.start(), match.end()):
                flags.append(
                    RiskFlag(
                        code="deterministic_return_promise",
                        severity="high",
                        message="不允许作出确定性收益承诺。",
                        evidence=target,
                    )
                )
                break
        for pattern in self._unsuitable:
            match = pattern.search(text)
            if match and not self._is_guarded_match(text, match.start(), match.end()):
                flags.append(
                    RiskFlag(
                        code="unsuitable_trading",
                        severity="high",
                        message="不允许提供满仓或杠杆交易指令。",
                        evidence=target,
                    )
                )
                break
        return flags

    @staticmethod
    def _is_guarded_match(text: str, start: int, end: int) -> bool:
        window = text[max(0, start - 28): min(len(text), end + 8)]
        return bool(GUARDRAIL_CONTEXT_RE.search(window))

    @staticmethod
    def _looks_like_guardrail(text: str) -> bool:
        lowered = text.lower()
        return any(
            token in lowered
            for token in (
                "\u6211\u4e0d\u80fd\u63d0\u4f9b",
                "\u4e0d\u80fd\u4f5c\u4e3a\u6295\u8d44\u7ed3\u8bba",
                "\u89e6\u53d1\u5408\u89c4\u6539\u5199",
                "cannot provide",
            )
        )

    def _requires_auditable_sources(self, user_input: str) -> bool:
        return any(pattern.search(user_input) for pattern in self._source_required)

    @staticmethod
    def _topic_label(user_input: str) -> str:
        lowered = user_input.lower()
        if any(token in lowered for token in ("10-k", "10-q", "filing")) or "\u8d22\u62a5" in user_input:
            return "\u8d22\u62a5\u6216\u516c\u544a\u5206\u6790"
        if any(token in lowered for token in ("quote", "price")) or any(token in user_input for token in ("\u80a1\u4ef7", "\u884c\u60c5")):
            return "\u5b9e\u65f6\u6216\u5ef6\u65f6\u884c\u60c5"
        if any(token in lowered for token in ("portfolio", "holding")) or any(token in user_input for token in ("\u6301\u4ed3", "\u7ec4\u5408")):
            return "\u7ec4\u5408\u6216\u6301\u4ed3\u5206\u6790"
        if any(token in lowered for token in ("fomc", "fed")) or any(token in user_input for token in ("\u7f8e\u8054\u50a8", "\u52a0\u606f", "\u964d\u606f")):
            return "\u5b8f\u89c2\u4e8b\u4ef6\u5206\u6790"
        return "\u9700\u8981\u6570\u636e\u652f\u6491\u7684\u91d1\u878d\u5206\u6790"

    @staticmethod
    def _missing_data_hints(user_input: str) -> list[str]:
        lowered = user_input.lower()
        hints: list[str] = []
        if any(token in lowered for token in ("portfolio", "holding")) or any(token in user_input for token in ("\u6301\u4ed3", "\u7ec4\u5408")):
            hints.append("\u7ec4\u5408\u6301\u4ed3\u548c\u6210\u672c\u4ef7")
        if any(token in lowered for token in ("quote", "price")) or any(token in user_input for token in ("\u80a1\u4ef7", "\u884c\u60c5")):
            hints.append("\u884c\u60c5\u6e90\u6216\u62a5\u4ef7\u7f51\u5173")
        if any(token in lowered for token in ("10-k", "10-q", "filing")) or "\u8d22\u62a5" in user_input:
            hints.append("SEC/EDGAR \u62ab\u9732\u6587\u4ef6\u6216\u516c\u53f8\u516c\u544a")
        if any(token in lowered for token in ("fomc", "fed")) or any(token in user_input for token in ("\u7f8e\u8054\u50a8", "\u52a0\u606f", "\u964d\u606f")):
            hints.append("\u5b8f\u89c2\u6307\u6807\u6765\u6e90")
        if not hints:
            hints.extend(["\u53ef\u9a8c\u8bc1\u7684\u884c\u60c5\u6216\u516c\u544a", "\u5206\u6790\u65f6\u95f4\u8303\u56f4"])
        return hints
