from __future__ import annotations

import unittest

from finagent.packages.risk.rules import RiskService
from finagent.packages.tools.schemas import ComplianceCheckInput


class RiskRulesTest(unittest.TestCase):
    def test_blocks_deterministic_and_all_in_language(self) -> None:
        service = RiskService()
        review = service.review(
            candidate_text="This trade is guaranteed and risk-free.",
            user_input="\u63a8\u8350\u4e00\u53ea\u660e\u5929\u5fc5\u6da8\u7684\u80a1\uff0c\u6211\u60f3\u6ee1\u4ed3",
            sources=[],
        )

        self.assertEqual(review.verdict, "reject")
        self.assertIn("deterministic_return_promise", {flag.code for flag in review.flags})
        self.assertIn("unsuitable_trading", {flag.code for flag in review.flags})

    def test_passes_sourced_neutral_analysis(self) -> None:
        service = RiskService()
        review = service.review(
            candidate_text="Portfolio risk analysis notes drawdown and valuation sensitivity.",
            user_input="Please analyze my portfolio risk.",
            sources=[],
        )

        self.assertEqual(review.verdict, "rewrite")
        self.assertIn("missing_source", {flag.code for flag in review.flags})

    def test_general_education_does_not_require_sources(self) -> None:
        service = RiskService()
        review = service.review(
            candidate_text="\u5206\u6563\u6295\u8d44\u7684\u4f5c\u7528\u662f\u964d\u4f4e\u5355\u4e00\u8d44\u4ea7\u6ce2\u52a8\u5bf9\u7ec4\u5408\u7684\u5f71\u54cd\u3002",
            user_input="\u8bf4\u660e\u4e00\u4e0b\u5206\u6563\u6295\u8d44\u7684\u57fa\u672c\u539f\u7406",
            sources=[],
        )

        self.assertEqual(review.verdict, "pass")

    def test_negated_guardrail_language_is_not_flagged_as_promise(self) -> None:
        service = RiskService()
        review = service.review(
            candidate_text="我的回答不会预设结论，也不能保证无风险或稳赚；会根据当前问题调用模型生成。",
            user_input="你的回答是预设的吗？",
            sources=[],
        )

        self.assertEqual(review.verdict, "pass")
        self.assertNotIn("deterministic_return_promise", {flag.code for flag in review.flags})

    def test_missing_source_rewrite_is_contextual(self) -> None:
        service = RiskService()
        review = service.review(
            candidate_text="Portfolio risk analysis notes drawdown and valuation sensitivity.",
            user_input="\u8bf7\u5206\u6790\u6211\u7684\u6301\u4ed3\u98ce\u9669",
            sources=[],
        )
        rewritten = service.rewrite_response(
            review=review,
            candidate_text="Portfolio risk analysis notes drawdown and valuation sensitivity.",
            user_input="\u8bf7\u5206\u6790\u6211\u7684\u6301\u4ed3\u98ce\u9669",
        )

        self.assertIn("\u7ec4\u5408\u6216\u6301\u4ed3\u5206\u6790", rewritten)
        self.assertIn("\u7ec4\u5408\u6301\u4ed3\u548c\u6210\u672c\u4ef7", rewritten)

    def test_compliance_tool_reviews_original_user_input(self) -> None:
        service = RiskService()
        result = service.tool_check(
            ComplianceCheckInput(
                text="This answer only describes a neutral screening process.",
                user_input="\u63a8\u8350\u4e00\u53ea\u660e\u5929\u5fc5\u6da8\u7684\u80a1\uff0c\u6211\u60f3\u6ee1\u4ed3",
                source_count=1,
            ),
            context=None,
        )

        self.assertEqual(result["verdict"], "reject")
        self.assertTrue(result["user_input_checked"])
        self.assertIn("reviewed_text_hash", result)
        self.assertTrue(result["self_reflection"])
        self.assertIn("user_input", {flag.get("evidence") for flag in result["flags"]})

    def test_rewrite_response_does_not_release_unsafe_candidate(self) -> None:
        service = RiskService()
        candidate = "This trade is guaranteed and risk-free."
        review = service.review(
            candidate_text=candidate,
            user_input="Please explain diversification.",
            sources=[],
        )
        rewritten = service.rewrite_response(review=review, candidate_text=candidate)

        self.assertEqual(review.verdict, "rewrite")
        self.assertNotEqual(rewritten, candidate)
        self.assertIn("\u89e6\u53d1\u5408\u89c4\u6539\u5199", rewritten)
        self.assertNotIn("guaranteed", rewritten.lower())
        self.assertNotIn("risk-free", rewritten.lower())

    def test_guardrail_response_is_not_flagged_for_missing_sources(self) -> None:
        service = RiskService()
        response = service.safe_response(user_input="Please recommend a stock.")
        review = service.review(
            candidate_text=response,
            user_input="Please explain portfolio guardrails.",
            sources=[],
        )

        self.assertEqual(review.verdict, "pass")

    def test_review_records_self_reflection_steps(self) -> None:
        service = RiskService()
        review = service.review(
            candidate_text="Neutral educational answer.",
            user_input="Please explain diversification.",
            sources=[],
        )

        self.assertGreaterEqual(len(review.self_reflection), 3)
        self.assertTrue(any("确定性收益承诺" in step for step in review.self_reflection))


if __name__ == "__main__":
    unittest.main()
