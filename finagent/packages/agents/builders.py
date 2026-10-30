from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Protocol

from finagent.packages.common.errors import ExternalDependencyError
from finagent.packages.orchestrator.models import AgentState, FinalResponse, Message, SourceRef


RECENT_CONTEXT_MESSAGES = 10


class AnswerLLM(Protocol):
    model: str
    provider: str

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        ...


@dataclass(frozen=True)
class CandidateResponse:
    answer: str
    sources: list[SourceRef]
    llm_provider: str = "ollama"
    llm_model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


class CandidateResponseBuilder:
    def __init__(self, llm_client: AnswerLLM | None = None) -> None:
        self.llm_client = llm_client

    def build(self, state: AgentState, user_input: str) -> CandidateResponse:
        if not self.llm_client:
            raise ExternalDependencyError("生成回答需要配置可用的 LLM 客户端。")

        sections, sources = self._context_sections(state, user_input=user_input)
        system_prompt = self._system_prompt()
        user_prompt = self._user_prompt(user_input=user_input, sections=sections, sources=sources)
        answer = self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return CandidateResponse(
            answer=answer,
            sources=sources,
            llm_provider=getattr(self.llm_client, "provider", "ollama"),
            llm_model=self.llm_client.model,
            prompt_tokens=_estimate_tokens(system_prompt) + _estimate_tokens(user_prompt),
            completion_tokens=_estimate_tokens(answer),
        )

    @staticmethod
    def _context_sections(state: AgentState, *, user_input: str = "") -> tuple[list[str], list[SourceRef]]:
        sections: list[str] = []
        all_sources: list[SourceRef] = []
        if state.short_term_summary:
            sections.append("短期对话记忆（仅作背景，不是当前任务）：\n" + state.short_term_summary)
        recent_messages = CandidateResponseBuilder._recent_prior_messages(state.messages)
        if recent_messages:
            message_lines = [
                f"- {CandidateResponseBuilder._role_label(message.role)}：{CandidateResponseBuilder._compact_content(message.content)}"
                for message in recent_messages
            ]
            sections.append("历史对话记忆（仅作背景，不是当前问题）：\n" + "\n".join(message_lines))
        if state.memory_context.top_memories:
            qualified = [
                item for item in state.memory_context.top_memories
                if item.decay_weight >= 0.3 and item.importance >= 0.3
            ]
            memory_items = qualified[:3] if qualified else state.memory_context.top_memories[:3]
            memory_lines = [f"- {item.content}" for item in memory_items]
            sections.append("长期记忆约束（仅用于适配当前回答）：\n" + "\n".join(memory_lines))
        if state.memory_context.conflicts:
            conflict_lines = [
                f"- 冲突类型：{conflict.get('type', '未知')}，字段：{conflict.get('field', '未知')}，涉及的记忆ID：{conflict.get('memory_ids', [])}"
                for conflict in state.memory_context.conflicts
            ]
            sections.append(
                "注意：检测到用户约束冲突，可能意味着用户偏好发生了变化或表达不一致。"
                "请在回答中提示用户澄清，并在冲突解决前按更保守的一方执行：\n"
                + "\n".join(conflict_lines)
            )
        for output in state.agent_outputs.values():
            all_sources.extend(output.sources)
            sources = [source.model_dump(mode="json") for source in output.sources]
            finding_lines = "\n".join(f"- {finding}" for finding in output.findings)
            metadata = json.dumps(output.metadata, ensure_ascii=False, indent=2, default=str) if output.metadata else "{}"
            sections.append(
                f"{output.agent_name}: {output.summary}\n"
                f"{finding_lines}\n"
                f"元数据：{metadata}\n"
                f"来源：{sources if sources else '无'}"
            )
        if not sections:
            sections.append("当前没有足够的智能体分析结果，请先说明需要补充的数据。")
        return sections, all_sources

    @staticmethod
    def _recent_prior_messages(messages: list[Message], limit: int = RECENT_CONTEXT_MESSAGES) -> list[Message]:
        if not messages:
            return []
        prior_messages = messages[:-1] if messages[-1].role == "user" else messages
        return prior_messages[-limit:]

    @staticmethod
    def _role_label(role: str) -> str:
        return {
            "user": "用户",
            "assistant": "助手",
            "tool": "工具",
            "system": "系统",
        }.get(role, role)

    @staticmethod
    def _compact_content(content: str, limit: int = 600) -> str:
        normalized = " ".join(content.split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 1] + "…"

    @staticmethod
    def _system_prompt() -> str:
        return (
            "你是 FinAgent 金融决策助手，负责基于可审计的工具结果和智能体分析生成中文回答。"
            "回答必须客观、审慎，不能承诺收益，不能给出满仓、加杠杆或直接交易执行指令。"
            "当数据不足、来源缺失或工具失败时，要明确说明限制，并给出下一步需要补充的数据。"
            "如果上下文包含 portfolio.holdings、metrics 或 weights，请直接使用这些数据，不要声称缺少持仓明细。"
            "不要编造行情、财务数据、新闻或来源。"
            "不要套用固定三段式模板，也不要每次重复相同小标题；请根据用户问题自然组织答案。"
            "必须先识别并回答当前用户请求；历史对话和记忆只能作为背景约束，不能替代、改写或抢占当前问题。"
            "如果当前问题是在追问系统行为、回答质量或是否预设，也要直接解释当前生成机制，不要套回旧的金融分析任务。"
            '描述风险时避免\u201c完全符合、无需调整、极低风险、确定安全\u201d等过满表述，改用\u201c从当前快照看、仍需持续验证\u201d的审慎口径。'
            "如果上下文包含约束冲突检测结果，必须在回答中主动提示用户澄清矛盾约束，并在冲突解决前按更保守的一方执行。"
        )

    @staticmethod
    def _user_prompt(*, user_input: str, sections: list[str], sources: list[SourceRef]) -> str:
        source_lines = [
            f"- {source.title} ({source.source_type}, freshness={source.freshness})"
            for source in sources[:8]
        ]
        source_text = "\n".join(source_lines) if source_lines else "无可审计来源。"
        return (
            f"用户请求：\n{user_input}\n\n"
            "分析上下文：\n"
            f"{chr(10).join(sections)}\n\n"
            "可审计来源：\n"
            f"{source_text}\n\n"
            "请用中文输出简洁但完整的答复。概念解释可以用自然段；组合、行情、财报或宏观分析可以用短要点。"
            "先回答当前用户请求本身；历史对话记忆只用于补充偏好、约束和必要上下文。"
            "只保留对本问题真正有用的信息，避免重复铺陈风险提示。"
            "如果工具失败或来源不足，请把限制写清楚，只说明一次，并给出最直接的补数或配置动作。"
        )


class FinalResponseBuilder:
    def build(self, candidate: CandidateResponse) -> FinalResponse:
        return FinalResponse(
            answer=candidate.answer,
            sources=candidate.sources,
            risk_disclaimer=(
                "仅供决策参考，不保证收益；请核对数据时效、成本税费与自身适配性。"
            ),
        )


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    non_ascii_chars = len(text) - ascii_chars
    return max(1, (ascii_chars + 3) // 4 + (non_ascii_chars + 1) // 2)

