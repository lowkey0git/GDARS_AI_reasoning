"""
Synthesis Node
==============
Combines personalization and final assembly into one node.

Why merge them?
  The original design had separate personalization_node and final_response_node,
  each making independent LLM calls. But they were reading the same inputs and
  their outputs were tightly coupled — personalization produced text that
  final_response would then reformat.

  Merging them into one node with the synthesis model (balanced speed/quality)
  saves one LLM call per request with no quality loss.

Personalization is handled by injecting the UserProfile into the system prompt
rather than running a separate five-lens pipeline. The five-lens approach
(DepthLens, ModalityLens, etc.) is architecturally elegant but costs 5 LLM
calls for what is essentially a formatting task.

The synthesis model (llama-3.1-8b-instant) is used here because:
  - The hard thinking is already done by the 70B reasoning node
  - This node only needs to format and adapt an existing answer
  - 8B is fast and cheap and perfectly capable of formatting
"""

from __future__ import annotations

from agents.state import AgentState, QueryComplexity, SkillLevel
from utils.llm_client import get_synthesis_client
from utils.logging import get_logger

log = get_logger(__name__)


def _build_style_instructions(state: AgentState) -> str:
    """
    Convert UserProfile into concrete style instructions for the LLM.
    These are specific enough that the model can actually follow them,
    unlike vague instructions like "be appropriate for the skill level".
    """
    p = state.user_profile
    lines = []

    if p.skill_level == SkillLevel.BEGINNER:
        lines.append("Use plain language. Define every technical term the first time it appears in parentheses.")
        lines.append("Use short sentences (under 20 words). One concept per paragraph.")
    elif p.skill_level == SkillLevel.INTERMEDIATE:
        lines.append("Use technical vocabulary. Assume familiarity with basic concepts but explain advanced ones.")
    elif p.skill_level == SkillLevel.ADVANCED:
        lines.append("Use expert-level vocabulary. Skip introductory scaffolding entirely.")
    else:  # EXPERT
        lines.append("Peer-level communication. Reference specific tools, papers, or techniques directly.")

    if p.code_oriented:
        lines.append("Include working code examples where they clarify the point. Use proper language tags.")

    if p.math_depth == "none":
        lines.append("No equations or mathematical notation.")
    elif p.math_depth == "intuition":
        lines.append("Use intuitive descriptions of mathematical concepts, not equations.")
    elif p.math_depth == "equations":
        lines.append("Include equations with brief explanations of each term.")

    tone_map = {
        "encouraging": "Use warm, supportive language. Acknowledge that this is challenging.",
        "direct":      "Be direct. Imperatives are fine. Cut all hedging language.",
        "formal":      "Use formal academic register. No contractions.",
        "neutral":     "Professional and clear. No emotional loading.",
    }
    if p.tone in tone_map:
        lines.append(tone_map[p.tone])

    return "\n".join(lines)


_SYNTHESIS_SYSTEM_TMPL = """You are a technical communication specialist.

Your task: Take the reasoning output and produce a final, polished response.

Style instructions:
{style_instructions}

Content rules:
- Lead with the most important information (inverted pyramid)
- Be direct: state the answer, then justify it
- If there are concrete next steps, list them at the end under ## Next Steps
- Do NOT include meta-commentary about the reasoning process
- Do NOT pad with unnecessary caveats or generic disclaimers
- Preserve all specific facts, numbers, and tool names exactly

Format: clean markdown. Length should match complexity —
short answer for simple queries, structured document for complex ones."""


class SynthesisNode:
    """LangGraph node: synthesis_node"""

    def __init__(self) -> None:
        self._llm = get_synthesis_client()

    async def run(self, state: AgentState) -> AgentState:
        state.log_node("synthesis_node")

        style = _build_style_instructions(state)
        system = _SYNTHESIS_SYSTEM_TMPL.format(style_instructions=style)

        # Gather the best content from the reasoning phase
        source_content = (
            state.chain_of_thought
            or "\n\n".join(s.conclusion for s in state.thought_steps[-5:])
            or state.evidence_synthesis
            or "No reasoning output available."
        )

        # Include any generated code from tool execution
        code_blocks = "\n\n".join(
            f"```\n{v}\n```" for k, v in state.tool_outputs.items()
            if k.startswith("code_")
        )

        goal = state.inferred_goal.true_goal if state.inferred_goal else state.user_query

        user_msg = (
            f"True Goal: {goal}\n\n"
            f"Reasoning Output:\n{source_content[:3000]}\n"
            + (f"\nGenerated Code:\n{code_blocks}" if code_blocks else "") +
            f"\n\nKey Next Actions:\n"
            + "\n".join(f"- {a}" for a in state.next_actions[:5])
        )

        try:
            final = await self._llm.ainvoke(system, user_msg)
            state.adaptive_response = final
            state.final_solution    = final
            state.is_complete       = True
            state.record_model(self._llm.model_name)
            log.info("Synthesis complete", length=len(final))
        except Exception as exc:
            state.record_error(f"Synthesis failed: {exc}")
            log.error("Synthesis error", error=str(exc))
            # Fallback: use raw reasoning output
            state.final_solution = source_content
            state.is_complete    = True

        return state


async def synthesis_node(state: AgentState) -> AgentState:
    return await SynthesisNode().run(state)
