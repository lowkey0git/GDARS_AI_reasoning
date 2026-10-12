"""
Verification + Reflection Nodes
=================================
Verification: scores the final draft and decides if reflection is needed.
Reflection:   rewrites the draft to fix identified problems.

Both use the fast 8B router model — we're doing quality checking, not
generating new deep reasoning.

Engineering note on when reflection triggers:
  Reflection adds ~1.5 full pipeline passes in token cost.
  We only trigger it when:
    1. confidence_score < reflection_threshold (0.60 default)
    2. complexity == NOVEL (SIMPLE and COMPOUND answers don't benefit enough)
    3. iteration < max_iterations

  For COMPOUND queries with confidence > 0.60, we accept the answer as-is.
  Users get a faster response; accuracy loss is minimal for non-novel queries.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agents.state import AgentState, QueryComplexity
from config.settings import get_settings
from utils.llm_client import get_router_client
from utils.logging import get_logger

log = get_logger(__name__)


class VerificationOutput(BaseModel):
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    hallucination_flags: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    goal_aligned: bool = True
    should_reflect: bool = False
    notes: str = ""


_VERIFY_SYSTEM = """You are a quality evaluator for AI responses.

Check the response draft for:
1. Hallucinations: specific claims (names, numbers, URLs, dates) with no grounding in evidence
2. Contradictions: claims that conflict with each other or with the evidence
3. Goal alignment: does the response actually address the true goal?
4. Overall confidence: how much should we trust this response?

Be specific about any problems. If there are no problems, say so.
Respond with JSON only."""

_REFLECT_SYSTEM = """You are a response quality improver.

You are given a response draft with identified problems.
Rewrite the response to fix those problems.

Rules:
- Fix only what is flagged. Don't change correct content.
- Ground every claim in the provided evidence.
- If a claim can't be grounded, remove it or mark it as uncertain.
- Keep the same structure and style as the original.
- Return only the improved response, no meta-commentary."""


class VerificationNode:
    """LangGraph node: verification_node"""

    def __init__(self) -> None:
        self._llm = get_router_client()

    async def run(self, state: AgentState) -> AgentState:
        state.log_node("verification_node")
        cfg = get_settings().planner

        evidence = "\n".join(
            f.content[:200] for f in state.evidence[:4]
        ) or "No external evidence."

        draft = state.final_solution[:1500] or state.adaptive_response[:1500]
        if not draft:
            log.info("No draft to verify")
            return state

        user_msg = (
            f"True Goal: {state.inferred_goal.true_goal if state.inferred_goal else state.user_query}\n\n"
            f"Evidence:\n{evidence}\n\n"
            f"Response Draft:\n{draft}\n\n"
            f"Current Confidence: {state.confidence_score}"
        )

        try:
            output = await self._llm.aextract(
                system=_VERIFY_SYSTEM,
                user=user_msg,
                schema=VerificationOutput,
            )
            state.confidence_score = output.confidence
            state.record_model(self._llm.model_name)

            # Only reflect for NOVEL queries below the reflection threshold
            state.should_reflect = (
                output.should_reflect
                and state.complexity == QueryComplexity.NOVEL
                and state.confidence_score < cfg.reflection_threshold
                and state.iteration < state.max_iterations
            )

            log.info(
                "Verification complete",
                confidence=output.confidence,
                flags=len(output.hallucination_flags),
                should_reflect=state.should_reflect,
            )
        except Exception as exc:
            state.record_error(f"Verification failed: {exc}")
            log.warning("Verification error — keeping current confidence", error=str(exc))

        return state


class ReflectionNode:
    """LangGraph node: reflection_node"""

    def __init__(self) -> None:
        self._llm = get_router_client()

    async def run(self, state: AgentState) -> AgentState:
        state.log_node("reflection_node")
        state.iteration += 1

        log.info("Reflection triggered", iteration=state.iteration,
                 confidence=state.confidence_score)

        evidence = "\n".join(f.content[:300] for f in state.evidence[:4]) or "No evidence."
        draft    = state.final_solution or state.adaptive_response or "No draft."

        user_msg = (
            f"True Goal: {state.inferred_goal.true_goal if state.inferred_goal else state.user_query}\n\n"
            f"Evidence:\n{evidence}\n\n"
            f"Current Draft:\n{draft[:2000]}\n\n"
            f"Identified Problems:\n"
            + "\n".join(f"- {e}" for e in state.error_log[-3:])
        )

        try:
            improved = await self._llm.ainvoke(_REFLECT_SYSTEM, user_msg)
            state.final_solution    = improved
            state.adaptive_response = improved
            state.record_model(self._llm.model_name)
            log.info("Reflection complete — draft improved")
        except Exception as exc:
            state.record_error(f"Reflection failed: {exc}")
            log.error("Reflection error", error=str(exc))

        return state


async def verification_node(state: AgentState) -> AgentState:
    return await VerificationNode().run(state)


async def reflection_node(state: AgentState) -> AgentState:
    return await ReflectionNode().run(state)
