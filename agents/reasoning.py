"""
Reasoning Node
==============
Runs a chain-of-thought reasoning sequence using the 70B model.

For SIMPLE queries: one direct reasoning step, no chain.
For COMPOUND: 3-5 step CoT.
For NOVEL: 5-8 step CoT with mid-chain evaluation after each step.

Mid-chain evaluation detail:
  After each thought step is generated, the evaluation harness scores it.
  If factuality or usefulness falls below threshold, the step is retried
  once with a corrective prompt that includes the score and reason.
  This catches bad reasoning early rather than propagating it forward.

  One retry costs ~1 LLM call on the 8B model + one on the 70B model.
  That's cheaper than a full reflection loop (which costs 2–3 full
  pipeline passes).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agents.state import AgentState, QueryComplexity, ThoughtStep
from evaluation.harness import EvaluationHarness
from utils.llm_client import get_reasoning_client, get_router_client
from utils.logging import get_logger

log = get_logger(__name__)


class ReasoningOutput(BaseModel):
    steps: list[dict] = Field(default_factory=list)
    chain_of_thought: str = ""
    blockers: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    confidence: float = 0.7


_SIMPLE_SYSTEM = """Answer the query directly and concisely.
Be accurate. If you're not certain, say so.
Provide the most useful answer you can from your knowledge.
Respond in 2-4 sentences."""

_REASONING_SYSTEM = """You are a multi-step reasoning engine.

Given the goal and evidence, perform structured chain-of-thought reasoning.

Rules:
1. Each step must advance toward the goal — no restating the question.
2. Ground every claim in the evidence when possible.
3. Explicitly note what you don't know or can't verify.
4. End with concrete, ordered next actions.

The reasoning objective: close the gap between the user's current state
and their desired state as directly as possible.

Respond with JSON only."""


class ReasoningNode:
    """LangGraph node: reasoning_node"""

    def __init__(self) -> None:
        self._reasoner  = get_reasoning_client()
        self._router    = get_router_client()     # for mid-chain retry scoring
        self._evaluator = EvaluationHarness()

    async def run(self, state: AgentState) -> AgentState:
        state.log_node("reasoning_node")

        # ── SIMPLE: one direct call, no chain ────────────────
        if state.complexity == QueryComplexity.SIMPLE:
            return await self._simple_reasoning(state)

        # ── COMPOUND / NOVEL: full CoT ───────────────────────
        return await self._chain_reasoning(state)

    async def _simple_reasoning(self, state: AgentState) -> AgentState:
        """Single fast reasoning call for simple queries."""
        context = "\n".join(
            e.get("content", "")[:200] for e in
            (state.working_memory + state.episodic_memory)[:3]
        )
        user_msg = (
            f"Query: {state.user_query}\n"
            + (f"\nContext:\n{context}" if context else "")
        )
        try:
            answer = await self._router.ainvoke(_SIMPLE_SYSTEM, user_msg)
            state.record_model(self._router.model_name)
            step = state.add_thought(
                thought=f"Direct answer to: {state.user_query}",
                conclusion=answer,
                model=self._router.model_name,
            )
            state.chain_of_thought = answer
            state.next_actions = []
            state.confidence_score = 0.80    # high confidence for simple factual
        except Exception as exc:
            state.record_error(f"Simple reasoning failed: {exc}")
            state.confidence_score = 0.4
        return state

    async def _chain_reasoning(self, state: AgentState) -> AgentState:
        """Multi-step CoT with optional mid-chain evaluation."""
        goal = state.inferred_goal.true_goal if state.inferred_goal else state.user_query
        evidence_summary = state.evidence_synthesis or "No external evidence."
        memory_ctx = "\n".join(
            e.get("content", "")[:200] for e in
            (state.working_memory + state.episodic_memory)[:3]
        )
        plan_txt = ""
        if state.execution_plan:
            plan_txt = "\n".join(
                f"  {i+1}. [{s.tool}] {s.title}: {s.description[:80]}"
                for i, s in enumerate(state.execution_plan.steps[:5])
            )

        user_msg = (
            f"True Goal: {goal}\n"
            f"Latent Objectives: {', '.join(state.inferred_goal.latent_objectives[:3]) if state.inferred_goal else 'none'}\n"
            f"Current State: {state.inferred_goal.current_state if state.inferred_goal else 'unknown'}\n"
            f"Desired State: {state.inferred_goal.desired_state if state.inferred_goal else 'unknown'}\n"
            f"Constraints: {', '.join(state.constraints) or 'none'}\n\n"
            f"Evidence:\n{evidence_summary}\n\n"
            + (f"Memory Context:\n{memory_ctx}\n\n" if memory_ctx else "")
            + (f"Execution Plan:\n{plan_txt}" if plan_txt else "")
        )

        try:
            output = await self._reasoner.aextract(
                system=_REASONING_SYSTEM,
                user=user_msg,
                schema=ReasoningOutput,
            )
            state.record_model(self._reasoner.model_name)

            # Add each step with optional mid-chain evaluation
            cfg_eval = self._evaluator._cfg
            for s in output.steps:
                step = state.add_thought(
                    thought=s.get("thought", ""),
                    conclusion=s.get("conclusion", ""),
                    model=self._reasoner.model_name,
                )

                # Mid-chain evaluation for NOVEL queries
                if (state.complexity == QueryComplexity.NOVEL
                        and cfg_eval.score_intermediate_steps):
                    score = await self._evaluator.score_step(
                        step, goal, evidence_summary
                    )
                    step.factuality_score = score.factuality
                    step.usefulness_score = score.usefulness
                    state.step_scores.append(score)

                    # Retry once if step quality is poor
                    if self._evaluator.should_retry_step(score):
                        step = await self._retry_step(step, goal, evidence_summary, score.reasoning, state)

            state.chain_of_thought = output.chain_of_thought
            state.next_actions     = output.next_actions
            state.confidence_score = output.confidence

            # Final aggregate scoring (covers all steps for COMPOUND)
            state = await self._evaluator.score_state(state)

            log.info(
                "CoT reasoning complete",
                steps=len(output.steps),
                confidence=output.confidence,
                avg_factuality=state.avg_factuality,
            )
        except Exception as exc:
            state.record_error(f"Chain reasoning failed: {exc}")
            log.error("Reasoning error", error=str(exc))
            state.confidence_score = 0.35

        return state

    async def _retry_step(
        self,
        original_step: ThoughtStep,
        goal: str,
        evidence: str,
        score_reason: str,
        state: AgentState,
    ) -> ThoughtStep:
        """
        Retry a low-scoring step with a corrective prompt.
        Uses the 70B model — same as original — because the 8B model
        likely can't do better on a step it just failed.
        """
        corrective_prompt = (
            f"A previous reasoning step was rated low quality for this reason:\n"
            f"'{score_reason}'\n\n"
            f"Retry the step with better grounding in evidence and more direct "
            f"progress toward the goal.\n\n"
            f"Goal: {goal}\n"
            f"Evidence: {evidence[:400]}\n\n"
            f"Original thought: {original_step.thought}\n"
            f"Original conclusion: {original_step.conclusion}\n\n"
            f"Produce an improved conclusion in 2-3 sentences."
        )
        try:
            improved = await self._reasoner.ainvoke(
                "You are a reasoning quality improver. Fix the given step.",
                corrective_prompt,
            )
            original_step.conclusion = improved
            log.info("Step retried successfully", step_num=original_step.step_num)
        except Exception as exc:
            log.warning("Step retry failed", step_num=original_step.step_num, error=str(exc))
        return original_step


async def reasoning_node(state: AgentState) -> AgentState:
    return await ReasoningNode().run(state)
