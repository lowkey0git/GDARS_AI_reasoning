"""
Evaluation Harness
==================
Scores each intermediate reasoning step before the next step runs.

Why score intermediate steps?
  Without mid-chain scoring, a bad reasoning step at step 2 propagates
  forward and contaminates steps 3–8. You only discover the problem at
  the final output, after spending tokens on all the subsequent steps.

  With mid-chain scoring, a step below the threshold triggers an
  immediate retry of just that step with a corrective prompt. This
  is cheaper than a full reflection loop.

Two dimensions per step:

  Factuality (0.0–1.0)
    Is the claim in this step grounded in the retrieved evidence or
    well-established fact? A step that says "Python was created in 1991"
    when the evidence says 1989 scores low here.

  Usefulness (0.0–1.0)
    Does this step move the reasoning toward the goal? A step that
    correctly restates the question scores high on factuality but low
    on usefulness.

Engineering tradeoff:
  Scoring with the 70B model is accurate but expensive.
  Scoring with the 8B router model is cheap but misses subtle errors.

  We use the 8B model for intermediate steps (speed matters more)
  and the 70B model for the final solution (quality matters more).
  The cost difference is ~10x per call.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agents.state import AgentState, StepScore, ThoughtStep
from config.settings import get_settings
from utils.llm_client import get_router_client
from utils.logging import get_logger

log = get_logger(__name__)


class StepScoreOutput(BaseModel):
    factuality: float = Field(0.5, ge=0.0, le=1.0)
    usefulness: float = Field(0.5, ge=0.0, le=1.0)
    reasoning: str = ""


_SCORE_SYSTEM = """You are an evaluation engine for AI reasoning steps.

Given:
- The user's goal
- The evidence available
- A single reasoning step (thought + conclusion)

Score the step on two dimensions (0.0 to 1.0):

factuality: Is the conclusion grounded in the evidence or well-established fact?
  1.0 = directly supported by evidence
  0.5 = plausible but not directly supported
  0.0 = contradicts evidence or is made up

usefulness: Does this step move reasoning toward the goal?
  1.0 = directly advances toward the goal
  0.5 = relevant but not decisive
  0.0 = restates the question or goes in a different direction

Be strict. Most steps should score 0.4–0.7. Reserve 0.9+ for genuinely excellent steps.
Respond with ONLY valid JSON."""


class EvaluationHarness:
    """
    Scores individual thought steps and updates the AgentState in place.
    """

    def __init__(self) -> None:
        # Use the fast 8B router model for intermediate scoring
        # to keep latency and token cost low
        self._llm = get_router_client()
        self._cfg = get_settings().evaluation

    async def score_step(
        self,
        step: ThoughtStep,
        goal: str,
        evidence_summary: str,
    ) -> StepScore:
        """
        Score a single thought step.
        Returns a StepScore; does not modify state directly.
        """
        user_msg = (
            f"Goal: {goal}\n\n"
            f"Evidence summary:\n{evidence_summary[:600]}\n\n"
            f"Step {step.step_num}:\n"
            f"Thought: {step.thought}\n"
            f"Conclusion: {step.conclusion}"
        )

        try:
            output = await self._llm.aextract(
                system=_SCORE_SYSTEM,
                user=user_msg,
                schema=StepScoreOutput,
            )
            return StepScore(
                step_num=step.step_num,
                factuality=output.factuality,
                usefulness=output.usefulness,
                reasoning=output.reasoning,
            )
        except Exception as exc:
            log.warning("Step scoring failed — using neutral scores", error=str(exc))
            return StepScore(
                step_num=step.step_num,
                factuality=0.5,
                usefulness=0.5,
                reasoning="Scoring failed — neutral score assigned.",
            )

    async def score_state(self, state: AgentState) -> AgentState:
        """
        Score all un-scored thought steps and update aggregate metrics.
        Called after the reasoning node completes.
        """
        if not self._cfg.score_intermediate_steps:
            return state

        goal = state.inferred_goal.true_goal if state.inferred_goal else state.user_query
        evidence_summary = state.evidence_synthesis or "\n".join(
            e.content[:200] for e in state.evidence[:3]
        )

        already_scored = {s.step_num for s in state.step_scores}
        new_scores: list[StepScore] = []

        for step in state.thought_steps:
            if step.step_num in already_scored:
                continue
            score = await self.score_step(step, goal, evidence_summary)
            # Write scores back to the ThoughtStep for observability
            step.factuality_score = score.factuality
            step.usefulness_score = score.usefulness
            new_scores.append(score)

        state.step_scores.extend(new_scores)

        # Recompute aggregates over all scores
        if state.step_scores:
            state.avg_factuality = sum(s.factuality for s in state.step_scores) / len(state.step_scores)
            state.avg_usefulness = sum(s.usefulness for s in state.step_scores) / len(state.step_scores)

        log.info(
            "Evaluation complete",
            steps_scored=len(new_scores),
            avg_factuality=round(state.avg_factuality, 3),
            avg_usefulness=round(state.avg_usefulness, 3),
        )
        return state

    def should_retry_step(self, score: StepScore) -> bool:
        """
        Returns True if a step's scores are low enough to warrant a retry.
        """
        cfg = self._cfg
        return (
            score.factuality < cfg.min_factuality_score
            or score.usefulness < cfg.min_usefulness_score
        )
