"""
Hierarchical Planner
=====================
Two-stage node: route first, then plan.

Stage 1 — Complexity Router
  Classifies the query as SIMPLE / COMPOUND / NOVEL using the fast 8B model.
  This single cheap call determines which nodes execute downstream.

  SIMPLE:   memory lookup + fast reasoning. Skips web search, execution, reflection.
  COMPOUND: full pipeline minus reflection. Good for most technical questions.
  NOVEL:    full pipeline including reflection loop. For genuinely new problems.

Stage 2 — Goal Inference + Plan Generation
  Skipped entirely for SIMPLE queries (saves ~2 LLM calls).
  For COMPOUND and NOVEL, uses the 70B model to:
    - Infer the true goal from the surface query
    - Build a typed ExecutionPlan

Engineering note on the flat vs hierarchical tradeoff:
  A flat multi-agent loop calls every agent on every query.
  For "what is a Python list?" that means 9 LLM calls.
  The hierarchical planner routes that to 3 calls.

  The cost of the router call itself is ~200 tokens on the 8B model,
  which is negligible. The savings on SIMPLE queries are large.

  The risk is misclassification: a NOVEL query routed as SIMPLE gets
  a shallow answer. We mitigate this with a confidence cutoff —
  anything below cfg.complexity_confidence_cutoff defaults to COMPOUND.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agents.state import (
    AgentState, ExecutionPlan, ExecutionStep,
    InferredGoal, QueryComplexity, SkillLevel,
)
from utils.llm_client import get_router_client, get_reasoning_client
from utils.logging import get_logger

log = get_logger(__name__)


# ── Router schemas ────────────────────────────────────────────────────────────

class ComplexityOutput(BaseModel):
    complexity: str = "compound"    # simple | compound | novel
    confidence: float = 0.7
    reason: str = ""


class GoalOutput(BaseModel):
    true_goal: str
    latent_objectives: list[str] = Field(default_factory=list)
    current_state: str = ""
    desired_state: str = ""
    success_criteria: list[str] = Field(default_factory=list)
    distance_estimate: float = 0.8
    needs_web_search: bool = False
    needs_code: bool = False
    constraints: list[str] = Field(default_factory=list)


class PlanOutput(BaseModel):
    objective: str
    steps: list[dict] = Field(default_factory=list)
    needs_web_search: bool = False
    needs_code: bool = False


# ── Prompts ───────────────────────────────────────────────────────────────────

_ROUTER_SYSTEM = """You are a query complexity classifier for an AI reasoning system.

Classify the query into exactly one tier:

SIMPLE   — Can be answered from memory or basic reasoning alone.
           Examples: definitions, simple how-to, factual lookups.
           No web search or multi-step planning needed.

COMPOUND — Needs retrieval + structured reasoning but not deep exploration.
           Examples: technical explanations, comparisons, implementation guides.

NOVEL    — Genuinely new problem requiring exploration, hypothesis, reflection.
           Examples: open research questions, ambiguous goals, strategy design.

Be conservative: when unsure, default to COMPOUND.
Respond with JSON only."""

_GOAL_SYSTEM = """You are a goal inference engine for an autonomous AI system.

Transform the surface query into a deep goal representation.
Infer what the user ACTUALLY wants beyond the literal words.
Ground your inferences in the vocabulary and framing of the query.
Do not invent objectives that aren't suggested by the query.
Respond with JSON only."""

_PLAN_SYSTEM = """You are an execution planner for an autonomous AI reasoning system.

Given a true goal, build a concrete execution plan.
Each step needs: title (short action label), description (what happens),
tool: one of [web_search | code_generation | memory_retrieval | reasoning | none]

Order: information gathering → reasoning → synthesis.
Keep plans concise: 3-6 steps. More steps = more tokens = higher cost.
Respond with JSON only."""


# ── Planner node ──────────────────────────────────────────────────────────────

class HierarchicalPlanner:
    """LangGraph node: planner_node"""

    def __init__(self) -> None:
        self._router    = get_router_client()
        self._reasoner  = get_reasoning_client()
        self._cfg = None

    def _get_cfg(self):
        if self._cfg is None:
            from config.settings import get_settings
            self._cfg = get_settings().planner
        return self._cfg

    async def run(self, state: AgentState) -> AgentState:
        state.log_node("planner_node")
        cfg = self._get_cfg()

        # ── Stage 1: Route ────────────────────────────────────
        try:
            complexity_out = await self._router.aextract(
                system=_ROUTER_SYSTEM,
                user=f"Query: {state.user_query}",
                schema=ComplexityOutput,
            )
            state.record_model(self._router.model_name)

            # Apply confidence cutoff — low confidence → default to COMPOUND
            if complexity_out.confidence < cfg.complexity_confidence_cutoff:
                state.complexity = QueryComplexity.COMPOUND
                state.complexity_confidence = complexity_out.confidence
            else:
                try:
                    state.complexity = QueryComplexity(complexity_out.complexity.lower())
                except ValueError:
                    state.complexity = QueryComplexity.COMPOUND
                state.complexity_confidence = complexity_out.confidence

            log.info(
                "Query classified",
                complexity=state.complexity.value,
                confidence=state.complexity_confidence,
                reason=complexity_out.reason[:80],
            )
        except Exception as exc:
            state.record_error(f"Complexity routing failed: {exc}")
            state.complexity = QueryComplexity.COMPOUND

        # ── Stage 2: Goal + Plan (skipped for SIMPLE) ─────────
        if state.complexity == QueryComplexity.SIMPLE:
            log.info("SIMPLE query — skipping full planning")
            return state

        try:
            history_txt = "\n".join(
                f"{m['role'].upper()}: {m['content']}"
                for m in state.conversation_history[-4:]
            )
            goal_out = await self._reasoner.aextract(
                system=_GOAL_SYSTEM,
                user=f"Query: {state.user_query}\nHistory:\n{history_txt}",
                schema=GoalOutput,
            )
            state.record_model(self._reasoner.model_name)

            state.inferred_goal = InferredGoal(
                surface_query=state.user_query,
                true_goal=goal_out.true_goal,
                latent_objectives=goal_out.latent_objectives,
                current_state=goal_out.current_state,
                desired_state=goal_out.desired_state,
                success_criteria=goal_out.success_criteria,
                distance_estimate=goal_out.distance_estimate,
            )
            state.constraints = goal_out.constraints
            state.add_thought(
                thought=f"Surface query: '{state.user_query}'",
                conclusion=f"True goal: '{goal_out.true_goal}'",
                model=self._reasoner.model_name,
            )
            log.info("Goal inferred", true_goal=goal_out.true_goal[:80])

        except Exception as exc:
            state.record_error(f"Goal inference failed: {exc}")
            # Fallback — treat query as goal
            state.inferred_goal = InferredGoal(
                surface_query=state.user_query,
                true_goal=state.user_query,
            )

        try:
            goal = state.inferred_goal
            plan_out = await self._reasoner.aextract(
                system=_PLAN_SYSTEM,
                user=(
                    f"True goal: {goal.true_goal}\n"
                    f"Objectives: {', '.join(goal.latent_objectives[:3])}\n"
                    f"Constraints: {', '.join(state.constraints) or 'none'}"
                ),
                schema=PlanOutput,
            )
            state.execution_plan = ExecutionPlan(
                objective=plan_out.objective,
                steps=[
                    ExecutionStep(
                        title=s.get("title", f"Step {i+1}"),
                        description=s.get("description", ""),
                        tool=s.get("tool", "none"),
                    )
                    for i, s in enumerate(plan_out.steps)
                ],
                complexity=state.complexity,
                needs_web_search=plan_out.needs_web_search,
                needs_code=plan_out.needs_code,
            )
            log.info("Plan built", steps=len(plan_out.steps))

        except Exception as exc:
            state.record_error(f"Plan generation failed: {exc}")

        return state


async def planner_node(state: AgentState) -> AgentState:
    return await HierarchicalPlanner().run(state)
