"""
Agent State
===========
The single Pydantic model that flows through every LangGraph node.

Engineering decisions:
  1. All fields Optional or with defaults so partial population is safe.
     Nodes write their slice and pass forward — earlier nodes don't know
     what later nodes will add.

  2. ThoughtStep is a first-class data structure, not a string.
     Each step carries factuality_score and usefulness_score from the
     evaluation harness. This makes reasoning observable and debuggable.

  3. QueryComplexity drives the hierarchical planner. The router sets it
     once at the start; every downstream node reads it to decide whether
     to run or skip itself.

  4. model_dump() / model_validate() roundtrips are used at LangGraph
     node boundaries because LangGraph's StateGraph operates on dicts
     internally. We convert at the boundary, not inside nodes.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


# ── Enums ────────────────────────────────────────────────────────────────────

class QueryComplexity(str, Enum):
    """
    Set by the hierarchical planner router.
    Determines which nodes execute and which are skipped.

    SIMPLE   → memory_node → reasoning_node → final_node
    COMPOUND → full pipeline minus reflection loop
    NOVEL    → full pipeline including reflection loop
    """
    SIMPLE   = "simple"
    COMPOUND = "compound"
    NOVEL    = "novel"


class SkillLevel(str, Enum):
    BEGINNER     = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED     = "advanced"
    EXPERT       = "expert"


class NodeStatus(str, Enum):
    PENDING  = "pending"
    COMPLETE = "complete"
    FAILED   = "failed"
    SKIPPED  = "skipped"


# ── Sub-models ───────────────────────────────────────────────────────────────

class ThoughtStep(BaseModel):
    """
    A single step in a chain-of-thought reasoning sequence.

    factuality_score: 0.0–1.0, set by the evaluation harness.
      Measures whether the claim in this step is grounded in
      retrieved context or well-established fact.

    usefulness_score: 0.0–1.0, set by the evaluation harness.
      Measures whether this step actually moves the reasoning
      toward the goal. A step can be factually correct but useless
      (e.g. restating the question).
    """
    step_num: int
    thought: str
    conclusion: str
    factuality_score: float = 0.0
    usefulness_score: float = 0.0
    model_used: str = ""        # which Groq model produced this step


class EvidenceFragment(BaseModel):
    """A single retrieved passage with provenance and scoring."""
    content: str
    source_type: str            # web | faiss | episodic | working
    source_url: str = ""
    source_title: str = ""
    relevance_score: float = 0.0
    credibility_score: float = 0.5


class ExecutionStep(BaseModel):
    step_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    title: str
    description: str
    tool: str = "none"          # web_search | code_generation | memory | reasoning | none
    status: NodeStatus = NodeStatus.PENDING
    result: str = ""
    error: str = ""


class ExecutionPlan(BaseModel):
    objective: str
    steps: list[ExecutionStep] = Field(default_factory=list)
    complexity: QueryComplexity = QueryComplexity.COMPOUND
    needs_web_search: bool = False
    needs_code: bool = False


class UserProfile(BaseModel):
    skill_level: SkillLevel = SkillLevel.INTERMEDIATE
    domain: str = "general"
    tone: str = "neutral"           # neutral | encouraging | direct | formal
    math_depth: str = "none"        # none | intuition | equations
    code_oriented: bool = True


class InferredGoal(BaseModel):
    surface_query: str
    true_goal: str
    latent_objectives: list[str] = Field(default_factory=list)
    current_state: str = ""
    desired_state: str = ""
    success_criteria: list[str] = Field(default_factory=list)
    distance_estimate: float = 1.0  # 0 = achieved, 1 = far


class StepScore(BaseModel):
    """Evaluation harness output for a single thought step."""
    step_num: int
    factuality: float
    usefulness: float
    reasoning: str              # why the harness gave these scores


# ── Master AgentState ────────────────────────────────────────────────────────

class AgentState(BaseModel):
    """
    The canonical state object.

    Flows through every LangGraph node as model_dump() dict.
    Reconstructed at each node boundary with model_validate().

    Field grouping follows the pipeline order so it's easy to
    see what each phase produces.
    """

    # Identity
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    request_id: str = Field(default_factory=lambda: str(uuid4())[:12])

    # ── Phase 0: Input ──────────────────────────────────────────
    user_query: str = ""
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    user_profile: UserProfile = Field(default_factory=UserProfile)

    # ── Phase 1: Routing ────────────────────────────────────────
    complexity: QueryComplexity = QueryComplexity.COMPOUND
    complexity_confidence: float = 0.0

    # ── Phase 2: Planning ───────────────────────────────────────
    inferred_goal: Optional[InferredGoal] = None
    execution_plan: Optional[ExecutionPlan] = None
    constraints: list[str] = Field(default_factory=list)

    # ── Phase 3: Retrieval ──────────────────────────────────────
    sub_queries: list[str] = Field(default_factory=list)
    evidence: list[EvidenceFragment] = Field(default_factory=list)
    evidence_synthesis: str = ""
    coverage_gaps: list[str] = Field(default_factory=list)

    # ── Phase 4: Memory ─────────────────────────────────────────
    working_memory: list[dict[str, Any]] = Field(default_factory=list)   # from Redis
    episodic_memory: list[dict[str, Any]] = Field(default_factory=list)  # from ChromaDB
    semantic_memory: list[dict[str, Any]] = Field(default_factory=list)  # from FAISS

    # ── Phase 5: Reasoning ──────────────────────────────────────
    thought_steps: list[ThoughtStep] = Field(default_factory=list)
    chain_of_thought: str = ""
    next_actions: list[str] = Field(default_factory=list)

    # ── Phase 6: Evaluation ─────────────────────────────────────
    step_scores: list[StepScore] = Field(default_factory=list)
    avg_factuality: float = 0.0
    avg_usefulness: float = 0.0

    # ── Phase 7: Execution ──────────────────────────────────────
    execution_results: list[dict[str, Any]] = Field(default_factory=list)
    tool_outputs: dict[str, str] = Field(default_factory=dict)

    # ── Phase 8: Response ───────────────────────────────────────
    adaptive_response: str = ""
    final_solution: str = ""
    confidence_score: float = 0.0

    # ── Control flow ────────────────────────────────────────────
    iteration: int = 0
    max_iterations: int = 8
    should_reflect: bool = False
    is_complete: bool = False

    # ── Observability ───────────────────────────────────────────
    node_trace: list[str] = Field(default_factory=list)
    error_log: list[str] = Field(default_factory=list)
    models_used: list[str] = Field(default_factory=list)  # MoE audit trail

    # ── Validators ──────────────────────────────────────────────

    @field_validator("confidence_score", "avg_factuality", "avg_usefulness")
    @classmethod
    def clamp_to_unit(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    @field_validator("iteration")
    @classmethod
    def iteration_non_negative(cls, v: int) -> int:
        return max(0, v)

    # ── Helpers ─────────────────────────────────────────────────

    def log_node(self, name: str) -> None:
        self.node_trace.append(name)

    def record_error(self, msg: str) -> None:
        self.error_log.append(msg)

    def record_model(self, model: str) -> None:
        if model not in self.models_used:
            self.models_used.append(model)

    def add_thought(self, thought: str, conclusion: str, model: str = "") -> ThoughtStep:
        step = ThoughtStep(
            step_num=len(self.thought_steps) + 1,
            thought=thought,
            conclusion=conclusion,
            model_used=model,
        )
        self.thought_steps.append(step)
        return step

    def should_run_full_pipeline(self) -> bool:
        return self.complexity in (QueryComplexity.COMPOUND, QueryComplexity.NOVEL)

    def should_run_reflection(self) -> bool:
        return (
            self.complexity == QueryComplexity.NOVEL
            and self.should_reflect
            and self.iteration < self.max_iterations
        )
