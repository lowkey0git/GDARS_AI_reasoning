"""
LangGraph Graph
================
Defines the StateGraph with hierarchical routing.

Three execution paths determined by the planner:

SIMPLE path (3 nodes, ~2s):
  planner → memory → reasoning → synthesis → END

COMPOUND path (6 nodes, ~8s):
  planner → memory → retrieval → reasoning → synthesis → verification → END

NOVEL path (full pipeline, ~20s, with optional reflection):
  planner → memory → retrieval → execution → reasoning
         → synthesis → verification
         → [reflection → reasoning loop if needed]
         → END

Engineering note on the node wrapping:
  LangGraph StateGraph operates on plain dicts internally.
  We convert AgentState ↔ dict at each node boundary using
  _wrap(). This keeps the node implementations clean — they
  receive and return AgentState, not raw dicts.
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from agents.state import AgentState, QueryComplexity
from agents.planner import planner_node
from agents.retrieval import RetrievalNode
from agents.reasoning import reasoning_node
from agents.synthesis import synthesis_node
from agents.verification import verification_node, reflection_node
from config.settings import get_settings
from memory.memory_system import MemorySystem
from utils.logging import get_logger

log = get_logger(__name__)


def _wrap(async_fn):
    """
    Converts an async (AgentState) → AgentState function into
    an async (dict) → dict function for LangGraph.
    """
    async def wrapper(state_dict: dict) -> dict:
        state  = AgentState.model_validate(state_dict)
        result = await async_fn(state)
        return result.model_dump()
    return wrapper


def _memory_node_fn(memory_system: MemorySystem):
    """Inline memory node — reads all three memory layers into state."""
    async def _run(state: AgentState) -> AgentState:
        state.log_node("memory_node")
        try:
            ctx = await memory_system.retrieve_context(
                query=state.inferred_goal.true_goal if state.inferred_goal else state.user_query,
                session_id=state.session_id,
            )
            state.working_memory  = ctx["working"]
            state.episodic_memory = ctx["episodic"]
            state.semantic_memory = ctx["semantic"]
            log.info("Memory node complete",
                     working=len(state.working_memory),
                     episodic=len(state.episodic_memory),
                     semantic=len(state.semantic_memory))
        except Exception as exc:
            state.record_error(f"Memory retrieval failed: {exc}")
            log.warning("Memory node error — continuing", error=str(exc))
        return state
    return _run


def _execution_node_fn():
    """
    Inline execution node — runs tool steps from the plan.
    Kept inline because it's thin: tool dispatch logic lives in ToolRegistry.
    """
    async def _run(state: AgentState) -> AgentState:
        state.log_node("execution_node")
        if not state.execution_plan:
            return state

        from tools.registry import ToolRegistry
        for step in state.execution_plan.steps:
            from agents.state import NodeStatus
            if step.status != NodeStatus.PENDING:
                continue
            tool = step.tool.lower()
            try:
                if tool == "web_search":
                    from tools.registry import ToolRegistry
                    results = await ToolRegistry.tavily().search(step.description)
                    step.result = "\n".join(r.get("content","")[:300] for r in results[:3])
                    for r in results[:2]:
                        state.evidence.append(
                            state.evidence[0].__class__(
                                content=r.get("content",""),
                                source_type="web",
                                source_url=r.get("url",""),
                            ) if state.evidence else __import__("agents.state", fromlist=["EvidenceFragment"]).EvidenceFragment(
                                content=r.get("content",""),
                                source_type="web",
                                source_url=r.get("url",""),
                            )
                        )
                elif tool == "code_generation":
                    code = await ToolRegistry.codegen().generate(
                        description=step.description,
                        context=state.evidence_synthesis[:500],
                    )
                    step.result = code
                    state.tool_outputs[f"code_{step.step_id}"] = code
                else:
                    step.status = NodeStatus.SKIPPED
                    continue
                step.status = NodeStatus.COMPLETE
                state.execution_results.append({
                    "step_id": step.step_id, "tool": tool,
                    "result": step.result[:400],
                })
            except Exception as exc:
                step.status = NodeStatus.FAILED
                step.error  = str(exc)
                state.record_error(f"Step '{step.title}' failed: {exc}")
        return state
    return _run


# ── Routing functions ─────────────────────────────────────────────────────────

def _complexity_router(state_dict: dict) -> str:
    """Routes after planner based on query complexity."""
    state = AgentState.model_validate(state_dict)
    if state.complexity == QueryComplexity.SIMPLE:
        return "memory_node"      # SIMPLE: skip retrieval
    return "memory_node"          # all paths go through memory


def _post_memory_router(state_dict: dict) -> str:
    """Routes after memory based on complexity."""
    state = AgentState.model_validate(state_dict)
    if state.complexity == QueryComplexity.SIMPLE:
        return "reasoning_node"   # SIMPLE: skip retrieval
    return "retrieval_node"


def _post_retrieval_router(state_dict: dict) -> str:
    """Routes after retrieval based on whether plan has tool steps."""
    state = AgentState.model_validate(state_dict)
    if (state.complexity == QueryComplexity.NOVEL
            and state.execution_plan
            and any(s.tool in ("web_search","code_generation")
                    for s in state.execution_plan.steps)):
        return "execution_node"
    return "reasoning_node"


def _reflection_router(state_dict: dict) -> Literal["reflection_node", "__end__"]:
    """Routes after verification — reflect or finish."""
    state = AgentState.model_validate(state_dict)
    if state.should_reflect:
        return "reflection_node"
    return "__end__"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(memory_system: MemorySystem) -> Any:
    """
    Constructs and compiles the GDARS StateGraph.
    Returns a compiled graph ready to call with .ainvoke() or .astream().
    """
    retrieval_agent = RetrievalNode(memory_system)

    graph = StateGraph(dict)

    # Register all nodes
    graph.add_node("planner_node",      _wrap(planner_node))
    graph.add_node("memory_node",       _wrap(_memory_node_fn(memory_system)))
    graph.add_node("retrieval_node",    _wrap(retrieval_agent.run))
    graph.add_node("execution_node",    _wrap(_execution_node_fn()))
    graph.add_node("reasoning_node",    _wrap(reasoning_node))
    graph.add_node("synthesis_node",    _wrap(synthesis_node))
    graph.add_node("verification_node", _wrap(verification_node))
    graph.add_node("reflection_node",   _wrap(reflection_node))

    # Edges
    graph.add_edge(START, "planner_node")
    graph.add_edge("planner_node", "memory_node")

    graph.add_conditional_edges(
        "memory_node", _post_memory_router,
        {"retrieval_node": "retrieval_node", "reasoning_node": "reasoning_node"},
    )

    graph.add_conditional_edges(
        "retrieval_node", _post_retrieval_router,
        {"execution_node": "execution_node", "reasoning_node": "reasoning_node"},
    )

    graph.add_edge("execution_node",    "reasoning_node")
    graph.add_edge("reasoning_node",    "synthesis_node")
    graph.add_edge("synthesis_node",    "verification_node")

    graph.add_conditional_edges(
        "verification_node", _reflection_router,
        {"reflection_node": "reflection_node", "__end__": END},
    )

    graph.add_edge("reflection_node", "reasoning_node")  # loop back

    return graph.compile()
