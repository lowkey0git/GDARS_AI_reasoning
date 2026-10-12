"""
Local demo — runs the full pipeline without starting a server.
Usage:
    python scripts/run_local.py
    python scripts/run_local.py 1   # runs example index 1
    python scripts/run_local.py 2   # runs example index 2
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.graph import build_graph
from agents.state import AgentState, UserProfile, SkillLevel
from memory.memory_system import MemorySystem
from config.settings import get_settings
from utils.logging import setup_logging


EXAMPLES = [
    {
        "query":   "How do I build a production REST API in Python?",
        "profile": {"skill_level": "intermediate", "code_oriented": True, "tone": "direct"},
    },
    {
        "query":   "What is the difference between episodic and semantic memory in AI?",
        "profile": {"skill_level": "beginner", "code_oriented": False, "tone": "encouraging"},
    },
    {
        "query":   "Design the architecture for a real-time ML feature store",
        "profile": {"skill_level": "advanced", "code_oriented": True, "math_depth": "intuition"},
    },
]


async def run(example: dict) -> None:
    cfg = get_settings()
    setup_logging(level=cfg.api.log_level, environment=cfg.environment)

    print(f"\n{'='*65}")
    print("GDARS v2 — Goal-Directed Adaptive Reasoning System")
    print(f"{'='*65}")
    print(f"Query: {example['query']}\n")

    memory = MemorySystem()
    graph  = build_graph(memory)

    profile = UserProfile(**example["profile"])
    initial = AgentState(
        user_query=example["query"],
        user_profile=profile,
        max_iterations=cfg.planner.max_iterations,
    )

    result_dict = await graph.ainvoke(initial.model_dump())

    from agents.state import AgentState as AS
    result = AS.model_validate(result_dict)

    print(f"\n{'─'*65}")
    print(f"COMPLEXITY:    {result.complexity.value.upper()}")
    print(f"CONFIDENCE:    {result.confidence_score:.2f}")
    print(f"FACTUALITY:    {result.avg_factuality:.2f}")
    print(f"USEFULNESS:    {result.avg_usefulness:.2f}")
    print(f"MODELS USED:   {', '.join(result.models_used)}")
    print(f"NODE TRACE:    {' → '.join(result.node_trace)}")

    if result.inferred_goal:
        print(f"\nINFERRED GOAL: {result.inferred_goal.true_goal}")
        print(f"LATENT:        {', '.join(result.inferred_goal.latent_objectives[:3])}")

    print(f"\n{'─'*65}")
    print("FINAL ANSWER")
    print(f"{'─'*65}\n")
    print(result.final_solution)

    if result.next_actions:
        print(f"\n{'─'*65}")
        print("NEXT STEPS")
        for i, a in enumerate(result.next_actions[:5], 1):
            print(f"  {i}. {a}")

    if result.error_log:
        print(f"\nWARNINGS: {'; '.join(result.error_log)}")

    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    ex  = EXAMPLES[min(idx, len(EXAMPLES) - 1)]
    asyncio.run(run(ex))
