"""
Hierarchical Planner tests.
All LLM calls are mocked — these tests verify routing logic and
state transformations, not LLM output quality.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agents.state import AgentState, QueryComplexity


class TestComplexityRouter:
    @pytest.mark.asyncio
    async def test_routes_simple_query(self, sample_state):
        with patch("agents.planner.get_router_client") as mock_router_fn, \
             patch("agents.planner.get_reasoning_client") as mock_reason_fn:

            mock_router = MagicMock()
            mock_router.model_name = "llama3-8b-8192"
            mock_router.aextract = AsyncMock(return_value=MagicMock(
                complexity="simple", confidence=0.92, reason="Basic factual query"
            ))
            mock_router_fn.return_value  = mock_router
            mock_reason_fn.return_value  = MagicMock()

            from agents.planner import HierarchicalPlanner
            result = await HierarchicalPlanner().run(sample_state)

        assert result.complexity == QueryComplexity.SIMPLE
        assert result.complexity_confidence == 0.92
        # SIMPLE queries must not invoke the reasoning client for planning
        mock_reason_fn.return_value.aextract.assert_not_called()

    @pytest.mark.asyncio
    async def test_routes_novel_query(self, sample_state):
        with patch("agents.planner.get_router_client") as mock_router_fn, \
             patch("agents.planner.get_reasoning_client") as mock_reason_fn:

            mock_router = MagicMock()
            mock_router.model_name = "llama3-8b-8192"
            mock_router.aextract = AsyncMock(return_value=MagicMock(
                complexity="novel", confidence=0.85, reason="Open-ended design question"
            ))
            mock_router_fn.return_value = mock_router

            mock_reasoner = MagicMock()
            mock_reasoner.model_name = "llama-3.3-70b-versatile"
            mock_reasoner.aextract = AsyncMock(side_effect=[
                # Goal inference
                MagicMock(
                    true_goal="Design a scalable system",
                    latent_objectives=["scalability", "reliability"],
                    current_state="no system", desired_state="working system",
                    success_criteria=["handles 10k rps"],
                    distance_estimate=0.9, needs_web_search=True,
                    needs_code=False, constraints=[],
                ),
                # Plan generation
                MagicMock(
                    objective="Design scalable system",
                    steps=[{"title": "Research", "description": "Look up patterns", "tool": "web_search"}],
                    needs_web_search=True, needs_code=False,
                ),
            ])
            mock_reason_fn.return_value = mock_reasoner

            from agents.planner import HierarchicalPlanner
            result = await HierarchicalPlanner().run(sample_state)

        assert result.complexity == QueryComplexity.NOVEL
        assert result.inferred_goal is not None
        assert result.inferred_goal.true_goal == "Design a scalable system"
        assert result.execution_plan is not None
        assert len(result.execution_plan.steps) == 1

    @pytest.mark.asyncio
    async def test_low_confidence_defaults_to_compound(self, sample_state):
        """Router confidence below cutoff → COMPOUND regardless of router output."""
        with patch("agents.planner.get_router_client") as mock_fn, \
             patch("agents.planner.get_reasoning_client") as mock_reason_fn:

            mock_router = MagicMock()
            mock_router.model_name = "llama3-8b-8192"
            mock_router.aextract = AsyncMock(return_value=MagicMock(
                complexity="simple", confidence=0.50,  # below 0.70 cutoff
                reason="Not sure"
            ))
            mock_fn.return_value = mock_router

            mock_reasoner = MagicMock()
            mock_reasoner.model_name = "test"
            mock_reasoner.aextract = AsyncMock(side_effect=[
                MagicMock(true_goal="g", latent_objectives=[], current_state="",
                          desired_state="", success_criteria=[], distance_estimate=0.5,
                          needs_web_search=False, needs_code=False, constraints=[]),
                MagicMock(objective="o", steps=[], needs_web_search=False, needs_code=False),
            ])
            mock_reason_fn.return_value = mock_reasoner

            from agents.planner import HierarchicalPlanner
            result = await HierarchicalPlanner().run(sample_state)

        assert result.complexity == QueryComplexity.COMPOUND

    @pytest.mark.asyncio
    async def test_router_failure_defaults_to_compound(self, sample_state):
        """LLM failure during routing → defaults to COMPOUND safely."""
        with patch("agents.planner.get_router_client") as mock_fn, \
             patch("agents.planner.get_reasoning_client") as mock_reason_fn:

            mock_router = MagicMock()
            mock_router.model_name = "test"
            mock_router.aextract = AsyncMock(side_effect=Exception("network error"))
            mock_fn.return_value = mock_router

            mock_reasoner = MagicMock()
            mock_reasoner.model_name = "test"
            mock_reasoner.aextract = AsyncMock(side_effect=Exception("network error"))
            mock_reason_fn.return_value = mock_reasoner

            from agents.planner import HierarchicalPlanner
            result = await HierarchicalPlanner().run(sample_state)

        assert result.complexity == QueryComplexity.COMPOUND
        assert len(result.error_log) > 0

    @pytest.mark.asyncio
    async def test_planner_node_trace(self, sample_state):
        with patch("agents.planner.get_router_client") as mock_fn, \
             patch("agents.planner.get_reasoning_client") as mock_reason_fn:

            mock_router = MagicMock()
            mock_router.model_name = "test"
            mock_router.aextract = AsyncMock(side_effect=Exception("fail"))
            mock_fn.return_value = mock_router
            mock_reason_fn.return_value = MagicMock(aextract=AsyncMock(side_effect=Exception("fail")))

            from agents.planner import HierarchicalPlanner
            result = await HierarchicalPlanner().run(sample_state)

        assert "planner_node" in result.node_trace

    @pytest.mark.asyncio
    async def test_goal_inference_fallback_on_error(self, sample_state):
        """Goal inference failure → falls back to query as goal."""
        with patch("agents.planner.get_router_client") as mock_router_fn, \
             patch("agents.planner.get_reasoning_client") as mock_reason_fn:

            mock_router = MagicMock()
            mock_router.model_name = "test"
            mock_router.aextract = AsyncMock(return_value=MagicMock(
                complexity="compound", confidence=0.85, reason="ok"
            ))
            mock_router_fn.return_value = mock_router

            mock_reasoner = MagicMock()
            mock_reasoner.model_name = "test"
            mock_reasoner.aextract = AsyncMock(side_effect=Exception("LLM down"))
            mock_reason_fn.return_value = mock_reasoner

            from agents.planner import HierarchicalPlanner
            result = await HierarchicalPlanner().run(sample_state)

        # Should fall back to query as true goal
        assert result.inferred_goal is not None
        assert result.inferred_goal.true_goal == sample_state.user_query
