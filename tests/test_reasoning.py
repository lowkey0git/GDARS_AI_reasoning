"""
Reasoning node tests.
Tests SIMPLE vs CoT paths, mid-chain evaluation integration,
and the step retry mechanism.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agents.state import AgentState, QueryComplexity, ThoughtStep, StepScore


def _make_reasoning_output(steps=2, confidence=0.80):
    """Returns a mock ReasoningOutput with n steps."""
    return MagicMock(
        steps=[
            {"thought": f"thought {i}", "conclusion": f"conclusion {i}"}
            for i in range(1, steps + 1)
        ],
        chain_of_thought="Full reasoning narrative.",
        blockers=[],
        next_actions=["action 1", "action 2"],
        confidence=confidence,
    )


class TestSimpleReasoningPath:
    @pytest.mark.asyncio
    async def test_simple_uses_router_model(self, sample_state):
        """SIMPLE queries must use the cheap 8B router, not the 70B."""
        sample_state.complexity = QueryComplexity.SIMPLE

        with patch("agents.reasoning.get_router_client") as mock_router_fn, \
             patch("agents.reasoning.get_reasoning_client") as mock_reasoner_fn:

            mock_router = MagicMock()
            mock_router.model_name = "llama3-8b-8192"
            mock_router.ainvoke = AsyncMock(return_value="Python uses reference counting.")
            mock_router_fn.return_value = mock_router
            mock_reasoner_fn.return_value = MagicMock()  # should not be called

            from agents.reasoning import ReasoningNode
            result = await ReasoningNode().run(sample_state)

        assert "reasoning_node" in result.node_trace
        # Reasoner should not have been used for SIMPLE
        mock_reasoner_fn.return_value.aextract.assert_not_called()

    @pytest.mark.asyncio
    async def test_simple_sets_high_confidence(self, sample_state):
        sample_state.complexity = QueryComplexity.SIMPLE

        with patch("agents.reasoning.get_router_client") as mock_fn, \
             patch("agents.reasoning.get_reasoning_client"):
            mock_router = MagicMock()
            mock_router.model_name = "test"
            mock_router.ainvoke = AsyncMock(return_value="Direct answer here.")
            mock_fn.return_value = mock_router

            from agents.reasoning import ReasoningNode
            result = await ReasoningNode().run(sample_state)

        # SIMPLE path hardcodes 0.80 confidence
        assert result.confidence_score == pytest.approx(0.80, abs=0.01)

    @pytest.mark.asyncio
    async def test_simple_adds_one_thought_step(self, sample_state):
        sample_state.complexity = QueryComplexity.SIMPLE

        with patch("agents.reasoning.get_router_client") as mock_fn, \
             patch("agents.reasoning.get_reasoning_client"):
            mock_router = MagicMock()
            mock_router.model_name = "test"
            mock_router.ainvoke = AsyncMock(return_value="The answer is 42.")
            mock_fn.return_value = mock_router

            from agents.reasoning import ReasoningNode
            result = await ReasoningNode().run(sample_state)

        assert len(result.thought_steps) == 1

    @pytest.mark.asyncio
    async def test_simple_fallback_on_error(self, sample_state):
        sample_state.complexity = QueryComplexity.SIMPLE

        with patch("agents.reasoning.get_router_client") as mock_fn, \
             patch("agents.reasoning.get_reasoning_client"):
            mock_router = MagicMock()
            mock_router.model_name = "test"
            mock_router.ainvoke = AsyncMock(side_effect=Exception("LLM timeout"))
            mock_fn.return_value = mock_router

            from agents.reasoning import ReasoningNode
            result = await ReasoningNode().run(sample_state)

        assert result.confidence_score == pytest.approx(0.4, abs=0.01)
        assert len(result.error_log) > 0


class TestChainReasoningPath:
    @pytest.mark.asyncio
    async def test_compound_adds_thought_steps(self, sample_state_with_goal):
        sample_state_with_goal.complexity = QueryComplexity.COMPOUND

        with patch("agents.reasoning.get_reasoning_client") as mock_fn, \
             patch("agents.reasoning.get_router_client") as mock_router_fn, \
             patch("agents.reasoning.EvaluationHarness") as mock_eval:

            mock_reasoner = MagicMock()
            mock_reasoner.model_name = "llama-3.3-70b-versatile"
            mock_reasoner.aextract = AsyncMock(
                return_value=_make_reasoning_output(steps=3, confidence=0.82)
            )
            mock_fn.return_value = mock_reasoner

            mock_router = MagicMock()
            mock_router.model_name = "llama3-8b-8192"
            mock_router_fn.return_value = mock_router

            mock_harness = MagicMock()
            mock_harness.score_state = AsyncMock(return_value=sample_state_with_goal)
            mock_harness._cfg = MagicMock(
                score_intermediate_steps=True,
                min_factuality_score=0.55,
                min_usefulness_score=0.50,
            )
            mock_eval.return_value = mock_harness

            from agents.reasoning import ReasoningNode
            result = await ReasoningNode().run(sample_state_with_goal)

        assert len(result.thought_steps) == 3
        assert result.chain_of_thought == "Full reasoning narrative."
        assert result.next_actions == ["action 1", "action 2"]
        assert result.confidence_score == pytest.approx(0.82, abs=0.01)

    @pytest.mark.asyncio
    async def test_compound_records_model_used(self, sample_state_with_goal):
        sample_state_with_goal.complexity = QueryComplexity.COMPOUND

        with patch("agents.reasoning.get_reasoning_client") as mock_fn, \
             patch("agents.reasoning.get_router_client"), \
             patch("agents.reasoning.EvaluationHarness") as mock_eval:

            mock_reasoner = MagicMock()
            mock_reasoner.model_name = "llama-3.3-70b-versatile"
            mock_reasoner.aextract = AsyncMock(
                return_value=_make_reasoning_output(steps=2)
            )
            mock_fn.return_value = mock_reasoner

            mock_harness = MagicMock()
            mock_harness.score_state = AsyncMock(return_value=sample_state_with_goal)
            mock_harness._cfg = MagicMock(score_intermediate_steps=False)
            mock_eval.return_value = mock_harness

            from agents.reasoning import ReasoningNode
            result = await ReasoningNode().run(sample_state_with_goal)

        assert "llama-3.3-70b-versatile" in result.models_used

    @pytest.mark.asyncio
    async def test_novel_triggers_mid_chain_eval(self, sample_state_with_goal):
        """NOVEL queries run the evaluator after each step."""
        sample_state_with_goal.complexity = QueryComplexity.NOVEL

        with patch("agents.reasoning.get_reasoning_client") as mock_fn, \
             patch("agents.reasoning.get_router_client"), \
             patch("agents.reasoning.EvaluationHarness") as mock_eval_cls:

            mock_reasoner = MagicMock()
            mock_reasoner.model_name = "test"
            mock_reasoner.aextract = AsyncMock(
                return_value=_make_reasoning_output(steps=2)
            )
            mock_fn.return_value = mock_reasoner

            # Good scores — no retry needed
            mock_harness = MagicMock()
            mock_harness.score_step = AsyncMock(return_value=StepScore(
                step_num=1, factuality=0.85, usefulness=0.80, reasoning="good"
            ))
            mock_harness.should_retry_step = MagicMock(return_value=False)
            mock_harness.score_state = AsyncMock(return_value=sample_state_with_goal)
            mock_harness._cfg = MagicMock(
                score_intermediate_steps=True,
                min_factuality_score=0.55,
                min_usefulness_score=0.50,
            )
            mock_eval_cls.return_value = mock_harness

            from agents.reasoning import ReasoningNode
            result = await ReasoningNode().run(sample_state_with_goal)

        # score_step should have been called once per thought step
        assert mock_harness.score_step.call_count == 2

    @pytest.mark.asyncio
    async def test_novel_retries_low_scoring_step(self, sample_state_with_goal):
        """A step below threshold triggers one retry."""
        sample_state_with_goal.complexity = QueryComplexity.NOVEL

        with patch("agents.reasoning.get_reasoning_client") as mock_fn, \
             patch("agents.reasoning.get_router_client"), \
             patch("agents.reasoning.EvaluationHarness") as mock_eval_cls:

            mock_reasoner = MagicMock()
            mock_reasoner.model_name = "test"
            mock_reasoner.aextract = AsyncMock(
                return_value=_make_reasoning_output(steps=1)
            )
            # ainvoke used for the retry corrective call
            mock_reasoner.ainvoke = AsyncMock(return_value="Improved conclusion after retry.")
            mock_fn.return_value = mock_reasoner

            mock_harness = MagicMock()
            mock_harness.score_step = AsyncMock(return_value=StepScore(
                step_num=1, factuality=0.30, usefulness=0.25,
                reasoning="Very low quality step"
            ))
            mock_harness.should_retry_step = MagicMock(return_value=True)  # triggers retry
            mock_harness.score_state = AsyncMock(return_value=sample_state_with_goal)
            mock_harness._cfg = MagicMock(
                score_intermediate_steps=True,
                min_factuality_score=0.55,
                min_usefulness_score=0.50,
            )
            mock_eval_cls.return_value = mock_harness

            from agents.reasoning import ReasoningNode
            result = await ReasoningNode().run(sample_state_with_goal)

        # The retry ainvoke should have been called
        mock_reasoner.ainvoke.assert_called_once()
        # The conclusion should be the improved one
        assert result.thought_steps[0].conclusion == "Improved conclusion after retry."

    @pytest.mark.asyncio
    async def test_reasoning_fallback_on_llm_error(self, sample_state_with_goal):
        """LLM error during chain reasoning → low confidence, no crash."""
        sample_state_with_goal.complexity = QueryComplexity.COMPOUND

        with patch("agents.reasoning.get_reasoning_client") as mock_fn, \
             patch("agents.reasoning.get_router_client"), \
             patch("agents.reasoning.EvaluationHarness") as mock_eval:

            mock_reasoner = MagicMock()
            mock_reasoner.model_name = "test"
            mock_reasoner.aextract = AsyncMock(side_effect=Exception("timeout"))
            mock_fn.return_value = mock_reasoner
            mock_eval.return_value = MagicMock()

            from agents.reasoning import ReasoningNode
            result = await ReasoningNode().run(sample_state_with_goal)

        assert result.confidence_score == pytest.approx(0.35, abs=0.01)
        assert len(result.error_log) > 0


class TestReasoningNodeTrace:
    @pytest.mark.asyncio
    async def test_always_logs_node(self, sample_state):
        sample_state.complexity = QueryComplexity.SIMPLE

        with patch("agents.reasoning.get_router_client") as mock_fn, \
             patch("agents.reasoning.get_reasoning_client"):
            mock_fn.return_value = MagicMock(
                model_name="test",
                ainvoke=AsyncMock(side_effect=Exception("fail")),
            )

            from agents.reasoning import ReasoningNode
            result = await ReasoningNode().run(sample_state)

        assert "reasoning_node" in result.node_trace
