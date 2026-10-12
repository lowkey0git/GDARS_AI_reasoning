"""
Evaluation Harness tests.
Tests scoring logic, threshold decisions, and state aggregation.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agents.state import AgentState, ThoughtStep, StepScore, QueryComplexity


class TestStepScoring:
    @pytest.mark.asyncio
    async def test_score_step_returns_step_score(self):
        with patch("evaluation.harness.get_router_client") as mock_fn:
            mock_llm = MagicMock()
            mock_llm.aextract = AsyncMock(return_value=MagicMock(
                factuality=0.85, usefulness=0.75, reasoning="Well grounded."
            ))
            mock_fn.return_value = mock_llm

            from evaluation.harness import EvaluationHarness
            harness = EvaluationHarness()
            step = ThoughtStep(step_num=1, thought="Python uses GC", conclusion="Memory is managed")
            score = await harness.score_step(step, "understand Python memory", "Python uses garbage collection")

        assert isinstance(score, StepScore)
        assert score.factuality == 0.85
        assert score.usefulness == 0.75
        assert score.step_num == 1

    @pytest.mark.asyncio
    async def test_score_step_fallback_on_error(self):
        """LLM failure returns neutral 0.5/0.5 scores — doesn't crash."""
        with patch("evaluation.harness.get_router_client") as mock_fn:
            mock_llm = MagicMock()
            mock_llm.aextract = AsyncMock(side_effect=Exception("LLM down"))
            mock_fn.return_value = mock_llm

            from evaluation.harness import EvaluationHarness
            harness = EvaluationHarness()
            step  = ThoughtStep(step_num=1, thought="t", conclusion="c")
            score = await harness.score_step(step, "goal", "evidence")

        assert score.factuality == 0.5
        assert score.usefulness == 0.5

    @pytest.mark.asyncio
    async def test_score_state_updates_aggregates(self, sample_state_with_goal):
        with patch("evaluation.harness.get_router_client") as mock_fn:
            mock_llm = MagicMock()
            mock_llm.aextract = AsyncMock(return_value=MagicMock(
                factuality=0.80, usefulness=0.70, reasoning="ok"
            ))
            mock_fn.return_value = mock_llm

            # Add two thought steps to the state
            sample_state_with_goal.add_thought("t1", "c1")
            sample_state_with_goal.add_thought("t2", "c2")

            from evaluation.harness import EvaluationHarness
            result = await EvaluationHarness().score_state(sample_state_with_goal)

        assert result.avg_factuality == pytest.approx(0.80, abs=0.01)
        assert result.avg_usefulness == pytest.approx(0.70, abs=0.01)
        assert len(result.step_scores) == 2

    @pytest.mark.asyncio
    async def test_score_state_skips_already_scored(self, sample_state_with_goal):
        """Steps that already have scores in step_scores are not re-scored."""
        with patch("evaluation.harness.get_router_client") as mock_fn:
            mock_llm = MagicMock()
            mock_llm.aextract = AsyncMock(return_value=MagicMock(
                factuality=0.9, usefulness=0.9, reasoning="great"
            ))
            mock_fn.return_value = mock_llm

            sample_state_with_goal.add_thought("t", "c")
            # Pre-populate step_scores as if it was already scored
            sample_state_with_goal.step_scores = [
                StepScore(step_num=1, factuality=0.6, usefulness=0.6, reasoning="already scored")
            ]

            from evaluation.harness import EvaluationHarness
            result = await EvaluationHarness().score_state(sample_state_with_goal)

        # Should not have scored again — still just 1 entry
        assert len(result.step_scores) == 1
        # The original score should be preserved
        assert result.step_scores[0].factuality == 0.6

    @pytest.mark.asyncio
    async def test_score_state_skips_when_disabled(self, sample_state_with_goal):
        with patch("evaluation.harness.get_router_client") as mock_fn, \
             patch("evaluation.harness.get_settings") as mock_cfg_fn:

            mock_cfg = MagicMock()
            mock_cfg.evaluation.score_intermediate_steps = False
            mock_cfg_fn.return_value = mock_cfg

            mock_llm = MagicMock()
            mock_llm.aextract = AsyncMock()
            mock_fn.return_value = mock_llm

            sample_state_with_goal.add_thought("t", "c")

            from evaluation.harness import EvaluationHarness
            result = await EvaluationHarness().score_state(sample_state_with_goal)

        # No scoring should have happened
        mock_llm.aextract.assert_not_called()


class TestShouldRetryStep:
    def test_retry_when_factuality_low(self):
        with patch("evaluation.harness.get_router_client"):
            from evaluation.harness import EvaluationHarness
            harness = EvaluationHarness()
            score   = StepScore(step_num=1, factuality=0.40, usefulness=0.80, reasoning="low fact")
            assert harness.should_retry_step(score) is True

    def test_retry_when_usefulness_low(self):
        with patch("evaluation.harness.get_router_client"):
            from evaluation.harness import EvaluationHarness
            harness = EvaluationHarness()
            score   = StepScore(step_num=1, factuality=0.80, usefulness=0.30, reasoning="low use")
            assert harness.should_retry_step(score) is True

    def test_no_retry_when_both_above_threshold(self):
        with patch("evaluation.harness.get_router_client"):
            from evaluation.harness import EvaluationHarness
            harness = EvaluationHarness()
            score   = StepScore(step_num=1, factuality=0.80, usefulness=0.80, reasoning="good")
            assert harness.should_retry_step(score) is False

    def test_no_retry_at_exact_threshold(self):
        with patch("evaluation.harness.get_router_client"):
            from evaluation.harness import EvaluationHarness
            harness = EvaluationHarness()
            # 0.55 is the min_factuality_score threshold
            score   = StepScore(step_num=1, factuality=0.55, usefulness=0.55, reasoning="exact")
            assert harness.should_retry_step(score) is False
