"""
Synthesis, Verification, and Reflection node tests.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agents.state import AgentState, QueryComplexity


class TestSynthesisNode:
    @pytest.mark.asyncio
    async def test_sets_final_solution(self, sample_state_with_goal):
        with patch("agents.synthesis.get_synthesis_client") as mock_fn:
            mock_llm = MagicMock()
            mock_llm.model_name = "llama-3.1-8b-instant"
            mock_llm.ainvoke = AsyncMock(return_value="## Answer\n\nHere is the final answer.")
            mock_fn.return_value = mock_llm

            from agents.synthesis import SynthesisNode
            result = await SynthesisNode().run(sample_state_with_goal)

        assert result.final_solution == "## Answer\n\nHere is the final answer."
        assert result.is_complete is True

    @pytest.mark.asyncio
    async def test_marks_complete_on_success(self, sample_state_with_goal):
        with patch("agents.synthesis.get_synthesis_client") as mock_fn:
            mock_llm = MagicMock()
            mock_llm.model_name = "test"
            mock_llm.ainvoke = AsyncMock(return_value="Done.")
            mock_fn.return_value = mock_llm

            from agents.synthesis import SynthesisNode
            result = await SynthesisNode().run(sample_state_with_goal)

        assert result.is_complete is True

    @pytest.mark.asyncio
    async def test_fallback_to_chain_of_thought_on_error(self, sample_state_with_goal):
        sample_state_with_goal.chain_of_thought = "Fallback content from reasoning."

        with patch("agents.synthesis.get_synthesis_client") as mock_fn:
            mock_llm = MagicMock()
            mock_llm.model_name = "test"
            mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM down"))
            mock_fn.return_value = mock_llm

            from agents.synthesis import SynthesisNode
            result = await SynthesisNode().run(sample_state_with_goal)

        # Should fall back to chain_of_thought
        assert result.final_solution == "Fallback content from reasoning."
        assert result.is_complete is True

    @pytest.mark.asyncio
    async def test_records_synthesis_model(self, sample_state_with_goal):
        with patch("agents.synthesis.get_synthesis_client") as mock_fn:
            mock_llm = MagicMock()
            mock_llm.model_name = "llama-3.1-8b-instant"
            mock_llm.ainvoke = AsyncMock(return_value="Answer.")
            mock_fn.return_value = mock_llm

            from agents.synthesis import SynthesisNode
            result = await SynthesisNode().run(sample_state_with_goal)

        assert "llama-3.1-8b-instant" in result.models_used

    @pytest.mark.asyncio
    async def test_logs_synthesis_node(self, sample_state_with_goal):
        with patch("agents.synthesis.get_synthesis_client") as mock_fn:
            mock_fn.return_value = MagicMock(
                model_name="test",
                ainvoke=AsyncMock(return_value="done"),
            )

            from agents.synthesis import SynthesisNode
            result = await SynthesisNode().run(sample_state_with_goal)

        assert "synthesis_node" in result.node_trace

    def test_beginner_style_instructions_include_definitions(self):
        from agents.synthesis import _build_style_instructions
        from agents.state import UserProfile, SkillLevel
        state = AgentState(user_query="test")
        state.user_profile = UserProfile(skill_level=SkillLevel.BEGINNER)
        instructions = _build_style_instructions(state)
        assert "Define" in instructions or "plain" in instructions.lower()

    def test_expert_style_instructions_skip_basics(self):
        from agents.synthesis import _build_style_instructions
        from agents.state import UserProfile, SkillLevel
        state = AgentState(user_query="test")
        state.user_profile = UserProfile(skill_level=SkillLevel.EXPERT)
        instructions = _build_style_instructions(state)
        assert "scaffolding" in instructions.lower() or "expert" in instructions.lower()

    def test_direct_tone_in_instructions(self):
        from agents.synthesis import _build_style_instructions
        from agents.state import UserProfile, SkillLevel
        state = AgentState(user_query="test")
        state.user_profile = UserProfile(skill_level=SkillLevel.INTERMEDIATE, tone="direct")
        instructions = _build_style_instructions(state)
        assert "direct" in instructions.lower() or "hedging" in instructions.lower()


class TestVerificationNode:
    @pytest.mark.asyncio
    async def test_updates_confidence_score(self, sample_state_with_goal):
        sample_state_with_goal.final_solution = "The answer is FastAPI."

        with patch("agents.verification.get_router_client") as mock_fn, \
             patch("agents.verification.get_settings") as mock_cfg:

            mock_cfg.return_value = MagicMock(
                planner=MagicMock(reflection_threshold=0.60)
            )
            mock_llm = MagicMock()
            mock_llm.model_name = "test"
            mock_llm.aextract = AsyncMock(return_value=MagicMock(
                confidence=0.88,
                hallucination_flags=[],
                contradictions=[],
                goal_aligned=True,
                should_reflect=False,
                notes="Looks good.",
            ))
            mock_fn.return_value = mock_llm

            from agents.verification import VerificationNode
            result = await VerificationNode().run(sample_state_with_goal)

        assert result.confidence_score == pytest.approx(0.88, abs=0.01)
        assert result.should_reflect is False

    @pytest.mark.asyncio
    async def test_triggers_reflection_for_novel_low_confidence(self, sample_state_with_goal):
        sample_state_with_goal.complexity  = QueryComplexity.NOVEL
        sample_state_with_goal.final_solution = "Some uncertain answer."
        sample_state_with_goal.iteration   = 1
        sample_state_with_goal.max_iterations = 8

        with patch("agents.verification.get_router_client") as mock_fn, \
             patch("agents.verification.get_settings") as mock_cfg:

            mock_cfg.return_value = MagicMock(
                planner=MagicMock(reflection_threshold=0.60)
            )
            mock_llm = MagicMock()
            mock_llm.model_name = "test"
            mock_llm.aextract = AsyncMock(return_value=MagicMock(
                confidence=0.45,
                hallucination_flags=["unverified claim"],
                contradictions=[],
                goal_aligned=False,
                should_reflect=True,
                notes="Several issues found.",
            ))
            mock_fn.return_value = mock_llm

            from agents.verification import VerificationNode
            result = await VerificationNode().run(sample_state_with_goal)

        assert result.should_reflect is True
        assert result.confidence_score < 0.60

    @pytest.mark.asyncio
    async def test_no_reflection_for_compound_regardless_of_confidence(self, sample_state_with_goal):
        """Reflection is NOVEL-only — COMPOUND never reflects."""
        sample_state_with_goal.complexity  = QueryComplexity.COMPOUND
        sample_state_with_goal.final_solution = "An answer."

        with patch("agents.verification.get_router_client") as mock_fn, \
             patch("agents.verification.get_settings") as mock_cfg:

            mock_cfg.return_value = MagicMock(
                planner=MagicMock(reflection_threshold=0.60)
            )
            mock_llm = MagicMock()
            mock_llm.model_name = "test"
            mock_llm.aextract = AsyncMock(return_value=MagicMock(
                confidence=0.40,          # below threshold
                hallucination_flags=["flag"],
                contradictions=[],
                goal_aligned=False,
                should_reflect=True,      # LLM says yes
                notes="Problems found.",
            ))
            mock_fn.return_value = mock_llm

            from agents.verification import VerificationNode
            result = await VerificationNode().run(sample_state_with_goal)

        # COMPOUND: should_reflect must be False regardless
        assert result.should_reflect is False

    @pytest.mark.asyncio
    async def test_skips_verification_when_no_draft(self, sample_state):
        """No draft available → skip verification gracefully."""
        with patch("agents.verification.get_router_client") as mock_fn:
            mock_fn.return_value = MagicMock(aextract=AsyncMock())

            from agents.verification import VerificationNode
            result = await VerificationNode().run(sample_state)

        # aextract should not have been called — no draft to verify
        mock_fn.return_value.aextract.assert_not_called()


class TestReflectionNode:
    @pytest.mark.asyncio
    async def test_increments_iteration(self, sample_state_with_goal):
        sample_state_with_goal.final_solution = "Current draft."
        sample_state_with_goal.iteration = 2

        with patch("agents.verification.get_router_client") as mock_fn:
            mock_llm = MagicMock()
            mock_llm.model_name = "test"
            mock_llm.ainvoke = AsyncMock(return_value="Improved answer.")
            mock_fn.return_value = mock_llm

            from agents.verification import ReflectionNode
            result = await ReflectionNode().run(sample_state_with_goal)

        assert result.iteration == 3

    @pytest.mark.asyncio
    async def test_updates_final_solution(self, sample_state_with_goal):
        sample_state_with_goal.final_solution = "Old answer."

        with patch("agents.verification.get_router_client") as mock_fn:
            mock_llm = MagicMock()
            mock_llm.model_name = "test"
            mock_llm.ainvoke = AsyncMock(return_value="Improved answer after reflection.")
            mock_fn.return_value = mock_llm

            from agents.verification import ReflectionNode
            result = await ReflectionNode().run(sample_state_with_goal)

        assert result.final_solution == "Improved answer after reflection."
        assert result.adaptive_response == "Improved answer after reflection."

    @pytest.mark.asyncio
    async def test_reflection_fallback_on_error(self, sample_state_with_goal):
        """LLM error during reflection → keeps existing draft, no crash."""
        original = "Original draft must survive."
        sample_state_with_goal.final_solution = original

        with patch("agents.verification.get_router_client") as mock_fn:
            mock_llm = MagicMock()
            mock_llm.model_name = "test"
            mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM down"))
            mock_fn.return_value = mock_llm

            from agents.verification import ReflectionNode
            result = await ReflectionNode().run(sample_state_with_goal)

        # Draft should be unchanged
        assert result.final_solution == original
        assert len(result.error_log) > 0

    @pytest.mark.asyncio
    async def test_reflection_logs_node(self, sample_state_with_goal):
        sample_state_with_goal.final_solution = "draft"

        with patch("agents.verification.get_router_client") as mock_fn:
            mock_fn.return_value = MagicMock(
                model_name="test",
                ainvoke=AsyncMock(return_value="improved"),
            )

            from agents.verification import ReflectionNode
            result = await ReflectionNode().run(sample_state_with_goal)

        assert "reflection_node" in result.node_trace
