"""
AgentState validation tests.
These run with zero LLM calls — pure data model tests.
"""
import pytest
from agents.state import (
    AgentState, ThoughtStep, EvidenceFragment, ExecutionStep,
    ExecutionPlan, UserProfile, InferredGoal, StepScore,
    QueryComplexity, SkillLevel, NodeStatus,
)


class TestAgentStateDefaults:
    def test_creates_with_query_only(self):
        s = AgentState(user_query="test")
        assert s.user_query == "test"
        assert s.iteration == 0
        assert s.is_complete is False
        assert s.complexity == QueryComplexity.COMPOUND

    def test_session_id_auto_generated(self):
        s1 = AgentState(user_query="a")
        s2 = AgentState(user_query="b")
        assert s1.session_id != s2.session_id

    def test_request_id_auto_generated(self):
        s = AgentState(user_query="test")
        assert len(s.request_id) == 12

    def test_all_list_fields_default_empty(self):
        s = AgentState(user_query="test")
        for field in ["conversation_history", "thought_steps", "evidence",
                      "node_trace", "error_log", "models_used", "next_actions",
                      "step_scores", "constraints"]:
            assert getattr(s, field) == [], f"{field} should default to []"

    def test_serialisation_roundtrip(self):
        s = AgentState(user_query="roundtrip test")
        s.log_node("planner_node")
        s.add_thought("thinking", "concluded X", model="test-model")
        dumped   = s.model_dump()
        restored = AgentState.model_validate(dumped)
        assert restored.user_query    == s.user_query
        assert restored.node_trace    == s.node_trace
        assert len(restored.thought_steps) == 1


class TestAgentStateValidators:
    def test_confidence_clamped_above_one(self):
        s = AgentState(user_query="test", confidence_score=1.5)
        assert s.confidence_score == 1.0

    def test_confidence_clamped_below_zero(self):
        s = AgentState(user_query="test", confidence_score=-0.3)
        assert s.confidence_score == 0.0

    def test_avg_factuality_clamped(self):
        s = AgentState(user_query="test", avg_factuality=2.0)
        assert s.avg_factuality == 1.0

    def test_iteration_non_negative(self):
        s = AgentState(user_query="test", iteration=-5)
        assert s.iteration == 0


class TestAgentStateHelpers:
    def test_log_node_appends(self):
        s = AgentState(user_query="test")
        s.log_node("planner_node")
        s.log_node("retrieval_node")
        assert s.node_trace == ["planner_node", "retrieval_node"]

    def test_record_error_appends(self):
        s = AgentState(user_query="test")
        s.record_error("something broke")
        assert "something broke" in s.error_log

    def test_record_model_deduplicates(self):
        s = AgentState(user_query="test")
        s.record_model("llama3-8b-8192")
        s.record_model("llama3-8b-8192")
        s.record_model("llama-3.3-70b-versatile")
        assert s.models_used == ["llama3-8b-8192", "llama-3.3-70b-versatile"]

    def test_add_thought_returns_step(self):
        s    = AgentState(user_query="test")
        step = s.add_thought("thinking hard", "I concluded X", model="test-model")
        assert isinstance(step, ThoughtStep)
        assert step.step_num == 1
        assert step.model_used == "test-model"
        assert len(s.thought_steps) == 1

    def test_add_thought_increments_step_num(self):
        s = AgentState(user_query="test")
        s.add_thought("step 1 thought", "step 1 conclusion")
        s.add_thought("step 2 thought", "step 2 conclusion")
        assert s.thought_steps[0].step_num == 1
        assert s.thought_steps[1].step_num == 2

    def test_should_run_full_pipeline_compound(self):
        s = AgentState(user_query="test", complexity=QueryComplexity.COMPOUND)
        assert s.should_run_full_pipeline() is True

    def test_should_run_full_pipeline_simple(self):
        s = AgentState(user_query="test", complexity=QueryComplexity.SIMPLE)
        assert s.should_run_full_pipeline() is False

    def test_should_run_reflection_novel_only(self):
        s = AgentState(
            user_query="test",
            complexity=QueryComplexity.NOVEL,
            should_reflect=True,
            iteration=2,
            max_iterations=8,
        )
        assert s.should_run_reflection() is True

    def test_should_run_reflection_false_for_compound(self):
        s = AgentState(
            user_query="test",
            complexity=QueryComplexity.COMPOUND,
            should_reflect=True,
        )
        assert s.should_run_reflection() is False

    def test_should_run_reflection_false_at_max_iterations(self):
        s = AgentState(
            user_query="test",
            complexity=QueryComplexity.NOVEL,
            should_reflect=True,
            iteration=8,
            max_iterations=8,
        )
        assert s.should_run_reflection() is False


class TestSubModels:
    def test_thought_step_default_scores(self):
        step = ThoughtStep(step_num=1, thought="t", conclusion="c")
        assert step.factuality_score == 0.0
        assert step.usefulness_score == 0.0

    def test_evidence_fragment_default_credibility(self):
        ef = EvidenceFragment(content="some text", source_type="web")
        assert ef.credibility_score == 0.5

    def test_execution_step_default_status(self):
        step = ExecutionStep(title="Do something", description="desc")
        assert step.status == NodeStatus.PENDING
        assert step.tool == "none"

    def test_inferred_goal_defaults(self):
        goal = InferredGoal(surface_query="q", true_goal="g")
        assert goal.distance_estimate == 1.0
        assert goal.latent_objectives == []

    def test_user_profile_defaults(self):
        p = UserProfile()
        assert p.skill_level == SkillLevel.INTERMEDIATE
        assert p.code_oriented is True
        assert p.tone == "neutral"

    def test_step_score_creation(self):
        score = StepScore(step_num=1, factuality=0.8, usefulness=0.7, reasoning="ok")
        assert score.factuality == 0.8
        assert score.usefulness == 0.7


class TestQueryComplexityRouting:
    def test_simple_skips_full_pipeline(self):
        s = AgentState(user_query="what is 2+2?",
                       complexity=QueryComplexity.SIMPLE)
        assert not s.should_run_full_pipeline()
        assert not s.should_run_reflection()

    def test_novel_allows_reflection(self):
        s = AgentState(
            user_query="design a new programming language",
            complexity=QueryComplexity.NOVEL,
            should_reflect=True,
            iteration=0,
            max_iterations=8,
        )
        assert s.should_run_full_pipeline()
        assert s.should_run_reflection()
