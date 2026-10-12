"""
Shared test fixtures.

Key design decisions:
  1. _silence_logging (autouse) — patches structlog before any import
     touches it. Prevents the PrintLogger.name AttributeError that
     crashes tests when setup_logging hasn't been called yet.

  2. _mock_settings (autouse) — provides a consistent Settings object
     so tests don't need real API keys and config doesn't vary by
     which .env file happens to exist on the test machine.

  3. mock_llm_client — a reusable fixture for tests that exercise
     agent nodes. Returns configurable AsyncMock responses.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture(autouse=True)
def _silence_logging():
    """
    Patches logging setup before every test.
    structlog's PrintLogger has no .name attribute; this prevents it
    from being created at all during tests.
    """
    with patch("utils.logging.setup_logging"), \
         patch("utils.logging.get_logger", return_value=MagicMock(
             info=MagicMock(), warning=MagicMock(),
             error=MagicMock(), debug=MagicMock(),
         )):
        yield


@pytest.fixture(autouse=True)
def _mock_settings():
    """
    Provides a consistent Settings mock for all tests.
    Tests that need specific settings values can override individual
    attributes on the returned mock.
    """
    from config.settings import get_settings

    mock_cfg = MagicMock()

    # LLM config
    mock_cfg.llm.api_key              = "test-key"
    mock_cfg.llm.router_model         = "llama3-8b-8192"
    mock_cfg.llm.reasoning_model      = "llama-3.3-70b-versatile"
    mock_cfg.llm.synthesis_model      = "llama-3.1-8b-instant"
    mock_cfg.llm.code_model           = "llama-3.3-70b-versatile"
    mock_cfg.llm.temperature_router   = 0.0
    mock_cfg.llm.temperature_reasoning = 0.1
    mock_cfg.llm.temperature_synthesis = 0.2
    mock_cfg.llm.max_tokens           = 4096

    # Memory config
    mock_cfg.memory.redis_url                  = "redis://localhost:6379/0"
    mock_cfg.memory.working_memory_ttl_seconds = 3600
    mock_cfg.memory.chroma_persist_dir         = "./data/test_chroma"
    mock_cfg.memory.faiss_index_path           = "./data/test_faiss"
    mock_cfg.memory.embedding_model            = "all-MiniLM-L6-v2"
    mock_cfg.memory.working_top_k              = 3
    mock_cfg.memory.episodic_top_k             = 5
    mock_cfg.memory.semantic_top_k             = 5

    # Retrieval config
    mock_cfg.retrieval.tavily_api_key    = ""
    mock_cfg.retrieval.max_web_results   = 4
    mock_cfg.retrieval.min_relevance_score = 0.35

    # Planner config
    mock_cfg.planner.max_iterations              = 8
    mock_cfg.planner.confidence_threshold        = 0.78
    mock_cfg.planner.reflection_threshold        = 0.60
    mock_cfg.planner.complexity_confidence_cutoff = 0.70

    # Evaluation config
    mock_cfg.evaluation.min_factuality_score      = 0.55
    mock_cfg.evaluation.min_usefulness_score      = 0.50
    mock_cfg.evaluation.score_intermediate_steps  = True

    # API config
    mock_cfg.api.host      = "0.0.0.0"
    mock_cfg.api.port      = 8000
    mock_cfg.api.log_level = "WARNING"
    mock_cfg.api.log_file  = "./logs/test.log"

    mock_cfg.environment = "development"

    get_settings.cache_clear()
    with patch("config.settings.get_settings", return_value=mock_cfg):
        yield mock_cfg
    get_settings.cache_clear()


@pytest.fixture
def mock_groq_client():
    """
    Returns a mock GroqClient that returns configurable responses.
    Usage:
        mock_groq_client.ainvoke.return_value = "some response"
        mock_groq_client.aextract.return_value = SomePydanticModel(...)
    """
    client = MagicMock()
    client.model_name = "test-model"
    client.ainvoke    = AsyncMock(return_value="default test response")
    client.aextract   = AsyncMock()
    return client


@pytest.fixture
def sample_state():
    """A minimal valid AgentState for use in tests."""
    from agents.state import AgentState
    return AgentState(user_query="How does Python handle memory?")


@pytest.fixture
def sample_state_with_goal():
    """AgentState with inferred goal and evidence pre-populated."""
    from agents.state import AgentState, InferredGoal, EvidenceFragment, QueryComplexity
    state = AgentState(user_query="How do I build a REST API in Python?")
    state.complexity = QueryComplexity.COMPOUND
    state.inferred_goal = InferredGoal(
        surface_query="How do I build a REST API in Python?",
        true_goal="Build a production-ready REST API using Python",
        latent_objectives=["ship working code", "follow best practices"],
        current_state="knows Python basics",
        desired_state="has a working API deployed",
    )
    state.evidence = [
        EvidenceFragment(
            content="FastAPI is a modern Python framework for building APIs with automatic OpenAPI docs.",
            source_type="semantic",
            relevance_score=0.9,
        ),
        EvidenceFragment(
            content="Use Pydantic models for request/response validation in FastAPI.",
            source_type="semantic",
            relevance_score=0.85,
        ),
    ]
    state.evidence_synthesis = "FastAPI is the recommended framework. Use Pydantic for validation."
    return state
