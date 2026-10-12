"""
FastAPI endpoint tests.
Graph and memory are fully mocked — these test HTTP contract only.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


def _completed_state(query: str = "test query") -> dict:
    """Minimal completed AgentState dict."""
    return {
        "session_id":         "test-session-abc",
        "request_id":         "req-001-xyz",
        "iteration":          0,
        "max_iterations":     8,
        "user_query":         query,
        "conversation_history": [],
        "user_profile": {
            "skill_level": "intermediate", "domain": "general",
            "tone": "neutral", "math_depth": "none", "code_oriented": True,
        },
        "complexity":              "compound",
        "complexity_confidence":   0.85,
        "inferred_goal": {
            "surface_query": query, "true_goal": "Understand the topic deeply",
            "latent_objectives": ["learn", "apply"],
            "current_state": "curious", "desired_state": "knowledgeable",
            "success_criteria": [], "distance_estimate": 0.6,
        },
        "execution_plan": None,
        "constraints":    [],
        "sub_queries":    [],
        "evidence":       [],
        "evidence_synthesis": "Some relevant evidence found.",
        "coverage_gaps":  [],
        "working_memory":  [],
        "episodic_memory": [],
        "semantic_memory": [],
        "thought_steps":   [{"step_num": 1, "thought": "t", "conclusion": "c",
                             "factuality_score": 0.8, "usefulness_score": 0.75, "model_used": "test"}],
        "chain_of_thought": "Reasoning complete.",
        "next_actions":     ["action 1", "action 2"],
        "step_scores":      [],
        "avg_factuality":   0.80,
        "avg_usefulness":   0.75,
        "execution_results": [],
        "tool_outputs":     {},
        "adaptive_response":  "The answer is X.",
        "final_solution":     "Here is a complete answer to your query.",
        "confidence_score":   0.82,
        "should_reflect":     False,
        "is_complete":        True,
        "node_trace":         ["planner_node", "memory_node", "retrieval_node",
                               "reasoning_node", "synthesis_node", "verification_node"],
        "error_log":          [],
        "models_used":        ["llama3-8b-8192", "llama-3.3-70b-versatile"],
    }


@pytest.fixture
def client():
    """TestClient with graph and memory fully mocked."""
    mock_graph  = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=_completed_state())
    mock_memory = MagicMock()
    mock_memory.store_interaction = AsyncMock()
    mock_memory.health_check      = AsyncMock(return_value={
        "working_memory": True, "episodic_memory": True, "semantic_memory": True
    })
    mock_memory.episodic = MagicMock()
    mock_memory.episodic.retrieve.return_value = []
    mock_memory.semantic = MagicMock()
    mock_memory.semantic.add.return_value = 0

    with patch("api.app._graph",         mock_graph), \
         patch("api.app._memory_system", mock_memory), \
         patch("api.app._startup_time",  1000.0):
        from api.app import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c, mock_graph, mock_memory


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_returns_200(self, client):
        c, _, _ = client
        resp = c.get("/health")
        assert resp.status_code == 200

    def test_status_ok(self, client):
        c, _, _ = client
        assert c.get("/health").json()["status"] == "ok"

    def test_contains_models(self, client):
        c, _, _ = client
        data = c.get("/health").json()
        assert "models" in data
        assert "router" in data["models"]
        assert "reasoning" in data["models"]

    def test_contains_memory_status(self, client):
        c, _, _ = client
        data = c.get("/health").json()
        assert "memory" in data

    def test_contains_uptime(self, client):
        c, _, _ = client
        data = c.get("/health").json()
        assert "uptime_seconds" in data

    def test_no_auth_required(self, client):
        c, _, _ = client
        assert c.get("/health").status_code == 200


# ── /ready ────────────────────────────────────────────────────────────────────

class TestReadyEndpoint:
    def test_returns_ready_when_initialised(self, client):
        c, _, _ = client
        resp = c.get("/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    def test_returns_503_when_graph_none(self):
        with patch("api.app._graph",         None), \
             patch("api.app._memory_system", MagicMock()):
            from api.app import app
            c = TestClient(app, raise_server_exceptions=False)
            resp = c.get("/ready")
        assert resp.status_code == 503


# ── /query ────────────────────────────────────────────────────────────────────

class TestQueryEndpoint:
    def test_basic_query_returns_200(self, client):
        c, mock_graph, _ = client
        mock_graph.ainvoke = AsyncMock(return_value=_completed_state("hello"))
        assert c.post("/query", json={"query": "hello"}).status_code == 200

    def test_final_solution_in_response(self, client):
        c, mock_graph, _ = client
        mock_graph.ainvoke = AsyncMock(return_value=_completed_state("test"))
        data = c.post("/query", json={"query": "test"}).json()
        assert data["final_solution"] == "Here is a complete answer to your query."

    def test_confidence_score_in_response(self, client):
        c, mock_graph, _ = client
        mock_graph.ainvoke = AsyncMock(return_value=_completed_state("test"))
        data = c.post("/query", json={"query": "test"}).json()
        assert data["confidence_score"] == pytest.approx(0.82, abs=0.01)

    def test_complexity_in_response(self, client):
        c, mock_graph, _ = client
        mock_graph.ainvoke = AsyncMock(return_value=_completed_state("test"))
        data = c.post("/query", json={"query": "test"}).json()
        assert data["complexity"] == "compound"

    def test_models_used_in_response(self, client):
        c, mock_graph, _ = client
        mock_graph.ainvoke = AsyncMock(return_value=_completed_state("test"))
        data = c.post("/query", json={"query": "test"}).json()
        assert isinstance(data["models_used"], list)
        assert len(data["models_used"]) > 0

    def test_factuality_scores_in_response(self, client):
        c, mock_graph, _ = client
        mock_graph.ainvoke = AsyncMock(return_value=_completed_state("test"))
        data = c.post("/query", json={"query": "test"}).json()
        assert "avg_factuality" in data
        assert "avg_usefulness" in data

    def test_empty_query_rejected_422(self, client):
        c, _, _ = client
        assert c.post("/query", json={"query": ""}).status_code == 422

    def test_query_too_long_rejected_422(self, client):
        c, _, _ = client
        assert c.post("/query", json={"query": "x" * 5000}).status_code == 422

    def test_all_required_fields_present(self, client):
        c, mock_graph, _ = client
        mock_graph.ainvoke = AsyncMock(return_value=_completed_state("field check"))
        data = c.post("/query", json={"query": "field check"}).json()
        for field in ["session_id", "request_id", "query", "final_solution",
                      "confidence_score", "complexity", "next_actions",
                      "node_trace", "models_used", "reasoning_steps",
                      "execution_time_ms", "avg_factuality", "avg_usefulness"]:
            assert field in data, f"Missing: {field}"

    def test_execution_time_positive(self, client):
        c, mock_graph, _ = client
        mock_graph.ainvoke = AsyncMock(return_value=_completed_state("time"))
        data = c.post("/query", json={"query": "time"}).json()
        assert data["execution_time_ms"] >= 0

    def test_with_user_profile(self, client):
        c, mock_graph, _ = client
        mock_graph.ainvoke = AsyncMock(return_value=_completed_state("profile"))
        resp = c.post("/query", json={
            "query": "profile test",
            "user_profile": {"skill_level": "beginner", "tone": "encouraging"},
        })
        assert resp.status_code == 200

    def test_with_conversation_history(self, client):
        c, mock_graph, _ = client
        mock_graph.ainvoke = AsyncMock(return_value=_completed_state("follow-up"))
        resp = c.post("/query", json={
            "query": "follow-up",
            "conversation_history": [
                {"role": "user",      "content": "first question"},
                {"role": "assistant", "content": "first answer"},
            ],
        })
        assert resp.status_code == 200

    def test_memory_store_called(self, client):
        c, mock_graph, mock_memory = client
        mock_graph.ainvoke = AsyncMock(return_value=_completed_state("store test"))
        c.post("/query", json={"query": "store test"})
        mock_memory.store_interaction.assert_called_once()

    def test_graph_error_returns_500(self, client):
        c, mock_graph, _ = client
        mock_graph.ainvoke = AsyncMock(side_effect=Exception("graph failed"))
        resp = c.post("/query", json={"query": "error trigger"})
        assert resp.status_code == 500


# ── /memory/ingest ────────────────────────────────────────────────────────────

class TestMemoryIngest:
    def test_ingest_returns_ok(self, client):
        c, _, mock_memory = client
        mock_memory.semantic.add.return_value = 3
        resp = c.post("/memory/ingest", json=["doc 1", "doc 2", "doc 3"])
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["ingested"] == 3

    def test_ingest_empty_list(self, client):
        c, _, _ = client
        resp = c.post("/memory/ingest", json=[])
        assert resp.status_code == 200
        assert resp.json()["ingested"] == 0


# ── /session ──────────────────────────────────────────────────────────────────

class TestSessionEndpoint:
    def test_returns_session_id(self, client):
        c, _, mock_memory = client
        mock_memory.episodic.retrieve.return_value = []
        data = c.get("/session/sess-xyz").json()
        assert data["session_id"] == "sess-xyz"

    def test_returns_memories_list(self, client):
        c, _, mock_memory = client
        mock_memory.episodic.retrieve.return_value = [
            {"content": "past Q&A", "source": "episodic", "relevance_score": 0.8, "metadata": {}}
        ]
        data = c.get("/session/sess-xyz").json()
        assert isinstance(data["memories"], list)
        assert len(data["memories"]) == 1


# ── /metrics ──────────────────────────────────────────────────────────────────

class TestMetricsEndpoint:
    def test_returns_metrics(self, client):
        c, _, _ = client
        data = c.get("/metrics").json()
        assert "requests_total" in data
        assert "errors_total" in data
        assert "uptime_seconds" in data
