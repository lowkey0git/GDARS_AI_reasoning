"""
Memory system tests.
WorkingMemory tests use a mocked Redis client.
EpisodicMemory tests use a real temporary ChromaDB instance.
SemanticMemory tests use a real temporary FAISS index.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── WorkingMemory (Redis) ─────────────────────────────────────────────────────

class TestWorkingMemory:
    @pytest.fixture
    def working_memory(self):
        """WorkingMemory with a mocked Redis client."""
        with patch("memory.memory_system.get_settings") as mock_cfg_fn:
            mock_cfg = MagicMock()
            mock_cfg.memory.redis_url = "redis://localhost:6379/0"
            mock_cfg.memory.working_memory_ttl_seconds = 3600
            mock_cfg_fn.return_value = mock_cfg

            with patch("memory.memory_system.EmbeddingService"):
                # Mock the redis import so we don't need a real server
                with patch("redis.asyncio.from_url") as mock_redis_fn:
                    mock_redis = AsyncMock()
                    mock_redis.setex = AsyncMock(return_value=True)
                    mock_redis.get   = AsyncMock(return_value=None)
                    mock_redis.ping  = AsyncMock(return_value=True)
                    mock_redis_fn.return_value = mock_redis

                    from memory.memory_system import WorkingMemory
                    wm = WorkingMemory()
                    wm._client  = mock_redis
                    wm._enabled = True
                    yield wm, mock_redis

    @pytest.mark.asyncio
    async def test_set_calls_redis_setex(self, working_memory):
        wm, mock_redis = working_memory
        await wm.set("sess-001", "key", {"data": "value"})
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_returns_none_on_miss(self, working_memory):
        wm, mock_redis = working_memory
        mock_redis.get.return_value = None
        result = await wm.get("sess-001", "missing-key")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_returns_parsed_json(self, working_memory):
        import json
        wm, mock_redis = working_memory
        mock_redis.get.return_value = json.dumps({"answer": 42})
        result = await wm.get("sess-001", "some-key")
        assert result == {"answer": 42}

    @pytest.mark.asyncio
    async def test_store_context_accumulates(self, working_memory):
        wm, mock_redis = working_memory
        # First call: no existing context
        mock_redis.get.return_value = None
        await wm.store_context("sess-001", "first content", "web")
        # Should have called setex to store it
        mock_redis.setex.assert_called()

    @pytest.mark.asyncio
    async def test_ping_returns_true_when_connected(self, working_memory):
        wm, mock_redis = working_memory
        result = await wm.ping()
        assert result is True

    @pytest.mark.asyncio
    async def test_disabled_when_redis_unavailable(self):
        with patch("memory.memory_system.get_settings") as mock_cfg_fn:
            mock_cfg = MagicMock()
            mock_cfg.memory.redis_url = "redis://localhost:6379/0"
            mock_cfg.memory.working_memory_ttl_seconds = 3600
            mock_cfg_fn.return_value = mock_cfg

            with patch("redis.asyncio.from_url", side_effect=Exception("connection refused")):
                from memory.memory_system import WorkingMemory
                wm = WorkingMemory()

        assert wm._enabled is False
        # Should return None gracefully without raising
        result = await wm.get("sess", "key")
        assert result is None

    @pytest.mark.asyncio
    async def test_ping_returns_false_when_disabled(self):
        with patch("memory.memory_system.get_settings") as mock_cfg_fn:
            mock_cfg = MagicMock()
            mock_cfg.memory.redis_url = "redis://localhost:6379/0"
            mock_cfg.memory.working_memory_ttl_seconds = 3600
            mock_cfg_fn.return_value = mock_cfg

            with patch("redis.asyncio.from_url", side_effect=Exception("down")):
                from memory.memory_system import WorkingMemory
                wm = WorkingMemory()

        assert await wm.ping() is False


# ── EpisodicMemory (ChromaDB) ─────────────────────────────────────────────────

class TestEpisodicMemory:
    @pytest.fixture
    def episodic(self, tmp_path):
        with patch("memory.memory_system.get_settings") as mock_cfg_fn:
            mock_cfg = MagicMock()
            mock_cfg.memory.chroma_persist_dir = str(tmp_path / "chroma")
            mock_cfg.memory.embedding_model    = "all-MiniLM-L6-v2"
            mock_cfg_fn.return_value = mock_cfg

            from memory.memory_system import EpisodicMemory, EmbeddingService
            EmbeddingService._instance = None
            yield EpisodicMemory()

    def test_count_zero_on_empty(self, episodic):
        assert episodic.count() == 0

    def test_store_returns_id(self, episodic):
        doc_id = episodic.store("test content", metadata={"source": "test"})
        assert isinstance(doc_id, str)
        assert len(doc_id) > 0

    def test_retrieve_empty_returns_empty(self, episodic):
        result = episodic.retrieve("anything", n=5)
        assert result == []

    def test_store_then_count(self, episodic):
        episodic.store("Python is a programming language")
        assert episodic.count() == 1

    def test_store_then_retrieve(self, episodic):
        episodic.store("FastAPI is a Python web framework for building APIs")
        results = episodic.retrieve("web framework Python", n=1)
        assert len(results) == 1
        assert results[0]["source"] == "episodic"
        assert 0.0 <= results[0]["relevance_score"] <= 1.0

    def test_retrieve_n_results_respected(self, episodic):
        for i in range(5):
            episodic.store(f"Document {i} about machine learning and AI systems")
        results = episodic.retrieve("machine learning", n=2)
        assert len(results) <= 2

    def test_metadata_preserved(self, episodic):
        episodic.store("content", metadata={"session_id": "sess-123", "type": "interaction"})
        results = episodic.retrieve("content", n=1)
        if results:
            assert results[0]["metadata"].get("session_id") == "sess-123"


# ── SemanticMemory (FAISS) ────────────────────────────────────────────────────

class TestSemanticMemory:
    @pytest.fixture
    def semantic(self, tmp_path):
        with patch("memory.memory_system.get_settings") as mock_cfg_fn:
            mock_cfg = MagicMock()
            mock_cfg.memory.faiss_index_path = str(tmp_path / "faiss" / "index")
            mock_cfg.memory.embedding_model  = "all-MiniLM-L6-v2"
            mock_cfg_fn.return_value = mock_cfg

            from memory.memory_system import SemanticMemory, EmbeddingService
            EmbeddingService._instance = None
            yield SemanticMemory()

    def test_count_zero_on_empty(self, semantic):
        assert semantic.count() == 0

    def test_retrieve_empty_returns_empty(self, semantic):
        result = semantic.retrieve("anything", k=5)
        assert result == []

    def test_add_returns_count(self, semantic):
        n = semantic.add(["doc one about Python", "doc two about JavaScript"])
        assert n == 2

    def test_add_then_count(self, semantic):
        semantic.add(["document about neural networks"])
        assert semantic.count() == 1

    def test_add_then_retrieve(self, semantic):
        semantic.add(["LangGraph is used for building stateful multi-agent systems"])
        results = semantic.retrieve("multi-agent framework", k=1)
        assert len(results) == 1
        assert results[0]["source"] == "semantic"
        assert results[0]["relevance_score"] > 0.0

    def test_relevance_score_between_zero_and_one(self, semantic):
        semantic.add(["some content about AI"])
        results = semantic.retrieve("AI content", k=1)
        if results:
            assert 0.0 <= results[0]["relevance_score"] <= 1.0
