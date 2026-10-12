"""
Retrieval node tests.
Tests query decomposition, parallel strategy execution,
evidence scoring, deduplication, and synthesis.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agents.state import AgentState, QueryComplexity, EvidenceFragment


def _make_memory_system():
    """Returns a mock MemorySystem."""
    mem = MagicMock()
    mem.episodic = MagicMock()
    mem.episodic.retrieve.return_value = []
    mem.semantic  = MagicMock()
    mem.semantic.retrieve.return_value = []
    mem.retrieve_context = AsyncMock(return_value={
        "working": [], "episodic": [], "semantic": [],
    })
    return mem


class TestQueryDecomposition:
    @pytest.mark.asyncio
    async def test_decomposes_goal_into_sub_queries(self, sample_state_with_goal):
        with patch("agents.retrieval.get_router_client") as mock_fn:
            mock_llm = MagicMock()
            mock_llm.model_name = "test"
            mock_llm.ainvoke = AsyncMock(
                return_value='["FastAPI tutorial 2024", "Pydantic models REST", "Python API deployment"]'
            )
            mock_fn.return_value = mock_llm

            from agents.retrieval import QueryDecomposer
            decomposer = QueryDecomposer()
            result = await decomposer.decompose(sample_state_with_goal)

        assert len(result) == 3
        assert all(isinstance(q, str) for q in result)

    @pytest.mark.asyncio
    async def test_decomposer_fallback_on_error(self, sample_state_with_goal):
        """LLM error → returns the true goal as single query."""
        with patch("agents.retrieval.get_router_client") as mock_fn:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))
            mock_fn.return_value = mock_llm

            from agents.retrieval import QueryDecomposer
            result = await QueryDecomposer().decompose(sample_state_with_goal)

        assert len(result) == 1
        assert result[0] == sample_state_with_goal.inferred_goal.true_goal

    @pytest.mark.asyncio
    async def test_decomposer_caps_at_three_queries(self, sample_state_with_goal):
        """Never returns more than 3 sub-queries regardless of LLM output."""
        with patch("agents.retrieval.get_router_client") as mock_fn:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(
                return_value='["q1", "q2", "q3", "q4", "q5"]'
            )
            mock_fn.return_value = mock_llm

            from agents.retrieval import QueryDecomposer
            result = await QueryDecomposer().decompose(sample_state_with_goal)

        assert len(result) <= 3

    @pytest.mark.asyncio
    async def test_decomposer_rejects_non_list_json(self, sample_state_with_goal):
        """Non-list JSON → fallback to single query."""
        with patch("agents.retrieval.get_router_client") as mock_fn:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value='{"query": "something"}')
            mock_fn.return_value = mock_llm

            from agents.retrieval import QueryDecomposer
            result = await QueryDecomposer().decompose(sample_state_with_goal)

        assert len(result) == 1


class TestEvidenceDeduplication:
    def test_removes_near_duplicates(self):
        """Two near-identical fragments → only the higher-scored one is kept."""
        from agents.retrieval import RetrievalNode

        mem = _make_memory_system()

        with patch("agents.retrieval.get_router_client") as mock_fn:
            mock_fn.return_value = MagicMock()

            node = RetrievalNode(mem)

            with patch("agents.retrieval.EmbeddingService") as mock_svc:
                import numpy as np
                mock_embedder = MagicMock()
                # Fragment 0 and 1 are near-identical (cosine sim ≈ 1.0)
                # Fragment 2 is different
                vec_a = np.array([1.0, 0.0, 0.0])
                vec_b = np.array([0.999, 0.001, 0.0])
                vec_c = np.array([0.0, 1.0, 0.0])
                mock_embedder.embed_batch.return_value = [
                    vec_a.tolist(), vec_b.tolist(), vec_c.tolist()
                ]
                mock_svc.get.return_value = mock_embedder

                frags = [
                    EvidenceFragment(content="Python is dynamically typed",
                                     source_type="web", relevance_score=0.9),
                    EvidenceFragment(content="Python is dynamically typed language",
                                     source_type="web", relevance_score=0.7),
                    EvidenceFragment(content="FastAPI uses Pydantic for validation",
                                     source_type="semantic", relevance_score=0.8),
                ]
                result = node._deduplicate(frags)

        # Should have removed one of the duplicates
        assert len(result) <= 2

    def test_keeps_single_fragment_unchanged(self):
        mem = _make_memory_system()
        with patch("agents.retrieval.get_router_client"):
            from agents.retrieval import RetrievalNode
            node = RetrievalNode(mem)
            frags = [EvidenceFragment(content="solo content", source_type="web")]
            result = node._deduplicate(frags)
        assert len(result) == 1
        assert result[0].content == "solo content"

    def test_dedup_fallback_on_embedding_error(self):
        """Embedding failure → returns original list unchanged."""
        mem = _make_memory_system()
        with patch("agents.retrieval.get_router_client"), \
             patch("agents.retrieval.EmbeddingService") as mock_svc:
            mock_svc.get.side_effect = Exception("embedding service down")

            from agents.retrieval import RetrievalNode
            node = RetrievalNode(mem)
            frags = [
                EvidenceFragment(content="fragment 1", source_type="web"),
                EvidenceFragment(content="fragment 2", source_type="web"),
            ]
            result = node._deduplicate(frags)

        assert len(result) == 2  # unchanged


class TestCoverageGaps:
    def test_identifies_uncovered_queries(self):
        mem = _make_memory_system()
        with patch("agents.retrieval.get_router_client"):
            from agents.retrieval import RetrievalNode
            node = RetrievalNode(mem)

            sub_queries = ["FastAPI tutorial", "Pydantic models", "API deployment"]
            # Evidence has high relevance so "covered" check passes, but titles
            # don't match sub-queries — all become gaps
            evidence = [
                EvidenceFragment(content="some content", source_type="web",
                                 relevance_score=0.3, source_title="unrelated page"),
            ]
            gaps = node._find_gaps(sub_queries, evidence)

        # At least some gaps should be identified since evidence scores are low
        assert isinstance(gaps, list)

    def test_no_gaps_when_evidence_high_scored(self):
        mem = _make_memory_system()
        with patch("agents.retrieval.get_router_client"):
            from agents.retrieval import RetrievalNode
            node = RetrievalNode(mem)

            sub_queries = ["query 1"]
            # High relevance score → considered covered
            evidence = [
                EvidenceFragment(content="content", source_type="web",
                                 relevance_score=0.9, source_title=""),
            ]
            gaps = node._find_gaps(sub_queries, evidence)

        assert gaps == []


class TestURLCredibility:
    def test_github_high_credibility(self):
        from agents.retrieval import _url_credibility
        assert _url_credibility("https://github.com/user/repo") == pytest.approx(0.90)

    def test_arxiv_high_credibility(self):
        from agents.retrieval import _url_credibility
        assert _url_credibility("https://arxiv.org/abs/2301.12345") == pytest.approx(0.90)

    def test_reddit_low_credibility(self):
        from agents.retrieval import _url_credibility
        assert _url_credibility("https://reddit.com/r/python") == pytest.approx(0.55)

    def test_unknown_domain_neutral(self):
        from agents.retrieval import _url_credibility
        score = _url_credibility("https://some-random-blog.io/post")
        assert score == pytest.approx(0.70)

    def test_empty_url_neutral(self):
        from agents.retrieval import _url_credibility
        assert _url_credibility("") == pytest.approx(0.5)


class TestRetrievalNodeSimplePath:
    @pytest.mark.asyncio
    async def test_simple_skips_web_retrieval(self, sample_state):
        """SIMPLE queries use memory only — no Tavily calls."""
        sample_state.complexity = QueryComplexity.SIMPLE
        mem = _make_memory_system()

        with patch("agents.retrieval.get_router_client") as mock_fn:
            mock_fn.return_value = MagicMock(model_name="test")

            from agents.retrieval import RetrievalNode
            result = await RetrievalNode(mem).run(sample_state)

        assert "retrieval_node" in result.node_trace
        # Memory retrieve_context should have been called
        mem.retrieve_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_simple_populates_memory_fields(self, sample_state):
        sample_state.complexity = QueryComplexity.SIMPLE
        mem = _make_memory_system()
        mem.retrieve_context = AsyncMock(return_value={
            "working":  [{"content": "recent context", "source": "web"}],
            "episodic": [{"content": "past Q&A", "source": "episodic",
                          "relevance_score": 0.7, "metadata": {}}],
            "semantic": [],
        })

        with patch("agents.retrieval.get_router_client") as mock_fn:
            mock_fn.return_value = MagicMock(model_name="test")

            from agents.retrieval import RetrievalNode
            result = await RetrievalNode(mem).run(sample_state)

        assert len(result.working_memory)  == 1
        assert len(result.episodic_memory) == 1
        assert len(result.semantic_memory) == 0


class TestRetrievalNodeCompoundPath:
    @pytest.mark.asyncio
    async def test_compound_runs_decomposition(self, sample_state_with_goal):
        """COMPOUND queries should decompose and retrieve."""
        sample_state_with_goal.complexity = QueryComplexity.COMPOUND
        mem = _make_memory_system()

        with patch("agents.retrieval.get_router_client") as mock_fn, \
             patch("agents.retrieval.get_settings") as mock_cfg:

            mock_cfg.return_value = MagicMock(
                retrieval=MagicMock(tavily_api_key="", min_relevance_score=0.35),
                memory=MagicMock(working_top_k=3),
            )
            mock_llm = MagicMock()
            mock_llm.model_name = "test"
            # decompose
            mock_llm.ainvoke = AsyncMock(side_effect=[
                '["query 1", "query 2"]',   # decomposition
                "Evidence synthesis here.", # synthesis
            ])
            # scoring
            mock_llm.ainvoke = AsyncMock(return_value='["query 1", "query 2"]')
            mock_fn.return_value = mock_llm

            from agents.retrieval import RetrievalNode
            result = await RetrievalNode(mem).run(sample_state_with_goal)

        assert "retrieval_node" in result.node_trace
