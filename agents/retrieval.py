"""
Retrieval Node
==============
Retrieves evidence from three sources in parallel:
  1. Tavily web search (for COMPOUND and NOVEL queries that need fresh info)
  2. Semantic memory (FAISS — domain knowledge you've injected)
  3. Episodic memory (ChromaDB — past interactions)

Then scores and deduplicates.

Engineering decisions:

  Parallel retrieval with asyncio.gather
    All three sources run concurrently. Total time = slowest source,
    not sum of all three. Typically saves 1–2 seconds per request.

  Query decomposition (2–3 sub-queries only)
    We don't decompose into 4+ queries because each extra query adds
    one Tavily API call (rate limit risk) and more tokens. 2–3 covers
    the information space without waste.

  LLM-based relevance scoring
    We score with the fast 8B model. It's not as accurate as a
    cross-encoder reranker, but it's fast and free on Groq.
    A cross-encoder (like ms-marco-MiniLM) would be more accurate
    but requires running a local model, which adds memory overhead
    and complexity we don't need yet.

  Deduplication threshold 0.92 cosine similarity
    Anything above 0.92 is treated as a duplicate. This is deliberately
    conservative — we'd rather keep a near-duplicate than discard a
    subtly different piece of evidence.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from agents.state import AgentState, EvidenceFragment
from config.settings import get_settings
from memory.memory_system import MemorySystem
from utils.llm_client import get_router_client
from utils.logging import get_logger

log = get_logger(__name__)


_DECOMPOSE_SYSTEM = """You are a research query strategist.
Break the user's goal into 2-3 distinct sub-queries covering different aspects.
Each must target a different facet. Be specific. Include year for tech topics.
Return ONLY a JSON array of strings. Example: ["query 1", "query 2"]"""

_SCORE_SYSTEM = """Score each passage for relevance to the goal (0.0 to 1.0).
0.0 = irrelevant, 0.5 = somewhat relevant, 1.0 = directly answers the goal.
Return ONLY a JSON array of numbers, one per passage, same order as input."""


class RetrievalNode:
    """LangGraph node: retrieval_node"""

    def __init__(self, memory_system: MemorySystem) -> None:
        self._memory = memory_system
        self._llm    = get_router_client()
        self._cfg    = get_settings()

    async def run(self, state: AgentState) -> AgentState:
        state.log_node("retrieval_node")

        # SIMPLE queries skip web retrieval — use memory only
        if state.complexity.value == "simple":
            ctx = await self._memory.retrieve_context(
                state.user_query, state.session_id,
                cfg_top_k=self._cfg.memory.working_top_k,
            )
            state.working_memory  = ctx["working"]
            state.episodic_memory = ctx["episodic"]
            state.semantic_memory = ctx["semantic"]
            log.info("SIMPLE retrieval — memory only")
            return state

        goal = state.inferred_goal.true_goal if state.inferred_goal else state.user_query

        # ── Decompose goal into sub-queries ───────────────────
        sub_queries = await self._decompose(goal, state)
        state.sub_queries = sub_queries

        # ── Parallel retrieval from all sources ───────────────
        needs_web = bool(state.execution_plan and state.execution_plan.needs_web_search)
        tasks = []
        if needs_web:
            tasks.append(self._web_retrieve(sub_queries))
        tasks.append(self._memory_retrieve(goal, state.session_id))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        raw_fragments: list[EvidenceFragment] = []
        for r in results:
            if isinstance(r, list):
                raw_fragments.extend(r)

        log.info("Raw fragments retrieved", count=len(raw_fragments))

        # ── Score and filter ──────────────────────────────────
        if raw_fragments:
            scored = await self._score(raw_fragments, goal)
            min_score = self._cfg.retrieval.min_relevance_score
            filtered = [f for f in scored if f.relevance_score >= min_score]
            deduplicated = self._deduplicate(filtered)
            state.evidence = sorted(deduplicated, key=lambda f: f.relevance_score, reverse=True)[:8]
            state.evidence_synthesis = await self._synthesise(state.evidence, goal)
            state.coverage_gaps = self._find_gaps(sub_queries, state.evidence)

        # ── Store memory layer refs on state ──────────────────
        ctx = await self._memory.retrieve_context(goal, state.session_id)
        state.working_memory  = ctx["working"]
        state.episodic_memory = ctx["episodic"]
        state.semantic_memory = ctx["semantic"]

        log.info(
            "Retrieval complete",
            final_fragments=len(state.evidence),
            coverage_gaps=state.coverage_gaps,
        )
        state.record_model(self._llm.model_name)
        return state

    async def _decompose(self, goal: str, state: AgentState) -> list[str]:
        try:
            raw = await self._llm.ainvoke(_DECOMPOSE_SYSTEM, f"GOAL: {goal}")
            text = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            queries = json.loads(text)
            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                return queries[:3]
        except Exception as exc:
            log.warning("Query decomposition failed", error=str(exc))
        return [goal]

    async def _web_retrieve(self, sub_queries: list[str]) -> list[EvidenceFragment]:
        try:
            from tavily import AsyncTavilyClient
            cfg = get_settings()
            client = AsyncTavilyClient(api_key=cfg.retrieval.tavily_api_key)
            tasks = [client.search(q, max_results=3, search_depth="advanced")
                     for q in sub_queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            frags = []
            for r in results:
                if isinstance(r, dict):
                    for item in r.get("results", []):
                        frags.append(EvidenceFragment(
                            content=item.get("content", ""),
                            source_type="web",
                            source_url=item.get("url", ""),
                            source_title=item.get("title", ""),
                            credibility_score=_url_credibility(item.get("url", "")),
                        ))
            return frags
        except Exception as exc:
            log.warning("Web retrieval failed", error=str(exc))
            return []

    async def _memory_retrieve(
        self, query: str, session_id: str
    ) -> list[EvidenceFragment]:
        frags = []
        for entry in self._memory.episodic.retrieve(query, n=4):
            frags.append(EvidenceFragment(
                content=entry["content"],
                source_type="episodic",
                relevance_score=entry["relevance_score"],
            ))
        for entry in self._memory.semantic.retrieve(query, k=4):
            frags.append(EvidenceFragment(
                content=entry["content"],
                source_type="semantic",
                relevance_score=entry["relevance_score"],
            ))
        return frags

    async def _score(
        self, fragments: list[EvidenceFragment], goal: str
    ) -> list[EvidenceFragment]:
        """Score relevance using the fast 8B model. Batch up to 8 at a time."""
        for i in range(0, len(fragments), 8):
            batch = fragments[i : i + 8]
            passages = "\n\n---\n\n".join(
                f"[{j}] {f.content[:300]}" for j, f in enumerate(batch)
            )
            try:
                raw = await self._llm.ainvoke(
                    _SCORE_SYSTEM,
                    f"GOAL: {goal}\n\nPASSAGES:\n{passages}",
                )
                text = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
                scores = json.loads(text)
                if isinstance(scores, list):
                    for frag, score in zip(batch, scores):
                        frag.relevance_score = float(score)
            except Exception as exc:
                log.warning("Batch scoring failed", error=str(exc))
                for frag in batch:
                    frag.relevance_score = 0.5
        return fragments

    def _deduplicate(self, fragments: list[EvidenceFragment]) -> list[EvidenceFragment]:
        if len(fragments) <= 1:
            return fragments
        try:
            from memory.memory_system import EmbeddingService
            import numpy as np

            embedder = EmbeddingService.get()
            texts = [f.content[:300] for f in fragments]
            vecs  = np.array(embedder.embed_batch(texts), dtype="float32")
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs  = vecs / np.where(norms == 0, 1, norms)

            kept, dropped = [], set()
            for i in range(len(fragments)):
                if i in dropped:
                    continue
                for j in range(i + 1, len(fragments)):
                    if j in dropped:
                        continue
                    if float(np.dot(vecs[i], vecs[j])) >= 0.92:
                        # Keep the higher-scored one
                        if fragments[i].relevance_score >= fragments[j].relevance_score:
                            dropped.add(j)
                        else:
                            dropped.add(i)
                            break
                if i not in dropped:
                    kept.append(fragments[i])
            return kept
        except Exception as exc:
            log.warning("Deduplication skipped", error=str(exc))
            return fragments

    async def _synthesise(
        self, fragments: list[EvidenceFragment], goal: str
    ) -> str:
        if not fragments:
            return "No relevant evidence retrieved."
        top = fragments[:6]
        passages = "\n\n---\n\n".join(
            f"[score={f.relevance_score:.2f}|{f.source_type}] {f.content[:400]}"
            for f in top
        )
        system = (
            "Synthesise the evidence into a 150-200 word summary "
            "directly serving the goal. Lead with the most important finding. "
            "Keep specific numbers, names, and tool references exact. "
            "Return only the synthesis."
        )
        try:
            return await self._llm.ainvoke(system, f"GOAL: {goal}\n\nEVIDENCE:\n{passages}")
        except Exception as exc:
            log.warning("Synthesis failed", error=str(exc))
            return "\n\n".join(f.content[:200] for f in top[:3])

    def _find_gaps(
        self, sub_queries: list[str], evidence: list[EvidenceFragment]
    ) -> list[str]:
        """Sub-queries with no high-scoring evidence are coverage gaps."""
        gaps = []
        for q in sub_queries:
            relevant = [
                f for f in evidence
                if q.lower()[:20] in f.source_title.lower()
                or f.relevance_score > 0.6
            ]
            if not relevant:
                gaps.append(q)
        return gaps


def _url_credibility(url: str) -> float:
    high = ("arxiv.org", "github.com", "docs.", ".edu", ".gov",
            "stackoverflow.com", "python.org", "pytorch.org")
    low  = ("reddit.com", "quora.com", "medium.com")
    u = url.lower()
    if any(h in u for h in high):
        return 0.90
    if any(l in u for l in low):
        return 0.55
    return 0.70


async def retrieval_node(state: AgentState, memory_system: MemorySystem) -> AgentState:
    return await RetrievalNode(memory_system).run(state)
