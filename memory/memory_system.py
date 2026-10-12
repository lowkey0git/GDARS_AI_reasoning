"""
Three-Layer Memory System
==========================

Layer 1 — Working Memory (Redis)
  Fast key/value store for the current session.
  Stores: recent turns, intermediate results, user preferences this session.
  TTL: 1 hour after last write.
  Why Redis: sub-millisecond reads, built-in TTL, survives container restarts.

Layer 2 — Episodic Memory (ChromaDB)
  Persisted semantic store of past interactions.
  Stores: full Q+A pairs, inferred goals, past solutions.
  Retrieval: cosine similarity on sentence embeddings.
  Why ChromaDB: runs locally, no external service needed, good Python API.

Layer 3 — Semantic Memory (FAISS)
  Local vector index of your domain knowledge base.
  Stores: documents you inject via /memory/ingest.
  Retrieval: L2 distance on sentence embeddings.
  Why FAISS: fastest local ANN search, battle-tested by Meta.

Engineering note on the escalation pattern:
  The planner queries Working first. If it finds sufficient context
  (relevance > threshold), it stops there. Only on a cache miss does
  it escalate to Episodic, then Semantic.
  This matters because each layer has a different cost:
    Working:  ~1ms   (Redis get)
    Episodic: ~20ms  (ChromaDB query + embedding)
    Semantic: ~15ms  (FAISS search + embedding)
  For a simple query this saves 35ms and two embedding calls per request.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional
from uuid import uuid4

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from config.settings import get_settings
from utils.logging import get_logger

log = get_logger(__name__)


# ── Embedding service — shared singleton across all memory layers ─────────────

class EmbeddingService:
    _instance: Optional["EmbeddingService"] = None

    def __init__(self) -> None:
        cfg = get_settings()
        log.info("Loading embedding model", model=cfg.memory.embedding_model)
        self._model = SentenceTransformer(cfg.memory.embedding_model)

    @classmethod
    def get(cls) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def embed(self, text: str) -> list[float]:
        return self._model.encode(text, convert_to_numpy=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, convert_to_numpy=True).tolist()


# ── Layer 1: Working Memory (Redis) ──────────────────────────────────────────

class WorkingMemory:
    """
    Session-scoped short-term memory.

    We store everything under a session namespace so different users
    never see each other's data. Keys look like:
      gdars:{session_id}:history
      gdars:{session_id}:goal
      gdars:{session_id}:preference:{key}
    """

    def __init__(self) -> None:
        cfg = get_settings()
        self._ttl = cfg.memory.working_memory_ttl_seconds
        self._client = None
        self._enabled = False
        try:
            import redis.asyncio as aioredis
            self._client = aioredis.from_url(
                cfg.memory.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            self._enabled = True
            log.info("WorkingMemory (Redis) connected", url=cfg.memory.redis_url)
        except Exception as exc:
            log.warning("Redis unavailable — working memory disabled", error=str(exc))

    async def set(self, session_id: str, key: str, value: Any) -> None:
        if not self._enabled or self._client is None:
            return
        try:
            full_key = f"gdars:{session_id}:{key}"
            await self._client.setex(full_key, self._ttl, json.dumps(value))
        except Exception as exc:
            log.warning("WorkingMemory.set failed", key=key, error=str(exc))

    async def get(self, session_id: str, key: str) -> Optional[Any]:
        if not self._enabled or self._client is None:
            return None
        try:
            full_key = f"gdars:{session_id}:{key}"
            raw = await self._client.get(full_key)
            return json.loads(raw) if raw else None
        except Exception as exc:
            log.warning("WorkingMemory.get failed", key=key, error=str(exc))
            return None

    async def get_recent_context(self, session_id: str) -> list[dict[str, Any]]:
        """Returns the last N stored context fragments for this session."""
        raw = await self.get(session_id, "recent_context")
        return raw if isinstance(raw, list) else []

    async def store_context(self, session_id: str, content: str, source: str) -> None:
        existing = await self.get_recent_context(session_id)
        existing.append({"content": content, "source": source})
        # Keep only the last 10 fragments to cap Redis memory usage
        await self.set(session_id, "recent_context", existing[-10:])

    async def ping(self) -> bool:
        if not self._enabled or self._client is None:
            return False
        try:
            return await self._client.ping()
        except Exception:
            return False


# ── Layer 2: Episodic Memory (ChromaDB) ──────────────────────────────────────

class EpisodicMemory:
    """
    Persisted semantic store of past interactions.
    Each stored entry is a full Q+A pair with metadata.
    """

    COLLECTION = "gdars_episodes"

    def __init__(self) -> None:
        cfg = get_settings()
        os.makedirs(cfg.memory.chroma_persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=cfg.memory.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._embedder = EmbeddingService.get()
        self._col = self._client.get_or_create_collection(self.COLLECTION)
        log.info("EpisodicMemory (ChromaDB) ready",
                 path=cfg.memory.chroma_persist_dir, docs=self._col.count())

    def store(self, content: str, metadata: Optional[dict] = None) -> str:
        doc_id = str(uuid4())
        try:
            self._col.add(
                ids=[doc_id],
                embeddings=[self._embedder.embed(content)],
                documents=[content],
                metadatas=[metadata or {}],
            )
        except Exception as exc:
            log.warning("EpisodicMemory.store failed", error=str(exc))
        return doc_id

    def retrieve(self, query: str, n: int = 5) -> list[dict[str, Any]]:
        count = self._col.count()
        if count == 0:
            return []
        try:
            results = self._col.query(
                query_embeddings=[self._embedder.embed(query)],
                n_results=min(n, count),
            )
            entries = []
            if results.get("documents"):
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    entries.append({
                        "content": doc,
                        "source": "episodic",
                        "relevance_score": float(1.0 - dist),
                        "metadata": meta,
                    })
            return entries
        except Exception as exc:
            log.warning("EpisodicMemory.retrieve failed", error=str(exc))
            return []

    def count(self) -> int:
        return self._col.count()


# ── Layer 3: Semantic Memory (FAISS) ─────────────────────────────────────────

class SemanticMemory:
    """
    Vector index over your domain knowledge base.
    Documents are added via the /memory/ingest API endpoint.
    """

    def __init__(self) -> None:
        cfg = get_settings()
        self._index_path = cfg.memory.faiss_index_path
        self._index = None
        self._documents: list[str] = []
        self._embedder = EmbeddingService.get()
        os.makedirs(os.path.dirname(self._index_path)
                    if os.path.dirname(self._index_path) else ".", exist_ok=True)
        self._try_load()

    def _try_load(self) -> None:
        try:
            import faiss, pickle
            idx_file  = self._index_path + ".index"
            docs_file = self._index_path + ".docs"
            if os.path.exists(idx_file) and os.path.exists(docs_file):
                self._index = faiss.read_index(idx_file)
                with open(docs_file, "rb") as f:
                    self._documents = pickle.load(f)
                log.info("SemanticMemory (FAISS) loaded", docs=len(self._documents))
        except Exception as exc:
            log.debug("FAISS load skipped", reason=str(exc))

    def add(self, docs: list[str]) -> int:
        import faiss, numpy as np, pickle
        if not docs:
            return 0
        vectors = np.array(self._embedder.embed_batch(docs), dtype="float32")
        if self._index is None:
            self._index = faiss.IndexFlatL2(vectors.shape[1])
        self._index.add(vectors)
        self._documents.extend(docs)
        # Persist after every add so restarts don't lose data
        idx_file  = self._index_path + ".index"
        docs_file = self._index_path + ".docs"
        os.makedirs(os.path.dirname(idx_file)
                    if os.path.dirname(idx_file) else ".", exist_ok=True)
        faiss.write_index(self._index, idx_file)
        with open(docs_file, "wb") as f:
            pickle.dump(self._documents, f)
        return len(docs)

    def retrieve(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        if self._index is None or not self._documents:
            return []
        import numpy as np
        vector = np.array([self._embedder.embed(query)], dtype="float32")
        k_actual = min(k, len(self._documents))
        distances, indices = self._index.search(vector, k_actual)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self._documents):
                results.append({
                    "content": self._documents[idx],
                    "source": "semantic",
                    "relevance_score": float(1.0 / (1.0 + dist)),  # convert L2 to similarity
                    "metadata": {},
                })
        return results

    def count(self) -> int:
        return len(self._documents)


# ── Unified MemorySystem facade ───────────────────────────────────────────────

class MemorySystem:
    """
    Single entry point for all memory operations.
    Callers don't need to know which layer handles what.
    """

    def __init__(self) -> None:
        self.working  = WorkingMemory()
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()

    async def retrieve_context(
        self,
        query: str,
        session_id: str,
        cfg_top_k: int = 5,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Retrieve from all three layers.
        Returns dict with keys "working", "episodic", "semantic".
        """
        working  = await self.working.get_recent_context(session_id)
        episodic = self.episodic.retrieve(query, n=cfg_top_k)
        semantic = self.semantic.retrieve(query, k=cfg_top_k)
        return {"working": working, "episodic": episodic, "semantic": semantic}

    async def store_interaction(
        self, query: str, response: str, session_id: str
    ) -> None:
        content = f"Q: {query}\nA: {response}"
        self.episodic.store(content, metadata={"session_id": session_id, "type": "interaction"})
        await self.working.store_context(session_id, content, source="interaction")

    async def health_check(self) -> dict[str, bool]:
        return {
            "working_memory":  await self.working.ping(),
            "episodic_memory": self.episodic.count() >= 0,
            "semantic_memory": self.semantic.count() >= 0,
        }
