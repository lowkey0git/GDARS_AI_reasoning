"""
FastAPI Application
====================
Endpoints:
  GET  /health          — system health with per-component status
  GET  /ready           — readiness check (fails if memory unavailable)
  POST /query           — full reasoning (returns when complete)
  POST /query/stream    — streaming SSE (progressive results as nodes complete)
  POST /memory/ingest   — add documents to semantic memory
  GET  /session/{id}    — retrieve session history
  GET  /metrics         — lightweight request stats

Progressive streaming design:
  Each node completion emits an SSE event with partial state.
  The client can render: inferred goal → evidence summary → reasoning
  steps → final answer, as they arrive.
  This means the user sees something useful within 2-3 seconds
  instead of waiting 15-20 seconds for the full pipeline.

Health check design:
  /health returns 200 even if Redis is down — the system degrades
  gracefully without working memory.
  /ready returns 503 if the graph is not initialised — used by
  Kubernetes/ECS to gate traffic during startup.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional
from uuid import uuid4

import orjson
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from agents.graph import build_graph
from agents.state import AgentState, UserProfile
from config.settings import get_settings
from memory.memory_system import MemorySystem
from utils.logging import get_logger, setup_logging

log = get_logger(__name__)

# ── Shared state (initialised in lifespan) ────────────────────────────────────
_memory_system: Optional[MemorySystem] = None
_graph:         Optional[Any]          = None
_request_count: int                    = 0
_error_count:   int                    = 0
_startup_time:  float                  = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _memory_system, _graph, _startup_time
    cfg = get_settings()
    setup_logging(
        level=cfg.api.log_level,
        log_file=cfg.api.log_file,
        environment=cfg.environment,
    )
    log.info("GDARS starting up", environment=cfg.environment)

    _memory_system = MemorySystem()
    _graph         = build_graph(_memory_system)
    _startup_time  = time.time()

    log.info("GDARS ready",
             reasoning_model=cfg.llm.reasoning_model,
             router_model=cfg.llm.router_model)
    yield
    log.info("GDARS shutting down")


app = FastAPI(
    title="GDARS",
    description="Goal-Directed Adaptive Reasoning System",
    version="2.0.0",
    lifespan=lifespan,
    default_response_class=JSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str                  = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str]   = None
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    user_profile: Optional[dict[str, Any]]     = None
    max_iterations: int         = Field(default=8, ge=1, le=15)


class QueryResponse(BaseModel):
    session_id:       str
    request_id:       str
    query:            str
    final_solution:   str
    confidence_score: float
    complexity:       str
    inferred_goal:    Optional[dict]    = None
    next_actions:     list[str]         = Field(default_factory=list)
    node_trace:       list[str]         = Field(default_factory=list)
    models_used:      list[str]         = Field(default_factory=list)
    avg_factuality:   float             = 0.0
    avg_usefulness:   float             = 0.0
    reasoning_steps:  int               = 0
    execution_time_ms: float            = 0.0
    errors:           list[str]         = Field(default_factory=list)


# ── Helper ────────────────────────────────────────────────────────────────────

def _build_initial_state(req: QueryRequest) -> AgentState:
    cfg     = get_settings()
    profile = UserProfile(**(req.user_profile or {})) if req.user_profile else UserProfile()
    return AgentState(
        session_id=req.session_id or str(uuid4()),
        user_query=req.query,
        conversation_history=req.conversation_history,
        user_profile=profile,
        max_iterations=req.max_iterations,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    """
    Returns 200 always. Shows per-component status.
    Use /ready for traffic gating.
    """
    global _memory_system
    cfg = get_settings()
    mem_health = await _memory_system.health_check() if _memory_system else {}
    uptime = round(time.time() - _startup_time, 1) if _startup_time else 0

    return {
        "status":          "ok",
        "version":         "2.0.0",
        "uptime_seconds":  uptime,
        "environment":     cfg.environment,
        "models": {
            "router":    cfg.llm.router_model,
            "reasoning": cfg.llm.reasoning_model,
            "synthesis": cfg.llm.synthesis_model,
        },
        "memory": mem_health,
        "requests_total": _request_count,
        "errors_total":   _error_count,
    }


@app.get("/ready")
async def ready() -> dict:
    """Kubernetes/ECS readiness probe. Returns 503 if not ready."""
    if _graph is None or _memory_system is None:
        raise HTTPException(503, "System not initialised")
    return {"status": "ready"}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    global _request_count, _error_count, _memory_system, _graph

    if _graph is None or _memory_system is None:
        raise HTTPException(503, "System not initialised")

    _request_count += 1
    start = time.monotonic()

    initial_state = _build_initial_state(req)
    log.info("Query received", session_id=initial_state.session_id,
             query=req.query[:80])

    try:
        result_dict = await _graph.ainvoke(initial_state.model_dump())
        result      = AgentState.model_validate(result_dict)
    except Exception as exc:
        _error_count += 1
        log.error("Graph execution failed", error=str(exc))
        raise HTTPException(500, f"Reasoning engine error: {exc}")

    # Persist interaction to memory
    await _memory_system.store_interaction(
        query=req.query,
        response=result.final_solution,
        session_id=result.session_id,
    )

    elapsed = (time.monotonic() - start) * 1000
    log.info("Query complete",
             session_id=result.session_id,
             confidence=result.confidence_score,
             complexity=result.complexity.value,
             time_ms=round(elapsed, 1))

    return QueryResponse(
        session_id       = result.session_id,
        request_id       = result.request_id,
        query            = req.query,
        final_solution   = result.final_solution,
        confidence_score = result.confidence_score,
        complexity       = result.complexity.value,
        inferred_goal    = result.inferred_goal.model_dump() if result.inferred_goal else None,
        next_actions     = result.next_actions,
        node_trace       = result.node_trace,
        models_used      = result.models_used,
        avg_factuality   = result.avg_factuality,
        avg_usefulness   = result.avg_usefulness,
        reasoning_steps  = len(result.thought_steps),
        execution_time_ms = round(elapsed, 2),
        errors           = result.error_log,
    )


@app.post("/query/stream")
async def query_stream(req: QueryRequest) -> StreamingResponse:
    """
    Server-Sent Events streaming endpoint.
    Emits one event per node completion so the client can render
    progressive results.

    Event format:
      data: {"node": "planner_node", "payload": {...}}
    """
    global _graph, _memory_system

    if _graph is None or _memory_system is None:
        raise HTTPException(503, "System not initialised")

    initial_state = _build_initial_state(req)
    cfg = get_settings()

    async def event_stream() -> AsyncGenerator[bytes, None]:
        try:
            async for event in _graph.astream(initial_state.model_dump()):
                if not event:
                    continue
                node_name  = list(event.keys())[0]
                node_state = list(event.values())[0]

                # Extract the most useful progressive data per node
                payload: dict[str, Any] = {"node": node_name}

                if node_name == "planner_node":
                    goal = node_state.get("inferred_goal")
                    payload["inferred_goal"] = goal.get("true_goal","") if goal else ""
                    payload["complexity"]    = node_state.get("complexity","")

                elif node_name == "retrieval_node":
                    payload["evidence_count"]   = len(node_state.get("evidence", []))
                    payload["evidence_summary"] = (node_state.get("evidence_synthesis","") or "")[:300]
                    payload["coverage_gaps"]    = node_state.get("coverage_gaps", [])

                elif node_name == "reasoning_node":
                    steps = node_state.get("thought_steps", [])
                    last  = steps[-1] if steps else {}
                    payload["step_num"]   = last.get("step_num", 0)
                    payload["conclusion"] = (last.get("conclusion","") or "")[:200]
                    payload["confidence"] = node_state.get("confidence_score", 0.0)

                elif node_name == "synthesis_node":
                    payload["partial_answer"] = (node_state.get("final_solution","") or "")[:500]

                elif node_name in ("verification_node", "reflection_node"):
                    payload["confidence"] = node_state.get("confidence_score", 0.0)
                    payload["iteration"]  = node_state.get("iteration", 0)

                yield b"data: " + orjson.dumps(payload) + b"\n\n"

                # Keepalive: send a comment every N seconds to prevent timeout
                await asyncio.sleep(0)

            # Terminal event
            yield b"data: " + orjson.dumps({"node": "__end__", "complete": True}) + b"\n\n"

        except Exception as exc:
            log.error("Streaming error", error=str(exc))
            yield b"data: " + orjson.dumps({"error": str(exc), "node": "__error__"}) + b"\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/memory/ingest")
async def ingest(documents: list[str]) -> dict:
    """Add documents to semantic (FAISS) memory."""
    if _memory_system is None:
        raise HTTPException(503, "Memory system not initialised")
    if not documents:
        return {"status": "ok", "ingested": 0}
    count = _memory_system.semantic.add(documents)
    log.info("Documents ingested", count=count)
    return {"status": "ok", "ingested": count}


@app.get("/session/{session_id}")
async def get_session(session_id: str) -> dict:
    if _memory_system is None:
        raise HTTPException(503, "Memory system not initialised")
    episodic = _memory_system.episodic.retrieve(session_id, n=10)
    return {"session_id": session_id, "memories": episodic}


@app.get("/metrics")
async def metrics() -> dict:
    return {
        "requests_total": _request_count,
        "errors_total":   _error_count,
        "uptime_seconds": round(time.time() - _startup_time, 1) if _startup_time else 0,
    }
