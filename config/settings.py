"""
Configuration Management
========================
All settings come from environment variables.
Grouped by concern so each module imports only what it needs.

Engineering note:
  We use pydantic-settings so every value is type-validated at startup.
  If a required key is missing the app crashes immediately with a clear
  message instead of failing silently mid-request.
"""

from __future__ import annotations
from functools import lru_cache
from typing import Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseSettings):
    """
    Groq model assignments per role.

    Why three models instead of one?
    - The router/classifier only needs a fast small model (8B).
      Using 70B there wastes ~400ms and tokens on a task that
      doesn't need that capacity.
    - The reasoner needs full capacity for multi-step CoT.
    - The synthesiser needs balance: quality output, not max depth.

    All three are Groq-hosted which means they're free-tier eligible
    and share the same API key.
    """
    model_config = SettingsConfigDict(env_prefix="GROQ_", extra="ignore")

    api_key: str = Field(..., description="Groq API key")

    # Fast, cheap — used for: query classification, step scoring, short synthesis
    router_model: str = Field("llama3-8b-8192")

    # Full capacity — used for: deep reasoning, goal inference, complex planning
    reasoning_model: str = Field("llama-3.3-70b-versatile")

    # Code-specialised — used for: code generation steps
    code_model: str = Field("llama-3.3-70b-versatile")

    # Balanced — used for: personalization, final synthesis
    synthesis_model: str = Field("llama-3.1-8b-instant")

    temperature_reasoning: float = Field(0.1)
    temperature_synthesis: float = Field(0.2)
    temperature_router: float = Field(0.0)   # deterministic classification
    max_tokens: int = Field(4096)


class MemoryConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MEMORY_", extra="ignore")

    # Working memory — Redis
    redis_url: str = Field("redis://redis:6379/0")
    working_memory_ttl_seconds: int = Field(3600)   # expire after 1 hour of inactivity

    # Episodic memory — ChromaDB
    chroma_persist_dir: str = Field("./data/chroma")

    # Semantic memory — FAISS
    faiss_index_path: str = Field("./data/faiss_index")
    embedding_model: str = Field("all-MiniLM-L6-v2")

    # How many entries to pull from each layer per retrieval
    working_top_k: int = Field(3)
    episodic_top_k: int = Field(5)
    semantic_top_k: int = Field(5)


class RetrievalConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_", extra="ignore")

    tavily_api_key: str = Field(default="", description="Tavily search API key")
    max_web_results: int = Field(4)
    min_relevance_score: float = Field(0.35)   # fragments below this are discarded


class PlannerConfig(BaseSettings):
    """
    Controls the hierarchical planner's routing thresholds.

    Engineering note on query complexity tiers:
      SIMPLE   — answerable from working/episodic memory alone.
                 Skips web search and full reasoning chain.
                 Target latency: < 2s
      COMPOUND — needs retrieval + reasoning but not all nodes.
                 Target latency: < 8s
      NOVEL    — full pipeline, all nodes, reflection loop allowed.
                 Target latency: < 20s
    """
    model_config = SettingsConfigDict(env_prefix="PLANNER_", extra="ignore")

    max_iterations: int = Field(8)
    confidence_threshold: float = Field(0.78)
    reflection_threshold: float = Field(0.60)

    # Complexity classification confidence cutoff.
    # Below this the planner defaults to COMPOUND rather than guessing.
    complexity_confidence_cutoff: float = Field(0.70)


class EvaluationConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EVAL_", extra="ignore")

    # Score thresholds — steps below these trigger a retry or reflection
    min_factuality_score: float = Field(0.55)
    min_usefulness_score: float = Field(0.50)

    # Whether to score every intermediate step (True) or only the final answer (False).
    # Full scoring catches bad reasoning earlier but costs ~1 LLM call per step.
    score_intermediate_steps: bool = Field(True)


class APIConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="API_", extra="ignore")

    host: str = Field("0.0.0.0")
    port: int = Field(8000)
    workers: int = Field(1)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field("INFO")
    log_file: str = Field("./logs/gdars.log")

    # How long a streaming connection can stay open without an event
    stream_keepalive_seconds: int = Field(15)


class Settings(BaseSettings):
    """
    Root settings object.
    Import this everywhere via get_settings().
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    environment: Literal["development", "staging", "production"] = Field("development")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns a cached singleton.
    The cache is intentional — settings don't change at runtime.
    In tests, call get_settings.cache_clear() before patching.
    """
    return Settings()
