"""
LLM Client — Mixture of Experts Routing
=========================================
Rather than using one model for every call, we route each type of
task to the model best suited for it.

  RouterClient    → llama3-8b-8192       fast, cheap, deterministic
  ReasoningClient → llama-3.3-70b-versatile  full capacity, deep CoT
  CodeClient      → llama-3.3-70b-versatile  code generation
  SynthesisClient → llama-3.1-8b-instant     balanced speed/quality

Engineering tradeoff:
  Adding a router call costs ~100ms and ~200 tokens.
  But it saves 3-5 expensive 70B calls for simple queries.
  Net result: lower average cost and latency across the query distribution.

  For a query like "what is Python" the router correctly identifies
  SIMPLE and we never touch the 70B model at all.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Type, TypeVar

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import get_settings
from utils.logging import get_logger

log = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)


class GroqClient:
    """
    Thin wrapper around LangChain ChatGroq.
    Handles retries, JSON extraction, and conversation history injection.
    """

    def __init__(self, model: str, temperature: float = 0.1,
                 max_tokens: int = 4096) -> None:
        cfg = get_settings()
        self.model_name = model
        self._llm = ChatGroq(
            api_key=cfg.llm.api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=8),
        reraise=True,
    )
    async def ainvoke(
        self,
        system: str,
        user: str,
        history: Optional[list[dict[str, str]]] = None,
    ) -> str:
        """
        Single LLM call. Returns raw string content.

        history: list of {"role": "user"|"assistant", "content": str}
        Only the last 6 messages are included to cap context cost.
        """
        messages: list[Any] = [SystemMessage(content=system)]

        for msg in (history or [])[-6:]:
            role, content = msg.get("role", "user"), msg.get("content", "")
            messages.append(
                HumanMessage(content=content) if role == "user"
                else AIMessage(content=content)
            )

        messages.append(HumanMessage(content=user))
        response = await self._llm.ainvoke(messages)
        return str(response.content)

    async def aextract(
        self,
        system: str,
        user: str,
        schema: Type[T],
        history: Optional[list[dict[str, str]]] = None,
    ) -> T:
        """
        Invoke LLM and parse response as a Pydantic model.
        Appends JSON schema instruction to system prompt.
        """
        json_prompt = (
            f"\n\nRespond with ONLY valid JSON matching this schema "
            f"(no markdown fences, no extra text):\n"
            f"{json.dumps(schema.model_json_schema(), indent=2)}"
        )
        raw = await self.ainvoke(system + json_prompt, user, history)
        return _parse_json(raw, schema)


def _parse_json(raw: str, schema: Type[T]) -> T:
    """
    Best-effort JSON extraction from LLM output.
    Handles markdown code fences and leading/trailing text.
    """
    text = raw.strip()
    # Strip ```json ... ``` fences
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

    try:
        return schema(**json.loads(text))
    except Exception:
        # Find the outermost { ... } block as fallback
        start, end = text.find("{"), text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return schema(**json.loads(text[start:end]))
            except Exception:
                pass
        raise ValueError(f"Cannot parse JSON from LLM output: {raw[:300]}")


# ── MoE client factory ────────────────────────────────────────────────────────

def get_router_client() -> GroqClient:
    """Fast, cheap, deterministic. Used for classification and short scoring."""
    cfg = get_settings()
    return GroqClient(
        model=cfg.llm.router_model,
        temperature=cfg.llm.temperature_router,
        max_tokens=512,
    )


def get_reasoning_client() -> GroqClient:
    """Full capacity. Used for deep CoT, goal inference, complex planning."""
    cfg = get_settings()
    return GroqClient(
        model=cfg.llm.reasoning_model,
        temperature=cfg.llm.temperature_reasoning,
        max_tokens=cfg.llm.max_tokens,
    )


def get_code_client() -> GroqClient:
    """Code-oriented. Used for code generation steps."""
    cfg = get_settings()
    return GroqClient(
        model=cfg.llm.code_model,
        temperature=0.05,
        max_tokens=cfg.llm.max_tokens,
    )


def get_synthesis_client() -> GroqClient:
    """Balanced speed/quality. Used for personalization and final synthesis."""
    cfg = get_settings()
    return GroqClient(
        model=cfg.llm.synthesis_model,
        temperature=cfg.llm.temperature_synthesis,
        max_tokens=cfg.llm.max_tokens,
    )
