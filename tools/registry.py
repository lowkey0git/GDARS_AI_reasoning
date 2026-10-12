"""
Tool Registry
=============
Singleton tool instances. Lazy-initialised on first use.

Tools available:
  TavilyTool    — live web search via Tavily AI
  CodeGenTool   — code synthesis via the code-specialised Groq model
"""

from __future__ import annotations

from typing import Any, Optional

from utils.logging import get_logger

log = get_logger(__name__)


class TavilyTool:
    def __init__(self) -> None:
        from config.settings import get_settings
        cfg = get_settings()
        self._enabled = bool(cfg.retrieval.tavily_api_key)
        self._client  = None
        if self._enabled:
            try:
                from tavily import AsyncTavilyClient
                self._client = AsyncTavilyClient(api_key=cfg.retrieval.tavily_api_key)
                log.info("TavilyTool ready")
            except Exception as exc:
                log.warning("Tavily client failed to init", error=str(exc))
                self._enabled = False

    async def search(self, query: str, max_results: int = 4) -> list[dict[str, Any]]:
        if not self._enabled or self._client is None:
            return []
        try:
            response = await self._client.search(
                query=query, max_results=max_results,
                search_depth="advanced", include_answer=True,
            )
            if isinstance(response, dict):
                return [
                    {"url": r.get("url",""), "title": r.get("title",""),
                     "content": r.get("content",""), "score": r.get("score",0.0)}
                    for r in response.get("results", [])
                ]
            return []
        except Exception as exc:
            log.error("Tavily search failed", query=query[:60], error=str(exc))
            return []


class CodeGenTool:
    def __init__(self) -> None:
        from utils.llm_client import get_code_client
        self._llm = get_code_client()

    async def generate(
        self, description: str, language: str = "python",
        context: Optional[str] = None,
    ) -> str:
        system = (
            f"You are an expert {language} engineer. "
            "Generate clean, production-quality, well-commented code. "
            "Return ONLY the code. No prose, no markdown fences."
        )
        user = f"Task: {description}" + (f"\n\nContext:\n{context}" if context else "")
        return await self._llm.ainvoke(system, user)


class ToolRegistry:
    _tavily:  Optional[TavilyTool]  = None
    _codegen: Optional[CodeGenTool] = None

    @classmethod
    def tavily(cls) -> TavilyTool:
        if cls._tavily is None:
            cls._tavily = TavilyTool()
        return cls._tavily

    @classmethod
    def codegen(cls) -> CodeGenTool:
        if cls._codegen is None:
            cls._codegen = CodeGenTool()
        return cls._codegen
