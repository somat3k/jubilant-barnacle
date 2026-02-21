"""
orchestrator.py — Multi-agent orchestrator with priority fallback chain.

Priority order (first available / successful wins):
  1. GroqAgent   — fastest cloud inference
  2. GeminiAgent — Google Gemini cloud
  3. OllamaAgent — local Ollama (always available offline)

If all agents fail (or timeout), the caller should fall back to the
Decision-Tree strategy (handled in algo.py / run_backtest).

10-minute timeout is passed to every agent.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd

from python_inference.agents.groq_agent import GroqAgent
from python_inference.agents.gemini_agent import GeminiAgent
from python_inference.agents.ollama_agent import OllamaAgent

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Tries agents in priority order, returning the first successful result.

    Args:
        timeout: Per-agent timeout in seconds (default: 600 = 10 minutes).
        groq_api_key: Groq API key (falls back to ``GROQ_API_KEY`` env var).
        gemini_api_key: Gemini API key (falls back to ``GEMINI_API_KEY`` env var).
        ollama_host: Ollama server URL (falls back to ``OLLAMA_HOST`` env var).
    """

    def __init__(
        self,
        timeout: int = 600,
        groq_api_key: str | None = None,
        gemini_api_key: str | None = None,
        ollama_host: str | None = None,
    ) -> None:
        self._timeout = timeout
        self._agents = [
            GroqAgent(api_key=groq_api_key, timeout=timeout),
            GeminiAgent(api_key=gemini_api_key, timeout=timeout),
            OllamaAgent(host=ollama_host, timeout=timeout),
        ]

    def optimise(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run optimisation, trying each agent until one succeeds.

        Args:
            df: OHLCV DataFrame.

        Returns:
            Strategy parameter dict from the first successful agent.

        Raises:
            RuntimeError: If all agents fail.
        """
        errors: list[str] = []
        for agent in self._agents:
            result = agent.safe_optimise(df)
            if result is not None:
                logger.info(
                    "Orchestrator: %s succeeded.", agent.__class__.__name__
                )
                return result
            errors.append(agent.__class__.__name__)

        raise RuntimeError(
            f"All AI agents failed ({', '.join(errors)}). "
            "Falling back to Decision-Tree strategy."
        )
