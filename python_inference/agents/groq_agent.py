"""
groq_agent.py — Strategy optimisation agent using the Groq cloud API.

Groq provides ultra-low-latency inference for Llama-based models.
The agent honours a configurable timeout (default 10 minutes).

Environment variable:
    GROQ_API_KEY — Groq API key (or pass api_key= constructor argument).
"""

from __future__ import annotations

import os

from python_inference.agents.base_agent import BaseAgent

_DEFAULT_MODEL = "llama3-8b-8192"


class GroqAgent(BaseAgent):
    """AI optimisation agent backed by the Groq inference API.

    Args:
        api_key: Groq API key.  Falls back to ``GROQ_API_KEY`` env var.
        model: Groq model identifier (default: ``llama3-8b-8192``).
        timeout: Seconds to wait for response (default: 600).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        timeout: int = BaseAgent.DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__(timeout=timeout, api_key=api_key or os.getenv("GROQ_API_KEY"), model=model)

    def _call_ai(self, prompt: str) -> str:
        try:
            from groq import Groq
        except ImportError as exc:
            raise RuntimeError(
                "groq package not installed.  Run: pip install groq"
            ) from exc

        if not self.api_key:
            raise ValueError(
                "Groq API key not set.  Pass api_key= or set GROQ_API_KEY env var."
            )

        client = Groq(api_key=self.api_key, timeout=float(self.timeout))
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512,
        )
        return response.choices[0].message.content
