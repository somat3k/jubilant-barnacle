"""
gemini_agent.py — Strategy optimisation agent using Google Gemini.

Uses the ``google-genai`` SDK (``google.genai``).

Environment variable:
    GEMINI_API_KEY — Google AI Studio API key (or pass api_key= constructor argument).
"""

from __future__ import annotations

import os

from python_inference.agents.base_agent import BaseAgent

_DEFAULT_MODEL = "gemini-2.0-flash"


class GeminiAgent(BaseAgent):
    """AI optimisation agent backed by Google Gemini.

    Args:
        api_key: Google AI Studio API key.  Falls back to ``GEMINI_API_KEY`` env var.
        model: Gemini model identifier (default: ``gemini-2.0-flash``).
        timeout: Seconds to wait for response (default: 600).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        timeout: int = BaseAgent.DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__(
            timeout=timeout,
            api_key=api_key or os.getenv("GEMINI_API_KEY"),
            model=model,
        )

    def _call_ai(self, prompt: str) -> str:
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError as exc:
            raise RuntimeError(
                "google-genai package not installed.  "
                "Run: pip install google-genai"
            ) from exc

        if not self.api_key:
            raise ValueError(
                "Gemini API key not set.  Pass api_key= or set GEMINI_API_KEY env var."
            )

        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=512,
                # google-genai HttpOptions.timeout is in milliseconds
                http_options=genai_types.HttpOptions(timeout=self.timeout * 1000),
            ),
        )
        return response.text
