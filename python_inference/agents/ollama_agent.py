"""
ollama_agent.py — Strategy optimisation agent using a locally hosted Ollama instance.

Ollama runs LLMs on-device (CPU / GPU) for fully offline inference.
Default endpoint: http://localhost:11434

No API key required — the model is served locally.

Environment variable:
    OLLAMA_HOST — Override the Ollama server URL (default: http://localhost:11434).
"""

from __future__ import annotations

import os

from python_inference.agents.base_agent import BaseAgent

_DEFAULT_MODEL = "llama3"
_DEFAULT_HOST = "http://localhost:11434"


class OllamaAgent(BaseAgent):
    """AI optimisation agent using a locally running Ollama server.

    Args:
        host: Ollama server base URL (default: ``http://localhost:11434``).
        model: Ollama model name (default: ``llama3``).
        timeout: Seconds to wait for response (default: 600).
    """

    def __init__(
        self,
        host: str | None = None,
        model: str = _DEFAULT_MODEL,
        timeout: int = BaseAgent.DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__(timeout=timeout, model=model)
        self.host = host or os.getenv("OLLAMA_HOST", _DEFAULT_HOST)

    def _call_ai(self, prompt: str) -> str:
        try:
            import ollama
        except ImportError as exc:
            raise RuntimeError(
                "ollama package not installed.  Run: pip install ollama"
            ) from exc

        client = ollama.Client(host=self.host, timeout=float(self.timeout))
        response = client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2, "num_predict": 512},
        )
        return response["message"]["content"]
