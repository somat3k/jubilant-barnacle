"""
base_agent.py — Abstract base class for all AI inference agents.

All agents share:
  - A configurable timeout (default 10 minutes = 600 s)
  - A ``optimise(df)`` method that returns strategy params dict
  - A fallback-safe ``safe_optimise(df)`` wrapper
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Default prompt template for strategy optimisation
_OPTIMISE_PROMPT_TEMPLATE = """
You are an expert algorithmic trader. Given the following OHLCV market data
summary, suggest optimal parameters for a moving-average crossover strategy.

Market summary:
{summary}

Respond ONLY with a valid JSON object containing the following keys:
- fast_period  (integer, 2-50)
- slow_period  (integer, 10-200)
- stop_loss    (float, 0.001-0.05, fraction of price)
- take_profit  (float, 0.002-0.10, fraction of price)
- reasoning    (string, brief explanation)

Example:
{{"fast_period": 10, "slow_period": 30, "stop_loss": 0.01, "take_profit": 0.03, "reasoning": "..."}}
"""


def _build_summary(df: pd.DataFrame) -> str:
    """Create a brief text summary of OHLCV data for the AI prompt."""
    rows = len(df)
    start = df.index[0] if hasattr(df.index, '__getitem__') else "N/A"
    end = df.index[-1] if hasattr(df.index, '__getitem__') else "N/A"
    close = df["Close"]
    return (
        f"Rows: {rows}, Date range: {start} to {end}, "
        f"Close min: {close.min():.5f}, max: {close.max():.5f}, "
        f"mean: {close.mean():.5f}, std: {close.std():.5f}, "
        f"last: {close.iloc[-1]:.5f}"
    )


def _parse_json_response(text: str) -> dict[str, Any]:
    """Extract and parse the first JSON object found in an AI response."""
    text = text.strip()
    # Try to find JSON block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"No valid JSON found in response: {text[:200]!r}")


class BaseAgent(ABC):
    """Abstract base for AI strategy optimisation agents.

    Args:
        timeout: Seconds to wait for the AI response (default: 600 = 10 min).
        api_key: API key for cloud providers (unused for local agents).
        model: Model identifier string.
    """

    DEFAULT_TIMEOUT: int = 600  # 10 minutes

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.timeout = timeout
        self.api_key = api_key
        self.model = model
        self._logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def _call_ai(self, prompt: str) -> str:
        """Send prompt to the AI backend and return raw text response.

        Implementations must respect ``self.timeout``.

        Args:
            prompt: The text prompt to send.

        Returns:
            Raw text response from the AI model.
        """

    def optimise(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run AI strategy optimisation on the supplied OHLCV DataFrame.

        Args:
            df: OHLCV DataFrame.

        Returns:
            Dict of strategy parameters.

        Raises:
            Exception: If the AI call or JSON parsing fails.
        """
        summary = _build_summary(df)
        prompt = _OPTIMISE_PROMPT_TEMPLATE.format(summary=summary)
        self._logger.info(
            "Sending optimisation prompt to %s (timeout=%ds)…",
            self.__class__.__name__, self.timeout,
        )
        raw = self._call_ai(prompt)
        params = _parse_json_response(raw)
        self._logger.info("Received params: %s", params)
        return params

    def safe_optimise(self, df: pd.DataFrame) -> dict[str, Any] | None:
        """Like ``optimise`` but returns *None* instead of raising on error."""
        try:
            return self.optimise(df)
        except Exception as exc:
            self._logger.warning(
                "%s optimisation failed: %s", self.__class__.__name__, exc
            )
            return None
