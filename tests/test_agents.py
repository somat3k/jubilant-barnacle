"""
test_agents.py — Tests for AI inference agents (no real API calls).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from python_inference.agents.base_agent import BaseAgent, _build_summary, _parse_json_response
from python_inference.agents.groq_agent import GroqAgent
from python_inference.agents.gemini_agent import GeminiAgent
from python_inference.agents.ollama_agent import OllamaAgent
from python_inference.agents.orchestrator import AgentOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(rows: int = 50) -> pd.DataFrame:
    import numpy as np
    rng = np.random.default_rng(0)
    closes = 1.1 + np.cumsum(rng.normal(0, 0.001, rows))
    return pd.DataFrame(
        {
            "Open": closes * 0.999,
            "High": closes * 1.001,
            "Low": closes * 0.998,
            "Close": closes,
            "Volume": rng.integers(50_000, 100_000, rows).astype(float),
        },
        index=pd.date_range("2024-01-01", periods=rows, freq="h"),
    )


# ---------------------------------------------------------------------------
# _build_summary
# ---------------------------------------------------------------------------

class TestBuildSummary:
    def test_returns_string(self):
        df = _make_df()
        result = _build_summary(df)
        assert isinstance(result, str)
        assert "Rows:" in result


# ---------------------------------------------------------------------------
# _parse_json_response
# ---------------------------------------------------------------------------

class TestParseJsonResponse:
    def test_parses_clean_json(self):
        raw = '{"fast_period": 10, "slow_period": 30}'
        result = _parse_json_response(raw)
        assert result["fast_period"] == 10

    def test_parses_json_embedded_in_text(self):
        raw = 'Here is my suggestion:\n{"fast_period": 8, "slow_period": 25}\nDone.'
        result = _parse_json_response(raw)
        assert result["slow_period"] == 25

    def test_raises_on_no_json(self):
        with pytest.raises(ValueError):
            _parse_json_response("This is not JSON at all.")


# ---------------------------------------------------------------------------
# BaseAgent (concrete stub)
# ---------------------------------------------------------------------------

class _StubAgent(BaseAgent):
    def __init__(self, response: str, **kwargs):
        super().__init__(**kwargs)
        self._response = response

    def _call_ai(self, prompt: str) -> str:
        return self._response


class TestBaseAgent:
    def test_optimise_returns_dict(self):
        agent = _StubAgent('{"fast_period": 12, "slow_period": 26, "stop_loss": 0.01, "take_profit": 0.03, "reasoning": "test"}')
        result = agent.optimise(_make_df())
        assert result["fast_period"] == 12

    def test_safe_optimise_returns_none_on_error(self):
        agent = _StubAgent("not json")
        result = agent.safe_optimise(_make_df())
        assert result is None

    def test_default_timeout_is_600(self):
        agent = _StubAgent("{}")
        assert agent.timeout == 600


# ---------------------------------------------------------------------------
# GroqAgent
# ---------------------------------------------------------------------------

class TestGroqAgent:
    def test_raises_without_api_key(self, tiny_df):
        agent = GroqAgent(api_key=None)
        # Make sure GROQ_API_KEY env var is not set
        import os
        os.environ.pop("GROQ_API_KEY", None)
        result = agent.safe_optimise(tiny_df)
        # Should return None because API key missing (not raise)
        assert result is None

    def test_model_default(self):
        agent = GroqAgent(api_key="fake")
        assert "llama" in agent.model.lower()

    def test_timeout_stored(self):
        agent = GroqAgent(api_key="fake", timeout=300)
        assert agent.timeout == 300


# ---------------------------------------------------------------------------
# GeminiAgent
# ---------------------------------------------------------------------------

class TestGeminiAgent:
    def test_raises_without_api_key(self, tiny_df):
        import os
        os.environ.pop("GEMINI_API_KEY", None)
        agent = GeminiAgent(api_key=None)
        result = agent.safe_optimise(tiny_df)
        assert result is None

    def test_model_default(self):
        agent = GeminiAgent(api_key="fake")
        assert "gemini" in agent.model.lower()


# ---------------------------------------------------------------------------
# OllamaAgent
# ---------------------------------------------------------------------------

class TestOllamaAgent:
    def test_default_host(self):
        import os
        os.environ.pop("OLLAMA_HOST", None)
        agent = OllamaAgent()
        assert "localhost" in agent.host

    def test_custom_host(self):
        agent = OllamaAgent(host="http://192.168.1.100:11434")
        assert "192.168.1.100" in agent.host

    def test_safe_optimise_returns_none_when_ollama_not_running(self, tiny_df):
        agent = OllamaAgent(host="http://localhost:19999", timeout=2)
        result = agent.safe_optimise(tiny_df)
        assert result is None


# ---------------------------------------------------------------------------
# AgentOrchestrator
# ---------------------------------------------------------------------------

class TestAgentOrchestrator:
    def test_raises_when_all_fail(self, tiny_df):
        orch = AgentOrchestrator(timeout=1)
        with pytest.raises(RuntimeError, match="All AI agents failed"):
            orch.optimise(tiny_df)

    def test_returns_first_successful_agent_result(self, tiny_df):
        orch = AgentOrchestrator(timeout=600)
        good_response = '{"fast_period": 7, "slow_period": 21, "stop_loss": 0.01, "take_profit": 0.03, "reasoning": "ok"}'
        # Patch the first agent to succeed
        orch._agents[0] = _StubAgent(good_response)
        result = orch.optimise(tiny_df)
        assert result["fast_period"] == 7

    def test_falls_back_to_second_agent(self, tiny_df):
        orch = AgentOrchestrator(timeout=600)
        good_response = '{"fast_period": 9, "slow_period": 27, "stop_loss": 0.01, "take_profit": 0.03, "reasoning": "ok"}'
        orch._agents[0] = _StubAgent("bad json")      # first agent fails
        orch._agents[1] = _StubAgent(good_response)   # second agent succeeds
        result = orch.optimise(tiny_df)
        assert result["fast_period"] == 9
