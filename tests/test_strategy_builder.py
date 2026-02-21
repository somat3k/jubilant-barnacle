"""
test_strategy_builder.py — Tests for DecisionTreeStrategyBuilder.
"""

from __future__ import annotations

import pytest
from python_inference.strategy_builder import DecisionTreeStrategyBuilder


class TestDecisionTreeStrategyBuilder:
    def test_fit_and_extract_returns_dict(self, tiny_df):
        builder = DecisionTreeStrategyBuilder()
        params = builder.fit_and_extract(tiny_df)
        assert isinstance(params, dict)

    def test_params_include_periods(self, tiny_df):
        builder = DecisionTreeStrategyBuilder()
        params = builder.fit_and_extract(tiny_df)
        assert "fast_period" in params
        assert "slow_period" in params

    def test_fast_less_than_slow(self, tiny_df):
        builder = DecisionTreeStrategyBuilder()
        params = builder.fit_and_extract(tiny_df)
        assert params["fast_period"] < params["slow_period"]

    def test_returns_defaults_on_bad_data(self):
        import pandas as pd
        bad_df = pd.DataFrame({"Close": [1.0], "Open": [1.0], "High": [1.0], "Low": [1.0], "Volume": [0]})
        builder = DecisionTreeStrategyBuilder()
        params = builder.fit_and_extract(bad_df)
        assert params == {"fast_period": 10, "slow_period": 30}
