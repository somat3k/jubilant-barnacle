"""
test_backtest_engine.py — Tests for the BacktestEngine wrapper.
"""

from __future__ import annotations

import pytest

from backtesting.backtest_engine import BacktestEngine


class TestBacktestEngine:
    def test_run_returns_dict(self, tiny_df):
        engine = BacktestEngine(tiny_df)
        stats = engine.run()
        assert isinstance(stats, dict)

    def test_stats_has_required_keys(self, tiny_df):
        engine = BacktestEngine(tiny_df)
        stats = engine.run()
        for key in ("Return [%]", "# Trades"):
            assert key in stats, f"Missing key: {key}"

    def test_run_with_custom_params(self, tiny_df):
        params = {"fast_period": 5, "slow_period": 20}
        engine = BacktestEngine(tiny_df, strategy_params=params)
        stats = engine.run()
        assert isinstance(stats, dict)

    def test_dummy_stats_on_insufficient_data(self):
        import pandas as pd
        tiny = pd.DataFrame(
            {"Open": [1.0]*5, "High": [1.1]*5, "Low": [0.9]*5,
             "Close": [1.05]*5, "Volume": [1000]*5},
            index=pd.date_range("2024-01-01", periods=5, freq="h"),
        )
        engine = BacktestEngine(tiny)
        stats = engine.run()
        # Should return dummy stats without raising
        assert stats["# Trades"] == 0

    def test_prepare_df_with_range_index(self, tiny_df):
        df = tiny_df.reset_index(drop=True)
        engine = BacktestEngine(df)
        prepared = engine._prepare_df()
        assert prepared is not None
        import pandas as pd
        assert isinstance(prepared.index, pd.DatetimeIndex)
