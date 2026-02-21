"""
test_decision_tree_strategy.py — Tests for the scikit-learn Decision Tree strategy.
"""

from __future__ import annotations

import numpy as np
import pytest

from backtesting.strategies.decision_tree_strategy import (
    DecisionTreeStrategy,
    build_features,
    build_labels,
)


class TestBuildFeatures:
    def test_returns_dataframe(self, tiny_df):
        feat = build_features(tiny_df)
        assert not feat.empty

    def test_expected_columns(self, tiny_df):
        feat = build_features(tiny_df)
        expected = {"rsi", "sma_10", "sma_30", "ema_12", "ema_26", "macd",
                    "macd_signal", "atr", "price_change", "volume_change"}
        assert expected.issubset(set(feat.columns))

    def test_no_all_nan_columns(self, tiny_df):
        feat = build_features(tiny_df)
        for col in feat.columns:
            assert feat[col].notna().any(), f"{col} is all NaN"


class TestBuildLabels:
    def test_binary_labels(self, tiny_df):
        labels = build_labels(tiny_df)
        unique = set(labels.dropna().unique())
        assert unique.issubset({0, 1})

    def test_same_length_as_df(self, tiny_df):
        labels = build_labels(tiny_df)
        assert len(labels) == len(tiny_df)


class TestDecisionTreeStrategy:
    def test_fit_returns_self(self, tiny_df):
        strategy = DecisionTreeStrategy(max_depth=3)
        result = strategy.fit(tiny_df)
        assert result is strategy

    def test_predict_shape(self, tiny_df):
        strategy = DecisionTreeStrategy(max_depth=3).fit(tiny_df)
        predictions = strategy.predict(tiny_df)
        features = build_features(tiny_df)
        assert len(predictions) == len(features)

    def test_predict_binary_values(self, tiny_df):
        strategy = DecisionTreeStrategy(max_depth=3).fit(tiny_df)
        predictions = strategy.predict(tiny_df)
        assert set(np.unique(predictions)).issubset({0, 1})

    def test_get_params_returns_dict(self, tiny_df):
        strategy = DecisionTreeStrategy(max_depth=3).fit(tiny_df)
        params = strategy.get_params()
        assert isinstance(params, dict)
        assert "fast_period" in params
        assert "slow_period" in params

    def test_predict_raises_before_fit(self, tiny_df):
        strategy = DecisionTreeStrategy()
        with pytest.raises(RuntimeError):
            strategy.predict(tiny_df)

    def test_insufficient_data_raises(self):
        import pandas as pd
        tiny = pd.DataFrame(
            {"Open": [1.0]*5, "High": [1.1]*5, "Low": [0.9]*5,
             "Close": [1.05]*5, "Volume": [1000]*5}
        )
        with pytest.raises(ValueError):
            DecisionTreeStrategy().fit(tiny)
