"""
strategy_builder.py — scikit-learn Decision Tree strategy builder for the
python_inference layer.

Wraps DecisionTreeStrategy from the backtesting.strategies module and exposes
a simpler interface for use in algo.py.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from backtesting.strategies.decision_tree_strategy import DecisionTreeStrategy

logger = logging.getLogger(__name__)


class DecisionTreeStrategyBuilder:
    """High-level wrapper that fits a DecisionTreeStrategy and extracts params.

    Args:
        max_depth: Maximum tree depth (default: 5).
        lookahead: Bars ahead to predict (default: 1).
    """

    def __init__(self, max_depth: int = 5, lookahead: int = 1) -> None:
        self.max_depth = max_depth
        self.lookahead = lookahead
        self._strategy: DecisionTreeStrategy | None = None

    def fit_and_extract(self, df: pd.DataFrame) -> dict[str, Any]:
        """Fit the Decision Tree on ``df`` and return strategy parameter dict.

        Args:
            df: OHLCV DataFrame.

        Returns:
            Dict with keys compatible with BacktestEngine strategy_params.
            Falls back to sensible defaults if fitting fails.
        """
        try:
            self._strategy = DecisionTreeStrategy(
                max_depth=self.max_depth,
                lookahead=self.lookahead,
            ).fit(df)
            params = self._strategy.get_params()
            logger.info("DecisionTreeStrategyBuilder params: %s", params)
            return params
        except Exception as exc:
            logger.warning(
                "DecisionTreeStrategy fitting failed (%s) — using defaults.", exc
            )
            return {"fast_period": 10, "slow_period": 30}
