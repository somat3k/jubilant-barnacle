"""
decision_tree_strategy.py — scikit-learn Decision Tree-based trading strategy.

The strategy:
  1. Computes technical indicator features (RSI, SMA, EMA, ATR, MACD).
  2. Trains a DecisionTreeClassifier on historical data to predict
     next-bar direction (up/down).
  3. Generates buy / sell signals based on classifier output.

This module can be used standalone or integrated into BacktestEngine via
the DecisionTreeStrategyBuilder in python_inference/strategy_builder.py.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.where(loss != 0, other=np.nan)
    rsi = 100 - (100 / (1 + rs))
    # When there are no losses at all RSI should be 100 (purely bullish).
    rsi = rsi.where(loss != 0, other=100.0)
    return rsi


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = _ema(series, fast)
    slow_ema = _ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = _ema(macd_line, signal)
    return macd_line, signal_line


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute indicator-based features from an OHLCV DataFrame.

    Returns a new DataFrame with NaN rows removed.
    """
    feat = pd.DataFrame(index=df.index)
    feat["rsi"] = _rsi(df["Close"])
    feat["sma_10"] = df["Close"].rolling(10).mean()
    feat["sma_30"] = df["Close"].rolling(30).mean()
    feat["ema_12"] = _ema(df["Close"], 12)
    feat["ema_26"] = _ema(df["Close"], 26)
    macd, macd_sig = _macd(df["Close"])
    feat["macd"] = macd
    feat["macd_signal"] = macd_sig
    feat["atr"] = _atr(df["High"], df["Low"], df["Close"])
    feat["price_change"] = df["Close"].pct_change()
    feat["volume_change"] = df["Volume"].pct_change() if "Volume" in df.columns else 0.0
    return feat.dropna()


def build_labels(df: pd.DataFrame, lookahead: int = 1) -> pd.Series:
    """Binary label: 1 if close price goes up in the next ``lookahead`` bars."""
    future_return = df["Close"].shift(-lookahead) / df["Close"] - 1
    return (future_return > 0).astype(int)


# ---------------------------------------------------------------------------
# DecisionTreeStrategy
# ---------------------------------------------------------------------------

class DecisionTreeStrategy:
    """Train a decision tree classifier on OHLCV data and generate signals.

    Args:
        max_depth: Maximum depth of the decision tree.
        lookahead: Number of bars ahead to predict.
        test_size: Fraction of data held out for validation.
    """

    def __init__(
        self,
        max_depth: int = 5,
        lookahead: int = 1,
        test_size: float = 0.2,
    ) -> None:
        self.max_depth = max_depth
        self.lookahead = lookahead
        self.test_size = test_size
        self._model: DecisionTreeClassifier | None = None
        self._feature_names: list[str] = []

    def fit(self, df: pd.DataFrame) -> "DecisionTreeStrategy":
        """Fit the decision tree on the supplied OHLCV DataFrame."""
        features = build_features(df)
        labels = build_labels(df, self.lookahead).reindex(features.index).dropna()
        features = features.loc[labels.index]

        if len(features) < 50:
            raise ValueError("Not enough data to train (need ≥ 50 rows after feature engineering).")

        split = int(len(features) * (1 - self.test_size))
        X_train, y_train = features.iloc[:split], labels.iloc[:split]

        self._model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            random_state=42,
        )
        self._model.fit(X_train, y_train)
        self._feature_names = list(features.columns)

        if len(features) > split:
            X_test, y_test = features.iloc[split:], labels.iloc[split:]
            acc = self._model.score(X_test, y_test)
            logger.info("DecisionTree validation accuracy: %.3f", acc)

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return a signal array aligned to df's index.

        Values: 1 = buy, 0 = hold/sell.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted yet — call fit() first.")
        features = build_features(df)
        return self._model.predict(features)

    def get_params(self) -> dict:
        """Return strategy parameters suitable for BacktestEngine."""
        if self._model is None:
            return {"fast_period": 10, "slow_period": 30}
        importances = self._model.feature_importances_
        top_idx = int(np.argmax(importances))
        top_feature = self._feature_names[top_idx] if self._feature_names else "unknown"
        return {
            "fast_period": self.lookahead * 5,
            "slow_period": self.lookahead * 20,
            "top_feature": top_feature,
            "max_depth": self.max_depth,
        }
