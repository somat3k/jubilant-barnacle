"""
conftest.py — shared pytest fixtures for the algotrading test suite.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SAMPLE_CSV = Path(__file__).parent.parent / "backtesting" / "data" / "sample.csv"
ONNX_PRETRAINED_DIR = Path(__file__).parent.parent / "onnx_models" / "pretrained"


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Return the bundled sample OHLCV CSV as a DataFrame."""
    from backtesting.data_loader import load_csv_ohlcv
    df = load_csv_ohlcv(SAMPLE_CSV)
    assert df is not None and not df.empty
    return df


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    """Return a minimal synthetic OHLCV DataFrame (200 rows) for fast tests."""
    rng = np.random.default_rng(42)
    n = 200
    closes = 1.10 + np.cumsum(rng.normal(0, 0.001, n))
    index = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "Open":   closes * (1 - rng.uniform(0, 0.001, n)),
            "High":   closes * (1 + rng.uniform(0, 0.002, n)),
            "Low":    closes * (1 - rng.uniform(0, 0.002, n)),
            "Close":  closes,
            "Volume": rng.integers(50_000, 150_000, n).astype(float),
        },
        index=index,
    )
