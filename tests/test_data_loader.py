"""
test_data_loader.py — Tests for the OHLCV data loading utilities.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from backtesting.data_loader import load_csv_ohlcv, _normalise_columns

SAMPLE_CSV = Path(__file__).parent.parent / "backtesting" / "data" / "sample.csv"


class TestNormaliseColumns:
    def test_rename_lowercase(self):
        df = pd.DataFrame({"open": [1], "high": [2], "low": [0.9], "close": [1.1], "vol": [1000]})
        result = _normalise_columns(df)
        assert set(result.columns) >= {"Open", "High", "Low", "Close", "Volume"}

    def test_rename_price_to_close(self):
        df = pd.DataFrame({"price": [1.5]})
        result = _normalise_columns(df)
        assert "Close" in result.columns

    def test_already_correct_columns_unchanged(self):
        df = pd.DataFrame({"Open": [1], "High": [2], "Low": [0.9], "Close": [1.1], "Volume": [500]})
        result = _normalise_columns(df)
        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]


class TestLoadCsvOHLCV:
    def test_loads_sample_csv(self):
        df = load_csv_ohlcv(SAMPLE_CSV)
        assert df is not None
        assert not df.empty
        assert {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns)

    def test_returns_none_for_missing_file(self, tmp_path):
        df = load_csv_ohlcv(tmp_path / "nonexistent.csv")
        assert df is None

    def test_datetime_index(self):
        df = load_csv_ohlcv(SAMPLE_CSV)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_numeric_types(self):
        df = load_csv_ohlcv(SAMPLE_CSV)
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} should be numeric"

    def test_no_nan_ohlc(self):
        df = load_csv_ohlcv(SAMPLE_CSV)
        for col in ("Open", "High", "Low", "Close"):
            assert df[col].notna().all(), f"{col} has NaN values"

    def test_volume_column_added_when_missing(self, tmp_path):
        csv_path = tmp_path / "no_volume.csv"
        csv_path.write_text("Date,Open,High,Low,Close\n2024-01-01,1.1,1.2,1.0,1.15\n")
        df = load_csv_ohlcv(csv_path)
        assert df is not None
        assert "Volume" in df.columns
        assert df["Volume"].iloc[0] == 0
