"""
data_loader.py — OHLCV data loaders for CSV files and live market data (yfinance).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Mapping from internal timeframe strings to yfinance interval strings
_YF_INTERVAL_MAP: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "60m",
    "4h": "1h",   # yfinance has no native 4h interval; use 1h as the closest available
    "1d": "1d",
}

_REQUIRED_COLS = {"Open", "High", "Low", "Close", "Volume"}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common column aliases to standard OHLCV names."""
    rename = {}
    for col in df.columns:
        low = col.strip().lower()
        if low in {"open", "o"}:
            rename[col] = "Open"
        elif low in {"high", "h"}:
            rename[col] = "High"
        elif low in {"low", "l"}:
            rename[col] = "Low"
        elif low in {"close", "c", "price"}:
            rename[col] = "Close"
        elif low in {"volume", "vol", "v"}:
            rename[col] = "Volume"
    return df.rename(columns=rename)


def load_csv_ohlcv(path: str | Path) -> pd.DataFrame | None:
    """Load OHLCV data from a CSV file.

    The CSV must contain at minimum Open, High, Low, Close columns (and
    optionally Volume).  A datetime index (or a ``Date``/``Time``/``Datetime``
    column) is expected but the function will fall back gracefully.

    Args:
        path: Filesystem path to the CSV file.

    Returns:
        Normalised DataFrame or *None* if the file cannot be read.
    """
    path = Path(path)
    if not path.exists():
        logger.error("CSV file not found: %s", path)
        return None
    try:
        df = pd.read_csv(path, parse_dates=True)
        df = _normalise_columns(df)

        # Try to set a datetime index
        for candidate in ("Datetime", "Date", "Time", "Timestamp", "timestamp"):
            if candidate in df.columns:
                df[candidate] = pd.to_datetime(df[candidate])
                df = df.set_index(candidate)
                break

        # Ensure numeric OHLCV columns
        for col in _REQUIRED_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "Volume" not in df.columns:
            df["Volume"] = 0

        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        logger.info("Loaded %d rows from %s", len(df), path)
        return df
    except Exception as exc:
        logger.error("Failed to load CSV %s: %s", path, exc)
        return None


def load_live_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    period: str = "60d",
) -> pd.DataFrame | None:
    """Download live OHLCV data using yfinance.

    Args:
        symbol: Ticker symbol, e.g. ``"AAPL"`` or ``"EURUSD=X"``.
        timeframe: Internal timeframe string (``"1m"`` … ``"1d"``).
        period: Look-back period supported by yfinance (e.g. ``"60d"``).

    Returns:
        Normalised DataFrame or *None* on failure.
    """
    try:
        import yfinance as yf  # optional at module level
    except ImportError:
        logger.error("yfinance is not installed — cannot fetch live data.")
        return None

    interval = _YF_INTERVAL_MAP.get(timeframe, "60m")
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            logger.warning("yfinance returned empty data for %s / %s", symbol, interval)
            return None
        df = _normalise_columns(df)
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        logger.info(
            "Fetched %d rows of live data for %s (%s)", len(df), symbol, interval
        )
        return df
    except Exception as exc:
        logger.error("yfinance download failed for %s: %s", symbol, exc)
        return None
