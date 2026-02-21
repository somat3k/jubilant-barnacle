"""
algo.py — Main entry point for the AI-powered algorithmic trading package.

Integrates:
  - AI inference agents (Groq, Gemini, Ollama / local)
  - scikit-learn Decision-Tree strategy building
  - ONNX pre-trained model inference
  - Backtesting engine (offline CSV + live OHLCV via yfinance)
  - MQL5 / cTrader signal bridge (file-based)

Usage (CLI):
    python algo.py --symbol EURUSD --timeframe 1h --mode backtest --data csv --csv data/EURUSD_1H.csv
    python algo.py --symbol AAPL   --timeframe 1d --mode live    --data live

The AI component operates with a 10-minute timeout; on timeout or error the
system falls back to the pure Decision-Tree strategy automatically.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from python_inference.agents.orchestrator import AgentOrchestrator
from python_inference.models.onnx_model import ONNXStrategyModel
from python_inference.strategy_builder import DecisionTreeStrategyBuilder
from backtesting.backtest_engine import BacktestEngine
from backtesting.data_loader import load_csv_ohlcv, load_live_ohlcv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("algo")

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def run_backtest(
    df: pd.DataFrame,
    strategy_params: dict | None = None,
    use_ai: bool = True,
    ai_timeout: int = 600,
) -> dict:
    """Run a full backtest cycle, optionally optimising via AI agents.

    Args:
        df: OHLCV DataFrame with columns Open/High/Low/Close/Volume.
        strategy_params: Optional dict of pre-computed strategy parameters.
        use_ai: Whether to invoke AI agents for parameter optimisation.
        ai_timeout: Seconds to wait for AI response before falling back.

    Returns:
        Dict containing backtest statistics and the final strategy_params used.
    """
    if strategy_params is None:
        strategy_params = {}

    if use_ai:
        orchestrator = AgentOrchestrator(timeout=ai_timeout)
        try:
            ai_params = orchestrator.optimise(df)
            logger.info("AI optimisation succeeded: %s", ai_params)
            strategy_params.update(ai_params)
        except Exception as exc:
            logger.warning(
                "AI optimisation failed (%s) — falling back to Decision Tree.", exc
            )

    # Fill in any missing params with Decision Tree defaults
    builder = DecisionTreeStrategyBuilder()
    dt_params = builder.fit_and_extract(df)
    for key, val in dt_params.items():
        strategy_params.setdefault(key, val)

    # Optionally enrich with ONNX signals
    onnx_path = Path(__file__).parent / "onnx_models" / "pretrained" / "strategy_model.onnx"
    if onnx_path.exists():
        try:
            onnx_model = ONNXStrategyModel(str(onnx_path))
            onnx_signal = onnx_model.predict(df)
            strategy_params["onnx_signal"] = onnx_signal
            logger.info(
                "ONNX model signal: %d predictions, unique values: %s",
                len(onnx_signal),
                sorted(set(onnx_signal.tolist())),
            )
        except Exception as exc:
            logger.warning("ONNX inference failed (%s) — skipping.", exc)

    engine = BacktestEngine(df, strategy_params)
    stats = engine.run()
    return {"stats": stats, "strategy_params": strategy_params}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AI Algorithmic Trader — Groq / Gemini / Ollama + ONNX + Backtesting"
    )
    p.add_argument("--symbol", default="EURUSD", help="Trading symbol (default: EURUSD)")
    p.add_argument(
        "--timeframe", default="1h",
        choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        help="Candlestick timeframe",
    )
    p.add_argument(
        "--mode", default="backtest",
        choices=["backtest", "live"],
        help="Execution mode",
    )
    p.add_argument(
        "--data", default="csv",
        choices=["csv", "live"],
        help="Data source",
    )
    p.add_argument("--csv", default="backtesting/data/sample.csv", help="Path to CSV file")
    p.add_argument("--no-ai", action="store_true", help="Disable AI optimisation")
    p.add_argument(
        "--ai-timeout", type=int, default=600,
        help="Seconds to wait for AI response (default: 600 = 10 min)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    logger.info(
        "Starting — symbol=%s timeframe=%s mode=%s data=%s",
        args.symbol, args.timeframe, args.mode, args.data,
    )

    # Load OHLCV data
    if args.data == "csv":
        df = load_csv_ohlcv(args.csv)
    else:
        df = load_live_ohlcv(args.symbol, args.timeframe)

    if df is None or df.empty:
        logger.error("No data loaded — aborting.")
        return 1

    result = run_backtest(
        df,
        use_ai=not args.no_ai,
        ai_timeout=args.ai_timeout,
    )

    logger.info("=== Backtest Results ===")
    for key, val in result["stats"].items():
        logger.info("  %s: %s", key, val)
    logger.info("=== Strategy Params Used ===")
    for key, val in result["strategy_params"].items():
        logger.info("  %s: %s", key, val)

    return 0


if __name__ == "__main__":
    sys.exit(main())
