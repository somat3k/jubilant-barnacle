"""
onnx_model.py — ONNX runtime wrapper for pre-trained trading strategy models.

Pre-trained ONNX models can be placed in ``onnx_models/pretrained/``.
The module exports ``ONNXStrategyModel`` for loading and running inference.

Local hosting: the model runs entirely on-device via onnxruntime — no network
calls required.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtesting.strategies.decision_tree_strategy import build_features

logger = logging.getLogger(__name__)

# Expected feature order (must match training pipeline)
_FEATURE_COLUMNS = [
    "rsi", "sma_10", "sma_30", "ema_12", "ema_26",
    "macd", "macd_signal", "atr", "price_change", "volume_change",
]


class ONNXStrategyModel:
    """Load and run inference with a pre-trained ONNX strategy model.

    The ONNX model is expected to:
      - Accept a float32 array of shape (N, num_features)
      - Output a float32 / int64 array of shape (N,) with values in {0, 1}
        or {-1, 0, 1} (sell / hold / buy).

    Args:
        model_path: Path to the ``.onnx`` model file.
    """

    def __init__(self, model_path: str | Path) -> None:
        self._path = Path(model_path)
        self._session = None
        self._input_name: str = ""
        self._load()

    def _load(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime not installed.  Run: pip install onnxruntime"
            ) from exc

        if not self._path.exists():
            raise FileNotFoundError(f"ONNX model file not found: {self._path}")

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(str(self._path), sess_options=opts)
        self._input_name = self._session.get_inputs()[0].name
        logger.info("Loaded ONNX model from %s", self._path)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Run model inference on an OHLCV DataFrame.

        Args:
            df: OHLCV DataFrame.

        Returns:
            1-D numpy array of signals aligned to the features rows.
            Values are typically 1 (buy) or 0 (hold/sell).
        """
        if self._session is None:
            raise RuntimeError("ONNX session not initialised.")

        features = build_features(df)
        available = [c for c in _FEATURE_COLUMNS if c in features.columns]
        X = features[available].astype(np.float32).to_numpy()

        outputs = self._session.run(None, {self._input_name: X})
        signals = np.array(outputs[0]).flatten()
        logger.info(
            "ONNX model produced %d signals (mean=%.3f)", len(signals), signals.mean()
        )
        return signals

    @classmethod
    def from_sklearn(
        cls,
        sklearn_model: Any,
        feature_dim: int,
        output_path: str | Path,
    ) -> "ONNXStrategyModel":
        """Export a fitted scikit-learn model to ONNX and load it.

        Args:
            sklearn_model: Fitted scikit-learn estimator (e.g.
                ``DecisionTreeClassifier``).
            feature_dim: Number of input features.
            output_path: Where to write the ``.onnx`` file.

        Returns:
            An ``ONNXStrategyModel`` instance ready for inference.
        """
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
        except ImportError as exc:
            raise RuntimeError(
                "skl2onnx not installed.  Run: pip install skl2onnx"
            ) from exc

        initial_type = [("float_input", FloatTensorType([None, feature_dim]))]
        onnx_model = convert_sklearn(sklearn_model, initial_types=initial_type)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        logger.info("Exported sklearn model to ONNX: %s", output_path)
        return cls(output_path)
