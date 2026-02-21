"""
test_onnx_model.py — Tests for the ONNX model wrapper (no real .onnx file required
for most tests; the export/load round-trip test is marked optional).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from python_inference.models.onnx_model import ONNXStrategyModel, _FEATURE_COLUMNS


class TestONNXStrategyModelLoad:
    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ONNXStrategyModel(tmp_path / "nonexistent.onnx")

    def test_raises_without_onnxruntime(self, tmp_path):
        fake_model = tmp_path / "model.onnx"
        fake_model.write_bytes(b"")
        with patch.dict("sys.modules", {"onnxruntime": None}):
            with pytest.raises((RuntimeError, ImportError)):
                ONNXStrategyModel(fake_model)


class TestONNXFromSklearn:
    """Test round-trip: sklearn → ONNX export → load → predict."""

    def test_export_and_predict(self, tiny_df, tmp_path):
        pytest.importorskip("onnxruntime")
        pytest.importorskip("skl2onnx")

        from backtesting.strategies.decision_tree_strategy import (
            DecisionTreeStrategy,
            build_features,
        )

        strategy = DecisionTreeStrategy(max_depth=3).fit(tiny_df)
        onnx_path = tmp_path / "model.onnx"

        model = ONNXStrategyModel.from_sklearn(
            strategy._model,
            feature_dim=len(_FEATURE_COLUMNS),
            output_path=onnx_path,
        )

        signals = model.predict(tiny_df)
        features = build_features(tiny_df)
        assert len(signals) == len(features)
        assert set(np.unique(signals)).issubset({0, 1})


class TestFeatureColumns:
    def test_feature_list_length(self):
        assert len(_FEATURE_COLUMNS) == 10

    def test_feature_names_unique(self):
        assert len(_FEATURE_COLUMNS) == len(set(_FEATURE_COLUMNS))
