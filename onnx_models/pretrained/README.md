# Pre-trained ONNX Models

Place pre-trained `.onnx` model files in this directory.

## Expected model interface

| Property | Value |
|---|---|
| Input name | `float_input` |
| Input shape | `(N, 10)` — 10 features (see below) |
| Input dtype | `float32` |
| Output shape | `(N,)` |
| Output values | `1` = buy, `0` = hold/sell |

## Feature order (must match training)

1. `rsi`
2. `sma_10`
3. `sma_30`
4. `ema_12`
5. `ema_26`
6. `macd`
7. `macd_signal`
8. `atr`
9. `price_change`
10. `volume_change`

## Generating a model from scratch

```python
from backtesting.data_loader import load_csv_ohlcv
from backtesting.strategies.decision_tree_strategy import DecisionTreeStrategy
from python_inference.models.onnx_model import ONNXStrategyModel

df = load_csv_ohlcv("backtesting/data/sample.csv")
strategy = DecisionTreeStrategy(max_depth=5).fit(df)

ONNXStrategyModel.from_sklearn(
    strategy._model,
    feature_dim=10,
    output_path="onnx_models/pretrained/strategy_model.onnx",
)
```
