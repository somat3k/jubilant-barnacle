# jubilant-barnacle

> **AI-powered Python algorithmic trading package** — Groq · Gemini · Ollama · ONNX · Backtesting · MQL5 · cTrader

---

## Overview

`jubilant-barnacle` is a modular algotrading package that combines:

| Component | Technology |
|---|---|
| AI Strategy Optimisation | Groq (Llama3), Google Gemini, Ollama (local) |
| Signal / Decision Making | scikit-learn Decision Tree |
| ONNX Inference | onnxruntime (locally hosted, no network) |
| Backtesting | backtesting.py — offline CSV & live yfinance |
| Forex / CFD Automation | MQL5 Expert Advisor (MetaTrader 5) |
| Derivatives Automation | cTrader cBot (C#, one-page headless) |
| Offline AI | Local Ollama LLM server |

---

## Directory Structure

```
jubilant-barnacle/
├── algo.py                        ← Main entry point / CLI
├── pyproject.toml
├── requirements.txt
│
├── backtesting/                   ← Offline backtesting engine
│   ├── backtest_engine.py
│   ├── data_loader.py
│   ├── data/
│   │   └── sample.csv             ← EURUSD 1-D sample data
│   └── strategies/
│       └── decision_tree_strategy.py
│
├── python_inference/              ← AI agents & ONNX inference
│   ├── strategy_builder.py
│   ├── agents/
│   │   ├── base_agent.py          ← Abstract agent (10-min timeout + fallback)
│   │   ├── groq_agent.py          ← Groq cloud (Llama3)
│   │   ├── gemini_agent.py        ← Google Gemini (google-genai SDK)
│   │   ├── ollama_agent.py        ← Local Ollama server
│   │   └── orchestrator.py        ← Priority fallback chain
│   └── models/
│       └── onnx_model.py          ← ONNX runtime wrapper + sklearn export
│
├── onnx_models/
│   └── pretrained/
│       └── README.md              ← Instructions for placing .onnx files
│
├── mql5/                          ← MetaTrader 5 Expert Advisor
│   ├── Include/
│   │   ├── AIStrategy.mqh         ← Signal bridge, SMA fallback, risk helpers
│   │   └── OHLCVData.mqh          ← OHLCV bar access + CSV export helper
│   ├── Experts/
│   │   └── AIAlgoTrader.mq5       ← Full EA with AI + SMA fallback
│   └── Properties/
│       └── settings.mqh           ← Configurable EA parameters
│
├── ctrader/
│   └── AIAlgoBot.cs               ← ONE-PAGE headless cTrader cBot (C#)
│
└── tests/
    ├── conftest.py
    ├── test_data_loader.py
    ├── test_decision_tree_strategy.py
    ├── test_backtest_engine.py
    ├── test_agents.py
    ├── test_onnx_model.py
    └── test_strategy_builder.py
```

---

## Installation

```bash
pip install -r requirements.txt
# or
pip install -e ".[dev]"
```

---

## Quick Start

### Run a CSV backtest with AI optimisation

```bash
python algo.py \
  --symbol EURUSD \
  --timeframe 1d \
  --mode backtest \
  --data csv \
  --csv backtesting/data/sample.csv
```

### Run without AI (Decision Tree only)

```bash
python algo.py --data csv --no-ai
```

### Run a live-data backtest (yfinance)

```bash
python algo.py --symbol AAPL --timeframe 1d --data live
```

---

## Configuration

| Environment Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Groq API key |
| `GEMINI_API_KEY` | Google AI Studio API key |
| `OLLAMA_HOST` | Ollama server URL (default: `http://localhost:11434`) |

Copy `.env.example` to `.env` and fill in your keys.

---

## AI Agent Timeout & Fallback

Every AI agent is constructed with a **600-second (10-minute) timeout**.  
The `AgentOrchestrator` tries agents in priority order:

1. **GroqAgent** — fastest cloud inference
2. **GeminiAgent** — Google Gemini cloud
3. **OllamaAgent** — local Ollama (works fully offline)

If all three fail or timeout, `algo.py` automatically falls back to the **scikit-learn Decision Tree** strategy — no manual intervention required.

---

## ONNX Models

Pre-trained `.onnx` files belong in `onnx_models/pretrained/`.  
See [`onnx_models/pretrained/README.md`](onnx_models/pretrained/README.md) for the expected model interface and a script to export a trained sklearn model to ONNX.

---

## MQL5 Expert Advisor

Copy the contents of `mql5/` into your MetaTrader 5 data directory:

```
<MT5 Data>\MQL5\Experts\AIAlgoTrader.mq5
<MT5 Data>\MQL5\Include\AIStrategy.mqh
<MT5 Data>\MQL5\Include\OHLCVData.mqh
```

The Python bridge writes `ai_signal.txt` (1 / -1 / 0) to the MT5 Common Files
folder; the EA reads it every `AI_SIGNAL_POLL_SEC` seconds.

---

## cTrader cBot

Open `ctrader/AIAlgoBot.cs` in cTrader Automate IDE and compile it.  
The cBot reads the same `ai_signal.txt` written by the Python bridge and falls
back to an EMA crossover after `AITimeoutSeconds` (default 600 s) of silence.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## License

MIT