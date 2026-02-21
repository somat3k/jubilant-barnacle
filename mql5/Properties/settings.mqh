//+------------------------------------------------------------------+
//|                                            settings.mqh           |
//|          AIAlgoTrader Expert Advisor — Configurable Properties    |
//|  Adjust these values to tune the strategy without recompiling.    |
//+------------------------------------------------------------------+
#ifndef SETTINGS_MQH
#define SETTINGS_MQH

// ---- Fast and slow moving average periods --------------------------
#define AI_FAST_MA_PERIOD   10    // Fast SMA period (bars)
#define AI_SLOW_MA_PERIOD   30    // Slow SMA period (bars)

// ---- Risk management -----------------------------------------------
#define AI_STOP_LOSS_PIPS   50    // Stop-loss distance in pips
#define AI_TAKE_PROFIT_PIPS 100   // Take-profit distance in pips
#define AI_LOT_SIZE         0.01  // Default micro lot size
#define AI_RISK_PERCENT     1.0   // % of balance risked per trade

// ---- AI signal file poll interval (seconds) -----------------------
#define AI_SIGNAL_POLL_SEC  5     // How often to re-read ai_signal.txt

// ---- OHLCV export for Python backtesting --------------------------
#define OHLCV_EXPORT_BARS   500   // Number of bars to export
#define OHLCV_EXPORT_FILE   "ohlcv_export.csv"

// ---- Magic number (unique EA identifier) --------------------------
#define EA_MAGIC_NUMBER     20240101

#endif // SETTINGS_MQH
