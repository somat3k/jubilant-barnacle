//+------------------------------------------------------------------+
//|                                               AIStrategy.mqh     |
//|          AI Algorithmic Trading Strategy Include                  |
//|  Provides signal constants, risk management helpers, and the     |
//|  interface contract between the Expert Advisor and AI signals     |
//+------------------------------------------------------------------+
#ifndef AI_STRATEGY_MQH
#define AI_STRATEGY_MQH

// Signal constants
#define SIGNAL_BUY   1
#define SIGNAL_SELL -1
#define SIGNAL_HOLD  0

// Default parameters (can be overridden in Properties/settings.mqh)
#ifndef AI_FAST_MA_PERIOD
   #define AI_FAST_MA_PERIOD   10
#endif
#ifndef AI_SLOW_MA_PERIOD
   #define AI_SLOW_MA_PERIOD   30
#endif
#ifndef AI_STOP_LOSS_PIPS
   #define AI_STOP_LOSS_PIPS   50
#endif
#ifndef AI_TAKE_PROFIT_PIPS
   #define AI_TAKE_PROFIT_PIPS 100
#endif
#ifndef AI_LOT_SIZE
   #define AI_LOT_SIZE         0.01
#endif

//+------------------------------------------------------------------+
//| Read the latest AI signal written by the Python bridge.           |
//| The bridge writes a file named "ai_signal.txt" with one of:       |
//|   "1"  — buy                                                      |
//|  "-1"  — sell                                                     |
//|   "0"  — hold                                                     |
//+------------------------------------------------------------------+
int ReadAISignal()
  {
   int    handle = FileOpen("ai_signal.txt", FILE_READ|FILE_TXT|FILE_COMMON);
   if(handle == INVALID_HANDLE)
      return SIGNAL_HOLD;

   string content = FileReadString(handle);
   FileClose(handle);

   int sig = (int)StringToInteger(content);
   if(sig == SIGNAL_BUY || sig == SIGNAL_SELL)
      return sig;
   return SIGNAL_HOLD;
  }

//+------------------------------------------------------------------+
//| Write a signal file consumed by the Python bridge (backtesting).  |
//+------------------------------------------------------------------+
void WriteSignal(int signal)
  {
   int handle = FileOpen("mql5_signal_out.txt", FILE_WRITE|FILE_TXT|FILE_COMMON);
   if(handle == INVALID_HANDLE)
      return;
   FileWrite(handle, IntegerToString(signal));
   FileClose(handle);
  }

//+------------------------------------------------------------------+
//| Simple SMA crossover fallback signal (used when AI is unavailable)|
//+------------------------------------------------------------------+
int SMAFallbackSignal(string symbol, ENUM_TIMEFRAMES tf,
                      int fastPeriod, int slowPeriod)
  {
   // In MQL5, iMA returns a handle; use CopyBuffer to get the actual values.
   int fastHandle = iMA(symbol, tf, fastPeriod, 0, MODE_SMA, PRICE_CLOSE);
   int slowHandle = iMA(symbol, tf, slowPeriod, 0, MODE_SMA, PRICE_CLOSE);
   if(fastHandle == INVALID_HANDLE || slowHandle == INVALID_HANDLE)
      return SIGNAL_HOLD;

   double fastBuf[2], slowBuf[2];
   if(CopyBuffer(fastHandle, 0, 0, 2, fastBuf) < 2) return SIGNAL_HOLD;
   if(CopyBuffer(slowHandle, 0, 0, 2, slowBuf) < 2) return SIGNAL_HOLD;

   // fastBuf[0] = most recent bar, fastBuf[1] = one bar earlier
   double fast0 = fastBuf[0], fast1 = fastBuf[1];
   double slow0 = slowBuf[0], slow1 = slowBuf[1];

   if(fast1 <= slow1 && fast0 > slow0) return SIGNAL_BUY;
   if(fast1 >= slow1 && fast0 < slow0) return SIGNAL_SELL;
   return SIGNAL_HOLD;
  }

//+------------------------------------------------------------------+
//| Compute position size based on risk percentage                    |
//+------------------------------------------------------------------+
double ComputeLotSize(double riskPercent, double stopLossPips)
  {
   double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   double pipValue       = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   if(pipValue <= 0 || stopLossPips <= 0)
      return AI_LOT_SIZE;
   double lots = (accountBalance * riskPercent / 100.0) / (stopLossPips * pipValue);
   lots = MathMax(SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN), lots);
   lots = MathMin(SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX), lots);
   return NormalizeDouble(lots, 2);
  }

#endif // AI_STRATEGY_MQH
