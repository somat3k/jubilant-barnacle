//+------------------------------------------------------------------+
//|                                          AIAlgoTrader.mq5         |
//|          AI-Powered Algorithmic Trading Expert Advisor            |
//|                                                                   |
//|  Architecture:                                                    |
//|    • Python bridge writes AI signal to "ai_signal.txt"           |
//|    • EA reads the file every AI_SIGNAL_POLL_SEC seconds           |
//|    • Falls back to SMA crossover when AI signal is unavailable    |
//|    • Exports live OHLCV to CSV for Python offline backtesting     |
//+------------------------------------------------------------------+
#property copyright "jubilant-barnacle"
#property version   "1.00"
#property strict

#include <Properties\settings.mqh>
#include <Include\AIStrategy.mqh>
#include <Include\OHLCVData.mqh>

//---- Input parameters
input int    InpFastPeriod    = AI_FAST_MA_PERIOD;    // Fast MA period
input int    InpSlowPeriod    = AI_SLOW_MA_PERIOD;    // Slow MA period
input double InpLotSize       = AI_LOT_SIZE;          // Lot size
input int    InpStopLossPips  = AI_STOP_LOSS_PIPS;    // Stop loss (pips)
input int    InpTakeProfitPips= AI_TAKE_PROFIT_PIPS;  // Take profit (pips)
input double InpRiskPercent   = AI_RISK_PERCENT;      // Risk % per trade
input bool   InpUseAI         = true;                 // Enable AI signals
input int    InpSignalPollSec = AI_SIGNAL_POLL_SEC;   // Signal poll interval (s)
input bool   InpExportOHLCV   = true;                 // Export OHLCV on init

//---- Internal state
datetime g_lastSignalTime = 0;
int      g_lastAISignal   = SIGNAL_HOLD;
ulong    g_openTicket     = 0;

//+------------------------------------------------------------------+
int OnInit()
  {
   Print("AIAlgoTrader EA initialised on ", _Symbol);

   if(InpExportOHLCV)
     {
      if(ExportOHLCVToCSV(_Symbol, _Period, OHLCV_EXPORT_BARS, OHLCV_EXPORT_FILE))
         Print("OHLCV exported to ", OHLCV_EXPORT_FILE);
      else
         Print("OHLCV export failed.");
     }
   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   Print("AIAlgoTrader EA removed. Reason code: ", reason);
  }

//+------------------------------------------------------------------+
void OnTick()
  {
   // Refresh AI signal on poll interval
   if(TimeCurrent() - g_lastSignalTime >= InpSignalPollSec)
     {
      if(InpUseAI)
         g_lastAISignal = ReadAISignal();
      g_lastSignalTime = TimeCurrent();
     }

   // Determine effective signal
   int signal;
   if(InpUseAI && g_lastAISignal != SIGNAL_HOLD)
     {
      signal = g_lastAISignal;
     }
   else
     {
      // Fallback to SMA crossover
      signal = SMAFallbackSignal(_Symbol, _Period, InpFastPeriod, InpSlowPeriod);
     }

   double lotSize = ComputeLotSize(InpRiskPercent, InpStopLossPips);
   double point   = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double ask     = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid     = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double slPips  = InpStopLossPips  * point * 10;
   double tpPips  = InpTakeProfitPips * point * 10;

   // Manage existing position
   if(PositionsTotal() > 0)
     {
      for(int i = PositionsTotal() - 1; i >= 0; i--)
        {
         ulong ticket = PositionGetTicket(i);
         if(PositionGetInteger(POSITION_MAGIC) != EA_MAGIC_NUMBER) continue;

         long posType = PositionGetInteger(POSITION_TYPE);
         if((posType == POSITION_TYPE_BUY  && signal == SIGNAL_SELL) ||
            (posType == POSITION_TYPE_SELL && signal == SIGNAL_BUY))
           {
            MqlTradeRequest req = {};
            MqlTradeResult  res = {};
            req.action   = TRADE_ACTION_DEAL;
            req.position = ticket;
            req.symbol   = _Symbol;
            req.volume   = PositionGetDouble(POSITION_VOLUME);
            req.type     = (posType == POSITION_TYPE_BUY)
                           ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
            req.price    = (req.type == ORDER_TYPE_SELL) ? bid : ask;
            req.magic    = EA_MAGIC_NUMBER;
            req.comment  = "AI close";
            if(!OrderSend(req, res))
               Print("Close order failed: ", res.retcode);
           }
        }
     }

   // Open new position
   if(PositionsTotal() == 0 && signal != SIGNAL_HOLD)
     {
      MqlTradeRequest req = {};
      MqlTradeResult  res = {};
      req.action  = TRADE_ACTION_DEAL;
      req.symbol  = _Symbol;
      req.volume  = lotSize;
      req.magic   = EA_MAGIC_NUMBER;
      req.comment = InpUseAI ? "AI signal" : "SMA fallback";

      if(signal == SIGNAL_BUY)
        {
         req.type  = ORDER_TYPE_BUY;
         req.price = ask;
         req.sl    = ask - slPips;
         req.tp    = ask + tpPips;
        }
      else
        {
         req.type  = ORDER_TYPE_SELL;
         req.price = bid;
         req.sl    = bid + slPips;
         req.tp    = bid - tpPips;
        }

      req.type_filling = ORDER_FILLING_IOC;
      if(!OrderSend(req, res))
         Print("Open order failed: ", res.retcode);
      else
         Print("Order opened #", res.order, " signal=", signal);
     }

   // Write our own signal back for the Python bridge
   WriteSignal(signal);
  }
//+------------------------------------------------------------------+
