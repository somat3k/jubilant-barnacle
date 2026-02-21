//+------------------------------------------------------------------+
//|                                               OHLCVData.mqh      |
//|          OHLCV Data Access Helpers                                |
//|  Wrappers for convenient access to bar data used by the           |
//|  AI strategy Expert Advisors.                                     |
//+------------------------------------------------------------------+
#ifndef OHLCV_DATA_MQH
#define OHLCV_DATA_MQH

//+------------------------------------------------------------------+
//| Return the last N bars as arrays for use in indicator calc.       |
//+------------------------------------------------------------------+
bool GetOHLCV(string symbol, ENUM_TIMEFRAMES tf, int bars,
              double &opens[], double &highs[], double &lows[],
              double &closes[], long &volumes[])
  {
   if(CopyOpen  (symbol, tf, 0, bars, opens)   <= 0) return false;
   if(CopyHigh  (symbol, tf, 0, bars, highs)   <= 0) return false;
   if(CopyLow   (symbol, tf, 0, bars, lows)    <= 0) return false;
   if(CopyClose (symbol, tf, 0, bars, closes)  <= 0) return false;
   if(CopyTickVolume(symbol, tf, 0, bars, volumes) <= 0)
     {
      ArrayResize(volumes, bars);
      ArrayInitialize(volumes, 0);
     }
   return true;
  }

//+------------------------------------------------------------------+
//| Export recent bars to a CSV file (for Python offline backtesting).|
//+------------------------------------------------------------------+
bool ExportOHLCVToCSV(string symbol, ENUM_TIMEFRAMES tf, int bars,
                      string filename)
  {
   double opens[], highs[], lows[], closes[];
   long   volumes[];
   datetime times[];

   if(!GetOHLCV(symbol, tf, bars, opens, highs, lows, closes, volumes))
      return false;
   if(CopyTime(symbol, tf, 0, bars, times) <= 0)
      return false;

   int handle = FileOpen(filename, FILE_WRITE|FILE_CSV|FILE_COMMON, ',');
   if(handle == INVALID_HANDLE)
      return false;

   FileWrite(handle, "Date", "Open", "High", "Low", "Close", "Volume");
   for(int i = bars - 1; i >= 0; i--)
     {
      string dt = TimeToString(times[i], TIME_DATE);
      FileWrite(handle, dt,
                DoubleToString(opens[i],  _Digits),
                DoubleToString(highs[i],  _Digits),
                DoubleToString(lows[i],   _Digits),
                DoubleToString(closes[i], _Digits),
                IntegerToString(volumes[i]));
     }
   FileClose(handle);
   return true;
  }

#endif // OHLCV_DATA_MQH
