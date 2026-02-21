// ==========================================================================
// AIAlgoBot.cs — cTrader AI-Powered Algorithmic Trading cBot (ONE PAGE)
// ==========================================================================
// Architecture:
//   • Reads AI signal from a shared file written by the Python bridge
//   • Falls back to EMA crossover strategy when AI is unavailable
//   • Runs headless (no UI) — suitable for server / VPS deployment
//   • 10-minute AI signal timeout with automatic fallback
// ==========================================================================

using System;
using System.IO;
using System.Text;
using System.Threading;
using cAlgo.API;
using cAlgo.API.Indicators;

namespace cAlgo.Robots
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.FileSystem)]
    public class AIAlgoBot : Robot
    {
        // ── Input parameters ───────────────────────────────────────────────
        [Parameter("AI Signal File", DefaultValue = "ai_signal.txt")]
        public string SignalFilePath { get; set; }

        [Parameter("Fast EMA Period", DefaultValue = 10, MinValue = 2)]
        public int FastPeriod { get; set; }

        [Parameter("Slow EMA Period", DefaultValue = 30, MinValue = 5)]
        public int SlowPeriod { get; set; }

        [Parameter("Lot Size (Units)", DefaultValue = 1000, MinValue = 1)]
        public double TradeVolume { get; set; }

        [Parameter("Stop Loss (Pips)", DefaultValue = 50, MinValue = 1)]
        public double StopLossPips { get; set; }

        [Parameter("Take Profit (Pips)", DefaultValue = 100, MinValue = 1)]
        public double TakeProfitPips { get; set; }

        [Parameter("Use AI Signals", DefaultValue = true)]
        public bool UseAI { get; set; }

        [Parameter("AI Signal Poll (seconds)", DefaultValue = 5, MinValue = 1)]
        public int SignalPollSeconds { get; set; }

        [Parameter("AI Timeout (seconds)", DefaultValue = 600, MinValue = 10)]
        public int AITimeoutSeconds { get; set; }

        // ── Internal state ─────────────────────────────────────────────────
        private ExponentialMovingAverage _fastEma;
        private ExponentialMovingAverage _slowEma;
        private int _lastAISignal = 0;            // -1=sell, 0=hold, 1=buy
        private DateTime _lastSignalReadTime = DateTime.MinValue;
        private DateTime _lastAISuccessTime  = DateTime.MinValue;
        private bool _aiAvailable = false;

        // ── Lifecycle ──────────────────────────────────────────────────────
        protected override void OnStart()
        {
            _fastEma = Indicators.ExponentialMovingAverage(Bars.ClosePrices, FastPeriod);
            _slowEma = Indicators.ExponentialMovingAverage(Bars.ClosePrices, SlowPeriod);
            Print($"AIAlgoBot started on {SymbolName}. UseAI={UseAI}, Timeout={AITimeoutSeconds}s");
        }

        protected override void OnBar()
        {
            RefreshAISignal();

            int signal = ResolveSignal();
            if (signal == 0) return;

            ManagePositions(signal);
            TryOpenPosition(signal);
        }

        protected override void OnStop()
        {
            Print("AIAlgoBot stopped.");
        }

        // ── Signal resolution ──────────────────────────────────────────────
        private void RefreshAISignal()
        {
            if (!UseAI) return;
            if ((DateTime.UtcNow - _lastSignalReadTime).TotalSeconds < SignalPollSeconds) return;

            _lastSignalReadTime = DateTime.UtcNow;
            try
            {
                string raw = File.ReadAllText(SignalFilePath, Encoding.UTF8).Trim();
                if (int.TryParse(raw, out int sig) && (sig == 1 || sig == -1 || sig == 0))
                {
                    _lastAISignal     = sig;
                    _lastAISuccessTime = DateTime.UtcNow;
                    _aiAvailable       = true;
                }
            }
            catch
            {
                // File missing or unreadable — let timeout logic handle it
            }

            // Mark AI unavailable after timeout
            if (_aiAvailable &&
                (DateTime.UtcNow - _lastAISuccessTime).TotalSeconds > AITimeoutSeconds)
            {
                Print($"AI signal timeout after {AITimeoutSeconds}s — falling back to EMA crossover.");
                _aiAvailable  = false;
                _lastAISignal = 0;
            }
        }

        private int ResolveSignal()
        {
            if (UseAI && _aiAvailable && _lastAISignal != 0)
                return _lastAISignal;

            // EMA crossover fallback
            int i = Bars.Count - 2;          // last complete bar
            if (i < 1) return 0;

            double fastCurr = _fastEma.Result[i];
            double fastPrev = _fastEma.Result[i - 1];
            double slowCurr = _slowEma.Result[i];
            double slowPrev = _slowEma.Result[i - 1];

            if (fastPrev <= slowPrev && fastCurr > slowCurr) return  1;  // bullish cross
            if (fastPrev >= slowPrev && fastCurr < slowCurr) return -1;  // bearish cross
            return 0;
        }

        // ── Trade management ───────────────────────────────────────────────
        private void ManagePositions(int newSignal)
        {
            foreach (var pos in Positions.FindAll("AIAlgoBot", SymbolName))
            {
                bool shouldClose =
                    (pos.TradeType == TradeType.Buy  && newSignal == -1) ||
                    (pos.TradeType == TradeType.Sell && newSignal ==  1);
                if (shouldClose)
                    ClosePosition(pos);
            }
        }

        private void TryOpenPosition(int signal)
        {
            if (Positions.FindAll("AIAlgoBot", SymbolName).Length > 0) return;

            string label    = "AIAlgoBot";
            string comment  = _aiAvailable ? "AI signal" : "EMA fallback";
            TradeType type  = signal == 1 ? TradeType.Buy : TradeType.Sell;
            double sl       = StopLossPips  * Symbol.PipSize;
            double tp       = TakeProfitPips * Symbol.PipSize;

            ExecuteMarketOrder(type, SymbolName, TradeVolume, label, StopLossPips, TakeProfitPips, comment);
        }
    }
}
