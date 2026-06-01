//+------------------------------------------------------------------+
//|                                                   AlgoTradeEA.mq5 |
//|                                    AlgoTrade Engine - Trading Bot |
//|                                             https://algotrade.com |
//+------------------------------------------------------------------+
#property copyright   "AlgoTrade Engine"
#property link        "https://algotrade.com"
#property version     "1.00"
#property description "AlgoTrade Engine - Multi-Strategy Expert Advisor"
#property description "Combines Trend Following, Mean Reversion, and Momentum"
#property description "strategies with configurable risk management."
//+------------------------------------------------------------------+
//| Includes                                                         |
//+------------------------------------------------------------------+
#include <Trade/Trade.mqh>
#include <Trade/PositionInfo.mqh>
#include <Trade/AccountInfo.mqh>
#include <Trade/SymbolInfo.mqh>
//+------------------------------------------------------------------+
//| Enums                                                            |
//+------------------------------------------------------------------+
enum ENUM_EXECUTION_MODE
{
   EXEC_MARKET_ONLY,      // Market orders only
   EXEC_PENDING_ONLY,     // Pending orders only
   EXEC_BOTH              // Both (market on signal, pending on level)
};

enum ENUM_POSITION_SIZING
{
   SIZING_RISK_PERCENT,   // Risk % of account per trade
   SIZING_FIXED_VOLUME,   // Fixed lot size
   SIZING_FIXED_FRACTION, // Fixed fraction of equity
   SIZING_KELLY           // Kelly criterion
};
//+------------------------------------------------------------------+
//| Input - General Settings                                         |
//+------------------------------------------------------------------+
input group "=== General Settings ==="
input ulong     InpMagicNumber      = 123456;           // Base Magic Number
input string    InpCommentPrefix    = "AlgoTrade";      // Trade Comment Prefix
input bool      InpTradeLog         = true;             // Enable CSV Trade Log
input bool      InpDrawOnChart      = true;             // Draw signals on chart
input ENUM_EXECUTION_MODE InpExecMode=EXEC_MARKET_ONLY; // Order execution mode
input int       InpSlippage         = 30;               // Max slippage (points)
//+------------------------------------------------------------------+
//| Input - Trend Following Module                                   |
//+------------------------------------------------------------------+
input group "=== Trend Following Module ==="
input bool      InpEnableTF         = true;             // Enable Trend Following
input int       InpTFMagicOffset    = 1;                // TF Magic offset
input int       InpTFEMAfast        = 12;               // TF Fast EMA
input int       InpTFEMAslow        = 26;               // TF Slow EMA
input int       InpTFADXperiod      = 14;               // TF ADX period
input double    InpTFADXthreshold   = 25.0;             // TF ADX threshold
input int       InpTFATRperiod      = 14;               // TF ATR period
input double    InpTFATRmult        = 2.0;              // TF ATR trailing mult
//+------------------------------------------------------------------+
//| Input - Mean Reversion Module                                    |
//+------------------------------------------------------------------+
input group "=== Mean Reversion Module ==="
input bool      InpEnableMR         = true;             // Enable Mean Reversion
input int       InpMRMagicOffset    = 2;                // MR Magic offset
input int       InpMRBBperiod       = 20;               // MR Bollinger period
input double    InpMRBBstddev       = 2.0;              // MR Bollinger std.dev
input int       InpMRRSIperiod      = 14;               // MR RSI period
input double    InpMRRSIoverbought  = 70.0;             // MR RSI overbought
input double    InpMRRSIoversold    = 30.0;             // MR RSI oversold
//+------------------------------------------------------------------+
//| Input - Momentum Module                                          |
//+------------------------------------------------------------------+
input group "=== Momentum Module ==="
input bool      InpEnableMO         = true;             // Enable Momentum
input int       InpMOMagicOffset    = 3;                // MO Magic offset
input int       InpMOROCperiod      = 12;               // MO ROC period
input double    InpMOROCthreshold   = 5.0;              // MO ROC threshold
//+------------------------------------------------------------------+
//| Input - Risk Management                                          |
//+------------------------------------------------------------------+
input group "=== Risk Management ==="
input ENUM_POSITION_SIZING InpSizingMethod=SIZING_RISK_PERCENT; // Position sizing
input double    InpRiskPercent      = 1.0;              // Risk % per trade
input double    InpFixedVolume      = 0.01;             // Fixed lot size (0=auto)
input double    InpFixedFraction    = 0.02;             // Fixed fraction of equity
input bool      InpUseKelly         = false;            // Use Kelly criterion
input double    InpKellyF           = 0.25;             // Kelly fraction cap
input double    InpKellyWinRate     = 0.55;             // Kelly win rate
input double    InpKellyAvgWin      = 1.5;              // Kelly avg win / risk
input double    InpKellyAvgLoss     = 1.0;              // Kelly avg loss / risk
input double    InpMaxDrawdown      = 20.0;             // Max drawdown (%)
input double    InpDailyLossLimit   = 5.0;              // Daily loss limit (%)
input double    InpTrailingStopPts  = 0.0;              // Fixed trailing stop (pts)
input bool      InpUseATRtrailing   = true;             // Use ATR trailing on TF
input int       InpPendingDistPts   = 20;               // Pending order distance (pts)
input double    InpMaxSpread        = 0.0;              // Max spread (pts, 0=any)
//+------------------------------------------------------------------+
//| Global variables                                                  |
//+------------------------------------------------------------------+
CTrade          m_trade;
CPositionInfo   m_position;
CAccountInfo    m_account;
CSymbolInfo     m_symbol;

// Indicator handles
int             hEMAfast   = INVALID_HANDLE;
int             hEMAslow   = INVALID_HANDLE;
int             hADX       = INVALID_HANDLE;
int             hATR       = INVALID_HANDLE;
int             hBB        = INVALID_HANDLE;
int             hRSI       = INVALID_HANDLE;
int             hROC       = INVALID_HANDLE;
int             hOBV       = INVALID_HANDLE;

// Data buffers (all as-series for index-0 = current)
double          bufEMAfast[];
double          bufEMAslow[];
double          bufADXmain[];
double          bufADXdiP[];
double          bufADXdiN[];
double          bufATR[];
double          bufBBupper[];
double          bufBBmiddle[];
double          bufBBlower[];
double          bufRSI[];
double          bufROC[];
double          bufOBV[];

// Previous tick values for crossover / divergence
double          g_prevEMAfast   = 0.0;
double          g_prevEMAslow   = 0.0;
double          g_prevROC       = 0.0;
double          g_prevOBV       = 0.0;
double          g_prevClose     = 0.0;

// Risk tracking
double          g_peakEquity      = 0.0;
double          g_dayStartEquity  = 0.0;
datetime        g_lastDayCheck    = 0;
datetime        g_lastBarTime     = 0;
bool            g_dailyLossHit    = false;
bool            g_drawdownHit     = false;

// File handle for log
int             g_logFile         = INVALID_HANDLE;
bool            g_logEnabled      = false;

// Indicator ready
bool            g_indicatorsOK    = false;
int             g_barsToCalc      = 100;  // Min bars needed

// Chart drawing label
string          g_labelName       = "AlgoTradeEA_Info";
string          g_labelSignal     = "AlgoTradeEA_Signal";
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit(void)
{
   //--- Validate inputs
   if(InpRiskPercent <= 0.0 && InpFixedVolume <= 0.0 && InpFixedFraction <= 0.0 && !InpUseKelly)
   {
      Print("ERROR: No position sizing method configured. Set RiskPercent, FixedVolume, FixedFraction, or enable Kelly.");
      return INIT_PARAMETERS_INCORRECT;
   }
   if(InpMaxDrawdown <= 0.0 || InpMaxDrawdown > 100.0)
   {
      Print("ERROR: MaxDrawdown must be between 0.1 and 100.0");
      return INIT_PARAMETERS_INCORRECT;
   }
   if(InpDailyLossLimit <= 0.0 || InpDailyLossLimit > 100.0)
   {
      Print("ERROR: DailyLossLimit must be between 0.1 and 100.0");
      return INIT_PARAMETERS_INCORRECT;
   }
   if(InpEnableTF && (InpTFEMAfast <= 0 || InpTFEMAslow <= 0 || InpTFEMAslow <= InpTFEMAfast))
   {
      Print("ERROR: TF EMA periods invalid. Slow must be > Fast.");
      return INIT_PARAMETERS_INCORRECT;
   }
   if(InpEnableMR && (InpMRBBperiod <= 1 || InpMRBBstddev <= 0.0))
   {
      Print("ERROR: MR Bollinger Bands parameters invalid.");
      return INIT_PARAMETERS_INCORRECT;
   }
   if(InpEnableMO && (InpMOROCperiod <= 1))
   {
      Print("ERROR: MO ROC period must be > 1.");
      return INIT_PARAMETERS_INCORRECT;
   }

   //--- Symbol info
   m_symbol.Name(_Symbol);
   m_symbol.Refresh();

   //--- Trade object
   m_trade.SetExpertMagicNumber(InpMagicNumber);
   m_trade.SetDeviationInPoints(InpSlippage);
   m_trade.SetAsyncMode(false);

   //--- Create indicator handles
   if(InpEnableTF)
   {
      hEMAfast = iMA(_Symbol, PERIOD_CURRENT, InpTFEMAfast, 0, MODE_EMA, PRICE_CLOSE);
      hEMAslow = iMA(_Symbol, PERIOD_CURRENT, InpTFEMAslow, 0, MODE_EMA, PRICE_CLOSE);
      hADX     = iADX(_Symbol, PERIOD_CURRENT, InpTFADXperiod);
      hATR     = iATR(_Symbol, PERIOD_CURRENT, InpTFATRperiod);
   }
   if(InpEnableMR)
   {
      hBB  = iBands(_Symbol, PERIOD_CURRENT, InpMRBBperiod, 0, InpMRBBstddev, PRICE_CLOSE);
      hRSI = iRSI(_Symbol, PERIOD_CURRENT, InpMRRSIperiod, PRICE_CLOSE);
   }
   if(InpEnableMO)
   {
      hROC = iMomentum(_Symbol, PERIOD_CURRENT, InpMOROCperiod, PRICE_CLOSE);
      hOBV = iOBV(_Symbol, PERIOD_CURRENT, VOLUME_TICK);
   }

   //--- Verify handles
   int failures = 0;
   if(InpEnableTF) { failures += (hEMAfast == INVALID_HANDLE); failures += (hEMAslow == INVALID_HANDLE); failures += (hADX == INVALID_HANDLE); failures += (hATR == INVALID_HANDLE); }
   if(InpEnableMR) { failures += (hBB == INVALID_HANDLE); failures += (hRSI == INVALID_HANDLE); }
   if(InpEnableMO) { failures += (hROC == INVALID_HANDLE); failures += (hOBV == INVALID_HANDLE); }
   if(failures > 0)
   {
      Print("ERROR: ", failures, " indicator handle(s) failed to create. Error: ", GetLastError());
      return INIT_FAILED;
   }

   //--- Set buffer as-series for convenient [0]=current
   ArraySetAsSeries(bufEMAfast,   true);
   ArraySetAsSeries(bufEMAslow,   true);
   ArraySetAsSeries(bufADXmain,   true);
   ArraySetAsSeries(bufADXdiP,    true);
   ArraySetAsSeries(bufADXdiN,    true);
   ArraySetAsSeries(bufATR,       true);
   ArraySetAsSeries(bufBBupper,   true);
   ArraySetAsSeries(bufBBmiddle,  true);
   ArraySetAsSeries(bufBBlower,   true);
   ArraySetAsSeries(bufRSI,       true);
   ArraySetAsSeries(bufROC,       true);
   ArraySetAsSeries(bufOBV,       true);

   //--- Calculate minimum bars needed
   int maxPeriod = 0;
   if(InpEnableTF) { maxPeriod = MathMax(maxPeriod, MathMax(InpTFEMAslow, InpTFADXperiod)); maxPeriod = MathMax(maxPeriod, InpTFATRperiod); }
   if(InpEnableMR) { maxPeriod = MathMax(maxPeriod, MathMax(InpMRBBperiod, InpMRRSIperiod)); }
   if(InpEnableMO) { maxPeriod = MathMax(maxPeriod, InpMOROCperiod); }
   g_barsToCalc = maxPeriod + 10; // generous buffer

   //--- Risk tracking
   g_peakEquity     = m_account.Balance();
   g_dayStartEquity = m_account.Balance();
   g_lastDayCheck   = TimeCurrent();
   g_dailyLossHit   = false;
   g_drawdownHit    = false;

   //--- Initialise previous EMA values from indicator data
   if(InpEnableTF)
   {
      CopyBuffer(hEMAfast, 0, 0, 3, bufEMAfast);
      CopyBuffer(hEMAslow, 0, 0, 3, bufEMAslow);
      if(BarsCalculated(hEMAfast) > 1)
      {
         g_prevEMAfast = bufEMAfast[1];
         g_prevEMAslow = bufEMAslow[1];
      }
   }
   if(InpEnableMO)
   {
      // Match Python: all stored values start at 0 for first-call guard
      g_prevROC   = 0.0;
      g_prevOBV   = 0.0;
      g_prevClose = 0.0;
   }

   //--- Chart label
   if(InpDrawOnChart)
   {
      if(ObjectFind(0, g_labelName) < 0)
         ObjectCreate(0, g_labelName, OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, g_labelName, OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetInteger(0, g_labelName, OBJPROP_XDISTANCE, 10);
      ObjectSetInteger(0, g_labelName, OBJPROP_YDISTANCE, 10);
      ObjectSetInteger(0, g_labelName, OBJPROP_COLOR, clrWhite);
      ObjectSetInteger(0, g_labelName, OBJPROP_FONTSIZE, 9);
      ObjectSetString(0, g_labelName, OBJPROP_TEXT, "AlgoTrade EA v1.00 | Loading...");
      ChartRedraw(0);
   }

   //--- Initial bar time for NewBar check
   datetime times[];
   if(CopyTime(_Symbol, PERIOD_CURRENT, 0, 1, times) > 0)
      g_lastBarTime = times[0];

   Print("AlgoTrade EA initialized on ", _Symbol, " | TF modules: TF=",
         InpEnableTF, " MR=", InpEnableMR, " MO=", InpEnableMO,
         " | Balance: ", m_account.Balance());
   return INIT_SUCCEEDED;
}
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   //--- Release indicator handles
   if(hEMAfast != INVALID_HANDLE) { IndicatorRelease(hEMAfast); hEMAfast = INVALID_HANDLE; }
   if(hEMAslow != INVALID_HANDLE) { IndicatorRelease(hEMAslow); hEMAslow = INVALID_HANDLE; }
   if(hADX     != INVALID_HANDLE) { IndicatorRelease(hADX);     hADX     = INVALID_HANDLE; }
   if(hATR     != INVALID_HANDLE) { IndicatorRelease(hATR);     hATR     = INVALID_HANDLE; }
   if(hBB      != INVALID_HANDLE) { IndicatorRelease(hBB);      hBB      = INVALID_HANDLE; }
   if(hRSI     != INVALID_HANDLE) { IndicatorRelease(hRSI);     hRSI     = INVALID_HANDLE; }
   if(hROC     != INVALID_HANDLE) { IndicatorRelease(hROC);     hROC     = INVALID_HANDLE; }
   if(hOBV     != INVALID_HANDLE) { IndicatorRelease(hOBV);     hOBV     = INVALID_HANDLE; }

   //--- Trade log file uses per-call handles; nothing to close here.

   //--- Remove chart objects
   if(InpDrawOnChart)
   {
      ObjectDelete(0, g_labelName);
      ObjectsDeleteAll(0, "AlgoTradeEA_");
      ChartRedraw(0);
   }

   Print("AlgoTrade EA deinitialized (reason=", reason, ")");
}
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick(void)
{
   //--- Only process on new bar (avoid re-processing same bar)
   if(!IsNewBar())
      return;

   //--- Check if trading is allowed by time/risk limits
   if(!IsTradingAllowed())
   {
      UpdateChartLabel();
      return;
   }

   //--- Refresh symbol info
   m_symbol.Refresh();
   m_trade.SetExpertMagicNumber(InpMagicNumber);

   //--- Refresh indicator data
   if(!RefreshIndicators())
   {
      if(g_indicatorsOK)
         Print("WARNING: Indicator refresh failed, skipping tick");
      return;
   }
   g_indicatorsOK = true;

   //--- Check spread
   if(InpMaxSpread > 0.0)
   {
      double spread = m_symbol.Spread();
      if(spread > InpMaxSpread)
      {
         Print("Spread too high: ", spread, " pts (max: ", InpMaxSpread, ")");
         UpdateChartLabel();
         return;
      }
   }

   //--- Update risk tracking (equity, drawdown)
   UpdateRiskTracking();

   //--- Manage trailing stops for existing positions (non-ATR)
   if(InpTrailingStopPts > 0.0)
      ManageFixedTrailingStop();

   //--- Process each enabled strategy module
   if(InpEnableTF) ProcessTrendFollowing();
   if(InpEnableMR) ProcessMeanReversion();
   if(InpEnableMO) ProcessMomentum();

   //--- Update chart label
   UpdateChartLabel();
}
//+------------------------------------------------------------------+
//| Check if a new bar has formed                                    |
//+------------------------------------------------------------------+
bool IsNewBar(void)
{
   datetime times[];
   if(CopyTime(_Symbol, PERIOD_CURRENT, 0, 1, times) < 1)
      return false;
   if(times[0] != g_lastBarTime)
   {
      g_lastBarTime = times[0];
      return true;
   }
   return false;
}
//+------------------------------------------------------------------+
//| Check if trading is allowed given risk limits                    |
//+------------------------------------------------------------------+
bool IsTradingAllowed(void)
{
   //--- Drawdown stop
   if(g_drawdownHit)
   {
      if(InpDrawOnChart)
         Print("MAX DRAWDOWN REACHED: ", InpMaxDrawdown, "%. Trading halted.");
      return false;
   }

   //--- Daily loss limit
   if(g_dailyLossHit)
   {
      if(InpDrawOnChart)
         Print("DAILY LOSS LIMIT REACHED: ", InpDailyLossLimit, "%. No new trades today.");
      return false;
   }

   return true;
}
//+------------------------------------------------------------------+
//| Refresh all indicator buffers from latest data                   |
//+------------------------------------------------------------------+
bool RefreshIndicators(void)
{
   int needed = g_barsToCalc;

   if(InpEnableTF)
   {
      if(CopyBuffer(hEMAfast, 0, 0, needed, bufEMAfast) < needed) return false;
      if(CopyBuffer(hEMAslow, 0, 0, needed, bufEMAslow) < needed) return false;
      if(CopyBuffer(hADX, 0, 0, needed, bufADXmain) < needed) return false;
      if(CopyBuffer(hADX, 1, 0, needed, bufADXdiP) < needed) return false;
      if(CopyBuffer(hADX, 2, 0, needed, bufADXdiN) < needed) return false;
      if(CopyBuffer(hATR, 0, 0, needed, bufATR) < needed) return false;
   }
   if(InpEnableMR)
   {
      if(CopyBuffer(hBB, 0, 0, needed, bufBBupper) < needed) return false;
      if(CopyBuffer(hBB, 1, 0, needed, bufBBmiddle) < needed) return false;
      if(CopyBuffer(hBB, 2, 0, needed, bufBBlower) < needed) return false;
      if(CopyBuffer(hRSI, 0, 0, needed, bufRSI) < needed) return false;
   }
   if(InpEnableMO)
   {
      if(CopyBuffer(hROC, 0, 0, needed, bufROC) < needed) return false;
      if(CopyBuffer(hOBV, 0, 0, needed, bufOBV) < needed) return false;
   }

   return true;
}
//+------------------------------------------------------------------+
//| Update risk tracking: drawdown, daily loss, equity               |
//+------------------------------------------------------------------+
void UpdateRiskTracking(void)
{
   double equity = m_account.Equity();
   double balance = m_account.Balance();

   //--- Peak equity tracking
   if(equity > g_peakEquity)
      g_peakEquity = equity;

   //--- Drawdown check
   if(g_peakEquity > 0.0)
   {
      double ddPct = (1.0 - equity / g_peakEquity) * 100.0;
      if(ddPct >= InpMaxDrawdown && InpMaxDrawdown > 0.0)
      {
         if(!g_drawdownHit)
         {
            Print("MAX DRAWDOWN EXCEEDED: ", DoubleToString(ddPct, 2), "% >= ", InpMaxDrawdown, "%");
            g_drawdownHit = true;
            CloseAllPositions("MaxDrawdown");
         }
      }
   }

   //--- Daily loss check
   datetime now = TimeCurrent();
   MqlDateTime dtNow, dtStart;
   TimeToStruct(g_lastDayCheck, dtStart);
   TimeToStruct(now, dtNow);
   if(dtNow.day != dtStart.day || dtNow.mon != dtStart.mon || dtNow.year != dtStart.year)
   {
      // New day - reset
      g_dayStartEquity = balance;
      g_lastDayCheck = now;
      g_dailyLossHit = false;
   }
   else
   {
      // Check daily loss
      if(g_dayStartEquity > 0.0)
      {
         double lossPct = (1.0 - equity / g_dayStartEquity) * 100.0;
         if(lossPct >= InpDailyLossLimit && InpDailyLossLimit > 0.0)
         {
            if(!g_dailyLossHit)
            {
               Print("DAILY LOSS LIMIT EXCEEDED: ", DoubleToString(lossPct, 2), "% >= ", InpDailyLossLimit, "%");
               g_dailyLossHit = true;
               CloseAllPositions("DailyLoss");
            }
         }
      }
   }
}
//+------------------------------------------------------------------+
//| Calculate lot size based on selected method                      |
//+------------------------------------------------------------------+
double CalculateLotSize(const double entryPrice, const double stopLossPrice, const string strategyTag)
{
   double lotSize = 0.0;
   double balance = m_account.Balance();
   double tickValue = m_symbol.TickValue();
   double tickSize  = m_symbol.TickSize();
   double lotStep   = m_symbol.LotStep();
   double minLot    = m_symbol.LotMin();
   double maxLot    = m_symbol.LotMax();

   if(lotStep <= 0.0) lotStep = 0.01;

   switch(InpSizingMethod)
   {
      case SIZING_FIXED_VOLUME:
      {
         lotSize = InpFixedVolume;
         break;
      }
      case SIZING_FIXED_FRACTION:
      {
         double contractSz = m_symbol.ContractSize();
         if(contractSz <= 0.0) contractSz = 100000.0;
         // Units = balance * fraction / entryPrice, then convert to lots
         double units = (balance * InpFixedFraction) / entryPrice;
         lotSize = units / contractSz;
         break;
      }
      case SIZING_KELLY:
      {
         double kellyF = InpKellyF;
         if(InpKellyAvgLoss > 0.0 && InpKellyWinRate > 0.0 && InpKellyWinRate < 1.0)
         {
            double b = InpKellyAvgWin / InpKellyAvgLoss;
            if(b > 0.0)
            {
               double p = InpKellyWinRate;
               double q = 1.0 - p;
               kellyF = (p * b - q) / b;
               kellyF = MathMax(0.01, MathMin(kellyF, InpKellyF));
            }
         }
         lotSize = (balance * kellyF) / (entryPrice * m_symbol.ContractSize());
         break;
      }
      case SIZING_RISK_PERCENT:
      default:
      {
         double riskAmount = balance * InpRiskPercent / 100.0;
         double stopDist = MathAbs(entryPrice - stopLossPrice);
         if(stopDist > 0.0 && tickValue > 0.0 && tickSize > 0.0)
         {
            double stopDistTicks = stopDist / tickSize;
            double riskPerLot = stopDistTicks * tickValue;
            if(riskPerLot > 0.0)
               lotSize = riskAmount / riskPerLot;
            else
               lotSize = balance * 0.01 / entryPrice;
         }
         else
         {
            // Fallback: fixed fraction 1%
            lotSize = balance * 0.01 / entryPrice;
         }
         break;
      }
   }

   //--- Normalise to lot step
   if(lotStep > 0.0)
      lotSize = MathFloor(lotSize / lotStep) * lotStep;
   lotSize = MathMax(minLot, MathMin(lotSize, maxLot));
   return lotSize;
}
//+------------------------------------------------------------------+
//| Core order sending wrapper with error handling                   |
//+------------------------------------------------------------------+
bool OrderSendWrapper(const ulong magic,          // Magic number for this order
                      const ENUM_ORDER_TYPE type, // ORDER_TYPE_BUY/SELL/BUY_LIMIT/etc
                      const double volume,
                      const double price,         // 0 = market price
                      const double sl,
                      const double tp,
                      const string comment,
                      const bool isPending)
{
   m_trade.SetExpertMagicNumber(magic);

   bool result = false;
   string typeDesc = "";

   if(!isPending)
   {
      //--- Market order
      if(type == ORDER_TYPE_BUY)
      {
         result = m_trade.Buy(volume, _Symbol, 0, sl, tp, comment);
         typeDesc = "BUY";
      }
      else if(type == ORDER_TYPE_SELL)
      {
         result = m_trade.Sell(volume, _Symbol, 0, sl, tp, comment);
         typeDesc = "SELL";
      }
      else
      {
         Print("ERROR: Invalid order type for market execution: ", EnumToString(type));
         return false;
      }
   }
   else
   {
      //--- Pending order
      if(price <= 0.0)
      {
         Print("ERROR: Pending order requires price > 0");
         return false;
      }

      switch(type)
      {
         case ORDER_TYPE_BUY_LIMIT:
            result = m_trade.BuyLimit(volume, price, _Symbol, sl, tp, ORDER_TIME_GTC, 0, comment);
            typeDesc = "BUY_LIMIT";
            break;
         case ORDER_TYPE_SELL_LIMIT:
            result = m_trade.SellLimit(volume, price, _Symbol, sl, tp, ORDER_TIME_GTC, 0, comment);
            typeDesc = "SELL_LIMIT";
            break;
         case ORDER_TYPE_BUY_STOP:
            result = m_trade.BuyStop(volume, price, _Symbol, sl, tp, ORDER_TIME_GTC, 0, comment);
            typeDesc = "BUY_STOP";
            break;
         case ORDER_TYPE_SELL_STOP:
            result = m_trade.SellStop(volume, price, _Symbol, sl, tp, ORDER_TIME_GTC, 0, comment);
            typeDesc = "SELL_STOP";
            break;
         default:
            Print("ERROR: Invalid pending order type: ", EnumToString(type));
            return false;
      }
   }

   //--- Error handling
   if(!result)
   {
      uint retcode = m_trade.ResultRetcode();
      string retDesc = m_trade.ResultRetcodeDescription();
      Print("OrderSend FAILED: ", typeDesc, " ", DoubleToString(volume, 2), " ", _Symbol,
            " | Retcode: ", retcode, " - ", retDesc);

      //--- Common error codes
      switch(retcode)
      {
         case TRADE_RETCODE_NO_MONEY:
            Print("   -> Insufficient money");
            break;
         case TRADE_RETCODE_INVALID_VOLUME:
            Print("   -> Invalid volume. Check lot step/min/max");
            break;
         case TRADE_RETCODE_MARKET_CLOSED:
            Print("   -> Market is closed");
            break;
         case TRADE_RETCODE_PRICE_OFF:
            Print("   -> Price is off (requote)");
            break;
         case TRADE_RETCODE_TOO_MANY_REQUESTS:
            Print("   -> Too many requests. Throttling");
            break;
         default:
            break;
      }
      return false;
   }

   //--- Success: log the trade
   ulong ticketResult = m_trade.ResultOrder();
   if(ticketResult > 0)
   {
      double fillPrice = m_trade.ResultPrice();
      Print("Order filled: #", ticketResult, " ", typeDesc, " ", DoubleToString(volume, 2),
            " @ ", DoubleToString(fillPrice, (int)m_symbol.Digits()), " [", comment, "]");

      LogTrade(ticketResult, _Symbol, comment, typeDesc, volume, fillPrice, sl, tp,
               TimeToString(TimeCurrent()), comment, (int)magic,
               (isPending ? "PENDING" : "MARKET"));
   }

   return result;
}
//+------------------------------------------------------------------+
//| CSV trade logging                                                 |
//+------------------------------------------------------------------+
void LogTrade(const ulong ticket, const string symbol, const string strategy,
              const string type, const double volume, const double price,
              const double sl, const double tp, const string timeStr,
              const string comment, const int magic, const string execType)
{
   if(!InpTradeLog)
      return;

   string fileName = "AlgoTradeEA\\trades.csv";

   //--- Ensure directory exists
   FolderCreate("AlgoTradeEA");

   int handle = FileOpen(fileName, FILE_READ|FILE_WRITE|FILE_CSV, ',');
   if(handle == INVALID_HANDLE)
   {
      //--- Create new file with header
      handle = FileOpen(fileName, FILE_WRITE|FILE_CSV, ',');
      if(handle == INVALID_HANDLE)
      {
         Print("WARNING: Could not create trade log file: ", fileName, " Error: ", GetLastError());
         return;
      }
      FileWrite(handle, "Ticket", "Symbol", "Strategy", "Type", "Volume",
                "Price", "SL", "TP", "Time", "Comment", "Magic", "ExecType");
      FileClose(handle);

      //--- Reopen for appending
      handle = FileOpen(fileName, FILE_READ|FILE_WRITE|FILE_CSV, ',');
      if(handle == INVALID_HANDLE)
         return;
   }

   //--- Seek to end and write
   FileSeek(handle, 0, SEEK_END);
   FileWrite(handle, ticket, symbol, strategy, type,
             DoubleToString(volume, 2), DoubleToString(price, (int)m_symbol.Digits()),
             DoubleToString(sl, (int)m_symbol.Digits()), DoubleToString(tp, (int)m_symbol.Digits()),
             timeStr, comment, magic, execType);
   FileFlush(handle);
   FileClose(handle);
}
//+------------------------------------------------------------------+
//| Close all open positions (emergency stop)                        |
//+------------------------------------------------------------------+
void CloseAllPositions(const string reason)
{
   if(InpTradeLog)
      Print("CLOSING ALL POSITIONS: ", reason);

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(m_position.SelectByIndex(i) && m_position.Symbol() == _Symbol)
      {
         ulong posMagic = m_position.Magic();
         // Only close positions with our magic number
         if(posMagic >= InpMagicNumber && posMagic <= InpMagicNumber + 10)
         {
            m_trade.SetExpertMagicNumber(posMagic);
            m_trade.PositionClose(_Symbol, InpSlippage);
         }
      }
   }
}
//+------------------------------------------------------------------+
//| Manage fixed trailing stop (non-ATR) for all open positions      |
//+------------------------------------------------------------------+
void ManageFixedTrailingStop(void)
{
   if(InpTrailingStopPts <= 0.0)
      return;

   double point = m_symbol.Point();
   double trailDist = InpTrailingStopPts * point;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!m_position.SelectByIndex(i)) continue;
      if(m_position.Symbol() != _Symbol) continue;
      ulong posMagic = m_position.Magic();
      if(posMagic < InpMagicNumber || posMagic > InpMagicNumber + 10) continue;

      double currentSL = m_position.StopLoss();
      double openPrice = m_position.PriceOpen();
      double currentPrice = (m_position.PositionType() == POSITION_TYPE_BUY) ?
                             m_symbol.Bid() : m_symbol.Ask();

      double newSL = 0.0;
      if(m_position.PositionType() == POSITION_TYPE_BUY)
         newSL = currentPrice - trailDist;
      else
         newSL = currentPrice + trailDist;

      // Only modify if new SL is better
      if(m_position.PositionType() == POSITION_TYPE_BUY)
      {
         if(newSL > currentSL)
         {
            m_trade.SetExpertMagicNumber(posMagic);
            m_trade.PositionModify(_Symbol, newSL, m_position.TakeProfit());
         }
      }
      else
      {
         if(newSL < currentSL || currentSL == 0.0)
         {
            m_trade.SetExpertMagicNumber(posMagic);
            m_trade.PositionModify(_Symbol, newSL, m_position.TakeProfit());
         }
      }
   }
}
//+------------------------------------------------------------------+
//| Manage ATR-based trailing stops for TF positions                 |
//+------------------------------------------------------------------+
void ManageATRTrailingStop(const ulong magic, const double atrValue)
{
   if(!InpUseATRtrailing || atrValue <= 0.0)
      return;

   double trailDist = InpTFATRmult * atrValue;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!m_position.SelectByIndex(i)) continue;
      if(m_position.Symbol() != _Symbol) continue;
      if(m_position.Magic() != magic) continue;

      double currentSL = m_position.StopLoss();
      double currentPrice = (m_position.PositionType() == POSITION_TYPE_BUY) ?
                             m_symbol.Bid() : m_symbol.Ask();

      double newSL = 0.0;
      if(m_position.PositionType() == POSITION_TYPE_BUY)
         newSL = currentPrice - trailDist;
      else
         newSL = currentPrice + trailDist;

      // Update only if better
      if(m_position.PositionType() == POSITION_TYPE_BUY)
      {
         if(newSL > currentSL || currentSL == 0.0)
         {
            m_trade.SetExpertMagicNumber(magic);
            m_trade.PositionModify(_Symbol, newSL, m_position.TakeProfit());
         }
      }
      else
      {
         if(newSL < currentSL || currentSL == 0.0)
         {
            m_trade.SetExpertMagicNumber(magic);
            m_trade.PositionModify(_Symbol, newSL, m_position.TakeProfit());
         }
      }
   }
}
//+------------------------------------------------------------------+
//| Update the on-chart info label                                   |
//+------------------------------------------------------------------+
void UpdateChartLabel(void)
{
   if(!InpDrawOnChart)
      return;

   string info = "AlgoTrade EA v1.00 | " + _Symbol + "\n";
   info += "Balance: " + DoubleToString(m_account.Balance(), 2) + " | Equity: " + DoubleToString(m_account.Equity(), 2) + "\n";
   info += "Spread: " + IntegerToString((int)m_symbol.Spread()) + " pts\n";

   if(g_drawdownHit)          info += "!! DRAWDOWN HALT !!\n";
   else if(g_dailyLossHit)    info += "!! DAILY LOSS LIMIT HIT !!\n";

   //--- Count positions per module
   int tfCount = 0, mrCount = 0, moCount = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!m_position.SelectByIndex(i)) continue;
      if(m_position.Symbol() != _Symbol) continue;
      ulong m = m_position.Magic();
      if(m == InpMagicNumber + InpTFMagicOffset) tfCount++;
      else if(m == InpMagicNumber + InpMRMagicOffset) mrCount++;
      else if(m == InpMagicNumber + InpMOMagicOffset) moCount++;
   }
   info += "Positions: TF=" + IntegerToString(tfCount) + " MR=" + IntegerToString(mrCount) + " MO=" + IntegerToString(moCount) + "\n";

   ObjectSetString(0, g_labelName, OBJPROP_TEXT, info);
   ChartRedraw(0);
}
//+------------------------------------------------------------------+
//| Draw signal arrows on the chart                                  |
//+------------------------------------------------------------------+
void DrawSignal(const datetime time, const double price, const string tag, const bool isBuy)
{
   if(!InpDrawOnChart)
      return;

   string objName = "AlgoTradeEA_Signal_" + tag + "_" + IntegerToString(time);
   ObjectCreate(0, objName, isBuy ? OBJ_ARROW_BUY : OBJ_ARROW_SELL, 0, time, price);
   ObjectSetInteger(0, objName, OBJPROP_COLOR, isBuy ? clrLimeGreen : clrRed);
   ObjectSetInteger(0, objName, OBJPROP_WIDTH, 2);
   ObjectSetInteger(0, objName, OBJPROP_ARROWCODE, isBuy ? 233 : 234); // ▲ ▼
   ChartRedraw(0);
}
//+------------------------------------------------------------------+
//| Check if position exists for a given magic number (symbol)       |
//+------------------------------------------------------------------+
bool HasPosition(const ulong magic)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(m_position.SelectByIndex(i))
      {
         if(m_position.Symbol() == _Symbol && m_position.Magic() == magic)
            return true;
      }
   }
   return false;
}
//+------------------------------------------------------------------+
//| Get position side for a given magic (POSITION_TYPE_BUY/SELL/both)|
//+------------------------------------------------------------------+
int GetPositionDirection(const ulong magic)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(m_position.SelectByIndex(i))
      {
         if(m_position.Symbol() == _Symbol && m_position.Magic() == magic)
         {
            if(m_position.PositionType() == POSITION_TYPE_BUY) return +1;
            else return -1;
         }
      }
   }
   return 0; // No position
}
//+------------------------------------------------------------------+
//| TREND FOLLOWING MODULE                                           |
//+------------------------------------------------------------------+
void ProcessTrendFollowing(void)
{
   ulong tfMagic = InpMagicNumber + InpTFMagicOffset;
   string commentPrefix = InpCommentPrefix + "-TF";

   //--- Check for existing TF position
   bool hasPos = HasPosition(tfMagic);
   int  posDir = GetPositionDirection(tfMagic);

   //--- Refresh indicator data (already done in OnTick, but ensure we have it)
   if(ArraySize(bufEMAfast) < 3) return;
   if(ArraySize(bufADXmain) < 1) return;

   //--- Current and previous values
   double emaF = bufEMAfast[0];
   double emaS = bufEMAslow[0];
   double emaF_prev = bufEMAfast[1];
   double emaS_prev = bufEMAslow[1];
   double adxVal = bufADXmain[0];
   double atrVal = bufATR[0];
   double bid = m_symbol.Bid();
   double ask = m_symbol.Ask();
   double currentPrice = (bid + ask) / 2.0;

   //--- Detect crossover
   bool goldenCross = (emaF_prev <= emaS_prev && emaF > emaS);
   bool deathCross  = (emaF_prev >= emaS_prev && emaF < emaS);

   //--- ENTRY
   if(!hasPos)
   {
      if(goldenCross && adxVal > InpTFADXthreshold)
      {
         // Long signal
         double sl = currentPrice - InpTFATRmult * atrVal;
         double lot = CalculateLotSize(ask, sl, "TF");
         if(lot > 0.0)
         {
            string comment = commentPrefix + "-LONG";
            if(InpExecMode == EXEC_MARKET_ONLY || InpExecMode == EXEC_BOTH)
            {
               OrderSendWrapper(tfMagic, ORDER_TYPE_BUY, lot, 0, sl, 0, comment, false);
               DrawSignal(TimeCurrent(), currentPrice, "TF_LONG", true);
            }
            if(InpExecMode == EXEC_PENDING_ONLY || InpExecMode == EXEC_BOTH)
            {
               double pendingPrice = ask + InpPendingDistPts * m_symbol.Point();
               OrderSendWrapper(tfMagic, ORDER_TYPE_BUY_STOP, lot, pendingPrice, sl, 0, comment, true);
               DrawSignal(TimeCurrent(), pendingPrice, "TF_LONG_PEND", true);
            }
         }
      }
      else if(deathCross && adxVal > InpTFADXthreshold)
      {
         // Short signal
         double sl = currentPrice + InpTFATRmult * atrVal;
         double lot = CalculateLotSize(bid, sl, "TF");
         if(lot > 0.0)
         {
            string comment = commentPrefix + "-SHORT";
            if(InpExecMode == EXEC_MARKET_ONLY || InpExecMode == EXEC_BOTH)
            {
               OrderSendWrapper(tfMagic, ORDER_TYPE_SELL, lot, 0, sl, 0, comment, false);
               DrawSignal(TimeCurrent(), currentPrice, "TF_SHORT", false);
            }
            if(InpExecMode == EXEC_PENDING_ONLY || InpExecMode == EXEC_BOTH)
            {
               double pendingPrice = bid - InpPendingDistPts * m_symbol.Point();
               OrderSendWrapper(tfMagic, ORDER_TYPE_SELL_STOP, lot, pendingPrice, sl, 0, comment, true);
               DrawSignal(TimeCurrent(), pendingPrice, "TF_SHORT_PEND", false);
            }
         }
      }
   }
   //--- EXIT
   else
   {
      if(posDir > 0)
      {
         // Long exit on death cross or ATR trailing
         if(deathCross)
         {
            ClosePositionByMagic(tfMagic, "TF death cross exit");
         }
         else
         {
            // ATR trailing stop management
            ManageATRTrailingStop(tfMagic, atrVal);
         }
      }
      else if(posDir < 0)
      {
         // Short exit on golden cross or ATR trailing
         if(goldenCross)
         {
            ClosePositionByMagic(tfMagic, "TF golden cross exit");
         }
         else
         {
            ManageATRTrailingStop(tfMagic, atrVal);
         }
      }
   }

   //--- Store previous values for next tick
   g_prevEMAfast = emaF;
   g_prevEMAslow = emaS;
}
//+------------------------------------------------------------------+
//| MEAN REVERSION MODULE                                            |
//+------------------------------------------------------------------+
void ProcessMeanReversion(void)
{
   ulong mrMagic = InpMagicNumber + InpMRMagicOffset;
   string commentPrefix = InpCommentPrefix + "-MR";

   bool hasPos = HasPosition(mrMagic);
   int  posDir = GetPositionDirection(mrMagic);

   if(ArraySize(bufBBupper) < 1) return;
   if(ArraySize(bufRSI) < 1) return;

   double bbU  = bufBBupper[0];
   double bbM  = bufBBmiddle[0];
   double bbL  = bufBBlower[0];
   double rsi  = bufRSI[0];
   double bid  = m_symbol.Bid();
   double ask  = m_symbol.Ask();
   double currentPrice = (bid + ask) / 2.0;

   if(bbU <= 0.0 || bbM <= 0.0 || bbL <= 0.0) return;

   //--- EXIT: price crosses middle band
   if(hasPos)
   {
      if(posDir > 0 && currentPrice >= bbM)
      {
         // Long exit: price above middle band
         ClosePositionByMagic(mrMagic, "MR middle band exit");
         return;
      }
      else if(posDir < 0 && currentPrice <= bbM)
      {
         // Short exit: price below middle band
         ClosePositionByMagic(mrMagic, "MR middle band exit");
         return;
      }
   }

   //--- ENTRY: no position
   if(!hasPos)
   {
      // Long: price at/below lower band AND RSI oversold
      if(currentPrice <= bbL && rsi < InpMRRSIoversold)
      {
         double sl = currentPrice - 2.0 * (bbM - bbL); // 2x BB width below entry
         double lot = CalculateLotSize(ask, sl, "MR");
         if(lot > 0.0)
         {
            string comment = commentPrefix + "-LONG";
            if(InpExecMode == EXEC_MARKET_ONLY || InpExecMode == EXEC_BOTH)
            {
               OrderSendWrapper(mrMagic, ORDER_TYPE_BUY, lot, 0, sl, bbM, comment, false);
               DrawSignal(TimeCurrent(), currentPrice, "MR_LONG", true);
            }
            if(InpExecMode == EXEC_PENDING_ONLY || InpExecMode == EXEC_BOTH)
            {
               // Place buy limit at the lower band level
               OrderSendWrapper(mrMagic, ORDER_TYPE_BUY_LIMIT, lot, bbL, sl, bbM, comment, true);
               DrawSignal(TimeCurrent(), bbL, "MR_LONG_LIMIT", true);
            }
         }
      }
      // Short: price at/above upper band AND RSI overbought
      else if(currentPrice >= bbU && rsi > InpMRRSIoverbought)
      {
         double sl = currentPrice + 2.0 * (bbU - bbM); // 2x BB width above entry
         double lot = CalculateLotSize(bid, sl, "MR");
         if(lot > 0.0)
         {
            string comment = commentPrefix + "-SHORT";
            if(InpExecMode == EXEC_MARKET_ONLY || InpExecMode == EXEC_BOTH)
            {
               OrderSendWrapper(mrMagic, ORDER_TYPE_SELL, lot, 0, sl, bbM, comment, false);
               DrawSignal(TimeCurrent(), currentPrice, "MR_SHORT", false);
            }
            if(InpExecMode == EXEC_PENDING_ONLY || InpExecMode == EXEC_BOTH)
            {
               OrderSendWrapper(mrMagic, ORDER_TYPE_SELL_LIMIT, lot, bbU, sl, bbM, comment, true);
               DrawSignal(TimeCurrent(), bbU, "MR_SHORT_LIMIT", false);
            }
         }
      }
   }
}
//+------------------------------------------------------------------+
//| MOMENTUM MODULE                                                  |
//+------------------------------------------------------------------+
void ProcessMomentum(void)
{
   ulong moMagic = InpMagicNumber + InpMOMagicOffset;
   string commentPrefix = InpCommentPrefix + "-MO";

   bool hasPos = HasPosition(moMagic);
   int  posDir = GetPositionDirection(moMagic);

   if(ArraySize(bufROC) < 1) return;
   if(ArraySize(bufOBV) < 1) return;

   // iMomentum returns (Close/Close-period)*100, centered at 100.
   // Convert to ROC-like percentage: roc = momentum - 100
   double momentum = bufROC[0];
   double roc      = momentum - 100.0;      // e.g. 105 -> +5%, 95 -> -5%
   double obv      = bufOBV[0];
   double bid      = m_symbol.Bid();
   double ask      = m_symbol.Ask();
   double currentPrice = (bid + ask) / 2.0;

   //--- Use stored previous values (matching Python logic)
   double prevClose = g_prevClose;
   double prevOBV   = g_prevOBV;

   //--- On first call with zero-defaults, treat as "no divergence"
   bool priceUp = (prevClose > 0.0) ? (currentPrice > prevClose) : true;
   bool obvUp   = (prevOBV != 0.0)  ? (obv > prevOBV)           : true;
   bool obvConfirms = (priceUp && obvUp) || (!priceUp && !obvUp);
   bool obvDiverges = (priceUp && !obvUp) || (!priceUp && obvUp);

   //--- ENTRY (mirrors Python: obv_confirms + ROC threshold)
   if(!hasPos && obvConfirms)
   {
      if(roc > InpMOROCthreshold)
      {
         // Bullish momentum
         double sl = currentPrice - (currentPrice * 0.02); // 2% stop
         double lot = CalculateLotSize(ask, sl, "MO");
         if(lot > 0.0)
         {
            string comment = commentPrefix + "-LONG";
            if(InpExecMode == EXEC_MARKET_ONLY || InpExecMode == EXEC_BOTH)
            {
               OrderSendWrapper(moMagic, ORDER_TYPE_BUY, lot, 0, sl, 0, comment, false);
               DrawSignal(TimeCurrent(), currentPrice, "MO_LONG", true);
            }
            if(InpExecMode == EXEC_PENDING_ONLY || InpExecMode == EXEC_BOTH)
            {
               double pendingPrice = ask + InpPendingDistPts * m_symbol.Point();
               OrderSendWrapper(moMagic, ORDER_TYPE_BUY_STOP, lot, pendingPrice, sl, 0, comment, true);
               DrawSignal(TimeCurrent(), pendingPrice, "MO_LONG_PEND", true);
            }
         }
      }
      else if(roc < -InpMOROCthreshold)
      {
         // Bearish momentum
         double sl = currentPrice + (currentPrice * 0.02);
         double lot = CalculateLotSize(bid, sl, "MO");
         if(lot > 0.0)
         {
            string comment = commentPrefix + "-SHORT";
            if(InpExecMode == EXEC_MARKET_ONLY || InpExecMode == EXEC_BOTH)
            {
               OrderSendWrapper(moMagic, ORDER_TYPE_SELL, lot, 0, sl, 0, comment, false);
               DrawSignal(TimeCurrent(), currentPrice, "MO_SHORT", false);
            }
            if(InpExecMode == EXEC_PENDING_ONLY || InpExecMode == EXEC_BOTH)
            {
               double pendingPrice = bid - InpPendingDistPts * m_symbol.Point();
               OrderSendWrapper(moMagic, ORDER_TYPE_SELL_STOP, lot, pendingPrice, sl, 0, comment, true);
               DrawSignal(TimeCurrent(), pendingPrice, "MO_SHORT_PEND", false);
            }
         }
      }
   }

   //--- EXIT (mirrors Python: ROC reversal, OBV divergence, ROC weakening)
   if(hasPos)
   {
      bool shouldExit = false;
      string exitReason = "";

      if(posDir > 0)
      {
         if(roc < -InpMOROCthreshold)
         {
            shouldExit = true;
            exitReason = "ROC reversal bearish";
         }
         else if(roc < 0)
         {
            shouldExit = true;
            exitReason = "ROC weakening";
         }
      }
      else if(posDir < 0)
      {
         if(roc > InpMOROCthreshold)
         {
            shouldExit = true;
            exitReason = "ROC reversal bullish";
         }
         else if(roc > 0)
         {
            shouldExit = true;
            exitReason = "ROC weakening";
         }
      }

      if(obvDiverges)
      {
         shouldExit = true;
         exitReason = (exitReason == "" ? "" : exitReason + " + ") + "OBV divergence";
      }

      if(shouldExit)
      {
         ClosePositionByMagic(moMagic, "MO " + exitReason);
      }
   }

   //--- Store values for next tick (matching Python pattern)
   g_prevROC   = roc;
   g_prevOBV   = obv;
   g_prevClose = currentPrice;
}
//+------------------------------------------------------------------+
//| Close position identified by magic number                        |
//+------------------------------------------------------------------+
void ClosePositionByMagic(const ulong magic, const string reason)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(m_position.SelectByIndex(i))
      {
         if(m_position.Symbol() == _Symbol && m_position.Magic() == magic)
         {
            m_trade.SetExpertMagicNumber(magic);
            bool closed = m_trade.PositionClose(_Symbol, InpSlippage);
            if(closed)
            {
               Print("Position closed: #", m_position.Ticket(), " [", reason, "]");
               LogTrade(m_position.Ticket(), _Symbol, reason, "CLOSE",
                        m_position.Volume(), m_position.PriceCurrent(),
                        0, 0, TimeToString(TimeCurrent()), reason, (int)magic, "CLOSE");
            }
            else
            {
               uint retcode = m_trade.ResultRetcode();
               Print("Position close FAILED: #", m_position.Ticket(), " [", reason,
                     "] Retcode: ", retcode, " ", m_trade.ResultRetcodeDescription());
            }
         }
      }
   }
}
//+------------------------------------------------------------------+
//| Get the last closing price (avoids conflict with built-in iClose)|
//+------------------------------------------------------------------+
double GetClosePrice(const string sym, const ENUM_TIMEFRAMES tf, const int shift)
{
   double close[];
   ArraySetAsSeries(close, true);
   if(CopyClose(sym, tf, shift, 1, close) > 0)
      return close[0];
   return 0.0;
}
//+------------------------------------------------------------------+
//| Get the last time                                                |
//+------------------------------------------------------------------+
datetime GetTime(const string sym, const ENUM_TIMEFRAMES tf, const int shift)
{
   datetime times[];
   ArraySetAsSeries(times, true);
   if(CopyTime(sym, tf, shift, 1, times) > 0)
      return times[0];
   return 0;
}
//+------------------------------------------------------------------+