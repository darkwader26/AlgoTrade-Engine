# AlgoTrade Engine — MetaTrader 5 Expert Advisor

This directory contains the MT5 counterpart of the AlgoTrade Engine algorithmic
trading framework. The EA (`AlgoTradeEA.mq5`) implements three trading strategies
with configurable risk management, matching the Python backend implementations.

---

## Quick Start

1. **Copy files** to your MetaTrader 5 installation:

   ```
   {MT5_DATA_DIR}/MQL5/Experts/AlgoTradeEA/
     AlgoTradeEA.mq5          # Main EA source
   ```

   Typical paths:
   - Windows: `%APPDATA%\MetaQuotes\Terminal\{INSTANCE}\MQL5\Experts\`
   - macOS (Wine): `~/.wine/drive_c/users/username/AppData/Roaming/MetaQuotes/...`
   - Linux (Wine): same structure as macOS

2. **Compile** in MetaEditor:
   - Open MetaEditor (Tools → MetaQuotes Language Editor)
   - Navigate to `AlgoTradeEA/AlgoTradeEA.mq5`
   - Click Compile (F7) — must produce zero errors and zero warnings

3. **Attach to chart**:
   - Drag `AlgoTradeEA` from Navigator onto any chart
   - Configure input parameters in the dialog
   - Enable automated trading (AutoTrading button or Tools → Options → Expert Advisors)

---

## Strategy Modules

### 1. Trend Following (`InpEnableTF = true`)
- **Entry**: EMA(12) crosses above EMA(26) (golden cross) with ADX > 25 filter
- **Exit**: Death cross OR ATR-based trailing stop (2× ATR)
- **Trailing**: ATR-multiplier trailing (configurable via `InpTFATRmult`)

### 2. Mean Reversion (`InpEnableMR = true`)
- **Entry (long)**: Price ≤ lower Bollinger Band (20,2) AND RSI < 30
- **Entry (short)**: Price ≥ upper Bollinger Band (20,2) AND RSI > 70
- **Exit**: Price crosses the middle Bollinger Band

### 3. Momentum (`InpEnableMO = true`)
- **Entry**: Rate of Change (ROC, period=12) exceeds threshold with OBV
  confirmation (price and OBV move in same direction)
- **Exit**: ROC reversal, OBV divergence (price/OBV diverge), or ROC weakening
  (positive ROC turns negative or vice versa)

---

## Risk Management

| Parameter | Default | Description |
|---|---|---|
| `InpSizingMethod` | SIZING_RISK_PERCENT | Position sizing method |
| `InpRiskPercent` | 1.0 | % of account at risk per trade |
| `InpFixedVolume` | 0.01 | Fixed lot size (0 = auto) |
| `InpFixedFraction` | 0.02 | Fraction of equity per trade |
| `InpUseKelly` | false | Enable Kelly criterion |
| `InpKellyF` | 0.25 | Kelly fraction cap (0.0 – 0.5) |
| `InpMaxDrawdown` | 20.0 | Stop trading at this drawdown % |
| `InpDailyLossLimit` | 5.0 | Stop new trades after this daily loss % |
| `InpTrailingStopPts` | 0.0 | Fixed trailing stop in points (0 = off) |
| `InpUseATRtrailing` | true | ATR trailing (TF module only) |

**Kelly Criterion**: The EA calculates `f* = (p × b − q) ÷ b` where:
- `p` = win rate (`InpKellyWinRate`)
- `q` = 1 − p
- `b` = avg win / avg loss (`InpKellyAvgWin / InpKellyAvgLoss`)

---

## Execution Modes

- **Market orders** (`EXEC_MARKET_ONLY`): Enters immediately on signal
- **Pending orders** (`EXEC_PENDING_ONLY`): Places stop/limit orders at specified
  levels
- **Both** (`EXEC_BOTH`): Places market orders AND pending orders on the same
  signal (not recommended for most use cases)

For mean reversion in pending mode, buy/sell limits are placed directly at the
Bollinger Band levels.

---

## Trade Logging

Each fill is recorded to `MQL5/Files/AlgoTradeEA/trades.csv` with:
`Ticket, Symbol, Strategy, Type, Volume, Price, SL, TP, Time, Comment, Magic, ExecType`

---

## Input Parameter Reference

### General
| Parameter | Type | Default | Range |
|---|---|---|---|
| `InpMagicNumber` | ulong | 123456 | Any |
| `InpCommentPrefix` | string | "AlgoTrade" | — |
| `InpTradeLog` | bool | true | — |
| `InpDrawOnChart` | bool | true | — |
| `InpExecMode` | enum | MARKET_ONLY | — |
| `InpSlippage` | int | 30 | 0–100 |

### Trend Following
| Parameter | Type | Default | Description |
|---|---|---|---|
| `InpTFEMAfast` | int | 12 | Fast EMA period |
| `InpTFEMAslow` | int | 26 | Slow EMA period |
| `InpTFADXperiod` | int | 14 | ADX period |
| `InpTFADXthreshold` | double | 25.0 | Min ADX for entry |
| `InpTFATRperiod` | int | 14 | ATR period |
| `InpTFATRmult` | double | 2.0 | ATR stop multiplier |

### Mean Reversion
| Parameter | Type | Default | Description |
|---|---|---|---|
| `InpMRBBperiod` | int | 20 | Bollinger Bands period |
| `InpMRBBstddev` | double | 2.0 | Bollinger Bands std.dev |
| `InpMRRSIperiod` | int | 14 | RSI period |
| `InpMRRSIoverbought` | double | 70.0 | RSI overbought threshold |
| `InpMRRSIoversold` | double | 30.0 | RSI oversold threshold |

### Momentum
| Parameter | Type | Default | Description |
|---|---|---|---|
| `InpMOROCperiod` | int | 12 | ROC period |
| `InpMOROCthreshold` | double | 5.0 | ROC entry threshold |

---

## Magic Number Scheme

Each strategy module gets a unique magic number derived from the base magic:

| Module | Magic = Base + Offset | Default |
|---|---|---|
| Trend Following | `InpMagicNumber + InpTFMagicOffset` | 123457 |
| Mean Reversion | `InpMagicNumber + InpMRMagicOffset` | 123458 |
| Momentum | `InpMagicNumber + InpMOMagicOffset` | 123459 |

---

## Compilation Notes

- The EA requires **MQL5** (MetaTrader 5). It will NOT compile in MetaTrader 4.
- Minimum MetaTrader 5 build: 2000+ (for `input group` syntax)
- No external DLLs or libraries required — pure MQL5 standard library
- All indicators are created with `iMA`, `iADX`, `iATR`, `iBands`, `iRSI`,
  `iMomentum`, and `iOBV` (no custom indicator files needed)

---

## Troubleshooting

| Symptom | Likely Cause |
|---|---|
| EA won't attach to chart | Compilation errors — check MetaEditor log |
| No trades taken | Risk limits hit (check Journal), spread too high |
| "Invalid volume" error | Lot size outside broker's min/max/step limits |
| "No money" error | Insufficient margin for the calculated lot size |
| Orders not closing | Market closed, or position belongs to different magic |
| CSV log not created | MT5 file permissions — check `Files` folder exists |

---

## Related

- Python backend: `~/trading-bot/strategies/trend_follow.py`
- Python backend: `~/trading-bot/strategies/mean_reversion.py`
- Python backend: `~/trading-bot/strategies/momentum.py`
- Python risk manager: `~/trading-bot/strategies/risk_manager.py`
