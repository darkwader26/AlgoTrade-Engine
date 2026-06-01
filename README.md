# =============================================================================
#  AlgoTrade Engine
# =============================================================================

**AlgoTrade Engine** is a modular, event-driven algorithmic trading framework
written in Python. It provides a complete pipeline for data ingestion, feature
engineering, signal generation, strategy execution, risk management, and order
submission — all designed for both backtesting and live paper trading.

![Python](https://img.shields.io/badge/python-3.11-blue)
![Build](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-245_passing-brightgreen)

---

## Architecture

The trading pipeline follows a modular, unidirectional data flow:

```
                    ┌──────────────┐
                    │   DataFeed   │
                    └──────┬───────┘
                           │ OHLCV bars
                           ▼
                 ┌──────────────────┐
                 │ FeaturePipeline  │  ← Technical indicators (SMA, RSI, MACD, etc.)
                 └──────┬───────────┘
                        │ Features
                        ▼
                ┌──────────────────┐
                │ SignalGenerator  │  ← Consensus signals from multiple indicators
                └──────┬───────────┘
                       │ Signals
                       ▼
                ┌──────────────────┐
                │    Strategy      │  ← User-defined trading strategies
                └──────┬───────────┘
                       │ Orders
                       ▼
                ┌──────────────────┐
                │   RiskManager    │  ← Position sizing, stop-loss, drawdown checks
                └──────┬───────────┘
                       │ Approved orders
                       ▼
                ┌──────────────────┐
                │  OrderManager    │  ← Order validation, execution, tracking
                └──────┬───────────┘
                       │ Filled orders
                       ▼
                ┌──────────────────┐
                │    Exchange      │  ← PaperExchange or LiveExchange
                └──────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) — fast Python package installer

### Setup

```bash
make setup
```

### Run Tests

```bash
make test
```

### Run in Paper Trading Mode

```bash
make run-paper
```

This starts the engine using the `PaperExchange`, reading configuration from
`config.yaml` (or defaults if the file is absent).

### Run with Coverage

```bash
make coverage
```

### Lint

```bash
make lint
```

### Build Docker Image

```bash
make docker
```

### Clean Caches

```bash
make clean
```

---

## Configuration

The system is configured via `config.yaml` in the project root. All fields have
sensible defaults so the bot runs out of the box.

| Parameter           | Default            | Description                                    |
|---------------------|--------------------|------------------------------------------------|
| `symbols`           | `[BTC/USD, ETH/USD]` | Trading symbols/pairs                        |
| `timeframes`        | `[1m, 5m, 1h]`     | Timeframes for bar data                        |
| `initial_capital`   | `100000.0`         | Starting capital for paper/backtest trading    |
| `commission`        | `0.001`            | Trading commission (0.1%)                      |
| `slippage`          | `0.0005`           | Slippage per order (0.05%)                     |
| `position_sizing`   | `fixed`            | Position sizing method (`fixed`, `percent`, `kelly`) |
| `max_position_size` | `10000.0`          | Maximum size per position                      |
| `stop_loss_pct`     | `0.02`             | Stop-loss threshold (2%)                       |
| `take_profit_pct`   | `0.05`             | Take-profit target (5%)                        |
| `max_drawdown_pct`  | `0.20`             | Maximum allowed drawdown (20%)                 |
| `use_paper_trading` | `true`             | Use PaperExchange instead of live exchange     |

Example `config.yaml`:

```yaml
symbols:
  - BTC/USD
  - ETH/USD
timeframes:
  - 1m
  - 5m
  - 1h
initial_capital: 100000.0
commission: 0.001
slippage: 0.0005
position_sizing: fixed
max_position_size: 10000.0
stop_loss_pct: 0.02
take_profit_pct: 0.05
max_drawdown_pct: 0.2
use_paper_trading: true
```

---

## Strategy Development

Writing a new strategy is straightforward. Subclass the `Strategy` base class
and implement the required methods.

### Step-by-Step

1. **Create a new file** in `strategies/`, e.g. `strategies/my_strategy.py`.

2. **Import and extend `Strategy`:**

```python
from typing import Any
from core.models import OHLCV, Order, OrderSide, OrderType
from strategies.base import Strategy


class MyCustomStrategy(Strategy):
    """My custom trading strategy."""

    @property
    def id(self) -> str:
        return "my_strategy"

    @property
    def name(self) -> str:
        return "My Custom Strategy"

    @property
    def symbols(self) -> list[str]:
        return ["BTC/USD", "ETH/USD"]

    @property
    def timeframe(self) -> str:
        return "5m"

    def on_bar(
        self,
        symbol: str,
        ohlcv: list[OHLCV],
        features: dict[str, Any],
        signals: dict[str, Any],
    ) -> list[Order]:
        # Use features and signals to decide trading actions
        if signals.get("direction") == 1 and signals.get("strength") == "strong":
            return [
                Order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=0.1,
                )
            ]
        return []
```

3. **Register your strategy** with the engine in `strategies/engine.py` or at
   startup:

```python
from strategies.my_strategy import MyCustomStrategy

engine = StrategyEngine(order_manager, config)
engine.register_strategy(MyCustomStrategy())
```

4. **Add your strategy** to `strategies/__init__.py`:

```python
from strategies.my_strategy import MyCustomStrategy

__all__ = [
    ...
    "MyCustomStrategy",
]
```

### Strategy Lifecycle Hooks

| Method               | When Called                          | Purpose                          |
|----------------------|--------------------------------------|----------------------------------|
| `on_init(config)`    | Once at startup                      | Set up indicators, load state    |
| `on_bar(...)`        | Every new bar for each symbol        | Generate orders                  |
| `on_order_filled(order)` | When an order is filled          | Track fills                      |
| `on_order_cancelled(order)` | When order is cancelled       | Handle cancellations             |
| `on_position_update(position)` | When position changes     | Update tracking                  |
| `on_stop()`          | During shutdown                      | Clean up resources               |

### Included Strategies

| Strategy                  | File                          | Description                              |
|---------------------------|-------------------------------|------------------------------------------|
| TrendFollowingStrategy    | `strategies/trend_follow.py`  | Follows EMA crossovers and trend signals |
| MeanReversionStrategy     | `strategies/mean_reversion.py`| Trades bounces from Bollinger Band edges |
| MomentumStrategy          | `strategies/momentum.py`      | Buys strength with RSI/MACD confirmation |

---

## Project Structure

```
trading-bot/
├── core/                        # Core backend execution
│   ├── __init__.py
│   ├── config.py                # Configuration (TradingConfig, load/save)
│   ├── data_feed.py             # Real-time and historical data feed
│   ├── exchange.py              # Exchange interface + PaperExchange
│   ├── models.py                # Domain models (OHLCV, Order, Position, etc.)
│   └── order_manager.py         # Order validation and execution
├── features/                    # Feature engineering & signal generation
│   ├── __init__.py
│   ├── cache.py                 # Feature computation cache
│   ├── feature_pipeline.py      # Orchestrated feature computation
│   ├── indicators.py            # Technical indicators (SMA, EMA, RSI, etc.)
│   ├── ml_signal.py             # ML-based signal generation (scikit-learn)
│   └── signal_generator.py      # Consensus signal aggregation
├── strategies/                  # Strategy engine & trading strategies
│   ├── __init__.py
│   ├── base.py                  # Abstract Strategy base class
│   ├── engine.py                # Strategy execution engine
│   ├── mean_reversion.py        # Mean-reversion strategy
│   ├── momentum.py              # Momentum strategy
│   ├── performance.py           # Performance metrics calculation
│   ├── risk_manager.py          # Risk checks and position sizing
│   └── trend_follow.py          # Trend-following strategy
├── tests/                       # Comprehensive test suite (245+ tests)
│   ├── __init__.py
│   ├── test_cache.py
│   ├── test_config.py
│   ├── test_core_models.py
│   ├── test_data_feed.py
│   ├── test_feature_pipeline.py
│   ├── test_indicators.py
│   ├── test_mean_reversion.py
│   ├── test_ml_signal.py
│   ├── test_momentum.py
│   ├── test_order_manager.py
│   ├── test_paper_exchange.py
│   ├── test_performance.py
│   ├── test_risk_manager.py
│   ├── test_signal_generator.py
│   ├── test_strategy_base.py
│   ├── test_strategy_engine.py
│   └── test_trend_follow.py
├── docker/                      # Docker & DevOps
│   ├── Dockerfile               # Multi-stage production image
│   ├── docker-compose.yml       # Service orchestration
│   ├── prometheus.yml           # Prometheus scraping config
│   └── grafana/
│       ├── dashboards/
│       │   └── trading-bot.json # Grafana dashboard
│       └── datasources/
│           └── prometheus.yml   # Datasource provisioning
├── .github/workflows/
│   └── ci.yml                   # CI pipeline (test, lint, docker build)
├── .dockerignore                # Build context exclusions
├── config.yaml                  # Trading configuration
├── docker-compose.yml           # Root compose → docker/docker-compose.yml
├── Makefile                     # Common development tasks
├── pyproject.toml               # Project metadata & tool config
└── README.md                    # This file
```

---

## Testing

The project includes **245+ tests** covering all components.

### Running Tests

```bash
# All tests with verbose output
make test

# With coverage report
make coverage

# Run a specific test file
uv run pytest tests/test_strategy_engine.py -v

# Run tests matching a keyword
uv run pytest tests/ -v -k "risk"
```

### Test Structure

- **Unit tests** test individual components in isolation
- **Integration tests** test the pipeline end-to-end
- Tests use the `PaperExchange` to avoid live market dependencies
- Feature computation and signal generation are tested against known inputs

### CI

Every push to `main` and every PR triggers the CI pipeline:

1. **test** — Runs all tests with coverage and uploads the report
2. **lint** — Runs `ruff check` on the entire codebase
3. **docker** — Verifies the Docker image builds successfully

---

## Docker

Build and run the full stack with Docker Compose:

```bash
# Build all services
make docker

# Start everything
docker compose -f docker/docker-compose.yml up -d

# View logs
docker compose -f docker/docker-compose.yml logs -f trading-bot
```

The compose stack includes:
- **trading-bot** — The trading engine (from Dockerfile)
- **prometheus** — Metrics collection
- **grafana** — Metrics dashboards (default: http://localhost:3000, admin/admin)

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
