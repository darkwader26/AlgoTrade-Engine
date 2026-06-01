"""Tests for the Strategy Engine."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from core.config import TradingConfig
from core.exchange import PaperExchange
from core.models import OHLCV, Order, OrderSide, OrderType
from core.order_manager import OrderManager
from strategies.base import Strategy
from strategies.engine import StrategyEngine
from strategies.momentum import MomentumConfig, MomentumStrategy


class MockStrategy(Strategy):
    """Strategy that always returns one buy order on each bar."""

    def __init__(self) -> None:
        self._symbols = ["BTC/USD"]
        self._bar_count = 0
        self._init_called = False

    @property
    def id(self) -> str:
        return "mock"

    @property
    def name(self) -> str:
        return "Mock Strategy"

    @property
    def symbols(self) -> list[str]:
        return self._symbols

    @property
    def timeframe(self) -> str:
        return "1m"

    def on_init(self, config: TradingConfig) -> None:
        self._init_called = True
        self._symbols = list(config.symbols)

    def on_bar(self, symbol: str, ohlcv, features, signals) -> list[Order]:
        self._bar_count += 1
        return [
            Order(
                symbol=symbol,
                side=OrderSide.BUY,
                type=OrderType.MARKET,
                quantity=0.01,
            )
        ]

    def on_stop(self) -> None:
        pass


class EmptyStrategy(Strategy):
    """Strategy that never creates orders."""

    def __init__(self) -> None:
        self._symbols = ["BTC/USD"]

    @property
    def id(self) -> str:
        return "empty"

    @property
    def name(self) -> str:
        return "Empty"

    @property
    def symbols(self) -> list[str]:
        return self._symbols

    @property
    def timeframe(self) -> str:
        return "1m"

    def on_bar(self, symbol: str, ohlcv, features, signals) -> list[Order]:
        return []

    def on_init(self, config: TradingConfig) -> None:
        self._symbols = list(config.symbols)


@pytest.fixture
def exchange() -> PaperExchange:
    return PaperExchange(initial_capital=100_000.0)


@pytest.fixture
def order_manager(exchange) -> OrderManager:
    return OrderManager(exchange, initial_cash=100_000.0)


@pytest.fixture
def config() -> TradingConfig:
    return TradingConfig(
        symbols=["BTC/USD"],
        initial_capital=100_000.0,
        max_position_size=20000.0,
    )


@pytest.fixture
def engine(order_manager, config) -> StrategyEngine:
    eng = StrategyEngine(order_manager, config)
    return eng


def make_ohlcv_bars(n: int = 100, start_price: float = 100.0) -> list[OHLCV]:
    """Create a list of OHLCV bars."""
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(n) * 0.5) + start_price
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [
        OHLCV(
            timestamp=ts,
            open=float(prices[i] * 0.99),
            high=float(prices[i] * 1.01),
            low=float(prices[i] * 0.99),
            close=float(prices[i]),
            volume=1000.0,
        )
        for i in range(n)
    ]


def make_ohlcv_df(n: int = 100, start_price: float = 100.0) -> pd.DataFrame:
    """Create a simple OHLCV DataFrame with n rows."""
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(n) * 0.5) + start_price
    dates = pd.date_range(start="2024-01-01", periods=n, freq="1h")

    return pd.DataFrame(
        {
            "open": prices * 0.99,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.full(n, 1000.0),
        },
        index=dates,
    )


class TestStrategyEngine:
    def test_initialization(self, engine):
        """Engine initializes with defaults."""
        assert not engine.is_running
        assert isinstance(engine.risk_manager, object)
        assert isinstance(engine.feature_pipeline, object)
        assert isinstance(engine.signal_generator, object)

    def test_register_strategy(self, engine):
        """Register a strategy."""
        strategy = MockStrategy()
        engine.register_strategy(strategy)
        assert "mock" in engine.strategies
        assert strategy._init_called

    def test_register_duplicate_raises(self, engine):
        """Registering same strategy twice raises."""
        strategy = MockStrategy()
        engine.register_strategy(strategy)
        with pytest.raises(KeyError, match="already registered"):
            engine.register_strategy(strategy)

    def test_remove_strategy(self, engine):
        """Remove a registered strategy."""
        strategy = MockStrategy()
        engine.register_strategy(strategy)
        engine.remove_strategy("mock")
        assert "mock" not in engine.strategies

    def test_run_bar_no_strategies(self, engine):
        """Running bar with no strategies returns empty list."""
        bars = make_ohlcv_bars(50)
        orders = engine.run_bar("BTC/USD", bars)
        assert orders == []

    def test_run_bar_with_strategy(self, engine, order_manager):
        """Running bar with a mock strategy submits orders."""
        engine.register_strategy(MockStrategy())

        # Need enough bars (at least 26 for EMA_26)
        bars = make_ohlcv_bars(50)
        orders = engine.run_bar("BTC/USD", bars)

        assert len(orders) == 1
        assert orders[0].symbol == "BTC/USD"
        assert orders[0].side == OrderSide.BUY

    def test_run_bar_empty_strategy(self, engine):
        """Running bar with empty strategy returns no orders."""
        engine.register_strategy(EmptyStrategy())

        bars = make_ohlcv_bars(50)
        orders = engine.run_bar("BTC/USD", bars)
        assert orders == []

    def test_get_strategy_metrics(self, engine):
        """Get metrics for unknown strategy returns empty dict."""
        engine.register_strategy(MockStrategy())
        metrics = engine.get_strategy_metrics("nonexistent")
        assert metrics == {}

    def test_start_stop(self, engine):
        """Start and stop lifecycle."""
        assert not engine.is_running
        engine.start()
        assert engine.is_running
        engine.stop()
        assert not engine.is_running

    def test_backtest_basic(self, engine):
        """Basic backtest runs without error."""
        strategy = MockStrategy()
        engine.register_strategy(strategy)

        df = make_ohlcv_df(50)
        result = engine.run_backtest({"BTC/USD": df})

        assert "trade_log" in result
        assert "equity_curve" in result
        assert "performance" in result
        assert "summary" in result

        assert len(result["equity_curve"]) >= 2

        summary = result["summary"]
        assert "total_return" in summary
        assert "sharpe_ratio" in summary
        assert "max_drawdown" in summary
        assert "win_rate" in summary
        assert "total_trades" in summary
        assert "profit_factor" in summary
        assert summary["start_equity"] == 100_000.0

    def test_backtest_with_momentum_strategy(self, engine):
        """Backtest with momentum strategy."""
        strategy = MomentumStrategy(
            MomentumConfig(roc_period=5, momentum_threshold=2.0)
        )
        strategy.symbols = ["BTC/USD"]
        engine.register_strategy(strategy)

        df = make_ohlcv_df(100, start_price=100.0)
        result = engine.run_backtest({"BTC/USD": df})

        assert "trade_log" in result
        assert "equity_curve" in result
        assert len(result["equity_curve"]) >= 2

    def test_backtest_date_filtering(self, engine):
        """Backtest with date range filtering."""
        strategy = MockStrategy()
        engine.register_strategy(strategy)

        df = make_ohlcv_df(100)
        # Use offset-naive datetimes (consistent with DataFrame index)
        start_date = datetime(2024, 1, 5)
        end_date = datetime(2024, 1, 10)

        result = engine.run_backtest(
            {"BTC/USD": df},
            start_date=start_date,
            end_date=end_date,
        )

        assert "equity_curve" in result

    def test_ohlcv_to_dataframe(self, engine):
        """Static helper converts OHLCV list to DataFrame."""
        ts = datetime.now(timezone.utc)
        ohlcv = [
            OHLCV(timestamp=ts, open=100.0, high=101.0, low=99.0, close=100.5, volume=1000.0),
            OHLCV(timestamp=ts, open=101.0, high=102.0, low=100.0, close=101.5, volume=1100.0),
        ]
        df = StrategyEngine._ohlcv_to_dataframe(ohlcv)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert len(df) == 2
        assert df["close"].iloc[-1] == 101.5

    def test_empty_ohlcv_to_dataframe(self, engine):
        """Empty list returns empty DataFrame."""
        df = StrategyEngine._ohlcv_to_dataframe([])
        assert df.empty

    def test_backtest_preserves_original_state(self, engine):
        """Backtest restores original order manager after completion."""
        original_om = engine._order_manager
        strategy = MockStrategy()
        engine.register_strategy(strategy)

        df = make_ohlcv_df(10)
        engine.run_backtest({"BTC/USD": df})

        assert engine._order_manager is original_om
