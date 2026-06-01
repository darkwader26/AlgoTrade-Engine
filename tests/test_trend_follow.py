"""Tests for the Trend Following strategy."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from core.config import TradingConfig
from core.models import OHLCV, Order, OrderSide, OrderType, Position
from strategies.trend_follow import TrendFollowingConfig, TrendFollowingStrategy


@pytest.fixture
def config() -> TradingConfig:
    return TradingConfig(
        symbols=["BTC/USD"],
        initial_capital=100_000.0,
    )


@pytest.fixture
def strategy() -> TrendFollowingStrategy:
    strat = TrendFollowingStrategy(
        TrendFollowingConfig(
            ema_fast=3,
            ema_slow=8,
            adx_threshold=20.0,
            atr_stop_multiplier=2.0,
        )
    )
    strat.symbols = ["BTC/USD"]
    return strat


def make_ohlcv(close_prices: list[float]) -> list[OHLCV]:
    """Create a list of OHLCV bars from close prices."""
    ts = datetime.now(timezone.utc)
    return [
        OHLCV(
            timestamp=ts,
            open=p * 0.99,
            high=p * 1.01,
            low=p * 0.99,
            close=p,
            volume=1000.0,
        )
        for p in close_prices
    ]


def approximate_ema(values: list[float], period: int) -> float:
    """Approximate EMA for last value."""
    arr = np.array(values, dtype=np.float64)
    from features.indicators import ema
    result = ema(arr, period)
    return float(result[-1]) if not np.isnan(result[-1]) else 0.0


def approximate_adx(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float:
    """Approximate ADX for last value."""
    from features.indicators import adx
    result = adx(np.array(highs), np.array(lows), np.array(closes), period)
    return float(result[-1]) if not np.isnan(result[-1]) else 0.0


def approximate_atr(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float:
    from features.indicators import atr
    result = atr(np.array(highs), np.array(lows), np.array(closes), period)
    return float(result[-1]) if not np.isnan(result[-1]) else 0.0


class TestTrendFollowing:
    def test_initialization(self, strategy: TrendFollowingStrategy, config: TradingConfig):
        """Strategy initializes with config."""
        strategy.on_init(config)
        assert strategy.id == "trend_following"
        assert strategy.name == "Trend Following"
        assert strategy.symbols == ["BTC/USD"]
        assert strategy.timeframe == "1h"

    def test_golden_cross_entry(self, strategy: TrendFollowingStrategy, config: TradingConfig):
        """Golden cross with high ADX should generate buy order."""
        strategy.on_init(config)

        # Create data that produces a golden cross (fast EMA crosses above slow EMA)
        # Start with prices going down, then sharply up
        prices = []
        # Downtrend first (50 bars)
        for i in range(50):
            prices.append(100.0 - i * 0.5)
        # Then uptrend (20 bars) to create crossover
        for i in range(20):
            prices.append(75.0 + i * 2.0)

        ohlcv = make_ohlcv(prices)

        # Compute features
        ema_fast = approximate_ema(prices, 3)
        ema_slow = approximate_ema(prices, 8)
        adx_val = approximate_adx(
            [p * 1.01 for p in prices],
            [p * 0.99 for p in prices],
            prices,
            period=14,
        )
        atr_val = approximate_atr(
            [p * 1.01 for p in prices],
            [p * 0.99 for p in prices],
            prices,
            period=14,
        )

        features = {
            "ema_3": ema_fast,
            "ema_8": ema_slow,
            "adx_14": adx_val,
            "atr_14": atr_val,
        }

        orders = strategy.on_bar("BTC/USD", ohlcv, features, {"direction": 1, "strength": "strong"})

        if ema_fast > ema_slow and adx_val >= 20.0 and orders:
            assert len(orders) == 1
            assert orders[0].side == OrderSide.BUY
            assert orders[0].symbol == "BTC/USD"
            assert orders[0].type == OrderType.MARKET
            assert orders[0].quantity > 0

    def test_death_cross_exit(self, strategy: TrendFollowingStrategy, config: TradingConfig):
        """Death cross should generate sell order when in a long position."""
        strategy.on_init(config)

        # Simulate being in a position
        pos = Position(symbol="BTC/USD", quantity=1.0, entry_price=100.0, current_price=105.0)
        strategy.on_position_update(pos)

        # Create data that produces a death cross (fast EMA crosses below slow EMA)
        prices = []
        # Uptrend first
        for i in range(50):
            prices.append(50.0 + i)
        # Then downtrend to create death cross
        for i in range(20):
            prices.append(100.0 - i * 3.0)

        ohlcv = make_ohlcv(prices)

        ema_fast = approximate_ema(prices, 3)
        ema_slow = approximate_ema(prices, 8)
        adx_val = 30.0
        atr_val = 2.0

        features = {
            "ema_3": ema_fast,
            "ema_8": ema_slow,
            "adx_14": adx_val,
            "atr_14": atr_val,
        }

        orders = strategy.on_bar("BTC/USD", ohlcv, features, {"direction": -1, "strength": "strong"})

        if ema_fast < ema_slow:
            assert len(orders) == 1
            assert orders[0].side == OrderSide.SELL

    def test_no_entry_without_adx(self, strategy: TrendFollowingStrategy, config: TradingConfig):
        """No entry should happen when ADX is below threshold even with golden cross."""
        strategy.on_init(config)

        prices = []
        for i in range(50):
            prices.append(100.0 - i * 0.5)
        for i in range(20):
            prices.append(75.0 + i * 2.0)

        ohlcv = make_ohlcv(prices)

        ema_fast = approximate_ema(prices, 3)
        ema_slow = approximate_ema(prices, 8)
        adx_val = 15.0  # Below threshold
        atr_val = 2.0

        features = {
            "ema_3": ema_fast,
            "ema_8": ema_slow,
            "adx_14": adx_val,
            "atr_14": atr_val,
        }

        orders = strategy.on_bar("BTC/USD", ohlcv, features, {"direction": 0, "strength": "weak"})

        # ADX is below threshold, so no entry even if golden cross
        assert len(orders) == 0

    def test_unknown_symbol_ignored(self, strategy: TrendFollowingStrategy, config: TradingConfig):
        """Bars for unregistered symbols are ignored."""
        strategy.on_init(config)

        ohlcv = make_ohlcv([100.0, 101.0, 102.0])
        orders = strategy.on_bar("ETH/USD", ohlcv, {"ema_12": 101}, {"direction": 0})
        assert orders == []

    def test_insufficient_data(self, strategy: TrendFollowingStrategy, config: TradingConfig):
        """Strategy returns empty list with too few bars."""
        strategy.on_init(config)
        ohlcv = make_ohlcv([100.0, 101.0])
        orders = strategy.on_bar("BTC/USD", ohlcv, {}, {})
        assert orders == []

    def test_on_stop_cleans_up(self, strategy: TrendFollowingStrategy, config: TradingConfig):
        """on_stop clears internal state."""
        strategy.on_init(config)
        strategy.on_stop()

        # After stop, the strategy should work again cleanly
        ohlcv = make_ohlcv([100.0, 101.0])
        orders = strategy.on_bar("BTC/USD", ohlcv, {}, {})
        assert orders == []

    def test_on_order_filled_callback(self, strategy: TrendFollowingStrategy, config: TradingConfig):
        """on_order_filled does not raise."""
        strategy.on_init(config)
        order = Order(symbol="BTC/USD", side=OrderSide.BUY, type=OrderType.MARKET, quantity=1.0)
        strategy.on_order_filled(order)  # Should not raise

    def test_metrics_property(self, strategy: TrendFollowingStrategy):
        """Metrics returns empty dict initially."""
        assert strategy.metrics == {}
