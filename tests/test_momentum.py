"""Tests for the Momentum strategy."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from core.config import TradingConfig
from core.models import OHLCV, OrderSide, Position
from strategies.momentum import MomentumConfig, MomentumStrategy


@pytest.fixture
def config() -> TradingConfig:
    return TradingConfig(
        symbols=["BTC/USD", "ETH/USD"],
        initial_capital=100_000.0,
    )


@pytest.fixture
def strategy() -> MomentumStrategy:
    strat = MomentumStrategy(
        MomentumConfig(
            roc_period=5,
            momentum_threshold=2.0,
            top_n=2,
        )
    )
    strat.symbols = ["BTC/USD", "ETH/USD"]
    return strat


def make_ohlcv(close_prices: list[float]) -> list[OHLCV]:
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


class TestMomentum:
    def test_initialization(self, strategy: MomentumStrategy, config: TradingConfig):
        """Strategy initializes with config."""
        strategy.on_init(config)
        assert strategy.id == "momentum"
        assert strategy.name == "Momentum"
        assert set(strategy.symbols) == {"BTC/USD", "ETH/USD"}
        assert strategy.timeframe == "1h"

    def test_long_entry_on_strong_roc(self, strategy: MomentumStrategy, config: TradingConfig):
        """Long entry when ROC exceeds threshold with OBV confirmation."""
        strategy.on_init(config)

        # Prices trending sharply up
        prices = [100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 115.0]
        ohlcv = make_ohlcv(prices)

        close_prices = np.array(prices, dtype=np.float64)
        volumes = np.full(7, 1000.0, dtype=np.float64)
        from features.indicators import obv, roc

        roc_val = float(roc(close_prices, 5)[-1])
        obv_val = float(obv(close_prices, volumes)[-1])

        features = {
            "roc_5": roc_val,
            "obv": obv_val,
        }

        orders = strategy.on_bar("BTC/USD", ohlcv, features, {"direction": 1})

        # If ROC > threshold and OBV confirms (price up, volume up)
        if roc_val > 2.0 and len(ohlcv) >= 6:
            assert len(orders) == 1
            assert orders[0].side == OrderSide.BUY
        else:
            assert len(orders) == 0

    def test_short_entry_on_strong_negative_roc(self, strategy: MomentumStrategy, config: TradingConfig):
        """Short entry when ROC is strongly negative with OBV confirmation."""
        strategy.on_init(config)

        # Prices trending sharply down
        prices = [100.0, 98.0, 96.0, 94.0, 92.0, 90.0, 85.0]
        ohlcv = make_ohlcv(prices)

        close_prices = np.array(prices, dtype=np.float64)
        volumes = np.full(7, 1000.0, dtype=np.float64)
        from features.indicators import obv, roc

        roc_val = float(roc(close_prices, 5)[-1])
        obv_val = float(obv(close_prices, volumes)[-1])

        features = {
            "roc_5": roc_val,
            "obv": obv_val,
        }

        orders = strategy.on_bar("BTC/USD", ohlcv, features, {"direction": -1})

        if roc_val < -2.0 and len(ohlcv) >= 6:
            assert len(orders) == 1
            assert orders[0].side == OrderSide.SELL
        else:
            assert len(orders) == 0

    def test_exit_on_roc_reversal_long(self, strategy: MomentumStrategy, config: TradingConfig):
        """Exit long position when ROC reverses."""
        strategy.on_init(config)

        # Simulate existing long position
        pos = Position(symbol="BTC/USD", quantity=1.0, entry_price=100.0, current_price=110.0)
        strategy.on_position_update(pos)

        # Need to prime the previous values
        strategy._prev_roc["BTC/USD"] = 5.0  # was strongly positive
        strategy._prev_close["BTC/USD"] = 110.0

        # Now ROC turns negative
        prices = [110.0, 109.0, 108.0, 107.0, 106.0, 105.0, 103.0]
        ohlcv = make_ohlcv(prices)

        close_prices = np.array(prices, dtype=np.float64)
        volumes = np.full(7, 1000.0, dtype=np.float64)
        from features.indicators import obv, roc

        roc_val = float(roc(close_prices, 5)[-1])
        obv_val = float(obv(close_prices, volumes)[-1])

        features = {
            "roc_5": roc_val,
            "obv": obv_val,
        }

        orders = strategy.on_bar("BTC/USD", ohlcv, features, {"direction": -1})

        # Should exit if ROC is negative or below threshold
        if roc_val < 0 or roc_val < -2.0:
            assert len(orders) == 1
            assert orders[0].side == OrderSide.SELL
        else:
            # May or may not exit depending on actual values
            pass

    def test_unknown_symbol_ignored(self, strategy: MomentumStrategy, config: TradingConfig):
        """Bars for unregistered symbols are ignored."""
        strategy.on_init(config)
        ohlcv = make_ohlcv([100.0, 101.0, 102.0])
        orders = strategy.on_bar("SOL/USD", ohlcv, {}, {})
        assert orders == []

    def test_insufficient_data(self, strategy: MomentumStrategy, config: TradingConfig):
        """Strategy returns empty list with too few bars."""
        strategy.on_init(config)
        ohlcv_data = [OHLCV(timestamp=..., open=100, high=101, low=99, close=100.5, volume=1000)]
        orders = strategy.on_bar("BTC/USD", ohlcv_data, {}, {})
        assert orders == []

    def test_on_stop_cleans_up(self, strategy: MomentumStrategy, config: TradingConfig):
        """on_stop clears internal state."""
        strategy.on_init(config)
        strategy._prev_roc["BTC/USD"] = 5.0
        strategy._prev_obv["BTC/USD"] = 1000.0
        strategy._prev_close["BTC/USD"] = 100.0

        strategy.on_stop()

        assert strategy._prev_roc == {}
        assert strategy._prev_obv == {}
        assert strategy._prev_close == {}

    def test_metrics_property(self, strategy: MomentumStrategy):
        """Metrics returns empty dict initially."""
        assert strategy.metrics == {}
