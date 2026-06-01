"""Tests for the Mean Reversion strategy."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from core.config import TradingConfig
from core.models import OHLCV, OrderSide, OrderType, Position
from strategies.mean_reversion import MeanReversionConfig, MeanReversionStrategy


@pytest.fixture
def config() -> TradingConfig:
    return TradingConfig(
        symbols=["BTC/USD"],
        initial_capital=100_000.0,
    )


@pytest.fixture
def strategy() -> MeanReversionStrategy:
    strat = MeanReversionStrategy(
        MeanReversionConfig(
            bb_period=5,  # Small for testing
            bb_stddev=2.0,
            rsi_period=5,
            rsi_oversold=30.0,
            rsi_overbought=70.0,
        )
    )
    strat.symbols = ["BTC/USD"]
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


class TestMeanReversion:
    def test_initialization(self, strategy: MeanReversionStrategy, config: TradingConfig):
        """Strategy initializes with config."""
        strategy.on_init(config)
        assert strategy.id == "mean_reversion"
        assert strategy.name == "Mean Reversion"
        assert strategy.symbols == ["BTC/USD"]
        assert strategy.timeframe == "5m"

    def test_long_entry_on_oversold(self, strategy: MeanReversionStrategy, config: TradingConfig):
        """Long entry when price touches lower band and RSI is oversold."""
        strategy.on_init(config)

        # Create data where price drops below lower band
        prices = [100.0] * 6 + [90.0, 88.0, 85.0, 82.0, 80.0, 78.0]
        ohlcv = make_ohlcv(prices)

        # Price at lower band or below
        close_prices = np.array(prices, dtype=np.float64)
        from features.indicators import bollinger_bands, rsi

        bb_upper, bb_middle, bb_lower = bollinger_bands(close_prices, 5, 2.0)
        rsi_val = rsi(close_prices, 5)

        features = {
            "bb_upper": float(bb_upper[-1]),
            "bb_middle": float(bb_middle[-1]),
            "bb_lower": float(bb_lower[-1]),
            "rsi_5": float(rsi_val[-1]),
        }

        orders = strategy.on_bar("BTC/USD", ohlcv, features, {"direction": -1})

        # If price <= lower band and RSI < 30, expect buy order
        current_price = prices[-1]
        if current_price <= features["bb_lower"] and features["rsi_5"] < 30.0:
            assert len(orders) == 1
            assert orders[0].side == OrderSide.BUY
            assert orders[0].type == OrderType.MARKET
        else:
            assert len(orders) == 0

    def test_short_entry_on_overbought(self, strategy: MeanReversionStrategy, config: TradingConfig):
        """Short entry when price touches upper band and RSI is overbought."""
        strategy.on_init(config)

        # Create data where price rises above upper band
        prices = [100.0] * 6 + [110.0, 112.0, 115.0, 118.0, 120.0, 122.0]
        ohlcv = make_ohlcv(prices)

        close_prices = np.array(prices, dtype=np.float64)
        from features.indicators import bollinger_bands, rsi

        bb_upper, bb_middle, bb_lower = bollinger_bands(close_prices, 5, 2.0)
        rsi_val = rsi(close_prices, 5)

        features = {
            "bb_upper": float(bb_upper[-1]),
            "bb_middle": float(bb_middle[-1]),
            "bb_lower": float(bb_lower[-1]),
            "rsi_5": float(rsi_val[-1]),
        }

        orders = strategy.on_bar("BTC/USD", ohlcv, features, {"direction": 1})

        current_price = prices[-1]
        if current_price >= features["bb_upper"] and features["rsi_5"] > 70.0:
            assert len(orders) == 1
            assert orders[0].side == OrderSide.SELL
        else:
            assert len(orders) == 0

    def test_exit_when_price_crosses_middle_band_long(self, strategy: MeanReversionStrategy, config: TradingConfig):
        """Exit long when price crosses above middle band."""
        strategy.on_init(config)

        # Simulate being in a long position
        pos = Position(symbol="BTC/USD", quantity=1.0, entry_price=80.0, current_price=85.0)
        strategy.on_position_update(pos)

        # Price crosses middle band (was below, now above)
        prices = [80.0, 82.0, 85.0, 90.0, 95.0, 100.0]
        ohlcv = make_ohlcv(prices)

        close_prices = np.array(prices, dtype=np.float64)
        from features.indicators import bollinger_bands, rsi

        bb_upper, bb_middle, bb_lower = bollinger_bands(close_prices, 5, 2.0)
        rsi_val = rsi(close_prices, 5)

        features = {
            "bb_upper": float(bb_upper[-1]),
            "bb_middle": float(bb_middle[-1]),
            "bb_lower": float(bb_lower[-1]),
            "rsi_5": float(rsi_val[-1]),
        }

        orders = strategy.on_bar("BTC/USD", ohlcv, features, {"direction": 0})

        # If current price >= middle band, should sell to exit
        current_price = prices[-1]
        if current_price >= features["bb_middle"]:
            assert len(orders) == 1
            assert orders[0].side == OrderSide.SELL

    def test_exit_when_price_crosses_middle_band_short(self, strategy: MeanReversionStrategy, config: TradingConfig):
        """Exit short when price crosses below middle band."""
        strategy.on_init(config)

        # Simulate being in a short position
        pos = Position(symbol="BTC/USD", quantity=-1.0, entry_price=120.0, current_price=115.0)
        strategy.on_position_update(pos)

        # Price crosses middle band (was above, now below)
        prices = [120.0, 118.0, 115.0, 110.0, 105.0, 100.0]
        ohlcv = make_ohlcv(prices)

        close_prices = np.array(prices, dtype=np.float64)
        from features.indicators import bollinger_bands, rsi

        bb_upper, bb_middle, bb_lower = bollinger_bands(close_prices, 5, 2.0)
        rsi_val = rsi(close_prices, 5)

        features = {
            "bb_upper": float(bb_upper[-1]),
            "bb_middle": float(bb_middle[-1]),
            "bb_lower": float(bb_lower[-1]),
            "rsi_5": float(rsi_val[-1]),
        }

        orders = strategy.on_bar("BTC/USD", ohlcv, features, {"direction": 0})

        current_price = prices[-1]
        if current_price <= features["bb_middle"]:
            assert len(orders) == 1
            assert orders[0].side == OrderSide.BUY

    def test_unknown_symbol_ignored(self, strategy: MeanReversionStrategy, config: TradingConfig):
        """Bars for unregistered symbols are ignored."""
        strategy.on_init(config)
        ohlcv = make_ohlcv([100.0, 101.0, 102.0])
        orders = strategy.on_bar("ETH/USD", ohlcv, {}, {})
        assert orders == []

    def test_insufficient_data(self, strategy: MeanReversionStrategy, config: TradingConfig):
        """Strategy returns empty list with too few bars."""
        strategy.on_init(config)
        ohlcv = make_ohlcv([100.0, 101.0])
        orders = strategy.on_bar("BTC/USD", ohlcv, {}, {})
        assert orders == []

    def test_on_stop(self, strategy: MeanReversionStrategy, config: TradingConfig):
        """on_stop does not raise."""
        strategy.on_init(config)
        strategy.on_stop()

    def test_metrics_property(self, strategy: MeanReversionStrategy):
        """Metrics returns empty dict initially."""
        assert strategy.metrics == {}
