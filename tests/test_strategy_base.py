"""Tests for the Strategy base class and lifecycle."""

from __future__ import annotations

from typing import Any

import pytest

from core.config import TradingConfig
from core.models import OHLCV, Order, OrderSide, OrderType, Position
from strategies.base import Strategy

# --- A minimal concrete strategy for testing the base class ---

class DummyStrategy(Strategy):
    """Minimal concrete strategy implementing all abstract methods."""

    def __init__(self) -> None:
        self._init_called = False
        self._bar_data: list[tuple] = []
        self._filled: list[Order] = []
        self._cancelled: list[Order] = []
        self._positions: list[Position] = []
        self._stopped = False

    @property
    def id(self) -> str:
        return "dummy"

    @property
    def name(self) -> str:
        return "Dummy Strategy"

    @property
    def symbols(self) -> list[str]:
        return ["BTC/USD"]

    @property
    def timeframe(self) -> str:
        return "1m"

    def on_init(self, config: TradingConfig) -> None:
        self._init_called = True

    def on_bar(
        self,
        symbol: str,
        ohlcv: list[OHLCV],
        features: dict[str, Any],
        signals: dict[str, Any],
    ) -> list[Order]:
        self._bar_data.append((symbol, ohlcv, features, signals))
        return []

    def on_order_filled(self, order: Order) -> None:
        self._filled.append(order)

    def on_order_cancelled(self, order: Order) -> None:
        self._cancelled.append(order)

    def on_position_update(self, position: Position) -> None:
        self._positions.append(position)

    def on_stop(self) -> None:
        self._stopped = True


class TestStrategyBase:
    def test_instantiation(self):
        """Strategy can be instantiated with abstract methods implemented."""
        s = DummyStrategy()
        assert s.id == "dummy"
        assert s.name == "Dummy Strategy"
        assert s.symbols == ["BTC/USD"]
        assert s.timeframe == "1m"
        assert s.metrics == {}

    def test_lifecycle_on_init(self):
        """on_init is called with TradingConfig."""
        s = DummyStrategy()
        config = TradingConfig.defaults()
        s.on_init(config)
        assert s._init_called

    def test_lifecycle_on_bar(self):
        """on_bar returns a list of orders (empty in this case)."""
        s = DummyStrategy()
        config = TradingConfig.defaults()
        s.on_init(config)

        ohlcv = [OHLCV(timestamp=..., open=100, high=101, low=99, close=100.5, volume=1000)]
        orders = s.on_bar("BTC/USD", ohlcv, {"rsi": 50}, {"direction": 0})
        assert orders == []
        assert len(s._bar_data) == 1

    def test_lifecycle_on_order_filled(self):
        """on_order_filled callback stores the order."""
        s = DummyStrategy()
        order = Order(symbol="BTC/USD", side=OrderSide.BUY, type=OrderType.MARKET, quantity=1.0)
        s.on_order_filled(order)
        assert len(s._filled) == 1
        assert s._filled[0] == order

    def test_lifecycle_on_order_cancelled(self):
        """on_order_cancelled callback stores the order."""
        s = DummyStrategy()
        order = Order(symbol="BTC/USD", side=OrderSide.BUY, type=OrderType.MARKET, quantity=1.0)
        s.on_order_cancelled(order)
        assert len(s._cancelled) == 1
        assert s._cancelled[0] == order

    def test_lifecycle_on_position_update(self):
        """on_position_update callback stores the position."""
        s = DummyStrategy()
        pos = Position(symbol="BTC/USD", quantity=1.0, entry_price=50000)
        s.on_position_update(pos)
        assert len(s._positions) == 1
        assert s._positions[0] == pos

    def test_lifecycle_on_stop(self):
        """on_stop cleanup is called."""
        s = DummyStrategy()
        s.on_stop()
        assert s._stopped

    def test_abstract_class_cannot_instantiate(self):
        """Cannot instantiate Strategy directly (it's abstract)."""
        with pytest.raises(TypeError):
            # noinspection PyTypeChecker,PyUnusedLocal
            Strategy()  # type: ignore[abstract]

    def test_metrics_property_default(self):
        """Default metrics returns empty dict."""
        s = DummyStrategy()
        assert s.metrics == {}
