"""Tests for core data models."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.models import (
    OHLCV,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Portfolio,
    Position,
    Trade,
)


class TestOHLCV:
    def test_creation(self):
        ts = datetime.now(timezone.utc)
        bar = OHLCV(timestamp=ts, open=100.0, high=105.0, low=99.0, close=102.0, volume=1000.0)
        assert bar.timestamp == ts
        assert bar.open == 100.0
        assert bar.high == 105.0
        assert bar.low == 99.0
        assert bar.close == 102.0
        assert bar.volume == 1000.0

    def test_immutable(self):
        bar = OHLCV(
            timestamp=datetime.now(timezone.utc),
            open=1.0, high=2.0, low=1.0, close=1.5, volume=100.0,
        )
        with pytest.raises(AttributeError):
            # noinspection PyDataclass
            bar.open = 99.0  # type: ignore[misc]

    def test_unpacking(self):
        ts = datetime.now(timezone.utc)
        bar = OHLCV(ts, 10.0, 11.0, 9.0, 10.5, 500.0)
        t, o, h, low_, c, v = bar
        assert t == ts
        assert o == 10.0
        assert h == 11.0
        assert low_ == 9.0
        assert c == 10.5
        assert v == 500.0


class TestOrderSide:
    def test_values(self):
        assert OrderSide.BUY.value == "BUY"
        assert OrderSide.SELL.value == "SELL"

    def test_comparison(self):
        assert OrderSide.BUY == OrderSide("BUY")
        assert OrderSide.SELL != OrderSide.BUY


class TestOrderType:
    def test_values(self):
        assert OrderType.MARKET.value == "MARKET"
        assert OrderType.LIMIT.value == "LIMIT"


class TestOrderStatus:
    def test_values(self):
        assert OrderStatus.PENDING.value == "PENDING"
        assert OrderStatus.FILLED.value == "FILLED"
        assert OrderStatus.PARTIAL.value == "PARTIAL"
        assert OrderStatus.CANCELLED.value == "CANCELLED"
        assert OrderStatus.REJECTED.value == "REJECTED"


class TestOrder:
    def test_default_creation(self):
        order = Order(symbol="BTC/USD", side=OrderSide.BUY, type=OrderType.MARKET, quantity=1.0)
        assert order.symbol == "BTC/USD"
        assert order.side == OrderSide.BUY
        assert order.type == OrderType.MARKET
        assert order.quantity == 1.0
        assert order.price is None
        assert order.status == OrderStatus.PENDING
        assert order.filled_qty == 0.0
        assert order.avg_price is None
        assert order.id is not None

    def test_limit_order_with_price(self):
        order = Order(
            symbol="ETH/USD",
            side=OrderSide.SELL,
            type=OrderType.LIMIT,
            quantity=2.5,
            price=2000.0,
        )
        assert order.price == 2000.0

    def test_unique_ids(self):
        o1 = Order(symbol="A", side=OrderSide.BUY, type=OrderType.MARKET, quantity=1)
        o2 = Order(symbol="A", side=OrderSide.BUY, type=OrderType.MARKET, quantity=1)
        assert o1.id != o2.id

    def test_filled_state(self):
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            quantity=1.0,
            price=50000.0,
            status=OrderStatus.FILLED,
            filled_qty=1.0,
            avg_price=50000.0,
        )
        assert order.status == OrderStatus.FILLED
        assert order.filled_qty == 1.0


class TestPosition:
    def test_creation(self):
        pos = Position(symbol="BTC/USD", quantity=1.0, entry_price=50000.0, current_price=51000.0)
        assert pos.symbol == "BTC/USD"
        assert pos.quantity == 1.0
        assert pos.entry_price == 50000.0
        assert pos.current_price == 51000.0

    def test_unrealized_pnl_long(self):
        pos = Position(symbol="BTC/USD", quantity=2.0, entry_price=50000.0, current_price=51000.0)
        pos.update_unrealized_pnl()
        assert pos.unrealized_pnl == 2000.0  # (51000-50000) * 2

    def test_unrealized_pnl_short(self):
        pos = Position(symbol="BTC/USD", quantity=-1.0, entry_price=50000.0, current_price=48000.0)
        pos.update_unrealized_pnl()
        assert pos.unrealized_pnl == 2000.0  # (48000-50000) * -1

    def test_unrealized_pnl_zero_qty(self):
        pos = Position(symbol="BTC/USD", quantity=0.0, entry_price=50000.0, current_price=99999.0)
        pos.update_unrealized_pnl()
        assert pos.unrealized_pnl == 0.0

    def test_mark_to_market(self):
        pos = Position(symbol="BTC/USD", quantity=1.0, entry_price=50000.0, current_price=50000.0)
        pos.mark_to_market(52000.0)
        assert pos.current_price == 52000.0
        assert pos.unrealized_pnl == 2000.0


class TestTrade:
    def test_creation(self):
        ts = datetime.now(timezone.utc)
        trade = Trade(
            order_id="abc123",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=1.0,
            price=50000.0,
            timestamp=ts,
            pnl=0.0,
        )
        assert trade.order_id == "abc123"
        assert trade.symbol == "BTC/USD"
        assert trade.side == OrderSide.BUY
        assert trade.pnl == 0.0


class TestPortfolio:
    def test_empty_portfolio(self):
        pf = Portfolio(cash=100000.0)
        assert pf.cash == 100000.0
        assert pf.positions == {}
        assert pf.total_equity == 0.0
        assert pf.total_pnl == 0.0

    def test_equity_with_position(self):
        pos = Position(symbol="BTC/USD", quantity=2.0, entry_price=50000.0, current_price=51000.0)
        pf = Portfolio(cash=50000.0, positions={"BTC/USD": pos})
        pf.update_equity()
        assert pf.total_equity == 50000.0 + 2.0 * 51000.0  # 152000.0

    def test_total_pnl(self):
        pos = Position(
            symbol="BTC/USD",
            quantity=1.0,
            entry_price=50000.0,
            current_price=52000.0,
            unrealized_pnl=2000.0,
            realized_pnl=500.0,
        )
        pf = Portfolio(cash=100000.0, positions={"BTC/USD": pos})
        pf.update_pnl()
        assert pf.total_pnl == 2500.0
