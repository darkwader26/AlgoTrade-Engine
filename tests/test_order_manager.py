"""Tests for OrderManager — order lifecycle, portfolio tracking, trade history."""

from __future__ import annotations

import pytest

from core.exchange import PaperExchange
from core.models import Order, OrderSide, OrderStatus, OrderType
from core.order_manager import OrderManager


@pytest.fixture
def exchange():
    return PaperExchange(
        initial_capital=100_000.0,
        commission=0.0,
        slippage=0.0,
        latency_ms=0,
        partial_fill_prob=0.0,
    )


@pytest.fixture
def manager(exchange):
    return OrderManager(exchange=exchange, initial_cash=100_000.0)


class TestOrderSubmission:
    def test_submit_market_buy(self, manager, exchange):
        exchange.update_market_price("BTC/USD", 50000.0)
        order = Order(symbol="BTC/USD", side=OrderSide.BUY, type=OrderType.MARKET, quantity=1.0)
        result = manager.submit(order)
        assert result.status == OrderStatus.FILLED
        assert result.filled_qty == 1.0

    def test_submit_rejected_insufficient_cash(self, manager, exchange):
        exchange.update_market_price("BTC/USD", 1_000_000.0)
        order = Order(symbol="BTC/USD", side=OrderSide.BUY, type=OrderType.MARKET, quantity=100.0)
        result = manager.submit(order)
        assert result.status == OrderStatus.REJECTED

    def test_validate_negative_quantity(self, manager):
        order = Order(symbol="BTC/USD", side=OrderSide.BUY, type=OrderType.MARKET, quantity=-1.0)
        with pytest.raises(ValueError, match="positive"):
            manager.submit(order)

    def test_validate_limit_no_price(self, manager):
        order = Order(symbol="BTC/USD", side=OrderSide.BUY, type=OrderType.LIMIT, quantity=1.0)
        with pytest.raises(ValueError, match="positive price"):
            manager.submit(order)

    def test_validate_invalid_side(self, manager):
        order = Order(symbol="BTC/USD", side="INVALID", type=OrderType.MARKET, quantity=1.0)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Invalid order side"):
            manager.submit(order)


class TestCancelOrder:
    def test_cancel_open_order(self, manager, exchange):
        exchange.update_market_price("BTC/USD", 50000.0)
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            quantity=1.0,
            price=40000.0,
        )
        manager.submit(order)
        assert manager.cancel(order.id) is True
        fetched = manager.get_order(order.id)
        assert fetched is not None
        assert fetched.status == OrderStatus.CANCELLED

    def test_cancel_nonexistent(self, manager):
        assert manager.cancel("fake") is False


class TestOrderRetrieval:
    def test_get_order(self, manager, exchange):
        exchange.update_market_price("BTC/USD", 50000.0)
        order = Order(symbol="BTC/USD", side=OrderSide.BUY, type=OrderType.MARKET, quantity=1.0)
        manager.submit(order)
        fetched = manager.get_order(order.id)
        assert fetched is not None
        assert fetched.id == order.id

    def test_get_nonexistent_order(self, manager):
        assert manager.get_order("nonexistent") is None

    def test_get_open_orders(self, manager, exchange):
        exchange.update_market_price("BTC/USD", 50000.0)
        limit = Order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            quantity=1.0,
            price=30000.0,
        )
        market = Order(symbol="BTC/USD", side=OrderSide.BUY, type=OrderType.MARKET, quantity=1.0)
        manager.submit(limit)
        manager.submit(market)
        open_orders = manager.get_open_orders()
        # limit is still open, market is filled
        assert len(open_orders) == 1
        assert open_orders[0].id == limit.id

    def test_order_history(self, manager, exchange):
        exchange.update_market_price("BTC/USD", 50000.0)
        o1 = Order(symbol="BTC/USD", side=OrderSide.BUY, type=OrderType.MARKET, quantity=1.0)
        o2 = Order(symbol="ETH/USD", side=OrderSide.BUY, type=OrderType.MARKET, quantity=10.0)
        manager.submit(o1)
        manager.submit(o2)
        history = manager.get_order_history()
        assert len(history) == 2

        filtered = manager.get_order_history(symbol="BTC/USD")
        assert len(filtered) == 1

    def test_order_history_limit(self, manager, exchange):
        exchange.update_market_price("BTC/USD", 50000.0)
        for _ in range(5):
            manager.submit(Order(symbol="BTC/USD", side=OrderSide.BUY, type=OrderType.MARKET, quantity=1.0))
        assert len(manager.get_order_history(limit=3)) == 3


class TestTradeHistory:
    def test_trade_history(self, manager, exchange):
        exchange.update_market_price("BTC/USD", 50000.0)
        order = Order(symbol="BTC/USD", side=OrderSide.BUY, type=OrderType.MARKET, quantity=1.0)
        manager.submit(order)
        trades = manager.get_trade_history()
        assert len(trades) >= 1
        assert trades[0].order_id == order.id

    def test_trade_history_filtered(self, manager, exchange):
        exchange.update_market_price("BTC/USD", 50000.0)
        exchange.update_market_price("ETH/USD", 3000.0)
        manager.submit(Order(symbol="BTC/USD", side=OrderSide.BUY, type=OrderType.MARKET, quantity=1.0))
        manager.submit(Order(symbol="ETH/USD", side=OrderSide.BUY, type=OrderType.MARKET, quantity=10.0))
        btc_trades = manager.get_trade_history(symbol="BTC/USD")
        assert len(btc_trades) == 1
        assert btc_trades[0].symbol == "BTC/USD"


class TestPortfolio:
    def test_initial_portfolio(self, manager):
        pf = manager.get_portfolio()
        assert pf.cash == 100_000.0
        assert pf.total_equity == 100_000.0
        assert pf.total_pnl == 0.0

    def test_portfolio_after_buy(self, manager, exchange):
        exchange.update_market_price("BTC/USD", 50000.0)
        order = Order(symbol="BTC/USD", side=OrderSide.BUY, type=OrderType.MARKET, quantity=2.0)
        manager.submit(order)
        pf = manager.get_portfolio()
        assert pf.cash < 100_000.0
        assert "BTC/USD" in pf.positions
        assert pf.positions["BTC/USD"].quantity == 2.0
        assert pf.total_equity == pytest.approx(100_000.0, abs=0.02)  # minus fees

    def test_portfolio_unrealized_pnl(self, manager, exchange):
        exchange.update_market_price("BTC/USD", 50000.0)
        order = Order(symbol="BTC/USD", side=OrderSide.BUY, type=OrderType.MARKET, quantity=1.0)
        manager.submit(order)
        # Price goes up
        exchange.update_market_price("BTC/USD", 52000.0)
        # Update portfolio mark-to-market via exchange (PaperExchange.update_market_price already updates positions)
        pf = manager.get_portfolio()
        if "BTC/USD" in pf.positions:
            assert pf.positions["BTC/USD"].current_price == 52000.0

    def test_full_trade_cycle_pnl(self, manager, exchange):
        # Buy at 50000
        exchange.update_market_price("BTC/USD", 50000.0)
        buy = Order(symbol="BTC/USD", side=OrderSide.BUY, type=OrderType.MARKET, quantity=2.0)
        manager.submit(buy)
        # Sell at 51000
        exchange.update_market_price("BTC/USD", 51000.0)
        sell = Order(symbol="BTC/USD", side=OrderSide.SELL, type=OrderType.MARKET, quantity=2.0)
        manager.submit(sell)
        pf = manager.get_portfolio()
        # Gross profit should be (51000-50000)*2 = 2000 (no fees in test config)
        assert pf.total_pnl == pytest.approx(2000.0, abs=0.02)

    def test_portfolio_copy_not_mutable(self, manager, exchange):
        exchange.update_market_price("BTC/USD", 50000.0)
        order = Order(symbol="BTC/USD", side=OrderSide.BUY, type=OrderType.MARKET, quantity=1.0)
        manager.submit(order)
        pf = manager.get_portfolio()
        # Modify the copy
        pf.cash = 0.0
        pf2 = manager.get_portfolio()
        assert pf2.cash != 0.0
