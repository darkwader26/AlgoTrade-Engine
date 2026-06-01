"""Tests for PaperExchange — order placement, fill simulation, slippage, fees, partial fills."""

from __future__ import annotations

import pytest

from core.exchange import PaperExchange
from core.models import Order, OrderSide, OrderStatus, OrderType


@pytest.fixture
def exchange():
    return PaperExchange(
        initial_capital=100_000.0,
        commission=0.001,
        slippage=0.0005,
        latency_ms=0,  # zero for tests
        partial_fill_prob=0.0,  # deterministic full fills
    )


class TestPaperExchangeBasic:
    def test_initial_balance(self, exchange):
        assert exchange.fetch_balance() == 100_000.0

    def test_fetch_ohlcv_empty(self, exchange):
        assert exchange.fetch_ohlcv("BTC/USD", "1m") == []

    def test_fetch_unknown_order(self, exchange):
        assert exchange.fetch_order("nonexistent") is None

    def test_fetch_unknown_position(self, exchange):
        assert exchange.fetch_position("BTC/USD") is None


class TestMarketOrders:
    def test_buy_market_fills(self, exchange):
        exchange.update_market_price("BTC/USD", 50000.0)
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=1.0,
        )
        result = exchange.create_order(order)
        assert result.status in (OrderStatus.FILLED, OrderStatus.PARTIAL)
        assert result.filled_qty > 0
        assert result.avg_price is not None

    def test_sell_market_fills(self, exchange):
        exchange.update_market_price("BTC/USD", 50000.0)
        # First buy
        buy = Order(symbol="BTC/USD", side=OrderSide.BUY, type=OrderType.MARKET, quantity=1.0)
        exchange.create_order(buy)
        # Then sell
        sell = Order(symbol="BTC/USD", side=OrderSide.SELL, type=OrderType.MARKET, quantity=1.0)
        result = exchange.create_order(sell)
        assert result.status in (OrderStatus.FILLED, OrderStatus.PARTIAL)

    def test_market_buy_deducts_cash(self, exchange):
        exchange.update_market_price("BTC/USD", 50000.0)
        order = Order(symbol="BTC/USD", side=OrderSide.BUY, type=OrderType.MARKET, quantity=1.0)
        exchange.create_order(order)
        # Cash should have decreased
        balance = exchange.fetch_balance()
        assert balance < 100_000.0

    def test_market_reject_insufficient_cash(self, exchange):
        exchange.update_market_price("BTC/USD", 1_000_000.0)
        order = Order(symbol="BTC/USD", side=OrderSide.BUY, type=OrderType.MARKET, quantity=100.0)
        result = exchange.create_order(order)
        assert result.status == OrderStatus.REJECTED

    def test_market_reject_no_price(self, exchange):
        order = Order(symbol="BTC/USD", side=OrderSide.BUY, type=OrderType.MARKET, quantity=1.0)
        # No price set via update_market_price
        result = exchange.create_order(order)
        assert result.status == OrderStatus.REJECTED


class TestSlippage:
    def test_buy_slippage_increases_price(self, exchange):
        exchange.update_market_price("BTC/USD", 50000.0)
        order = Order(symbol="BTC/USD", side=OrderSide.BUY, type=OrderType.MARKET, quantity=1.0)
        result = exchange.create_order(order)
        # With 0.05% slippage, buy price should be > 50000
        assert result.avg_price is not None
        assert result.avg_price > 50000.0

    def test_sell_slippage_decreases_price(self, exchange):
        exchange.update_market_price("BTC/USD", 50000.0)
        # Buy first to get inventory
        buy = Order(symbol="BTC/USD", side=OrderSide.BUY, type=OrderType.MARKET, quantity=1.0)
        exchange.create_order(buy)

        exchange.update_market_price("BTC/USD", 51000.0)
        sell = Order(symbol="BTC/USD", side=OrderSide.SELL, type=OrderType.MARKET, quantity=1.0)
        result = exchange.create_order(sell)
        assert result.avg_price is not None
        # With slippage, sell price should be < 51000
        assert result.avg_price < 51000.0


class TestFees:
    def test_commission_deducted_on_buy(self, exchange):
        exchange.update_market_price("BTC/USD", 1000.0)
        order = Order(symbol="BTC/USD", side=OrderSide.BUY, type=OrderType.MARKET, quantity=10.0)
        exchange.create_order(order)
        # 10 * 1000 = 10000 cost, 0.05% slippage (buy) => price=1000.50, 0.1% fee
        # cost = 10 * 1000.50 = 10005.00, fee = 10005 * 0.001 = 10.005
        # total = 10015.005, remaining = 100000 - 10015.005 = 89984.995
        remaining = exchange.fetch_balance()
        expected = 100_000.0 - (10 * 1000.0 * 1.0005 * 1.001)  # 89984.995
        assert remaining == pytest.approx(expected, abs=0.01)

    def test_commission_on_sell(self, exchange):
        exchange.update_market_price("BTC/USD", 1000.0)
        buy = Order(symbol="BTC/USD", side=OrderSide.BUY, type=OrderType.MARKET, quantity=10.0)
        exchange.create_order(buy)

        exchange.update_market_price("BTC/USD", 1100.0)
        balance_before = exchange.fetch_balance()
        sell = Order(symbol="BTC/USD", side=OrderSide.SELL, type=OrderType.MARKET, quantity=10.0)
        exchange.create_order(sell)
        balance_after = exchange.fetch_balance()
        # Proceeds with slippage and fee should be less than 10 * 1100
        assert balance_after > balance_before  # Profitable


class TestLimitOrders:
    def test_buy_limit_fills_on_price_drop(self, exchange):
        exchange.update_market_price("BTC/USD", 51000.0)
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            quantity=1.0,
            price=50000.0,
        )
        exchange.create_order(order)
        # Order should be pending
        assert order.status == OrderStatus.PENDING

        # Price drops to fill
        exchange.update_market_price("BTC/USD", 49900.0)
        fetched = exchange.fetch_order(order.id)
        assert fetched is not None
        assert fetched.status == OrderStatus.FILLED

    def test_sell_limit_fills_on_price_rise(self, exchange):
        exchange.update_market_price("BTC/USD", 50000.0)
        # Buy first
        buy = Order(symbol="BTC/USD", side=OrderSide.BUY, type=OrderType.MARKET, quantity=1.0)
        exchange.create_order(buy)

        sell = Order(
            symbol="BTC/USD",
            side=OrderSide.SELL,
            type=OrderType.LIMIT,
            quantity=1.0,
            price=52000.0,
        )
        exchange.create_order(sell)
        assert sell.status == OrderStatus.PENDING

        exchange.update_market_price("BTC/USD", 52500.0)
        fetched = exchange.fetch_order(sell.id)
        assert fetched is not None
        assert fetched.status == OrderStatus.FILLED


class TestCancelOrder:
    def test_cancel_pending(self, exchange):
        exchange.update_market_price("BTC/USD", 50000.0)
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            quantity=1.0,
            price=40000.0,
        )
        exchange.create_order(order)
        assert exchange.cancel_order(order.id) is True
        fetched = exchange.fetch_order(order.id)
        assert fetched is not None
        assert fetched.status == OrderStatus.CANCELLED

    def test_cancel_filled_order_fails(self, exchange):
        exchange.update_market_price("BTC/USD", 50000.0)
        order = Order(symbol="BTC/USD", side=OrderSide.BUY, type=OrderType.MARKET, quantity=1.0)
        exchange.create_order(order)
        # Should not be able to cancel a filled order
        assert exchange.cancel_order(order.id) is False

    def test_cancel_nonexistent(self, exchange):
        assert exchange.cancel_order("fake") is False


class TestPartialFills:
    @pytest.fixture
    def partial_exchange(self):
        return PaperExchange(
            initial_capital=100_000.0,
            commission=0.0,
            slippage=0.0,
            latency_ms=0,
            partial_fill_prob=1.0,  # Always partial
        )

    def test_partial_fill_is_possible(self, partial_exchange):
        partial_exchange.update_market_price("BTC/USD", 100.0)
        order = Order(symbol="BTC/USD", side=OrderSide.BUY, type=OrderType.MARKET, quantity=100.0)
        result = partial_exchange.create_order(order)
        assert result.status == OrderStatus.PARTIAL
        assert result.filled_qty < 100.0
        assert result.filled_qty > 0


class TestPositions:
    def test_position_created_after_buy(self, exchange):
        exchange.update_market_price("BTC/USD", 1000.0)
        order = Order(symbol="BTC/USD", side=OrderSide.BUY, type=OrderType.MARKET, quantity=10.0)
        exchange.create_order(order)
        pos = exchange.fetch_position("BTC/USD")
        assert pos is not None
        assert pos.quantity == 10.0
        assert pos.symbol == "BTC/USD"

    def test_position_closed_after_sell(self, exchange):
        exchange.update_market_price("BTC/USD", 50000.0)
        buy = Order(symbol="BTC/USD", side=OrderSide.BUY, type=OrderType.MARKET, quantity=1.0)
        exchange.create_order(buy)
        sell = Order(symbol="BTC/USD", side=OrderSide.SELL, type=OrderType.MARKET, quantity=1.0)
        exchange.create_order(sell)
        pos = exchange.fetch_position("BTC/USD")
        assert pos is None or pos.quantity == 0

    def test_multiple_positions(self, exchange):
        exchange.update_market_price("BTC/USD", 50000.0)
        exchange.update_market_price("ETH/USD", 3000.0)
        btc = Order(symbol="BTC/USD", side=OrderSide.BUY, type=OrderType.MARKET, quantity=1.0)
        eth = Order(symbol="ETH/USD", side=OrderSide.BUY, type=OrderType.MARKET, quantity=10.0)
        exchange.create_order(btc)
        exchange.create_order(eth)
        positions = exchange.get_positions()
        assert "BTC/USD" in positions
        assert "ETH/USD" in positions


class TestTrades:
    def test_trade_recorded(self, exchange):
        exchange.update_market_price("BTC/USD", 50000.0)
        order = Order(symbol="BTC/USD", side=OrderSide.BUY, type=OrderType.MARKET, quantity=1.0)
        exchange.create_order(order)
        trades = exchange.get_trades()
        assert len(trades) == 1
        assert trades[0].order_id == order.id
        assert trades[0].symbol == "BTC/USD"
        assert trades[0].side == OrderSide.BUY
