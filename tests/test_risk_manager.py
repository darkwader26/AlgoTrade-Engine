"""Tests for the Risk Manager."""

from __future__ import annotations

import threading

import pytest

from core.config import TradingConfig
from core.models import Order, OrderSide, OrderType, Portfolio, Position
from strategies.risk_manager import RiskManager


@pytest.fixture
def risk_manager() -> RiskManager:
    return RiskManager()


@pytest.fixture
def portfolio() -> Portfolio:
    pos = Position(
        symbol="BTC/USD",
        quantity=1.0,
        entry_price=50000.0,
        current_price=51000.0,
        unrealized_pnl=1000.0,
    )
    return Portfolio(
        cash=50000.0,
        positions={"BTC/USD": pos},
        total_equity=101000.0,
        total_pnl=1000.0,
    )


@pytest.fixture
def config() -> TradingConfig:
    return TradingConfig(
        symbols=["BTC/USD"],
        initial_capital=100_000.0,
        max_position_size=20000.0,
    )


class TestRiskManager:
    def test_check_order_allowed(self, risk_manager, portfolio, config):
        """Valid order should pass all checks."""
        order = Order(
            symbol="ETH/USD",  # Different symbol — no existing position
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=0.1,
            price=50000.0,
        )
        allowed, reason = risk_manager.check_order(order, portfolio, config)
        assert allowed, reason
        assert reason == "order allowed"

    def test_check_order_exceeds_max_position(self, risk_manager, portfolio, config):
        """Order exceeding max position size should be rejected."""
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=1.0,
            price=50000.0,  # value = 50000 > 20000
        )
        allowed, reason = risk_manager.check_order(order, portfolio, config)
        assert not allowed
        assert "max position size" in reason.lower()

    def test_apply_kelly(self, risk_manager):
        """Kelly Criterion calculation returns sensible fraction."""
        # High win rate, good win/loss ratio
        pos = Position(symbol="BTC/USD", quantity=1.0, entry_price=100.0)
        fraction = risk_manager.apply_kelly(pos, win_rate=0.6, avg_win=100, avg_loss=50)
        assert 0 < fraction <= 0.25

    def test_apply_kelly_zero_returns(self, risk_manager):
        """Zero win rate returns 0."""
        pos = Position(symbol="BTC/USD", quantity=1.0, entry_price=100.0)
        fraction = risk_manager.apply_kelly(pos, win_rate=0.0, avg_win=100, avg_loss=50)
        assert fraction == 0.0

    def test_apply_kelly_perfect_win_rate(self, risk_manager):
        """Perfect win rate is capped at 0.25."""
        pos = Position(symbol="BTC/USD", quantity=1.0, entry_price=100.0)
        fraction = risk_manager.apply_kelly(pos, win_rate=1.0, avg_win=100, avg_loss=50)
        assert fraction == 0.25

    def test_calculate_stop_loss_long(self, risk_manager):
        """Stop loss for long position."""
        stop = risk_manager.calculate_stop_loss(
            entry_price=100.0,
            side=OrderSide.BUY,
            atr_value=5.0,
            multiplier=2.0,
        )
        assert stop == pytest.approx(90.0)

    def test_calculate_stop_loss_short(self, risk_manager):
        """Stop loss for short position."""
        stop = risk_manager.calculate_stop_loss(
            entry_price=100.0,
            side=OrderSide.SELL,
            atr_value=5.0,
            multiplier=2.0,
        )
        assert stop == pytest.approx(110.0)

    def test_calculate_take_profit_long(self, risk_manager):
        """Take profit for long position."""
        tp = risk_manager.calculate_take_profit(
            entry_price=100.0,
            side=OrderSide.BUY,
            risk_reward_ratio=2.0,
        )
        assert tp > 100.0

    def test_calculate_take_profit_short(self, risk_manager):
        """Take profit for short position."""
        tp = risk_manager.calculate_take_profit(
            entry_price=100.0,
            side=OrderSide.SELL,
            risk_reward_ratio=2.0,
        )
        assert tp < 100.0

    def test_check_drawdown_not_exceeded(self, risk_manager):
        """Drawdown check returns False when within limits."""
        pf = Portfolio(cash=80000.0, total_equity=90000.0)
        # Set peak manually
        RiskManager._peak_equity = 100000.0

        stopped_out = risk_manager.check_drawdown(pf, 0.20)  # 20% max
        # drawdown = 1 - 90000/100000 = 10% < 20%
        assert not stopped_out

    def test_check_drawdown_exceeded(self, risk_manager):
        """Drawdown check returns True when exceeded."""
        pf = Portfolio(cash=70000.0, total_equity=75000.0)
        RiskManager._peak_equity = 100000.0

        stopped_out = risk_manager.check_drawdown(pf, 0.20)  # 20% max
        # drawdown = 1 - 75000/100000 = 25% > 20%
        assert stopped_out

    def test_thread_safety(self, risk_manager, portfolio, config):
        """Multiple threads can call check_order concurrently."""
        errors = []
        lock = threading.Lock()

        def worker():
            try:
                for _ in range(20):
                    order = Order(
                        symbol="BTC/USD",
                        side=OrderSide.BUY,
                        type=OrderType.MARKET,
                        quantity=0.01,
                        price=50000.0,
                    )
                    risk_manager.check_order(order, portfolio, config)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concentration_limit(self, risk_manager, config):
        """Order exceeding concentration limit should be rejected."""
        # Small equity, large order
        small_portfolio = Portfolio(
            cash=1000.0,
            total_equity=1000.0,
        )

        order = Order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=1.0,
            price=500.0,  # 50% of equity, exceeds 25% concentration limit
        )

        allowed, reason = risk_manager.check_order(order, small_portfolio, config)
        assert not allowed
        assert "concentration" in reason.lower()
