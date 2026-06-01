"""Tests for the Performance Metrics calculator."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.models import OrderSide, Trade
from strategies.performance import PerformanceMetrics


class TestPerformanceMetrics:
    def test_empty_trades(self):
        """Empty trades list returns zeros."""
        metrics = PerformanceMetrics.calculate([], [100000.0])
        assert metrics["total_trades"] == 0
        assert metrics["total_return_pct"] == 0.0
        assert metrics["sharpe_ratio"] == 0.0
        assert metrics["max_drawdown_pct"] == 0.0
        assert metrics["win_rate_pct"] == 0.0
        assert metrics["profit_factor"] == 0.0

    def test_all_winning_trades(self):
        """All winning trades produces correct metrics."""
        ts = datetime.now(timezone.utc)
        trades = [
            Trade(order_id="1", symbol="BTC/USD", side=OrderSide.BUY,
                  quantity=1.0, price=100.0, timestamp=ts, pnl=100.0),
            Trade(order_id="2", symbol="BTC/USD", side=OrderSide.BUY,
                  quantity=1.0, price=200.0, timestamp=ts, pnl=200.0),
        ]
        equity = [100000.0, 100100.0, 100300.0]
        metrics = PerformanceMetrics.calculate(trades, equity)

        assert metrics["total_trades"] == 2
        assert metrics["win_rate_pct"] == 100.0
        assert metrics["profit_factor"] == float("inf")  # no losses
        assert metrics["avg_win"] == 150.0
        assert metrics["avg_loss"] == 0.0
        assert metrics["expectancy"] > 0

    def test_all_losing_trades(self):
        """All losing trades produces correct metrics."""
        ts = datetime.now(timezone.utc)
        trades = [
            Trade(order_id="1", symbol="BTC/USD", side=OrderSide.BUY,
                  quantity=1.0, price=100.0, timestamp=ts, pnl=-50.0),
            Trade(order_id="2", symbol="BTC/USD", side=OrderSide.BUY,
                  quantity=1.0, price=100.0, timestamp=ts, pnl=-30.0),
        ]
        equity = [100000.0, 99950.0, 99920.0]
        metrics = PerformanceMetrics.calculate(trades, equity)

        assert metrics["total_trades"] == 2
        assert metrics["win_rate_pct"] == 0.0
        assert metrics["profit_factor"] == 0.0
        assert metrics["avg_win"] == 0.0
        assert metrics["avg_loss"] == 40.0
        assert metrics["expectancy"] < 0

    def test_mixed_trades(self):
        """Mixed winning and losing trades."""
        ts = datetime.now(timezone.utc)
        trades = [
            Trade(order_id="1", symbol="BTC/USD", side=OrderSide.BUY,
                  quantity=1.0, price=100.0, timestamp=ts, pnl=100.0),
            Trade(order_id="2", symbol="BTC/USD", side=OrderSide.BUY,
                  quantity=1.0, price=100.0, timestamp=ts, pnl=-50.0),
            Trade(order_id="3", symbol="BTC/USD", side=OrderSide.BUY,
                  quantity=1.0, price=100.0, timestamp=ts, pnl=50.0),
            Trade(order_id="4", symbol="BTC/USD", side=OrderSide.BUY,
                  quantity=1.0, price=100.0, timestamp=ts, pnl=-20.0),
        ]
        equity = [100000.0, 100100.0, 100050.0, 100100.0, 100080.0]
        metrics = PerformanceMetrics.calculate(trades, equity)

        assert metrics["total_trades"] == 4
        assert metrics["win_rate_pct"] == 50.0
        assert metrics["profit_factor"] == (100.0 + 50.0) / (50.0 + 20.0)  # 150/70 ≈ 2.14
        assert metrics["avg_win"] == 75.0
        assert metrics["avg_loss"] == 35.0
        assert 0 < metrics["sharpe_ratio"]  # positive since equity goes up

    def test_max_drawdown(self):
        """Max drawdown calculation."""
        equity = [100000.0, 110000.0, 105000.0, 95000.0, 98000.0, 92000.0, 100000.0]
        metrics = PerformanceMetrics.calculate([], equity)

        # Peak = 110000, trough = 92000, drawdown = (92000-110000)/110000 ≈ -16.36%
        assert metrics["max_drawdown_pct"] > 0
        assert metrics["max_drawdown_pct"] == pytest.approx(16.36, abs=0.1)

    def test_total_return_positive(self):
        """Total return percentage."""
        equity = [100000.0, 110000.0, 120000.0]
        metrics = PerformanceMetrics.calculate([], equity)

        assert metrics["total_return_pct"] == pytest.approx(20.0)  # (120000-100000)/100000 * 100

    def test_total_return_negative(self):
        """Total return negative."""
        equity = [100000.0, 90000.0, 80000.0]
        metrics = PerformanceMetrics.calculate([], equity)

        assert metrics["total_return_pct"] == pytest.approx(-20.0)

    def test_equity_curve_to_df(self):
        """Equity curve converted to DataFrame."""
        equity = [100000.0, 101000.0, 102000.0]
        timestamps = [
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 2, tzinfo=timezone.utc),
            datetime(2024, 1, 3, tzinfo=timezone.utc),
        ]

        df = PerformanceMetrics.equity_curve_to_df(equity, timestamps)
        assert list(df.columns) == ["equity", "timestamp"]
        assert len(df) == 3
        assert df["equity"].iloc[-1] == 102000.0

    def test_equity_curve_to_df_no_timestamps(self):
        """Equity curve without timestamps uses current time."""
        equity = [100000.0, 101000.0]
        df = PerformanceMetrics.equity_curve_to_df(equity)
        assert len(df) == 2
        assert "timestamp" in df.columns

    def test_plot_equity_curve(self):
        """Plot function returns path or None."""
        import os

        equity = [100000.0, 101000.0, 102000.0, 101500.0, 103000.0]
        save_path = os.path.expanduser("~/trading-bot/test_equity_curve.png")

        result = PerformanceMetrics.plot_equity_curve(
            equity,
            title="Test",
            save_path=save_path,
        )

        if result is not None:
            assert os.path.exists(result)
            os.remove(result)

    def test_sharpe_and_sortino(self):
        """Sharpe and Sortino ratios with realistic data."""
        # Steady upward equity curve
        equity = [100000.0]
        for i in range(1, 252):  # 1 year of daily data
            equity.append(equity[-1] * 1.001)  # 0.1% daily return

        metrics = PerformanceMetrics.calculate([], equity)
        assert metrics["sharpe_ratio"] > 0.5
        assert metrics["sortino_ratio"] > 0.5
