"""Performance calculation and visualisation for trading strategies."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from core.models import Trade


class PerformanceMetrics:
    """Calculates and visualises performance metrics from trades and equity curve."""

    @staticmethod
    def calculate(
        trades: list[Trade],
        equity_curve: list[float],
        timestamps: list[datetime] | None = None,
        risk_free_rate: float = 0.02,
    ) -> dict[str, Any]:
        """Compute comprehensive performance metrics.

        Parameters
        ----------
        trades : list[Trade]
            All completed trades.
        equity_curve : list[float]
            Equity values over time (one per bar/period).
        timestamps : list[datetime] or None
            Optional timestamps matching equity_curve.
        risk_free_rate : float
            Annual risk-free rate (default 2%).

        Returns
        -------
        dict
            Metric name -> value.
        """
        metrics: dict[str, Any] = {}

        total_trades = len(trades)
        metrics["total_trades"] = total_trades

        # --- Equity-based metrics (computed regardless of trades) ---
        # Total Return
        if len(equity_curve) < 2:
            metrics["total_return_pct"] = 0.0
            metrics["cagr"] = 0.0
        else:
            start_equity = equity_curve[0]
            end_equity = equity_curve[-1]
            total_return = (
                (end_equity - start_equity) / start_equity if start_equity > 0 else 0.0
            )
            metrics["total_return_pct"] = total_return * 100.0

            # CAGR
            if timestamps and len(timestamps) >= 2:
                t_start = timestamps[0]
                t_end = timestamps[-1]
                years = (t_end - t_start).total_seconds() / (365.25 * 86400)
                if years > 0:
                    cagr = (end_equity / start_equity) ** (1.0 / years) - 1.0
                    metrics["cagr"] = cagr * 100.0
                else:
                    metrics["cagr"] = 0.0
            else:
                metrics["cagr"] = 0.0

        # Sharpe Ratio (annualized)
        if len(equity_curve) > 1:
            eq_arr = np.array(equity_curve, dtype=np.float64)
            returns = np.diff(eq_arr) / eq_arr[:-1]
            returns = returns[~np.isnan(returns)]

            if len(returns) > 1:
                excess_returns = returns - (risk_free_rate / 252)
                sharpe = np.mean(excess_returns) / (np.std(returns, ddof=1) + 1e-10)
                metrics["sharpe_ratio"] = sharpe * math.sqrt(252)
            else:
                metrics["sharpe_ratio"] = 0.0
        else:
            metrics["sharpe_ratio"] = 0.0

        # Sortino Ratio
        if len(equity_curve) > 1:
            eq_arr = np.array(equity_curve, dtype=np.float64)
            returns = np.diff(eq_arr) / eq_arr[:-1]
            returns = returns[~np.isnan(returns)]

            if len(returns) > 1:
                downside = returns[returns < 0]
                downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-10
                excess_returns = returns - (risk_free_rate / 252)
                sortino = np.mean(excess_returns) / (downside_std + 1e-10)
                metrics["sortino_ratio"] = sortino * math.sqrt(252)
            else:
                metrics["sortino_ratio"] = 0.0
        else:
            metrics["sortino_ratio"] = 0.0

        # Max Drawdown
        if len(equity_curve) > 1:
            eq_arr = np.array(equity_curve, dtype=np.float64)
            running_max = np.maximum.accumulate(eq_arr)
            drawdowns = (eq_arr - running_max) / running_max
            max_dd = np.min(drawdowns)
            metrics["max_drawdown_pct"] = abs(max_dd) * 100.0
        else:
            metrics["max_drawdown_pct"] = 0.0

        # --- Trade-based metrics (need at least one trade) ---
        if total_trades == 0:
            metrics["win_rate_pct"] = 0.0
            metrics["profit_factor"] = 0.0
            metrics["avg_win"] = 0.0
            metrics["avg_loss"] = 0.0
            metrics["expectancy"] = 0.0
            metrics["avg_trade_duration"] = 0.0
            return metrics

        # Win/loss breakdown
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        win_count = len(winning_trades)
        loss_count = len(losing_trades)

        metrics["win_rate_pct"] = (win_count / total_trades) * 100 if total_trades > 0 else 0.0

        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))

        metrics["profit_factor"] = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )

        metrics["avg_win"] = gross_profit / win_count if win_count > 0 else 0.0
        metrics["avg_loss"] = gross_loss / loss_count if loss_count > 0 else 0.0

        # Expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        win_rate_dec = metrics["win_rate_pct"] / 100.0
        loss_rate_dec = 1.0 - win_rate_dec
        metrics["expectancy"] = (
            win_rate_dec * metrics["avg_win"] - loss_rate_dec * metrics["avg_loss"]
        )

        # Avg Trade Duration
        if timestamps and len(trades) >= 2:
            durations = []
            for i in range(1, len(trades)):
                dur = (trades[i].timestamp - trades[i - 1].timestamp).total_seconds()
                durations.append(dur)
            metrics["avg_trade_duration"] = (
                np.mean(durations) / 3600 if durations else 0.0
            )
        else:
            metrics["avg_trade_duration"] = 0.0

        return metrics

    # ------------------------------------------------------------------
    #  Plotting
    # ------------------------------------------------------------------

    @staticmethod
    def plot_equity_curve(
        equity_curve: list[float],
        title: str = "Equity Curve",
        save_path: str | None = None,
    ) -> str | None:
        """Plot the equity curve and save to a PNG file.

        Parameters
        ----------
        equity_curve : list[float]
            Equity values over time.
        title : str
            Plot title.
        save_path : str or None
            Path to save the image.  Defaults to ``~/trading-bot/equity_curve.png``.

        Returns
        -------
        str or None
            Path to saved file, or None if matplotlib is not available.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        if save_path is None:
            import os
            save_path = os.path.expanduser("~/trading-bot/equity_curve.png")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(equity_curve, linewidth=2, color="blue")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Bar")
        ax.set_ylabel("Equity ($)")
        ax.grid(True, alpha=0.3)

        if equity_curve:
            ax.axhline(y=equity_curve[0], color="gray", linestyle="--", alpha=0.5)

        plt.tight_layout()
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        return save_path

    # ------------------------------------------------------------------
    #  DataFrame conversion
    # ------------------------------------------------------------------

    @staticmethod
    def equity_curve_to_df(
        equity_curve: list[float],
        timestamps: list[datetime] | None = None,
    ) -> pd.DataFrame:
        """Convert an equity curve to a pandas DataFrame.

        Parameters
        ----------
        equity_curve : list[float]
            Equity values.
        timestamps : list[datetime] or None
            Timestamps for each equity value.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``equity`` and optionally ``timestamp``.
        """
        if timestamps is None:
            timestamps = [
                datetime.now(timezone.utc) for _ in range(len(equity_curve))
            ]

        df = pd.DataFrame({"equity": equity_curve, "timestamp": timestamps})
        return df
