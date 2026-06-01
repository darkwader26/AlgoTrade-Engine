"""Risk management system for the trading bot."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal

from core.config import TradingConfig
from core.models import Order, OrderSide, Portfolio, Position


class RiskManager:
    """Validates orders against risk rules and provides position sizing helpers.

    Thread-safe: all public methods acquire a lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Track daily loss (reset each day)
        self._daily_realized_pnl: dict[str, float] = {}  # symbol -> realized PnL
        self._current_day: int = datetime.now(timezone.utc).timetuple().tm_yday

    # ------------------------------------------------------------------
    #  Order validation
    # ------------------------------------------------------------------

    def check_order(
        self,
        order: Order,
        portfolio: Portfolio,
        config: TradingConfig,
    ) -> tuple[bool, str]:
        """Validate if *order* is allowed given current portfolio and config.

        Returns
        -------
        (allowed: bool, reason: str)
        """
        with self._lock:
            # 1. Max position size check
            max_pos = config.max_position_size
            if max_pos > 0:
                order_value = order.quantity * (order.price or portfolio.total_equity)
                if order_value > max_pos:
                    return False, (
                        f"Order value {order_value:.2f} exceeds max position size "
                        f"{max_pos:.2f}"
                    )

            # 2. Max open positions check
            open_count = sum(
                1 for p in portfolio.positions.values() if p.quantity != 0
            )
            # Count if this is a new position (not adding to existing)
            existing = portfolio.positions.get(order.symbol)
            if existing is None or existing.quantity == 0:
                open_count += 1  # this order would open a new position

            # 3. Concentration limit: no single position > 25% of equity
            equity = portfolio.total_equity
            concentration_limit = equity * 0.25 if equity > 0 else float("inf")
            if order.side == OrderSide.BUY:
                current_value = (
                    existing.quantity * existing.current_price
                    if existing and existing.quantity != 0
                    else 0.0
                )
                new_value = order.quantity * (order.price or equity)
                total_value = current_value + new_value
                if total_value > concentration_limit:
                    return False, (
                        f"Concentration limit exceeded: "
                        f"{total_value:.2f} > {concentration_limit:.2f}"
                    )

            # 4. Daily loss limit check (5% of initial capital per day)
            self._check_day_rollover()
            daily_loss_limit = config.initial_capital * 0.05
            daily_loss = abs(self._daily_realized_pnl.get(order.symbol, 0.0))
            if daily_loss > daily_loss_limit:
                return False, (
                    f"Daily loss limit reached for {order.symbol}: "
                    f"{daily_loss:.2f} > {daily_loss_limit:.2f}"
                )

            return True, "order allowed"

    # ------------------------------------------------------------------
    #  Position sizing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def apply_kelly(
        position: Position,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """Calculate Kelly Criterion position sizing fraction.

        ``f* = (p * avg_win - q * avg_loss) / (avg_win * avg_loss)``

        where:
        - p = win_rate
        - q = 1 - win_rate
        - avg_win = average winning trade PnL (as fraction of risk)
        - avg_loss = average losing trade PnL (as fraction)

        Returns
        -------
        float
            Fraction of capital to allocate (clipped to [0, 0.25]).
        """
        if avg_loss <= 0 or win_rate <= 0:
            return 0.0

        if win_rate >= 1.0:
            return 0.25  # cap at 25%

        q = 1.0 - win_rate
        # Kelly formula: (p * b - q) / b, where b = avg_win / avg_loss
        b = avg_win / avg_loss if avg_loss > 0 else 0.0
        if b <= 0:
            return 0.0

        kelly = (win_rate * b - q) / b

        # Clip to sensible range [0, 0.25]
        return max(0.0, min(kelly, 0.25))

    @staticmethod
    def calculate_stop_loss(
        entry_price: float,
        side: OrderSide,
        atr_value: float,
        multiplier: float = 2.0,
    ) -> float:
        """Calculate stop-loss price.

        For long positions: entry_price - multiplier * atr_value
        For short positions: entry_price + multiplier * atr_value
        """
        if side == OrderSide.BUY:
            return RiskManager._quantize(entry_price - multiplier * atr_value)
        else:
            return RiskManager._quantize(entry_price + multiplier * atr_value)

    @staticmethod
    def calculate_take_profit(
        entry_price: float,
        side: OrderSide,
        risk_reward_ratio: float = 2.0,
    ) -> float:
        """Calculate take-profit price.

        For long positions: entry_price + risk_reward_ratio * atr_value equivalent
        Uses risk_reward_ratio proportionally.
        """
        # We need a risk amount to compute the take profit.  If no ATR is
        # available, use a default 2% of entry price as risk.
        risk_amount = entry_price * 0.02  # 2% risk
        if side == OrderSide.BUY:
            return RiskManager._quantize(entry_price + risk_reward_ratio * risk_amount)
        else:
            return RiskManager._quantize(entry_price - risk_reward_ratio * risk_amount)

    # ------------------------------------------------------------------
    #  Drawdown check
    # ------------------------------------------------------------------

    @staticmethod
    def check_drawdown(
        portfolio: Portfolio,
        max_drawdown_pct: float,
    ) -> bool:
        """Check if drawdown exceeds the maximum allowed.

        Parameters
        ----------
        portfolio : Portfolio
            Current portfolio state.
        max_drawdown_pct : float
            Maximum allowable drawdown as fraction (e.g. 0.20 for 20%).

        Returns
        -------
        bool
            True if stopped out (drawdown exceeded max).
        """
        # Track peak equity internally
        if not hasattr(RiskManager, "_peak_equity"):
            RiskManager._peak_equity = portfolio.total_equity

        RiskManager._peak_equity = max(
            RiskManager._peak_equity, portfolio.total_equity
        )

        if RiskManager._peak_equity == 0:
            return False

        drawdown = 1.0 - (portfolio.total_equity / RiskManager._peak_equity)
        return drawdown > max_drawdown_pct

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _check_day_rollover(self) -> None:
        """Reset daily PnL tracking if the day has changed."""
        today = datetime.now(timezone.utc).timetuple().tm_yday
        if today != self._current_day:
            self._daily_realized_pnl.clear()
            self._current_day = today

    @staticmethod
    def _quantize(value: float) -> float:
        return float(
            Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        )

    @property
    def daily_loss(self) -> dict[str, float]:
        """Current daily realized loss by symbol."""
        with self._lock:
            return dict(self._daily_realized_pnl)
