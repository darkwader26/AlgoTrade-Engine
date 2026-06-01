"""Order lifecycle manager — submit, cancel, track orders and trades."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from core.models import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Portfolio,
    Position,
    Trade,
)

if TYPE_CHECKING:
    from core.exchange import Exchange


class OrderManager:
    """Manages the full lifecycle of orders and trades.

    Delegates execution to an Exchange instance, maintains an
    audit trail of all orders and trades, and computes portfolio
    state. Thread-safe.
    """

    def __init__(self, exchange: Exchange, initial_cash: float = 100_000.0) -> None:
        self._lock = threading.Lock()
        self._exchange: Exchange = exchange
        self._portfolio: Portfolio = Portfolio(
            cash=initial_cash,
            total_equity=initial_cash,
        )
        # order_id -> Order
        self._orders: dict[str, Order] = {}
        # Flat list of all trades
        self._trades: list[Trade] = []

    # ---- Order submission ----

    def submit(self, order: Order) -> Order:
        """Validate and submit an order to the exchange.

        Returns the order as processed by the exchange.
        """
        self._validate_order(order)
        submitted = self._exchange.create_order(order)
        with self._lock:
            self._orders[submitted.id] = submitted
            # Record any trades generated
            self._sync_trades(submitted)
            self._update_portfolio_locked()
        return submitted

    def cancel(self, order_id: str) -> bool:
        """Cancel an open order. Returns True if successfully cancelled."""
        result = self._exchange.cancel_order(order_id)
        if result:
            with self._lock:
                order = self._orders.get(order_id)
                if order and order.status in (
                    OrderStatus.PENDING,
                    OrderStatus.PARTIAL,
                ):
                    order.status = OrderStatus.CANCELLED
        return result

    # ---- Order retrieval ----

    def get_order(self, order_id: str) -> Order | None:
        """Fetch the latest state of an order."""
        # Try remote first for fresh state
        remote = self._exchange.fetch_order(order_id)
        if remote is not None:
            with self._lock:
                self._orders[order_id] = remote
            return remote
        with self._lock:
            return self._orders.get(order_id)

    def get_open_orders(self) -> list[Order]:
        """Return all currently open orders."""
        with self._lock:
            return [
                o
                for o in self._orders.values()
                if o.status in (OrderStatus.PENDING, OrderStatus.PARTIAL)
            ]

    def get_order_history(
        self, symbol: str | None = None, limit: int = 100
    ) -> list[Order]:
        """Return recent order history, optionally filtered by symbol."""
        with self._lock:
            orders = list(self._orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        # Most recent first
        orders.sort(key=lambda o: o.timestamp, reverse=True)
        return orders[:limit]

    # ---- Trade retrieval ----

    def get_trade_history(
        self, symbol: str | None = None, limit: int = 100
    ) -> list[Trade]:
        """Return recent trade history, optionally filtered by symbol."""
        with self._lock:
            trades = list(self._trades)
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        trades.sort(key=lambda t: t.timestamp, reverse=True)
        return trades[:limit]

    # ---- Portfolio ----

    def get_portfolio(self) -> Portfolio:
        """Get current portfolio snapshot."""
        self.update_portfolio()
        with self._lock:
            return Portfolio(
                cash=self._portfolio.cash,
                positions={
                    s: Position(**p.__dict__)
                    for s, p in self._portfolio.positions.items()
                },
                total_equity=self._portfolio.total_equity,
                total_pnl=self._portfolio.total_pnl,
            )

    def update_portfolio(self) -> None:
        """Recalculate portfolio state from exchange positions."""
        with self._lock:
            self._update_portfolio_locked()

    def _update_portfolio_locked(self) -> None:
        """Internal portfolio update (caller must hold lock)."""
        cash = self._exchange.fetch_balance()
        self._portfolio.cash = cash

        # Sync positions from exchange
        symbols_with_positions: set[str] = set()
        for order in self._orders.values():
            if order.symbol:
                symbols_with_positions.add(order.symbol)

        positions: dict[str, Position] = {}
        for symbol in symbols_with_positions:
            pos = self._exchange.fetch_position(symbol)
            if pos is not None and pos.quantity > 0:
                positions[symbol] = Position(**pos.__dict__)

        self._portfolio.positions = positions
        self._portfolio.update_equity()

        # Total PnL = unrealized PnL from open positions + realized PnL from trade history
        unrealized = sum(
            pos.unrealized_pnl for pos in positions.values()
        )
        realized = sum(t.pnl for t in self._trades)
        self._portfolio.total_pnl = realized + unrealized

    # ---- Internal helpers ----

    def _validate_order(self, order: Order) -> None:
        """Basic order validation before submission."""
        if order.quantity <= 0:
            raise ValueError("Order quantity must be positive")
        if order.type == OrderType.LIMIT and (
            order.price is None or order.price <= 0
        ):
            raise ValueError("Limit orders require a positive price")
        if order.side not in (OrderSide.BUY, OrderSide.SELL):
            raise ValueError(f"Invalid order side: {order.side}")

    def _sync_trades(self, order: Order) -> None:
        """Pull trades from the exchange for a completed order.

        Since PaperExchange records trades internally, we read them.
        For a real exchange this would query the order's fill history.
        """
        if hasattr(self._exchange, "get_trades"):
            all_trades = self._exchange.get_trades()  # type: ignore[attr-defined]
            # Only append trades we haven't seen yet
            seen_ids = {t.order_id for t in self._trades}
            for t in all_trades:
                if t.order_id not in seen_ids:
                    self._trades.append(t)
    def _add_trade(self, trade: Trade) -> None:
        """Manually record a trade (used by tests / direct injection)."""
        with self._lock:
            self._trades.append(trade)
