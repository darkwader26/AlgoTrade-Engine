"""Exchange connector abstraction and paper trading simulation."""

from __future__ import annotations

import random
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal

from core.models import (
    OHLCV,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Trade,
)


def _slippage_direction(side: OrderSide) -> int:
    """Return +1 for BUY (price increase), -1 for SELL (price decrease)."""
    return 1 if side == OrderSide.BUY else -1


class Exchange(ABC):
    """Abstract base class for exchange connectors."""

    @abstractmethod
    def fetch_ohlcv(self, symbol: str, interval: str, limit: int = 100) -> list[OHLCV]:
        """Fetch historical OHLCV data."""
        ...

    @abstractmethod
    def create_order(self, order: Order) -> Order:
        """Submit an order to the exchange."""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        ...

    @abstractmethod
    def fetch_order(self, order_id: str) -> Order | None:
        """Get the current state of an order."""
        ...

    @abstractmethod
    def fetch_position(self, symbol: str) -> Position | None:
        """Get the current position for a symbol."""
        ...

    @abstractmethod
    def fetch_balance(self) -> float:
        """Get the available cash balance."""
        ...


class PaperExchange(Exchange):
    """Paper/simulated exchange that fills orders with configurable
    latency, slippage, and fees. Thread-safe.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission: float = 0.001,  # 0.1%
        slippage: float = 0.0005,  # 0.05%
        latency_ms: float = 50.0,
        partial_fill_prob: float = 0.1,
    ) -> None:
        self._lock = threading.Lock()
        self._cash: float = initial_capital
        self._initial_capital: float = initial_capital
        self._commission: float = commission
        self._slippage: float = slippage
        self._latency_ms: float = latency_ms
        self._partial_fill_prob: float = partial_fill_prob

        # Market data: symbol -> current price
        self._last_prices: dict[str, float] = {}

        # Storage
        self._orders: dict[str, Order] = {}
        self._trades: list[Trade] = []
        self._positions: dict[str, Position] = {}

    # ---- Market data helpers ----

    def update_market_price(self, symbol: str, price: float) -> None:
        """Simulate an incoming market price update. Thread-safe."""
        with self._lock:
            old_price = self._last_prices.get(symbol, price)
            self._last_prices[symbol] = price
            # Update existing position's mark-to-market
            pos = self._positions.get(symbol)
            if pos is not None:
                pos.mark_to_market(price)

        # Check limit orders outside the lock scope to avoid reentrancy issues
        self._check_limit_orders(symbol, price, old_price)

    def _simulate_latency(self) -> None:
        """Simulate exchange latency."""
        if self._latency_ms > 0:
            time.sleep(self._latency_ms / 1000.0)

    # ---- Exchange interface ----

    def fetch_ohlcv(
        self, symbol: str, interval: str, limit: int = 100
    ) -> list[OHLCV]:
        self._simulate_latency()
        return []

    def create_order(self, order: Order) -> Order:
        """Submit an order to the paper exchange."""
        self._simulate_latency()
        with self._lock:
            if order.id in self._orders:
                raise ValueError(f"Order {order.id} already exists")
            new_order = Order(
                id=order.id,
                symbol=order.symbol,
                side=order.side,
                type=order.type,
                quantity=order.quantity,
                price=order.price,
                status=OrderStatus.PENDING,
                timestamp=order.timestamp,
            )
            self._orders[new_order.id] = new_order

        # Attempt immediate fill for MARKET orders
        if order.type == OrderType.MARKET:
            return self._fill_order(order.id)
        elif order.type == OrderType.LIMIT:
            with self._lock:
                current_price = self._last_prices.get(order.symbol)
            if current_price is not None:
                self._check_limit_orders(
                    order.symbol, current_price, current_price
                )

        with self._lock:
            return self._orders.get(order.id) or order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        self._simulate_latency()
        with self._lock:
            order = self._orders.get(order_id)
            if order is None or order.status not in (
                OrderStatus.PENDING,
                OrderStatus.PARTIAL,
            ):
                return False
            order.status = OrderStatus.CANCELLED
            return True

    def fetch_order(self, order_id: str) -> Order | None:
        self._simulate_latency()
        with self._lock:
            return self._orders.get(order_id)

    def fetch_position(self, symbol: str) -> Position | None:
        self._simulate_latency()
        with self._lock:
            return self._positions.get(symbol)

    def fetch_balance(self) -> float:
        self._simulate_latency()
        with self._lock:
            return self._cash

    # ---- Internal fill logic ----

    def _fill_order(self, order_id: str) -> Order:
        """Fill (or partially fill) an order after validation."""
        with self._lock:
            order = self._orders.get(order_id)
            if order is None:
                return None  # type: ignore[return-value]
            if order.status != OrderStatus.PENDING:
                return order

            symbol = order.symbol
            current_price = self._last_prices.get(symbol)

            # Determine fill price with slippage
            if order.type == OrderType.MARKET:
                if current_price is None:
                    order.status = OrderStatus.REJECTED
                    return order
                direction = _slippage_direction(order.side)
                fill_price = current_price * (1 + self._slippage * direction)
                fill_price = self._quantize(fill_price)
            else:  # LIMIT
                fill_price = order.price  # type: ignore[assignment]

            # Determine fill quantity (partial fill possible)
            if random.random() < self._partial_fill_prob:
                fill_qty = self._quantize(
                    order.quantity * random.uniform(0.1, 0.9)
                )
                if fill_qty <= 0:
                    fill_qty = self._quantize(order.quantity * 0.5)
            else:
                fill_qty = order.quantity

            # Validate against available balance/position
            if order.side == OrderSide.BUY:
                cost = fill_qty * fill_price
                total_cost = cost + cost * self._commission
                if total_cost > self._cash:
                    order.status = OrderStatus.REJECTED
                    return order
            else:  # SELL
                pos = self._positions.get(symbol)
                available = pos.quantity if pos else 0.0
                if fill_qty > available:
                    order.status = OrderStatus.REJECTED
                    return order

            # Execute the fill
            fee = self._quantize(fill_qty * fill_price * self._commission)
            trade_pnl = 0.0

            if order.side == OrderSide.BUY:
                self._cash -= fill_qty * fill_price + fee
                pos = self._positions.setdefault(
                    symbol, Position(symbol=symbol)
                )
                total_cost_before = pos.entry_price * pos.quantity
                new_cost = fill_qty * fill_price
                pos.quantity += fill_qty
                pos.entry_price = (
                    (total_cost_before + new_cost) / pos.quantity
                    if pos.quantity > 0
                    else 0.0
                )
                pos.current_price = fill_price
                pos.update_unrealized_pnl()
            else:  # SELL
                self._cash += fill_qty * fill_price - fee
                pos = self._positions.get(symbol)
                if pos:
                    trade_pnl = (fill_price - pos.entry_price) * fill_qty
                    pos.realized_pnl += trade_pnl
                    pos.quantity -= fill_qty
                    if pos.quantity <= 0:
                        del self._positions[symbol]
                    else:
                        pos.current_price = fill_price
                        pos.update_unrealized_pnl()

            # Record the trade
            trade = Trade(
                order_id=order.id,
                symbol=symbol,
                side=order.side,
                quantity=fill_qty,
                price=fill_price,
                timestamp=datetime.now(timezone.utc),
                pnl=trade_pnl,
            )
            self._trades.append(trade)

            # Update order status
            order.filled_qty = fill_qty
            order.avg_price = fill_price
            if fill_qty >= order.quantity:
                order.status = OrderStatus.FILLED
            else:
                order.status = OrderStatus.PARTIAL

            return order

    def _check_limit_orders(
        self, symbol: str, new_price: float, old_price: float
    ) -> None:
        """Check if any pending LIMIT orders should be filled based on price move."""
        with self._lock:
            fillable: list[str] = []
            for oid, order in self._orders.items():
                if (
                    order.symbol == symbol
                    and order.type == OrderType.LIMIT
                    and order.status == OrderStatus.PENDING
                    and order.price is not None
                ):
                    price = order.price
                    if order.side == OrderSide.BUY and old_price > price >= new_price:
                        fillable.append(oid)
                    elif (
                        order.side == OrderSide.SELL
                        and old_price < price <= new_price
                    ):
                        fillable.append(oid)

        for oid in fillable:
            self._fill_order(oid)

    # ---- Public accessors ----

    def get_trades(self) -> list[Trade]:
        with self._lock:
            return list(self._trades)

    def get_positions(self) -> dict[str, Position]:
        with self._lock:
            return {s: Position(**p.__dict__) for s, p in self._positions.items()}

    def get_cash(self) -> float:
        with self._lock:
            return self._cash

    @staticmethod
    def _quantize(value: float) -> float:
        return float(
            Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        )
