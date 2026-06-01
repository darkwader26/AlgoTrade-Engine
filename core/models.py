"""Data models for trading bot."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import NamedTuple


class OHLCV(NamedTuple):
    """Represents a single candlestick/bar of market data."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class OrderSide(str, Enum):
    """Order side — BUY or SELL."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type — MARKET or LIMIT."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    """Lifecycle status of an order."""

    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """A single order submitted to the exchange."""

    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: float | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    status: OrderStatus = OrderStatus.PENDING
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    filled_qty: float = 0.0
    avg_price: float | None = None


@dataclass
class Position:
    """An open position in a symbol."""

    symbol: str
    quantity: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    def update_unrealized_pnl(self) -> None:
        """Recalculate unrealized PnL from current price."""
        if self.quantity == 0:
            self.unrealized_pnl = 0.0
        else:
            self.unrealized_pnl = (self.current_price - self.entry_price) * self.quantity

    def mark_to_market(self, price: float) -> None:
        """Update current price and recalculate unrealized PnL."""
        self.current_price = price
        self.update_unrealized_pnl()


@dataclass
class Trade:
    """A fully or partially filled trade resulting from an order."""

    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    pnl: float = 0.0


@dataclass
class Portfolio:
    """Aggregate portfolio state."""

    cash: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    total_equity: float = 0.0
    total_pnl: float = 0.0

    def update_equity(self) -> None:
        """Recalculate total equity = cash + sum of position market values."""
        position_value = sum(
            pos.quantity * pos.current_price for pos in self.positions.values()
        )
        self.total_equity = self.cash + position_value

    def update_pnl(self) -> None:
        """Recalculate total PnL = realized + unrealized across all positions."""
        realized = sum(pos.realized_pnl for pos in self.positions.values())
        unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        self.total_pnl = realized + unrealized
