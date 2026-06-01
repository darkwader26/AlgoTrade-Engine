"""Abstract strategy base class for the trading bot."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.config import TradingConfig
from core.models import OHLCV, Order, Position


class Strategy(ABC):
    """Abstract base class for all trading strategies.

    Subclasses must implement the abstract methods and define the
    class-level attributes ``id``, ``name``, ``symbols``, and ``timeframe``.
    """

    # --- Subclass-overridable identity attributes ---------------------------

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique strategy identifier."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""
        ...

    @property
    @abstractmethod
    def symbols(self) -> list[str]:
        """List of symbols this strategy trades."""
        ...

    @property
    @abstractmethod
    def timeframe(self) -> str:
        """Timeframe this strategy operates on (e.g. '1m', '5m', '1h')."""
        ...

    # --- Lifecycle hooks ----------------------------------------------------

    def on_init(self, config: TradingConfig) -> None:
        """Called once at startup with the global trading configuration.

        Override to set up indicators, load state, etc.
        """
        pass

    @abstractmethod
    def on_bar(
        self,
        symbol: str,
        ohlcv: list[OHLCV],
        features: dict[str, Any],
        signals: dict[str, Any],
    ) -> list[Order]:
        """Called on each new bar for every symbol the strategy trades.

        Parameters
        ----------
        symbol : str
            The symbol being processed.
        ohlcv : list[OHLCV]
            Recent OHLCV bars (at least ``timeframe`` worth).
        features : dict
            Computed feature values (e.g. ``{'ema_12': ..., 'rsi_14': ...}``).
        signals : dict
            Aggregated signal values (e.g. ``{'direction': 1, 'strength': 'strong'}``).

        Returns
        -------
        list[Order]
            Orders to submit.  Return an empty list if no action.
        """
        ...

    def on_order_filled(self, order: Order) -> None:
        """Called when an order submitted by this strategy is filled."""
        pass

    def on_order_cancelled(self, order: Order) -> None:
        """Called when an order submitted by this strategy is cancelled."""
        pass

    def on_position_update(self, position: Position) -> None:
        """Called when a position managed by this strategy changes."""
        pass

    def on_stop(self) -> None:
        """Called during shutdown.  Override to clean up resources."""
        pass

    # --- Metrics ------------------------------------------------------------

    @property
    def metrics(self) -> dict[str, Any]:
        """Performance metrics calculated by the engine."""
        return {}
