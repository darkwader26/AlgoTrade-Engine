"""Trend Following strategy using EMA crossovers and ADX confirmation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.config import TradingConfig
from core.models import OHLCV, Order, OrderSide, OrderType, Position
from strategies.base import Strategy


@dataclass
class TrendFollowingConfig:
    """Configuration for the Trend Following strategy."""

    ema_fast: int = 12
    ema_slow: int = 26
    adx_threshold: float = 25.0
    atr_stop_multiplier: float = 2.0
    atr_period: int = 14
    adx_period: int = 14
    initial_capital_frac: float = 0.1  # fraction of initial capital per trade


class TrendFollowingStrategy(Strategy):
    """Trend Following strategy.

    Uses EMA crossovers (fast=12, slow=26) and ADX (>25 for trend strength).
    Enters on golden cross (fast EMA crosses above slow EMA) with ADX confirmation.
    Exits on death cross or trailing stop (ATR-based).
    """

    def __init__(self, config: TrendFollowingConfig | None = None) -> None:
        self._config = config or TrendFollowingConfig()
        self._symbols: list[str] = []
        self._positions: dict[str, Position] = {}
        self._prev_ema_fast: dict[str, float] = {}
        self._prev_ema_slow: dict[str, float] = {}
        self._trading_config: TradingConfig | None = None
        self._metrics: dict[str, Any] = {}

    # --- Strategy identity --------------------------------------------------

    @property
    def id(self) -> str:
        return "trend_following"

    @property
    def name(self) -> str:
        return "Trend Following"

    @property
    def symbols(self) -> list[str]:
        return self._symbols

    @symbols.setter
    def symbols(self, value: list[str]) -> None:
        self._symbols = value

    @property
    def timeframe(self) -> str:
        return "1h"

    # --- Lifecycle ----------------------------------------------------------

    def on_init(self, config: TradingConfig) -> None:
        self._trading_config = config
        if not self._symbols:
            self._symbols = list(config.symbols)
        # Initialise previous EMA tracking
        for sym in self._symbols:
            self._prev_ema_fast[sym] = 0.0
            self._prev_ema_slow[sym] = 0.0

    def on_bar(
        self,
        symbol: str,
        ohlcv: list[OHLCV],
        features: dict[str, Any],
        signals: dict[str, Any],
    ) -> list[Order]:
        if symbol not in self._symbols:
            return []

        if len(ohlcv) < 3:
            return []

        current_price = ohlcv[-1].close

        # Extract feature values
        ema_fast_val = features.get(f"ema_{self._config.ema_fast}", features.get("ema_12"))
        ema_slow_val = features.get(f"ema_{self._config.ema_slow}", features.get("ema_26"))
        adx_val = features.get(f"adx_{self._config.adx_period}", features.get("adx_14"))
        atr_val = features.get(f"atr_{self._config.atr_period}", features.get("atr_14"))

        # If features aren't named exactly, try to find them
        if ema_fast_val is None or ema_slow_val is None or adx_val is None or atr_val is None:
            # Try computing from raw data
            close_prices = np.array([b.close for b in ohlcv], dtype=np.float64)
            high_prices = np.array([b.high for b in ohlcv], dtype=np.float64)
            low_prices = np.array([b.low for b in ohlcv], dtype=np.float64)

            from features.indicators import adx as adx_func
            from features.indicators import atr as atr_func
            from features.indicators import ema as ema_func

            if ema_fast_val is None:
                ema_fast_arr = ema_func(close_prices, self._config.ema_fast)
                ema_fast_val = ema_fast_arr[-1] if not np.isnan(ema_fast_arr[-1]) else None
            if ema_slow_val is None:
                ema_slow_arr = ema_func(close_prices, self._config.ema_slow)
                ema_slow_val = ema_slow_arr[-1] if not np.isnan(ema_slow_arr[-1]) else None
            if adx_val is None:
                adx_arr = adx_func(high_prices, low_prices, close_prices, self._config.adx_period)
                adx_val = adx_arr[-1] if not np.isnan(adx_arr[-1]) else None
            if atr_val is None:
                atr_arr = atr_func(high_prices, low_prices, close_prices, self._config.atr_period)
                atr_val = atr_arr[-1] if not np.isnan(atr_arr[-1]) else None

        if ema_fast_val is None or ema_slow_val is None or adx_val is None or atr_val is None:
            return []

        orders: list[Order] = []
        pos = self._positions.get(symbol)

        prev_fast = self._prev_ema_fast.get(symbol, ema_fast_val)
        prev_slow = self._prev_ema_slow.get(symbol, ema_slow_val)

        # Detect crossover
        golden_cross = prev_fast <= prev_slow and ema_fast_val > ema_slow_val
        death_cross = prev_fast >= prev_slow and ema_fast_val < ema_slow_val

        self._prev_ema_fast[symbol] = ema_fast_val
        self._prev_ema_slow[symbol] = ema_slow_val

        # --- Entry logic ---
        if pos is None or pos.quantity == 0:
            if golden_cross and adx_val >= self._config.adx_threshold:
                # Calculate position size
                capital = (self._trading_config.initial_capital
                           if self._trading_config else 100_000.0)
                qty = capital * self._config.initial_capital_frac / current_price
                qty = round(qty, 6)
                if qty > 0:
                    orders.append(
                        Order(
                            symbol=symbol,
                            side=OrderSide.BUY,
                            type=OrderType.MARKET,
                            quantity=qty,
                        )
                    )
        # --- Exit logic ---
        else:
            is_long = pos.quantity > 0
            if is_long:
                if death_cross:
                    orders.append(
                        Order(
                            symbol=symbol,
                            side=OrderSide.SELL,
                            type=OrderType.MARKET,
                            quantity=abs(pos.quantity),
                        )
                    )
                else:
                    # Trailing stop based on ATR
                    stop_price = current_price - self._config.atr_stop_multiplier * atr_val
                    if hasattr(self, '_trailing_stops') and symbol in self._trailing_stops:
                        self._trailing_stops[symbol] = max(self._trailing_stops[symbol], stop_price)
                    else:
                        if not hasattr(self, '_trailing_stops'):
                            self._trailing_stops = {}
                        self._trailing_stops[symbol] = stop_price

                    if hasattr(self, '_trailing_stops') and current_price <= self._trailing_stops.get(symbol, 0):
                        orders.append(
                            Order(
                                symbol=symbol,
                                side=OrderSide.SELL,
                                type=OrderType.MARKET,
                                quantity=abs(pos.quantity),
                            )
                        )
                        self._trailing_stops.pop(symbol, None)

        return orders

    def on_order_filled(self, order: Order) -> None:
        pass

    def on_order_cancelled(self, order: Order) -> None:
        pass

    def on_position_update(self, position: Position) -> None:
        self._positions[position.symbol] = position

    def on_stop(self) -> None:
        self._prev_ema_fast.clear()
        self._prev_ema_slow.clear()
        if hasattr(self, '_trailing_stops'):
            self._trailing_stops.clear()

    @property
    def metrics(self) -> dict[str, Any]:
        return dict(self._metrics)

    @property
    def config(self) -> TrendFollowingConfig:
        return self._config
