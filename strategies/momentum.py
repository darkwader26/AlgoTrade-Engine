"""Momentum strategy using ROC and OBV divergence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.config import TradingConfig
from core.models import OHLCV, Order, OrderSide, OrderType, Position
from strategies.base import Strategy


@dataclass
class MomentumConfig:
    """Configuration for the Momentum strategy."""

    roc_period: int = 12
    momentum_threshold: float = 5.0
    top_n: int = 2  # number of top/bottom ranked symbols to trade
    trade_fraction: float = 0.1


class MomentumStrategy(Strategy):
    """Momentum strategy.

    Uses ROC (Rate of Change) + OBV divergence.
    Ranks multiple symbols by ROC, trades top/bottom ranked.
    Enters on strong ROC with OBV confirmation (rising price + rising volume).
    Exits when ROC reverses or OBV diverges.
    """

    def __init__(self, config: MomentumConfig | None = None) -> None:
        self._config = config or MomentumConfig()
        self._symbols: list[str] = []
        self._positions: dict[str, Position] = {}
        self._trading_config: TradingConfig | None = None
        self._metrics: dict[str, Any] = {}
        # Store last ROC and OBV values per symbol
        self._prev_roc: dict[str, float] = {}
        self._prev_obv: dict[str, float] = {}
        self._prev_close: dict[str, float] = {}

    # --- Identity -----------------------------------------------------------

    @property
    def id(self) -> str:
        return "momentum"

    @property
    def name(self) -> str:
        return "Momentum"

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
        for sym in self._symbols:
            self._prev_roc[sym] = 0.0
            self._prev_obv[sym] = 0.0
            self._prev_close[sym] = 0.0

    def on_bar(
        self,
        symbol: str,
        ohlcv: list[OHLCV],
        features: dict[str, Any],
        signals: dict[str, Any],
    ) -> list[Order]:
        if symbol not in self._symbols:
            return []

        if len(ohlcv) < self._config.roc_period + 1:
            return []

        current_price = ohlcv[-1].close

        # Extract feature values
        roc_val = features.get(f"roc_{self._config.roc_period}", features.get("roc_12"))
        obv_val = features.get("obv")

        # Fallback: compute from raw data
        if roc_val is None or obv_val is None:
            close_prices = np.array([b.close for b in ohlcv], dtype=np.float64)
            volumes = np.array([b.volume for b in ohlcv], dtype=np.float64)
            from features.indicators import obv as obv_func
            from features.indicators import roc as roc_func

            if roc_val is None:
                roc_arr = roc_func(close_prices, self._config.roc_period)
                roc_val = roc_arr[-1] if not np.isnan(roc_arr[-1]) else None
            if obv_val is None:
                obv_arr = obv_func(close_prices, volumes)
                obv_val = obv_arr[-1] if not np.isnan(obv_arr[-1]) else None

        if roc_val is None or obv_val is None:
            return []

        orders: list[Order] = []
        pos = self._positions.get(symbol)
        capital = self._trading_config.initial_capital if self._trading_config else 100_000.0
        qty = lambda: round(capital * self._config.trade_fraction / current_price, 6)

        prev_obv = self._prev_obv.get(symbol, 0.0)
        prev_close = self._prev_close.get(symbol, 0.0)

        # OBV confirmation: price rising AND OBV rising, or price falling AND OBV falling
        price_up = current_price > prev_close if prev_close > 0 else True
        obv_up = obv_val > prev_obv if prev_obv != 0 else True
        obv_confirms = (price_up and obv_up) or (not price_up and not obv_up)

        # OBV divergence: price up but OBV down, or price down but OBV up
        obv_diverges = (price_up and not obv_up and prev_close > 0 and prev_obv != 0) or \
                       (not price_up and obv_up and prev_close > 0 and prev_obv != 0)

        # Update stored values
        self._prev_roc[symbol] = roc_val
        self._prev_obv[symbol] = obv_val
        self._prev_close[symbol] = current_price

        # --- Exit logic ---
        if pos is not None and pos.quantity != 0:
            should_exit = False
            # ROC reversal
            if pos.quantity > 0 and roc_val < -self._config.momentum_threshold:
                should_exit = True
            elif pos.quantity < 0 and roc_val > self._config.momentum_threshold:
                should_exit = True
            # OBV divergence
            if obv_diverges:
                should_exit = True
            # ROC weakening
            if pos.quantity > 0 and roc_val < 0:
                should_exit = True
            elif pos.quantity < 0 and roc_val > 0:
                should_exit = True

            if should_exit:
                exit_side = OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY
                orders.append(
                    Order(
                        symbol=symbol,
                        side=exit_side,
                        type=OrderType.MARKET,
                        quantity=abs(pos.quantity),
                    )
                )
                return orders

        # --- Entry logic ---
        if (pos is None or pos.quantity == 0) and obv_confirms:
            if roc_val > self._config.momentum_threshold:
                q = qty()
                if q > 0:
                    orders.append(
                        Order(
                            symbol=symbol,
                            side=OrderSide.BUY,
                            type=OrderType.MARKET,
                            quantity=q,
                        )
                    )
            elif roc_val < -self._config.momentum_threshold:
                q = qty()
                if q > 0:
                    orders.append(
                        Order(
                            symbol=symbol,
                            side=OrderSide.SELL,
                            type=OrderType.MARKET,
                            quantity=q,
                        )
                    )

        return orders

    def on_order_filled(self, order: Order) -> None:
        pass

    def on_order_cancelled(self, order: Order) -> None:
        pass

    def on_position_update(self, position: Position) -> None:
        self._positions[position.symbol] = position

    def on_stop(self) -> None:
        self._prev_roc.clear()
        self._prev_obv.clear()
        self._prev_close.clear()

    @property
    def metrics(self) -> dict[str, Any]:
        return dict(self._metrics)

    @property
    def config(self) -> MomentumConfig:
        return self._config
