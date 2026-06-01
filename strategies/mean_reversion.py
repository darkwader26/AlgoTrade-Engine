"""Mean Reversion strategy using Bollinger Bands and RSI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.config import TradingConfig
from core.models import OHLCV, Order, OrderSide, OrderType, Position
from strategies.base import Strategy


@dataclass
class MeanReversionConfig:
    """Configuration for the Mean Reversion strategy."""

    bb_period: int = 20
    bb_stddev: float = 2.0
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    trade_fraction: float = 0.1  # fraction of capital per trade


class MeanReversionStrategy(Strategy):
    """Mean Reversion strategy.

    Uses Bollinger Bands + RSI.
    Enters long when price touches lower band and RSI < oversold threshold.
    Enters short when price touches upper band and RSI > overbought threshold.
    Exits when price crosses middle band.
    """

    def __init__(self, config: MeanReversionConfig | None = None) -> None:
        self._config = config or MeanReversionConfig()
        self._symbols: list[str] = []
        self._positions: dict[str, Position] = {}
        self._trading_config: TradingConfig | None = None
        self._metrics: dict[str, Any] = {}

    # --- Identity -----------------------------------------------------------

    @property
    def id(self) -> str:
        return "mean_reversion"

    @property
    def name(self) -> str:
        return "Mean Reversion"

    @property
    def symbols(self) -> list[str]:
        return self._symbols

    @symbols.setter
    def symbols(self, value: list[str]) -> None:
        self._symbols = value

    @property
    def timeframe(self) -> str:
        return "5m"

    # --- Lifecycle ----------------------------------------------------------

    def on_init(self, config: TradingConfig) -> None:
        self._trading_config = config
        if not self._symbols:
            self._symbols = list(config.symbols)

    def on_bar(
        self,
        symbol: str,
        ohlcv: list[OHLCV],
        features: dict[str, Any],
        signals: dict[str, Any],
    ) -> list[Order]:
        if symbol not in self._symbols:
            return []

        if len(ohlcv) < self._config.bb_period:
            return []

        current_price = ohlcv[-1].close

        # Extract feature values
        bb_upper = features.get("bb_upper")
        bb_middle = features.get("bb_middle")
        bb_lower = features.get("bb_lower")
        rsi_val = features.get(f"rsi_{self._config.rsi_period}", features.get("rsi_14"))

        # Fallback: compute from raw data if features not available
        if bb_upper is None or bb_middle is None or bb_lower is None or rsi_val is None:
            close_prices = np.array([b.close for b in ohlcv], dtype=np.float64)
            from features.indicators import bollinger_bands, rsi

            bb_upper_arr, bb_middle_arr, bb_lower_arr = bollinger_bands(
                close_prices, self._config.bb_period, self._config.bb_stddev
            )
            rsi_arr = rsi(close_prices, self._config.rsi_period)

            if bb_upper is None:
                bb_upper = bb_upper_arr[-1] if not np.isnan(bb_upper_arr[-1]) else None
            if bb_middle is None:
                bb_middle = bb_middle_arr[-1] if not np.isnan(bb_middle_arr[-1]) else None
            if bb_lower is None:
                bb_lower = bb_lower_arr[-1] if not np.isnan(bb_lower_arr[-1]) else None
            if rsi_val is None:
                rsi_val = rsi_arr[-1] if not np.isnan(rsi_arr[-1]) else None

        if bb_upper is None or bb_middle is None or bb_lower is None or rsi_val is None:
            return []

        orders: list[Order] = []
        pos = self._positions.get(symbol)
        capital = self._trading_config.initial_capital if self._trading_config else 100_000.0
        qty = lambda: round(capital * self._config.trade_fraction / current_price, 6)

        # --- Exit when price crosses middle band ---
        if pos is not None and pos.quantity != 0:
            if pos.quantity > 0 and current_price >= bb_middle:
                # Long exit: price crossed above middle band
                orders.append(
                    Order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        type=OrderType.MARKET,
                        quantity=abs(pos.quantity),
                    )
                )
                return orders
            elif pos.quantity < 0 and current_price <= bb_middle:
                # Short exit: price crossed below middle band
                orders.append(
                    Order(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        type=OrderType.MARKET,
                        quantity=abs(pos.quantity),
                    )
                )
                return orders

        # --- Entry logic ---
        if pos is None or pos.quantity == 0:
            if current_price <= bb_lower and rsi_val < self._config.rsi_oversold:
                # Oversold — go long
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
            elif current_price >= bb_upper and rsi_val > self._config.rsi_overbought:
                # Overbought — go short
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
        pass

    @property
    def metrics(self) -> dict[str, Any]:
        return dict(self._metrics)

    @property
    def config(self) -> MeanReversionConfig:
        return self._config
