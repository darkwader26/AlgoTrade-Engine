"""Strategy execution engine — the brain of the trading system."""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any

import pandas as pd

from core.config import TradingConfig
from core.models import (
    OHLCV,
    Order,
)
from core.order_manager import OrderManager
from features.feature_pipeline import FeaturePipeline
from features.signal_generator import SignalGenerator
from strategies.base import Strategy
from strategies.performance import PerformanceMetrics
from strategies.risk_manager import RiskManager


class StrategyEngine:
    """Orchestrates strategy execution, risk management, and order submission.

    For each new bar:
      1. Compute features via ``FeaturePipeline``
      2. Generate consensus signals via ``SignalGenerator``
      3. Call each registered strategy's ``on_bar``
      4. Validate resulting orders via ``RiskManager``
      5. Submit valid orders to ``OrderManager``
    """

    def __init__(
        self,
        order_manager: OrderManager,
        config: TradingConfig | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._order_manager = order_manager
        self._config = config or TradingConfig.defaults()
        self._risk_manager = RiskManager()
        self._performance = PerformanceMetrics()
        self._feature_pipeline = FeaturePipeline(lookback_buffer=500)
        self._signal_generator = SignalGenerator()

        self._strategies: dict[str, Strategy] = {}
        self._running = False
        self._thread: threading.Thread | None = None

        # Trade log for backtesting
        self._trade_log: list[dict[str, Any]] = []
        self._equity_curve: list[float] = []
        self._equity_timestamps: list[datetime] = []

        # Register default feature pipeline indicators
        self._init_default_features()

    def _init_default_features(self) -> None:
        """Register common indicators in the feature pipeline.

        Only single-output indicator functions are registered directly.
        Multi-output functions (bollinger_bands, macd, stoch) must be
        registered manually with separate column extraction or used
        directly inside strategies.
        """
        from features.indicators import (
            adx,
            atr,
            ema,
            obv,
            roc,
            rsi,
            sma,
        )

        fp = self._feature_pipeline

        # Moving averages
        fp.register("sma_20", sma, params={"period": 20})
        fp.register("ema_12", ema, params={"period": 12})
        fp.register("ema_26", ema, params={"period": 26})

        # RSI
        fp.register("rsi_14", rsi, params={"period": 14})

        # ATR and ADX
        fp.register("atr_14", atr, params={"period": 14})
        fp.register("adx_14", adx, params={"period": 14})

        # Volume indicators
        fp.register("obv", obv)
        fp.register("roc_12", roc, params={"period": 12})

    # ------------------------------------------------------------------
    #  Strategy registration
    # ------------------------------------------------------------------

    def register_strategy(self, strategy: Strategy) -> None:
        """Register a strategy with the engine."""
        with self._lock:
            if strategy.id in self._strategies:
                raise KeyError(f"Strategy '{strategy.id}' is already registered.")
            strategy.on_init(self._config)
            self._strategies[strategy.id] = strategy

    def remove_strategy(self, strategy_id: str) -> None:
        """Unregister a strategy."""
        with self._lock:
            strategy = self._strategies.pop(strategy_id, None)
            if strategy is not None:
                strategy.on_stop()

    def get_strategy(self, strategy_id: str) -> Strategy | None:
        """Get a registered strategy by id."""
        with self._lock:
            return self._strategies.get(strategy_id)

    # ------------------------------------------------------------------
    #  Bar processing
    # ------------------------------------------------------------------

    def run_bar(self, symbol: str, ohlcv: list[OHLCV]) -> list[Order]:
        """Process a new bar for *symbol* through all registered strategies.

        This is the main entry point called by the data feed.

        Returns
        -------
        list[Order]
            Orders that were submitted by strategies and passed risk checks.
        """
        with self._lock:
            return self._run_bar_locked(symbol, ohlcv)

    def _run_bar_locked(
        self, symbol: str, ohlcv: list[OHLCV]
    ) -> list[Order]:
        """Internal bar processing (caller holds lock)."""
        if not self._strategies:
            return []

        # Convert OHLCV list to DataFrame for feature computation
        df = self._ohlcv_to_dataframe(ohlcv)
        if df.empty:
            return []

        # 1. Compute features
        feature_df = self._feature_pipeline.compute(df)

        # 2. Generate signals
        signal_df = self._signal_generator.generate_signals(feature_df)

        # Get latest feature and signal values
        latest_features = feature_df.iloc[-1].to_dict() if not feature_df.empty else {}
        latest_signal = (
            signal_df.iloc[-1].to_dict() if not signal_df.empty else {}
        )

        # 3. Call each strategy's on_bar
        submitted_orders: list[Order] = []
        portfolio = self._order_manager.get_portfolio()

        for strategy in self._strategies.values():
            if symbol not in strategy.symbols:
                continue

            orders = strategy.on_bar(
                symbol=symbol,
                ohlcv=ohlcv,
                features=latest_features,
                signals=latest_signal,
            )

            for order in orders:
                # 4. Risk check
                allowed, reason = self._risk_manager.check_order(
                    order, portfolio, self._config
                )
                if not allowed:
                    continue

                # 5. Submit to OrderManager
                try:
                    submitted = self._order_manager.submit(order)
                    submitted_orders.append(submitted)
                except (ValueError, Exception):
                    pass

        return submitted_orders

    # ------------------------------------------------------------------
    #  Metrics
    # ------------------------------------------------------------------

    def get_strategy_metrics(self, strategy_id: str) -> dict[str, Any]:
        """Get performance metrics for a specific strategy."""
        strategy = self.get_strategy(strategy_id)
        if strategy is None:
            return {}
        return strategy.metrics

    # ------------------------------------------------------------------
    #  Lifecycle
    # ------------------------------------------------------------------

    def start(self, background: bool = False) -> None:
        """Start the engine.

        If *background* is True, runs in a separate daemon thread.
        Otherwise this is a no-op (run_bar is called externally by the data feed).
        """
        with self._lock:
            if self._running:
                return
            self._running = True

        if background:
            self._thread = threading.Thread(
                target=self._run_loop, daemon=True, name="strat-engine"
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop the engine and notify all strategies."""
        with self._lock:
            self._running = False
            for strategy in self._strategies.values():
                try:
                    strategy.on_stop()
                except Exception:
                    pass

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2)
            self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    def _run_loop(self) -> None:
        """Background loop — only used when ``start(background=True)``."""
        import time

        while self._running:
            time.sleep(0.1)

    # ------------------------------------------------------------------
    #  Backtesting
    # ------------------------------------------------------------------

    def run_backtest(
        self,
        ohlcv_data: dict[str, pd.DataFrame],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        initial_capital: float = 100_000.0,
    ) -> dict[str, Any]:
        """Run a backtest over historical data.

        Parameters
        ----------
        ohlcv_data : dict[str, pd.DataFrame]
            Symbol -> DataFrame with OHLCV columns ('open', 'high', 'low',
            'close', 'volume') and datetime index.
        start_date : datetime or None
            Optional start date filter.
        end_date : datetime or None
            Optional end date filter.
        initial_capital : float
            Starting capital for the backtest.

        Returns
        -------
        dict
            Backtest results with keys:
            - 'trade_log': list of filled trade dicts
            - 'equity_curve': list of equity values
            - 'performance': performance metrics dict
            - 'summary': simplified summary dict
        """

        from core.exchange import PaperExchange

        # Create isolated exchange and order manager for backtesting
        exchange = PaperExchange(initial_capital=initial_capital)
        bt_order_manager = OrderManager(exchange, initial_cash=initial_capital)

        # Store original state
        orig_om = self._order_manager

        try:
            self._order_manager = bt_order_manager

            # Ensure feature pipeline is clean
            self._feature_pipeline.clear_history()

            # Reset trade log and equity curve
            self._trade_log = []
            self._equity_curve = [initial_capital]
            self._equity_timestamps = []

            # Determine common date range
            all_dates: list[pd.Timestamp] = []
            for sym, df in ohlcv_data.items():
                if isinstance(df.index, pd.DatetimeIndex):
                    d = df.index.tz_localize(None) if df.index.tz is not None else df.index
                    all_dates.extend(d.tolist())
                else:
                    all_dates.extend(df.index.tolist())

            if not all_dates:
                return {"trade_log": [], "equity_curve": [], "performance": {}, "summary": {}}

            all_dates = sorted(set(all_dates))
            all_dates = [d for d in all_dates if isinstance(d, (datetime, pd.Timestamp))]

            # Filter by date range — normalize timezone awareness
            if start_date:
                sd = start_date.replace(tzinfo=None) if start_date.tzinfo else start_date
                all_dates = [d for d in all_dates
                             if (d.replace(tzinfo=None) if getattr(d, 'tzinfo', None) else d) >= sd]
            if end_date:
                ed = end_date.replace(tzinfo=None) if end_date.tzinfo else end_date
                all_dates = [d for d in all_dates
                             if (d.replace(tzinfo=None) if getattr(d, 'tzinfo', None) else d) <= ed]

            # Process bar by bar
            for idx, bar_time in enumerate(all_dates):
                bar_data: dict[str, list[OHLCV]] = {}

                for sym, df in ohlcv_data.items():
                    # Get all data up to this bar
                    df_bar = df[df.index <= bar_time]
                    if df_bar.empty:
                        continue

                    # Convert to OHLCV list
                    last_rows = df_bar.tail(100)
                    ohlcv_list: list[OHLCV] = []
                    for _, row in last_rows.iterrows():
                        ohlcv_list.append(
                            OHLCV(
                                timestamp=row.name if isinstance(row.name, datetime) else bar_time,
                                open=float(row.get("open", row.get("Open", 0))),
                                high=float(row.get("high", row.get("High", 0))),
                                low=float(row.get("low", row.get("Low", 0))),
                                close=float(row.get("close", row.get("Close", 0))),
                                volume=float(row.get("volume", row.get("Volume", 0))),
                            )
                        )

                    if ohlcv_list:
                        bar_data[sym] = ohlcv_list
                        # Update exchange price
                        close_price = ohlcv_list[-1].close
                        exchange.update_market_price(sym, close_price)

                # Process each symbol
                for sym, ohlcv_list in bar_data.items():
                    self._run_bar_locked(sym, ohlcv_list)

                    # Record trades
                    trades = bt_order_manager.get_trade_history(sym, limit=100)
                    for t in trades:
                        self._trade_log.append({
                            "order_id": t.order_id,
                            "symbol": t.symbol,
                            "side": t.side.value,
                            "quantity": t.quantity,
                            "price": t.price,
                            "pnl": t.pnl,
                            "timestamp": t.timestamp.isoformat()
                            if hasattr(t.timestamp, 'isoformat')
                            else str(t.timestamp),
                        })

                # Record equity
                portfolio = bt_order_manager.get_portfolio()
                self._equity_curve.append(portfolio.total_equity)
                self._equity_timestamps.append(
                    bar_time if isinstance(bar_time, datetime) else bar_time.to_pydatetime()
                )

            # Calculate performance
            trades_list = bt_order_manager.get_trade_history()
            perf = PerformanceMetrics.calculate(
                trades_list,
                self._equity_curve,
                self._equity_timestamps,
            )

            # Summary
            summary = {
                "total_return": perf.get("total_return_pct", 0.0),
                "sharpe_ratio": perf.get("sharpe_ratio", 0.0),
                "max_drawdown": perf.get("max_drawdown_pct", 0.0),
                "win_rate": perf.get("win_rate_pct", 0.0),
                "total_trades": perf.get("total_trades", 0),
                "profit_factor": perf.get("profit_factor", 0.0),
                "cagr": perf.get("cagr", 0.0),
                "sortino_ratio": perf.get("sortino_ratio", 0.0),
                "start_equity": initial_capital,
                "end_equity": self._equity_curve[-1] if self._equity_curve else initial_capital,
            }

            return {
                "trade_log": self._trade_log,
                "equity_curve": self._equity_curve,
                "performance": perf,
                "summary": summary,
            }

        finally:
            self._order_manager = orig_om

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ohlcv_to_dataframe(ohlcv: list[OHLCV]) -> pd.DataFrame:
        """Convert a list of OHLCV namedtuples to a DataFrame."""
        if not ohlcv:
            return pd.DataFrame()

        data = {
            "open": [b.open for b in ohlcv],
            "high": [b.high for b in ohlcv],
            "low": [b.low for b in ohlcv],
            "close": [b.close for b in ohlcv],
            "volume": [b.volume for b in ohlcv],
        }
        df = pd.DataFrame(data)

        # Use timestamps as index if available
        if ohlcv[0].timestamp is not None:
            df.index = [b.timestamp for b in ohlcv]

        return df

    @property
    def strategies(self) -> dict[str, Strategy]:
        """Return registered strategies dict."""
        with self._lock:
            return dict(self._strategies)

    @property
    def risk_manager(self) -> RiskManager:
        return self._risk_manager

    @property
    def feature_pipeline(self) -> FeaturePipeline:
        return self._feature_pipeline

    @property
    def signal_generator(self) -> SignalGenerator:
        return self._signal_generator
