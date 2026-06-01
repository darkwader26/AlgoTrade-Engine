"""Market data manager — subscribes to symbols, maintains rolling OHLCV buffers."""

from __future__ import annotations

import threading
from collections import deque
from typing import Callable

from core.models import OHLCV

# Type alias for bar event callbacks
BarCallback = Callable[[str, OHLCV], None]


class DataFeed:
    """Market data manager.

    Maintains rolling OHLCV buffers per symbol, and notifies
    registered callbacks when new bars arrive.
    Thread-safe.
    """

    def __init__(self, maxlen: int = 500) -> None:
        self._lock = threading.RLock()
        self._maxlen: int = maxlen
        # symbol -> deque[OHLCV]
        self._buffers: dict[str, deque[OHLCV]] = {}
        # strategy_id -> callback_fn
        self._callbacks: dict[str, BarCallback] = {}

    # ---- Subscription ----

    def subscribe(self, symbol: str) -> None:
        """Ensure a symbol buffer exists.

        Safe to call multiple times for the same symbol.
        """
        with self._lock:
            if symbol not in self._buffers:
                self._buffers[symbol] = deque(maxlen=self._maxlen)

    def unsubscribe(self, symbol: str) -> None:
        """Remove a symbol buffer and drop all its data."""
        with self._lock:
            self._buffers.pop(symbol, None)

    # ---- Data ingestion ----

    def update(self, symbol: str, ohlcv_list: list[OHLCV]) -> None:
        """Push new OHLCV bars for a specific symbol into the feed.

        Automatically subscribes the symbol if not yet tracked.
        Calls registered callbacks for each new bar.
        """
        if not ohlcv_list:
            return

        with self._lock:
            self.subscribe(symbol)
            buf = self._buffers[symbol]

            for bar in ohlcv_list:
                buf.append(bar)

        # Fire callbacks outside the lock
        for bar in ohlcv_list:
            self._notify(symbol, bar)

    # ---- Callbacks ----

    def register_callback(self, strategy_id: str, callback_fn: BarCallback) -> None:
        """Register a callback to receive new bar events.

        Callback signature: fn(symbol: str, bar: OHLCV) -> None
        """
        with self._lock:
            self._callbacks[strategy_id] = callback_fn

    def unregister_callback(self, strategy_id: str) -> None:
        """Remove a previously registered callback."""
        with self._lock:
            self._callbacks.pop(strategy_id, None)

    def _notify(self, symbol: str, bar: OHLCV) -> None:
        """Notify all callbacks of a new bar."""
        # Snapshot callbacks under lock, call outside
        with self._lock:
            cbs = dict(self._callbacks)
        for cb in cbs.values():
            try:
                cb(symbol, bar)
            except Exception:
                # Swallow callback exceptions to avoid breaking the feed
                pass

    # ---- Data access ----

    def latest(self, symbol: str, n: int = 1) -> list[OHLCV]:
        """Get the last *n* bars for a symbol (most recent first)."""
        with self._lock:
            buf = self._buffers.get(symbol)
            if buf is None:
                return []
            result = list(buf)[-n:]
            return list(reversed(result))

    def all_symbols(self) -> list[str]:
        """Return list of all subscribed symbols."""
        with self._lock:
            return list(self._buffers.keys())

    def buffer_size(self, symbol: str) -> int:
        """Return the number of bars currently stored for a symbol."""
        with self._lock:
            buf = self._buffers.get(symbol)
            return len(buf) if buf is not None else 0

    def clear(self) -> None:
        """Clear all buffers and callbacks."""
        with self._lock:
            self._buffers.clear()
            self._callbacks.clear()
