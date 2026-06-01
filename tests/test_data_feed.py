"""Tests for DataFeed — buffer management, callbacks, multi-symbol."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.data_feed import DataFeed
from core.models import OHLCV


@pytest.fixture
def feed():
    return DataFeed(maxlen=10)


@pytest.fixture
def sample_bar():
    ts = datetime.now(timezone.utc)
    return OHLCV(timestamp=ts, open=100.0, high=105.0, low=99.0, close=102.0, volume=1000.0)


class TestBufferManagement:
    def test_empty_feed(self, feed):
        assert feed.all_symbols() == []
        assert feed.latest("BTC/USD") == []

    def test_subscribe_and_unsubscribe(self, feed):
        feed.subscribe("BTC/USD")
        assert "BTC/USD" in feed.all_symbols()
        feed.unsubscribe("BTC/USD")
        assert "BTC/USD" not in feed.all_symbols()

    def test_update_creates_buffer_automatically(self, feed, sample_bar):
        feed.update("BTC/USD", [sample_bar])
        assert "BTC/USD" in feed.all_symbols()

    def test_latest_returns_most_recent(self, feed):
        ts1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts2 = datetime(2024, 1, 2, tzinfo=timezone.utc)
        bar1 = OHLCV(ts1, 100.0, 101.0, 99.0, 100.5, 1000.0)
        bar2 = OHLCV(ts2, 101.0, 102.0, 100.0, 101.5, 1100.0)
        feed.update("BTC/USD", [bar1, bar2])
        latest = feed.latest("BTC/USD", n=1)
        assert len(latest) == 1
        assert latest[0].timestamp == ts2

    def test_latest_multiple_bars(self, feed):
        bars = [
            OHLCV(datetime(2024, 1, i, tzinfo=timezone.utc), 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 1000.0)
            for i in range(1, 6)
        ]
        feed.update("BTC/USD", bars)
        latest = feed.latest("BTC/USD", n=3)
        assert len(latest) == 3

    def test_rolling_buffer_maxlen(self, feed):
        feed = DataFeed(maxlen=3)
        for i in range(10):
            bar = OHLCV(
                datetime(2024, 1, i + 1, tzinfo=timezone.utc),
                100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 1000.0,
            )
            feed.update("BTC/USD", [bar])
        assert feed.buffer_size("BTC/USD") == 3

    def test_buffer_size(self, feed):
        assert feed.buffer_size("NONEXISTENT") == 0
        bar = OHLCV(datetime.now(timezone.utc), 1.0, 2.0, 1.0, 1.5, 100.0)
        feed.update("BTC/USD", [bar])
        assert feed.buffer_size("BTC/USD") == 1


class TestCallbacks:
    def test_callback_receives_bar(self, feed, sample_bar):
        received = []

        def cb(symbol, bar):
            received.append((symbol, bar))

        feed.register_callback("strat1", cb)
        feed.update("BTC/USD", [sample_bar])
        assert len(received) == 1
        assert received[0][0] == "BTC/USD"
        assert received[0][1] == sample_bar

    def test_multiple_callbacks(self, feed, sample_bar):
        received = []

        def cb1(s, b):
            received.append((s, b, "cb1"))

        def cb2(s, b):
            received.append((s, b, "cb2"))

        feed.register_callback("s1", cb1)
        feed.register_callback("s2", cb2)
        feed.update("BTC/USD", [sample_bar])
        assert len(received) == 2

    def test_unregister_callback(self, feed, sample_bar):
        received = []

        def cb(s, b):
            received.append((s, b))

        feed.register_callback("strat1", cb)
        feed.unregister_callback("strat1")
        feed.update("BTC/USD", [sample_bar])
        assert len(received) == 0

    def test_callback_error_does_not_break_feed(self, feed, sample_bar):
        def broken_cb(s, b):
            raise RuntimeError("oops")

        feed.register_callback("broken", broken_cb)
        feed.update("BTC/USD", [sample_bar])  # should not raise


class TestMultiSymbol:
    def test_multiple_symbols(self, feed):
        btc = OHLCV(datetime.now(timezone.utc), 50000.0, 51000.0, 49000.0, 50500.0, 100.0)
        eth = OHLCV(datetime.now(timezone.utc), 3000.0, 3100.0, 2900.0, 3050.0, 500.0)
        feed.update("BTC/USD", [btc])
        feed.update("ETH/USD", [eth])
        assert set(feed.all_symbols()) == {"BTC/USD", "ETH/USD"}

    def test_latest_is_symbol_scoped(self, feed):
        btc = OHLCV(datetime.now(timezone.utc), 50000.0, 51000.0, 49000.0, 50500.0, 100.0)
        eth = OHLCV(datetime.now(timezone.utc), 3000.0, 3100.0, 2900.0, 3050.0, 500.0)
        feed.update("BTC/USD", [btc])
        feed.update("ETH/USD", [eth])
        assert feed.latest("BTC/USD")[0].close == 50500.0
        assert feed.latest("ETH/USD")[0].close == 3050.0


class TestClear:
    def test_clear_empties_everything(self, feed, sample_bar):
        feed.update("BTC/USD", [sample_bar])
        feed.register_callback("s1", lambda s, b: None)
        feed.clear()
        assert feed.all_symbols() == []
        assert feed.buffer_size("BTC/USD") == 0
        # Callbacks are gone — no error after clear
        feed.update("BTC/USD", [sample_bar])
        assert feed.buffer_size("BTC/USD") == 1
