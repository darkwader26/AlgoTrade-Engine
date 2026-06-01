"""Tests for the technical indicators library."""

import numpy as np
import pytest

from features.indicators import (
    adx,
    atr,
    bollinger_bands,
    ema,
    macd,
    obv,
    roc,
    rsi,
    sma,
    stoch,
    vwap,
)

# ======================================================================
#  Helpers
# ======================================================================

RTOL = 1e-5
ATOL = 1e-8


def _check_nan_prefix(result: np.ndarray, expected_nans: int) -> None:
    """Assert first *expected_nans* entries are NaN, rest are finite."""
    assert np.all(np.isnan(result[:expected_nans])), (
        f"Expected {expected_nans} leading NaN values, got "
        f"{np.sum(np.isnan(result[:expected_nans]))}"
    )
    if expected_nans < len(result):
        assert np.all(np.isfinite(result[expected_nans:])), (
            "Remaining values should be finite"
        )


@pytest.fixture
def flat_close():
    """A flat close series (all same values)."""
    return np.full(100, 100.0, dtype=np.float64)


@pytest.fixture
def trending_data():
    """Steady upward trend with some noise."""
    np.random.seed(42)
    n = 200
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5) + np.linspace(0, 10, n)
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)
    volume = np.random.randint(1000, 10000, size=n).astype(np.float64)
    return high, low, close, volume


# ======================================================================
#  SMA
# ======================================================================


class TestSMA:
    def test_basic_sma(self):
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(close, 3)
        _check_nan_prefix(result, 2)
        assert np.allclose(result[2:], [2.0, 3.0, 4.0])

    def test_sma_period_one(self):
        close = np.array([1.0, 2.0, 3.0])
        result = sma(close, 1)
        assert np.allclose(result, close)

    def test_sma_flat(self, flat_close):
        result = sma(flat_close, 10)
        assert np.allclose(result[9:], 100.0)

    def test_sma_insufficient_data(self):
        close = np.array([1.0, 2.0])
        result = sma(close, 5)
        assert np.all(np.isnan(result))

    def test_sma_nan_handling(self):
        close = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        result = sma(close, 3)
        # nancumsum should handle NaN
        assert len(result) == 5
        # At least it doesn't crash
        assert np.isnan(result[0]) and np.isnan(result[1])


# ======================================================================
#  EMA
# ======================================================================


class TestEMA:
    def test_basic_ema(self):
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ema(close, 3)
        _check_nan_prefix(result, 2)
        # Manual check: period=3 → alpha = 0.5
        # SMA of first 3 = 2.0
        # ema[2] = 2.0
        # ema[3] = 0.5*4 + 0.5*2 = 3.0
        # ema[4] = 0.5*5 + 0.5*3 = 4.0
        assert np.allclose(result[2:], [2.0, 3.0, 4.0])

    def test_ema_flat(self, flat_close):
        result = ema(flat_close, 10)
        assert np.allclose(result[9:], 100.0)

    def test_ema_period_one(self):
        close = np.array([1.0, 2.0, 3.0])
        result = ema(close, 1)
        # alpha = 2/2 = 1, so ema[i] = close[i]
        assert np.allclose(result[0], 1.0)
        assert np.allclose(result[1], 2.0)
        assert np.allclose(result[2], 3.0)

    def test_ema_insufficient_data(self):
        result = ema(np.array([1.0, 2.0]), 10)
        assert np.all(np.isnan(result))


# ======================================================================
#  RSI
# ======================================================================


class TestRSI:
    def test_rsi_oversold(self):
        """Strongly downward-trending → RSI near 0."""
        close = np.linspace(100, 50, 30)
        result = rsi(close, 14)
        _check_nan_prefix(result, 14)
        assert result[-1] < 30

    def test_rsi_overbought(self):
        """Strongly upward-trending → RSI near 100."""
        close = np.linspace(50, 100, 30)
        result = rsi(close, 14)
        _check_nan_prefix(result, 14)
        assert result[-1] > 70

    def test_rsi_flat(self, flat_close):
        result = rsi(flat_close, 14)
        _check_nan_prefix(result, 14)
        assert np.isnan(result[14]) or result[14] == 50.0
        # When close is flat, gains and losses are all zero → RSI = 50
        assert result[-1] == 50.0

    def test_rsi_alternating(self):
        """Alternating up/down → RSI around 50."""
        np.random.seed(0)
        close = 100.0 + np.random.randn(100)
        result = rsi(close, 14)
        _check_nan_prefix(result, 14)
        assert 20 < np.nanmean(result[14:]) < 80

    def test_rsi_monotonic_up(self):
        """Every step up → RSI = 100."""
        close = np.arange(10.0, 110.0)
        result = rsi(close, 14)
        _check_nan_prefix(result, 14)
        assert result[-1] == 100.0


# ======================================================================
#  MACD
# ======================================================================


class TestMACD:
    def test_macd_basic(self, trending_data):
        _, _, close, _ = trending_data
        macd_line, signal_line, histogram = macd(close)
        assert len(macd_line) == len(close)
        assert len(signal_line) == len(close)
        assert len(histogram) == len(close)
        # Histogram = MACD - Signal
        assert np.allclose(histogram[50:], macd_line[50:] - signal_line[50:])

    def test_macd_flat(self, flat_close):
        macd_line, signal_line, histogram = macd(flat_close)
        # macd_line is 0 from position 26 onwards (slow EMA starts at 25)
        assert np.allclose(macd_line[26:], 0.0, atol=1e-10)
        # signal_line = EMA(macd_line, 9); first valid at 26+9-1 = 34
        assert np.allclose(signal_line[34:], 0.0, atol=1e-10)
        # histogram = 0 from position 34 onwards
        assert np.allclose(histogram[34:], 0.0, atol=1e-10)


# ======================================================================
#  Bollinger Bands
# ======================================================================


class TestBollinger:
    def test_bb_basic(self, trending_data):
        _, _, close, _ = trending_data
        upper, middle, lower = bollinger_bands(close)
        assert len(upper) == len(close)
        assert len(middle) == len(close)
        assert len(lower) == len(close)
        # Upper > Middle > Lower (where defined)
        mask = ~np.isnan(upper)
        assert np.all(upper[mask] >= middle[mask])
        assert np.all(middle[mask] >= lower[mask])

    def test_bb_flat(self, flat_close):
        upper, middle, lower = bollinger_bands(flat_close)
        mask = ~np.isnan(upper)
        assert np.allclose(upper[mask], 100.0)
        assert np.allclose(middle[mask], 100.0)
        assert np.allclose(lower[mask], 100.0)


# ======================================================================
#  ATR
# ======================================================================


class TestATR:
    def test_atr_basic(self, trending_data):
        high, low, close, _ = trending_data
        result = atr(high, low, close, 14)
        _check_nan_prefix(result, 14)
        assert np.all(result[14:] > 0)

    def test_atr_no_range(self):
        """When high=low=close, ATR should be 0."""
        high = np.full(50, 100.0)
        low = np.full(50, 100.0)
        close = np.full(50, 100.0)
        result = atr(high, low, close, 14)
        assert np.allclose(result[14:], 0.0, atol=1e-10)


# ======================================================================
#  ADX
# ======================================================================


class TestADX:
    def test_adx_basic(self, trending_data):
        high, low, close, _ = trending_data
        result = adx(high, low, close, 14)
        # First 2*period-1 = 27 entries are NaN
        _check_nan_prefix(result, 27)
        # ADX should be between 0 and 100
        valid = result[27:]
        assert np.all(valid >= 0) and np.all(valid <= 100)

    def test_adx_strong_trend(self):
        """Strong unidirectional trend → high ADX."""
        n = 100
        close = np.linspace(50, 100, n)
        high = close + 0.5
        low = close - 0.5
        result = adx(high, low, close, 14)
        _check_nan_prefix(result, 27)
        # In a strong trend, ADX should be high
        assert np.nanmean(result[27:]) > 40

    def test_adx_no_trend(self, flat_close):
        """Flat prices → low ADX."""
        high = flat_close + 0.1
        low = flat_close - 0.1
        result = adx(high, low, flat_close, 14)
        _check_nan_prefix(result, 27)
        assert np.nanmean(result[27:]) < 30


# ======================================================================
#  OBV
# ======================================================================


class TestOBV:
    def test_obv_increasing(self):
        """Close always up → OBV always increases."""
        close = np.array([10.0, 11.0, 12.0, 13.0])
        volume = np.array([100.0, 200.0, 150.0, 300.0])
        result = obv(close, volume)
        assert np.all(np.diff(result) > 0)

    def test_obv_decreasing(self):
        """Close always down → OBV always decreases."""
        close = np.array([13.0, 12.0, 11.0, 10.0])
        volume = np.array([100.0, 200.0, 150.0, 300.0])
        result = obv(close, volume)
        assert np.all(np.diff(result) < 0)

    def test_obv_no_change(self):
        """Close unchanged → OBV unchanged."""
        close = np.full(5, 50.0)
        volume = np.array([100.0, 200.0, 150.0, 300.0, 50.0])
        result = obv(close, volume)
        # OBV should stay at volume[0] since no price change triggers addition
        assert np.allclose(result, volume[0])

    def test_obv_single_element(self):
        result = obv(np.array([10.0]), np.array([100.0]))
        assert result[0] == 100.0
        assert not np.isnan(result[0])


# ======================================================================
#  VWAP
# ======================================================================


class TestVWAP:
    def test_vwap_basic(self, trending_data):
        high, low, close, volume = trending_data
        result = vwap(high, low, close, volume)
        assert len(result) == len(close)
        assert np.all(np.isfinite(result))
        # VWAP of single bar = typical price
        typical0 = (high[0] + low[0] + close[0]) / 3.0
        assert np.allclose(result[0], typical0)

    def test_vwap_flat(self):
        """With identical prices and volumes, VWAP = typical price."""
        high = np.full(10, 101.0)
        low = np.full(10, 99.0)
        close = np.full(10, 100.0)
        volume = np.full(10, 1000.0)
        result = vwap(high, low, close, volume)
        typical = (101.0 + 99.0 + 100.0) / 3.0
        assert np.allclose(result, typical)


# ======================================================================
#  ROC
# ======================================================================


class TestROC:
    def test_roc_basic(self):
        close = np.array([100.0, 102.0, 104.0, 106.0, 108.0])
        result = roc(close, 2)
        _check_nan_prefix(result, 2)
        # ROC[2] = (104-100)/100*100 = 4.0
        # ROC[3] = (106-102)/102*100 ≈ 3.92
        # ROC[4] = (108-104)/104*100 ≈ 3.85
        assert np.allclose(result[2], 4.0)
        assert np.allclose(result[3], 106.0 / 102.0 * 100 - 100)
        assert np.allclose(result[4], 108.0 / 104.0 * 100 - 100)

    def test_roc_flat(self, flat_close):
        result = roc(flat_close, 12)
        _check_nan_prefix(result, 12)
        assert np.allclose(result[12:], 0.0, atol=1e-10)

    def test_roc_insufficient_data(self):
        result = roc(np.array([1.0, 2.0]), 5)
        assert np.all(np.isnan(result))


# ======================================================================
#  Stochastic
# ======================================================================


class TestStoch:
    def test_stoch_basic(self, trending_data):
        high, low, close, _ = trending_data
        k, d = stoch(high, low, close, 14, 3)
        _check_nan_prefix(k, 13)
        _check_nan_prefix(d, 15)  # 14 + 3 - 2 = 15
        # Both should be in [0, 100]
        mask_k = ~np.isnan(k)
        mask_d = ~np.isnan(d)
        assert np.all(k[mask_k] >= 0) and np.all(k[mask_k] <= 100)
        assert np.all(d[mask_d] >= 0) and np.all(d[mask_d] <= 100)

    def test_stoch_flat(self):
        """Flat prices → stochastic ≈ 50."""
        high = np.full(50, 101.0)
        low = np.full(50, 99.0)
        close = np.full(50, 100.0)
        k, d = stoch(high, low, close, 14, 3)
        # When range is 2 and close is exactly in the middle → 50%
        assert np.allclose(k[20:], 50.0)
        assert np.allclose(d[20:], 50.0)

    def test_stoch_at_extremes(self):
        """Close at the high of the window → K=100, close at low → K=0."""
        n = 50
        high = np.full(n, 110.0)
        low = np.full(n, 90.0)
        close = np.full(n, 110.0)  # at high
        k, d = stoch(high, low, close, 14, 3)
        assert np.allclose(k[13:], 100.0)

        close = np.full(n, 90.0)  # at low
        k, d = stoch(high, low, close, 14, 3)
        assert np.allclose(k[13:], 0.0)


# ======================================================================
#  Edge cases
# ======================================================================


class TestEdgeCases:
    def test_all_indicators_empty_array(self):
        """All functions gracefully handle empty arrays."""
        empty = np.array([])
        for fn in [sma, ema, rsi, roc]:
            result = fn(empty, 14)
            assert len(result) == 0

    def test_all_indicators_single_element(self):
        single = np.array([100.0])
        for fn in [sma, ema, rsi, roc]:
            result = fn(single, 14)
            assert len(result) == 1
            assert np.isnan(result[0])

    def test_atr_empty(self):
        result = atr(np.array([]), np.array([]), np.array([]))
        assert len(result) == 0
