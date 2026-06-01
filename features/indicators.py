"""Standalone technical indicator functions operating on numpy arrays.

Every function returns a :class:`numpy.ndarray` of the same length as the
input, with ``NaN`` values in positions where the indicator cannot be
computed (typically the first ``period - 1`` elements).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# ======================================================================
#  Moving averages
# ======================================================================


def sma(close: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average.

    Parameters
    ----------
    close : np.ndarray
        Close prices.
    period : int
        Look-back window.

    Returns
    -------
    np.ndarray
        SMA values; first ``period - 1`` entries are ``NaN``.  Any
        window that contains a ``NaN`` also yields ``NaN``.
    """
    out = np.full_like(close, np.nan, dtype=np.float64)
    if len(close) < period or period < 1:
        return out
    cumsum = np.nancumsum(close, dtype=np.float64)
    out[period - 1:] = (cumsum[period - 1:]
                        - np.concatenate([[0], cumsum[:-period]])) / period
    # Zero-out windows that contain any NaN
    nan_mask = np.isnan(close).astype(np.float64)
    nan_count = np.convolve(nan_mask, np.ones(period, dtype=np.float64), mode="valid")
    out[period - 1:] = np.where(nan_count > 0, np.nan, out[period - 1:])
    return out


def ema(close: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average (uses wilder-style smoothing by default).

    Parameters
    ----------
    close : np.ndarray
        Close prices.
    period : int
        Smoothing period.

    Returns
    -------
    np.ndarray
        EMA values; first ``period - 1`` entries are ``NaN`` (or more if
        the input has leading ``NaN`` values).
    """
    out = np.full_like(close, np.nan, dtype=np.float64)
    if len(close) < period or period < 1:
        return out
    alpha = 2.0 / (period + 1.0)

    # Find the first non-NaN index
    first_valid = 0
    while first_valid < len(close) and np.isnan(close[first_valid]):
        first_valid += 1
    if first_valid == len(close):  # all NaN
        return out

    # Seed position: need `period` consecutive valid values
    seed_idx = first_valid + period - 1
    if seed_idx >= len(close):
        return out
    # Ensure the seed window has no NaN
    if np.any(np.isnan(close[first_valid:seed_idx + 1])):
        return out

    out[seed_idx] = np.nanmean(close[first_valid:seed_idx + 1])
    for i in range(seed_idx + 1, len(close)):
        if np.isnan(close[i]):
            break
        out[i] = alpha * close[i] + (1.0 - alpha) * out[i - 1]
    return out


# ======================================================================
#  RSI
# ======================================================================


def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index.

    Uses Wilder's smoothed RSI method.

    Parameters
    ----------
    close : np.ndarray
        Close prices.
    period : int, default 14
        Look-back period.

    Returns
    -------
    np.ndarray
        RSI values (0–100 scale); first ``period`` entries are ``NaN``.
    """
    out = np.full_like(close, np.nan, dtype=np.float64)
    if len(close) < period + 1 or period < 1:
        return out

    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0).astype(np.float64)
    losses = np.where(deltas < 0, -deltas, 0.0).astype(np.float64)

    # Wilder's smoothing: start with SMA, then exponential
    avg_gain = np.full_like(close, np.nan, dtype=np.float64)
    avg_loss = np.full_like(close, np.nan, dtype=np.float64)

    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    alpha = 1.0 / period
    for i in range(period + 1, len(close)):
        avg_gain[i] = gains[i - 1] * alpha + avg_gain[i - 1] * (1.0 - alpha)
        avg_loss[i] = losses[i - 1] * alpha + avg_loss[i - 1] * (1.0 - alpha)

    rs = np.full_like(close, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    # When avg_loss ≈ 0, RSI is 100
    out[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    # If avg_loss is zero and avg_gain > 0 → RSI = 100
    zero_loss = (avg_loss == 0) & (avg_gain != 0)
    out[zero_loss] = 100.0
    # If both are zero → RSI = 50 (neutral)
    both_zero = (avg_loss == 0) & (avg_gain == 0)
    out[both_zero] = 50.0

    return out


# ======================================================================
#  MACD
# ======================================================================


def macd(
    close: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD (Moving Average Convergence Divergence).

    Returns
    -------
    macd_line : np.ndarray
        MACD line (fast EMA - slow EMA).
    signal_line : np.ndarray
        Signal line (EMA of MACD line).
    histogram : np.ndarray
        Histogram = MACD line - Signal line.
    """
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ======================================================================
#  Bollinger Bands
# ======================================================================


def bollinger_bands(
    close: np.ndarray,
    period: int = 20,
    stddev: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands.

    Returns
    -------
    upper : np.ndarray
        Upper band (middle + stddev * standard deviation).
    middle : np.ndarray
        Middle band (SMA).
    lower : np.ndarray
        Lower band (middle - stddev * standard deviation).
    """
    middle = sma(close, period)
    # Rolling standard deviation
    std = np.full_like(close, np.nan, dtype=np.float64)
    if len(close) >= period:
        for i in range(period - 1, len(close)):
            std[i] = np.nanstd(close[i - period + 1 : i + 1], ddof=0)
    upper = middle + stddev * std
    lower = middle - stddev * std
    return upper, middle, lower


# ======================================================================
#  ATR
# ======================================================================


def atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Average True Range (Wilder-smooothed).

    Parameters
    ----------
    high : np.ndarray
    low : np.ndarray
    close : np.ndarray
    period : int, default 14

    Returns
    -------
    np.ndarray
        ATR values; first ``period`` entries are ``NaN``.
    """
    out = np.full_like(close, np.nan, dtype=np.float64)
    n = len(close)
    if n < period + 1 or period < 1:
        return out

    tr = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
    tr[0] = high[0] - low[0]

    # Wilder's smoothing
    out[period] = np.nanmean(tr[1 : period + 1])  # first ATR = mean of TR
    alpha = 1.0 / period
    for i in range(period + 1, n):
        out[i] = tr[i] * alpha + out[i - 1] * (1.0 - alpha)
    return out


# ======================================================================
#  ADX
# ======================================================================


def adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Average Directional Index (Wilder).

    Parameters
    ----------
    high : np.ndarray
    low : np.ndarray
    close : np.ndarray
    period : int, default 14

    Returns
    -------
    np.ndarray
        ADX values; first ``2 * period - 1`` entries are ``NaN``.
    """
    out = np.full_like(close, np.nan, dtype=np.float64)
    n = len(close)
    if n < 2 * period or period < 1:
        return out

    # ---- True Range --------------------------------------------------
    tr = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
    tr[0] = high[0] - low[0]

    # ---- Directional Movement ----------------------------------------
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    # ---- Wilder-smoothed TR, +DM, -DM --------------------------------
    alpha = 1.0 / period
    smoothed_tr = np.full(n, np.nan, dtype=np.float64)
    smoothed_plus = np.full(n, np.nan, dtype=np.float64)
    smoothed_minus = np.full(n, np.nan, dtype=np.float64)

    smoothed_tr[period] = np.nansum(tr[1 : period + 1])
    smoothed_plus[period] = np.nansum(plus_dm[1 : period + 1])
    smoothed_minus[period] = np.nansum(minus_dm[1 : period + 1])

    for i in range(period + 1, n):
        smoothed_tr[i] = tr[i] * alpha + smoothed_tr[i - 1] * (1.0 - alpha)
        smoothed_plus[i] = plus_dm[i] * alpha + smoothed_plus[i - 1] * (1.0 - alpha)
        smoothed_minus[i] = minus_dm[i] * alpha + smoothed_minus[i - 1] * (1.0 - alpha)

    # ---- +DI / -DI ---------------------------------------------------
    plus_di = np.where(smoothed_tr != 0, 100.0 * smoothed_plus / smoothed_tr, 0.0)
    minus_di = np.where(smoothed_tr != 0, 100.0 * smoothed_minus / smoothed_tr, 0.0)

    # ---- DX ----------------------------------------------------------
    dx = np.full(n, np.nan, dtype=np.float64)
    di_sum = plus_di + minus_di
    mask = di_sum != 0
    dx[mask] = 100.0 * np.abs(plus_di[mask] - minus_di[mask]) / di_sum[mask]
    # DX is 0 when there is no directional movement
    dx[~mask] = 0.0

    # ---- ADX = smoothed DX -------------------------------------------
    out[2 * period - 1] = np.nanmean(dx[period : 2 * period])
    for i in range(2 * period, n):
        out[i] = dx[i] * alpha + out[i - 1] * (1.0 - alpha)

    return out


# ======================================================================
#  OBV
# ======================================================================


def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """On-Balance Volume.

    OBV[0] = volume[0]
    OBV[i] = OBV[i-1] + volume[i]  if close[i] > close[i-1]
    OBV[i] = OBV[i-1] - volume[i]  if close[i] < close[i-1]
    OBV[i] = OBV[i-1]              otherwise

    Returns
    -------
    np.ndarray
        OBV values (same length as input).
    """
    out = np.full_like(volume, np.nan, dtype=np.float64)
    if len(close) < 2:
        out[0] = volume[0] if len(close) == 1 else np.nan
        return out

    out[0] = volume[0]
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            out[i] = out[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            out[i] = out[i - 1] - volume[i]
        else:
            out[i] = out[i - 1]
    return out


# ======================================================================
#  VWAP
# ======================================================================


def vwap(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> np.ndarray:
    """Volume-Weighted Average Price (cumulative).

    VWAP = sum(typical_price * volume) / sum(volume), computed
    progressively from the start of the array.

    Typical Price = (high + low + close) / 3.

    Returns
    -------
    np.ndarray
        VWAP values (same length as input); first entry uses the first bar
        only (so it equals typical price of that bar).
    """
    out = np.full_like(close, np.nan, dtype=np.float64)
    n = len(close)
    if n == 0:
        return out
    typical = (high + low + close) / 3.0
    cum_pv = np.nancumsum(typical * volume, dtype=np.float64)
    cum_vol = np.nancumsum(volume, dtype=np.float64)
    mask = cum_vol != 0
    out[mask] = cum_pv[mask] / cum_vol[mask]
    return out


# ======================================================================
#  ROC
# ======================================================================


def roc(close: np.ndarray, period: int = 12) -> np.ndarray:
    """Rate of Change (percent).

    ROC[i] = (close[i] - close[i - period]) / close[i - period] * 100

    Parameters
    ----------
    close : np.ndarray
    period : int, default 12

    Returns
    -------
    np.ndarray
        ROC values; first ``period`` entries are ``NaN``.
    """
    out = np.full_like(close, np.nan, dtype=np.float64)
    if len(close) <= period or period < 1:
        return out
    out[period:] = (close[period:] - close[:-period]) / close[:-period] * 100.0
    return out


# ======================================================================
#  Stochastic Oscillator
# ======================================================================


def stoch(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_period: int = 14,
    d_period: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stochastic Oscillator.

    %K = (close - lowest_low) / (highest_high - lowest_low) * 100
    %D = SMA(%K, d_period)

    Parameters
    ----------
    high : np.ndarray
    low : np.ndarray
    close : np.ndarray
    k_period : int, default 14
        Look-back period for %K.
    d_period : int, default 3
        Smoothing period for %D.

    Returns
    -------
    k_line : np.ndarray
        %K values (first ``k_period - 1`` entries are ``NaN``).
    d_line : np.ndarray
        %D values (first ``k_period + d_period - 2`` entries are ``NaN``).
    """
    n = len(close)
    k_line = np.full(n, np.nan, dtype=np.float64)
    if n < k_period or k_period < 1:
        return k_line, np.full_like(k_line, np.nan)

    for i in range(k_period - 1, n):
        hh = np.nanmax(high[i - k_period + 1 : i + 1])
        ll = np.nanmin(low[i - k_period + 1 : i + 1])
        denom = hh - ll
        if denom != 0:
            k_line[i] = (close[i] - ll) / denom * 100.0
        else:
            k_line[i] = 50.0  # neutral when range is zero

    d_line = sma(k_line, d_period)
    return k_line, d_line
