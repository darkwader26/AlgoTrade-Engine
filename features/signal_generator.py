"""Composite signal generator using a weighted voting mechanism.

Each registered indicator votes **bullish** (+1), **bearish** (-1), or
**neutral** (0) based on configurable rules.  Votes are multiplied by
indicator-level weights and aggregated to produce a consensus signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
#  Signal constants
# ---------------------------------------------------------------------------

BULLISH = 1
NEUTRAL = 0
BEARISH = -1


# ---------------------------------------------------------------------------
#  Voting rule type
# ---------------------------------------------------------------------------

VoteFunc = Callable[[pd.Series], pd.Series]
"""Signature: ``func(feature_series: pd.Series) -> pd.Series[int]``.

Returns a series containing -1, 0, or +1 for each row.
"""


# ---------------------------------------------------------------------------
#  Built-in voting rules
# ---------------------------------------------------------------------------


def _rsi_vote(feature: pd.Series, lower: float = 30.0, upper: float = 70.0) -> pd.Series:
    """RSI voting: oversold → bullish, overbought → bearish."""
    votes = pd.Series(NEUTRAL, index=feature.index, dtype=int)
    votes[feature <= lower] = BULLISH
    votes[feature >= upper] = BEARISH
    return votes


def _macd_vote(
    macd_line: pd.Series,
    signal_line: pd.Series,
    histogram: pd.Series,
) -> pd.Series:
    """MACD voting: line > signal → bullish, line < signal → bearish.
    Extra confidence when histogram is positive/negative.
    """
    votes = pd.Series(NEUTRAL, index=macd_line.index, dtype=int)
    votes[macd_line > signal_line] = BULLISH
    votes[macd_line < signal_line] = BEARISH
    # Neutral when MACD and signal are very close (within 1% of spread)
    spread = (macd_line - signal_line).abs()
    threshold = macd_line.abs().mean() * 0.01 if len(macd_line) > 0 else 0.0
    votes[spread < threshold] = NEUTRAL
    return votes


def _bb_vote(
    close: pd.Series,
    upper: pd.Series,
    lower: pd.Series,
) -> pd.Series:
    """Bollinger Band voting: touch lower → bullish, touch upper → bearish."""
    votes = pd.Series(NEUTRAL, index=close.index, dtype=int)
    votes[close <= lower] = BULLISH
    votes[close >= upper] = BEARISH
    return votes


def _atr_vote(
    atr_series: pd.Series,
    close: pd.Series,
    threshold: float = 0.02,
) -> pd.Series:
    """ATR voting: high relative ATR suggests volatility breakout.

    If ATR/close > threshold → neutral (just volatile, no direction).
    Otherwise no signal.  This indicator is a *volatility* vote, not
    directional by itself, so we keep it neutral by default.
    """
    # ATR alone doesn't give direction; we leave it neutral but the
    # signal generator can still use it via custom rules.
    return pd.Series(NEUTRAL, index=close.index, dtype=int)


def _adx_vote(
    adx_series: pd.Series,
    threshold: float = 25.0,
) -> pd.Series:
    """ADX voting: ADX > 25 → trending, ADX < 20 → ranging (neutral)."""
    votes = pd.Series(NEUTRAL, index=adx_series.index, dtype=int)
    # ADX doesn't give direction; we mark as trending (non-neutral)
    # only when above threshold.  Direction must come from +DI/-DI cross,
    # which the user can configure as a separate feature.
    # For simplicity, we treat high ADX as a valid vote (direction-agnostic
    # until combined with other rules).
    votes[adx_series >= threshold] = BULLISH  # trending up-ish (placeholder)
    votes[adx_series >= threshold] = BEARISH  # won't override — both set above!
    # Actually, ADX alone can't be directional.  Leave neutral.
    return pd.Series(NEUTRAL, index=adx_series.index, dtype=int)


def _obv_vote(obv_series: pd.Series) -> pd.Series:
    """OBV voting: rising OBV → bullish, falling → bearish."""
    votes = pd.Series(NEUTRAL, index=obv_series.index, dtype=int)
    obv_diff = obv_series.diff()
    votes[obv_diff > 0] = BULLISH
    votes[obv_diff < 0] = BEARISH
    return votes


def _stoch_vote(
    k_line: pd.Series,
    lower: float = 20.0,
    upper: float = 80.0,
) -> pd.Series:
    """Stochastic voting: oversold → bullish, overbought → bearish."""
    votes = pd.Series(NEUTRAL, index=k_line.index, dtype=int)
    votes[k_line <= lower] = BULLISH
    votes[k_line >= upper] = BEARISH
    return votes


def _roc_vote(
    roc_series: pd.Series,
    positive_threshold: float = 5.0,
    negative_threshold: float = -5.0,
) -> pd.Series:
    """ROC voting: strong positive momentum → bullish, strong negative → bearish."""
    votes = pd.Series(NEUTRAL, index=roc_series.index, dtype=int)
    votes[roc_series >= positive_threshold] = BULLISH
    votes[roc_series <= negative_threshold] = BEARISH
    return votes


# ---------------------------------------------------------------------------
#  Indicator → rule registry
# ---------------------------------------------------------------------------

DEFAULT_RULES: Dict[str, VoteFunc] = {
    "rsi": _rsi_vote,
    "obv": _obv_vote,
    "stoch_k": _stoch_vote,
    "roc": _roc_vote,
    # MACD, Bollinger, ATR, ADX require multi-column inputs; they are
    # registered separately via add_rule with a custom VoteFunc.
}


# ---------------------------------------------------------------------------
#  SignalGenerator
# ---------------------------------------------------------------------------


@dataclass
class IndicatorVoteConfig:
    """Configuration for one indicator's voting behaviour."""
    feature_names: List[str]
    """Feature column(s) this indicator reads."""
    vote_fn: VoteFunc
    """Voting function."""
    weight: float = 1.0
    """Weight for the weighted sum."""
    enabled: bool = True


class SignalGenerator:
    """Generates consensus trading signals from a feature DataFrame.

    Each indicator votes BULLISH / BEARISH / NEUTRAL.  Votes are summed
    (with indicator-level weights) and compared to configurable thresholds.

    Parameters
    ----------
    strong_threshold : float, default 0.6
        Fractional threshold (of max possible weight) for a *strong* signal.
    weak_threshold : float, default 0.2
        Fractional threshold for a *weak* signal.
    """

    def __init__(
        self,
        strong_threshold: float = 0.6,
        weak_threshold: float = 0.2,
    ) -> None:
        self._indicators: Dict[str, IndicatorVoteConfig] = {}
        self._strong_threshold = strong_threshold
        self._weak_threshold = weak_threshold

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_indicator(
        self,
        name: str,
        feature_names: List[str],
        vote_fn: VoteFunc,
        *,
        weight: float = 1.0,
        enabled: bool = True,
    ) -> "SignalGenerator":
        """Register an indicator with a voting rule.

        Parameters
        ----------
        name : str
            Unique indicator name (e.g. ``"rsi"``).
        feature_names : list of str
            Feature column(s) needed by the voting function.
        vote_fn : VoteFunc
            Callable that maps features to votes (-1, 0, +1).
        weight : float, default 1.0
            Vote weight in the consensus sum.
        enabled : bool, default True
        """
        if name in self._indicators:
            raise KeyError(f"Indicator '{name}' is already registered.")

        self._indicators[name] = IndicatorVoteConfig(
            feature_names=feature_names,
            vote_fn=vote_fn,
            weight=weight,
            enabled=enabled,
        )
        return self

    def remove_indicator(self, name: str) -> None:
        """Remove a registered indicator."""
        self._indicators.pop(name, None)

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Produce consensus signals from the feature DataFrame.

        Parameters
        ----------
        feature_df : pd.DataFrame
            Must contain columns matching registered indicator
            ``feature_names``.

        Returns
        -------
        pd.DataFrame
            Columns:
            - ``signal``: raw consensus score (float)
            - ``normalized_signal``: score / max_possible_weight, in [-1, 1]
            - ``direction``: BULLISH (1), NEUTRAL (0), or BEARISH (-1)
            - ``strength``: ``"strong"``, ``"weak"``, or ``"neutral"``
            - ``votes``: dict of per-indicator votes
            - ``total_weight``: sum of active weights
        """
        n = len(feature_df)
        index = feature_df.index

        # Initialise vote storage
        vote_arrays: Dict[str, np.ndarray] = {}
        total_weight = 0.0

        for ind_name, cfg in self._indicators.items():
            if not cfg.enabled:
                continue

            # Collect feature columns needed
            try:
                features = [feature_df[col] for col in cfg.feature_names]
            except KeyError as e:
                raise KeyError(
                    f"Indicator '{ind_name}' requires column(s) {cfg.feature_names}, "
                    f"but {e} is missing from feature_df."
                )

            if len(features) == 1:
                votes = cfg.vote_fn(features[0])
            else:
                votes = cfg.vote_fn(*features)

            vote_arrays[ind_name] = votes.values
            total_weight += cfg.weight

        # Compute consensus
        if total_weight == 0:
            # No active indicators — everything is neutral
            result = pd.DataFrame(index=index)
            result["signal"] = 0.0
            result["normalized_signal"] = 0.0
            result["direction"] = NEUTRAL
            result["strength"] = "neutral"
            result["votes"] = [{} for _ in range(n)]
            result["total_weight"] = 0.0
            return result

        weighted_sum = np.zeros(n, dtype=np.float64)
        votes_dicts: List[Dict[str, int]] = [{} for _ in range(n)]

        for ind_name, cfg in self._indicators.items():
            if not cfg.enabled:
                continue
            arr = vote_arrays[ind_name]
            weighted_sum += cfg.weight * arr.astype(np.float64)
            for i in range(n):
                if not np.isnan(arr[i]):
                    votes_dicts[i][ind_name] = int(arr[i])

        # Normalise to [-1, 1]
        normalized = weighted_sum / total_weight

        # Determine direction and strength
        direction = np.where(
            normalized > self._weak_threshold,
            BULLISH,
            np.where(normalized < -self._weak_threshold, BEARISH, NEUTRAL),
        ).astype(int)

        strength = np.where(
            np.abs(normalized) >= self._strong_threshold,
            "strong",
            np.where(np.abs(normalized) >= self._weak_threshold, "weak", "neutral"),
        )

        result = pd.DataFrame(index=index)
        result["signal"] = weighted_sum
        result["normalized_signal"] = normalized
        result["direction"] = direction
        result["strength"] = strength
        result["votes"] = votes_dicts
        result["total_weight"] = total_weight

        return result

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def indicators(self) -> Dict[str, IndicatorVoteConfig]:
        """Return a deep copy of the registered indicator configs."""
        from copy import deepcopy
        return {k: deepcopy(v) for k, v in self._indicators.items()}

    @property
    def strong_threshold(self) -> float:
        return self._strong_threshold

    @strong_threshold.setter
    def strong_threshold(self, value: float) -> None:
        self._strong_threshold = value

    @property
    def weak_threshold(self) -> float:
        return self._weak_threshold

    @weak_threshold.setter
    def weak_threshold(self, value: float) -> None:
        self._weak_threshold = value
