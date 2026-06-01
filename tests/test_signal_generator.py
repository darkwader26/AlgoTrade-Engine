"""Tests for the SignalGenerator (composite signal voting)."""

import numpy as np
import pandas as pd
import pytest

from features.signal_generator import (
    BEARISH,
    BULLISH,
    NEUTRAL,
    SignalGenerator,
    _obv_vote,
    _roc_vote,
    _rsi_vote,
    _stoch_vote,
)


@pytest.fixture
def feature_df():
    """A simple feature DataFrame with a few indicator columns."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "rsi_14": np.where(
            np.arange(n) < 30,
            25.0,  # oversold first 30
            np.where(np.arange(n) > 70, 75.0, 50.0),  # overbought last 30, else neutral
        ),
        "obv": np.where(
            np.arange(n) < 50,
            1000.0 + np.arange(n) * 10,  # rising
            2000.0 - np.arange(n) * 5,  # falling
        ),
        "stoch_k": np.where(
            np.arange(n) < 30,
            15.0,  # oversold
            np.where(np.arange(n) > 70, 85.0, 50.0),  # overbought
        ),
        "roc": np.where(
            np.arange(n) < 30,
            10.0,  # strong positive
            np.where(np.arange(n) > 70, -10.0, 0.0),  # strong negative
        ),
        "close": 100.0 + np.cumsum(np.random.randn(n) * 0.5),
    })


class TestVoteFunctions:
    def test_rsi_vote(self):
        s = pd.Series([15.0, 50.0, 85.0, 30.0, 70.0])
        votes = _rsi_vote(s)
        assert votes.iloc[0] == BULLISH
        assert votes.iloc[1] == NEUTRAL
        assert votes.iloc[2] == BEARISH
        # Boundaries: ≤30 → bullish, ≥70 → bearish
        assert votes.iloc[3] == BULLISH
        assert votes.iloc[4] == BEARISH

    def test_obv_vote(self):
        s = pd.Series([100.0, 110.0, 105.0, 95.0, 95.0])
        votes = _obv_vote(s)
        assert votes.iloc[0] == NEUTRAL  # first value, no diff
        assert votes.iloc[1] == BULLISH  # rising
        assert votes.iloc[2] == BEARISH  # falling
        assert votes.iloc[3] == BEARISH  # falling
        assert votes.iloc[4] == NEUTRAL  # no change

    def test_stoch_vote(self):
        s = pd.Series([15.0, 50.0, 85.0, 20.0, 80.0])
        votes = _stoch_vote(s)
        assert votes.iloc[0] == BULLISH
        assert votes.iloc[1] == NEUTRAL
        assert votes.iloc[2] == BEARISH
        assert votes.iloc[3] == BULLISH
        assert votes.iloc[4] == BEARISH

    def test_roc_vote(self):
        s = pd.Series([10.0, 0.0, -10.0, 5.0, -5.0])
        votes = _roc_vote(s)
        assert votes.iloc[0] == BULLISH   # >= 5
        assert votes.iloc[1] == NEUTRAL   # between -5 and 5
        assert votes.iloc[2] == BEARISH   # <= -5
        assert votes.iloc[3] == BULLISH   # >= 5 (boundary)
        assert votes.iloc[4] == BEARISH   # <= -5 (boundary)


class TestSignalGenerator:
    def test_basic_signal(self, feature_df):
        """Single indicator → consensus matches its votes."""
        gen = SignalGenerator()
        gen.add_indicator("rsi", feature_names=["rsi_14"], vote_fn=_rsi_vote, weight=1.0)

        signals = gen.generate_signals(feature_df)
        assert "signal" in signals.columns
        assert "direction" in signals.columns
        assert "strength" in signals.columns
        assert "votes" in signals.columns

        # First row: rsi=25 → bullish → direction=1
        assert signals["direction"].iloc[0] == BULLISH
        # Middle row: rsi=50 → neutral → direction=0
        assert signals["direction"].iloc[30] == NEUTRAL  # rsi=25 range ends at 30
        assert signals["direction"].iloc[50] == NEUTRAL
        # Last row: rsi=75 → bearish → direction=-1
        assert signals["direction"].iloc[-1] == BEARISH

    def test_weighted_voting(self, feature_df):
        """Two indicators with different weights → weighted sum."""
        gen = SignalGenerator(strong_threshold=0.3, weak_threshold=0.1)

        # RSI weights 2x, OBV weights 1x
        gen.add_indicator("rsi", feature_names=["rsi_14"], vote_fn=_rsi_vote, weight=2.0)
        gen.add_indicator("obv", feature_names=["obv"], vote_fn=_obv_vote, weight=1.0)

        signals = gen.generate_signals(feature_df)

        # Row 0: RSI=25→bullish(+1)*2 + OBV(0)=neutral*1 = +2 (OBV first row is
        # neutral because there is no previous value to diff).
        assert np.isclose(signals["signal"].iloc[0], 2.0)
        assert np.isclose(signals["normalized_signal"].iloc[0], 2.0 / 3.0)
        assert signals["direction"].iloc[0] == BULLISH
        assert signals["strength"].iloc[0] == "strong"

    def test_conflicting_votes(self, feature_df):
        """Conflicting votes should neutralise."""
        gen = SignalGenerator(weak_threshold=0.1)
        # RSI says bullish (oversold), but we add a mock that always says bearish
        gen.add_indicator("rsi", feature_names=["rsi_14"], vote_fn=_rsi_vote, weight=1.0)
        gen.add_indicator(
            "always_bearish",
            feature_names=["close"],
            vote_fn=lambda s: pd.Series(BEARISH, index=s.index, dtype=int),
            weight=1.0,
        )

        signals = gen.generate_signals(feature_df)
        # Row 0: RSI=+1 + always_bearish=-1 = 0 → neutral
        assert signals["direction"].iloc[0] == NEUTRAL

    def test_no_indicators(self, feature_df):
        """With no indicators, everything is neutral."""
        gen = SignalGenerator()
        signals = gen.generate_signals(feature_df)
        assert np.all(signals["direction"] == NEUTRAL)
        assert np.all(signals["strength"] == "neutral")
        assert np.all(signals["signal"] == 0.0)

    def test_all_disabled(self, feature_df):
        """All indicators disabled → neutral."""
        gen = SignalGenerator()
        gen.add_indicator("rsi", feature_names=["rsi_14"], vote_fn=_rsi_vote, weight=1.0, enabled=False)
        signals = gen.generate_signals(feature_df)
        assert np.all(signals["direction"] == NEUTRAL)

    def test_threshold_configuration(self, feature_df):
        """Adjusting thresholds changes signal strength."""
        gen_loose = SignalGenerator(strong_threshold=0.3, weak_threshold=0.1)
        gen_loose.add_indicator("rsi", feature_names=["rsi_14"], vote_fn=_rsi_vote, weight=1.0)

        gen_tight = SignalGenerator(strong_threshold=0.9, weak_threshold=0.5)
        gen_tight.add_indicator("rsi", feature_names=["rsi_14"], vote_fn=_rsi_vote, weight=1.0)

        sig_loose = gen_loose.generate_signals(feature_df)
        sig_tight = gen_tight.generate_signals(feature_df)

        # Row 0: normalized=1.0 → strong in both
        assert sig_loose["strength"].iloc[0] == "strong"
        assert sig_tight["strength"].iloc[0] == "strong"
        # Row 50: rsi=50 → neutral for both
        assert sig_loose["strength"].iloc[50] == "neutral"
        assert sig_tight["strength"].iloc[50] == "neutral"

    def test_indicator_property(self, feature_df):
        """The 'indicators' property returns a copy of registered configs."""
        gen = SignalGenerator()
        gen.add_indicator("rsi", feature_names=["rsi_14"], vote_fn=_rsi_vote, weight=1.0)
        inds = gen.indicators
        assert "rsi" in inds
        assert inds["rsi"].weight == 1.0
        # Modifying the returned dict should not affect the original
        inds["rsi"].weight = 99.0
        assert gen.indicators["rsi"].weight == 1.0

    def test_remove_indicator(self, feature_df):
        """Removing an indicator works."""
        gen = SignalGenerator()
        gen.add_indicator("rsi", feature_names=["rsi_14"], vote_fn=_rsi_vote, weight=1.0)
        gen.remove_indicator("rsi")
        assert "rsi" not in gen.indicators

    def test_missing_feature_raises(self, feature_df):
        """Missing feature column raises KeyError."""
        gen = SignalGenerator()
        gen.add_indicator("rsi", feature_names=["nonexistent"], vote_fn=_rsi_vote, weight=1.0)
        with pytest.raises(KeyError, match="nonexistent"):
            gen.generate_signals(feature_df)

    def test_duplicate_register(self, feature_df):
        """Registering the same indicator name raises."""
        gen = SignalGenerator()
        gen.add_indicator("rsi", feature_names=["rsi_14"], vote_fn=_rsi_vote)
        with pytest.raises(KeyError, match="already registered"):
            gen.add_indicator("rsi", feature_names=["rsi_14"], vote_fn=_rsi_vote)

    def test_votes_column_content(self, feature_df):
        """The 'votes' column contains per-row dict of indicator votes."""
        gen = SignalGenerator()
        gen.add_indicator("rsi", feature_names=["rsi_14"], vote_fn=_rsi_vote, weight=1.0)
        signals = gen.generate_signals(feature_df)

        votes0 = signals["votes"].iloc[0]
        assert isinstance(votes0, dict)
        assert votes0["rsi"] == BULLISH

        votes_mid = signals["votes"].iloc[50]
        assert votes_mid["rsi"] == NEUTRAL
