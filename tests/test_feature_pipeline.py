"""Tests for the FeaturePipeline."""

import numpy as np
import pandas as pd
import pytest

from features.feature_pipeline import FeaturePipeline
from features.indicators import bollinger_bands, rsi, sma


@pytest.fixture
def ohlcv():
    """Standard OHLCV DataFrame with 200 rows of synthetic data."""
    np.random.seed(42)
    n = 200
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5) + np.linspace(0, 5, n)
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)
    open_ = close - np.random.randn(n) * 0.3
    volume = np.random.randint(1000, 10000, size=n).astype(np.float64)

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


class TestFeaturePipeline:
    def test_register_and_compute(self, ohlcv):
        """Basic registration and computation works."""
        pipeline = FeaturePipeline()
        pipeline.register("sma_20", sma, params={"period": 20})

        result = pipeline.compute(ohlcv)
        assert "sma_20" in result.columns
        assert len(result) == len(ohlcv)
        assert np.isnan(result["sma_20"].iloc[18])
        assert not np.isnan(result["sma_20"].iloc[19])

    def test_multiple_features(self, ohlcv):
        """Multiple features can be registered and computed."""
        pipeline = FeaturePipeline()
        pipeline.register("sma_10", sma, params={"period": 10})
        pipeline.register("sma_20", sma, params={"period": 20})
        pipeline.register("rsi_14", rsi, params={"period": 14})

        result = pipeline.compute(ohlcv)
        for col in ["sma_10", "sma_20", "rsi_14"]:
            assert col in result.columns

    def test_compute_single(self, ohlcv):
        """compute_single returns one feature."""
        pipeline = FeaturePipeline()
        pipeline.register("sma_20", sma, params={"period": 20})

        series = pipeline.compute_single("sma_20", ohlcv)
        assert isinstance(series, pd.Series)
        assert series.name == "sma_20"
        assert len(series) == len(ohlcv)

    def test_compute_single_unregistered(self, ohlcv):
        """Asking for an unregistered feature raises KeyError."""
        pipeline = FeaturePipeline()
        with pytest.raises(KeyError, match="not registered"):
            pipeline.compute_single("nonexistent", ohlcv)

    def test_double_register_raises(self, ohlcv):
        """Re-registering the same name raises KeyError."""
        pipeline = FeaturePipeline()
        pipeline.register("sma_20", sma, params={"period": 20})
        with pytest.raises(KeyError, match="already registered"):
            pipeline.register("sma_20", sma, params={"period": 20})

    def test_unregister(self, ohlcv):
        """Unregistering removes a feature."""
        pipeline = FeaturePipeline()
        pipeline.register("sma_20", sma, params={"period": 20})
        pipeline.unregister("sma_20")
        with pytest.raises(KeyError):
            pipeline.compute_single("sma_20", ohlcv)

    def test_normalization_zscore(self, ohlcv):
        """Z-score normalisation produces zero-mean, unit-variance output."""
        pipeline = FeaturePipeline()
        pipeline.register(
            "rsi_14",
            rsi,
            params={"period": 14},
            normalization="zscore",
        )
        result = pipeline.compute(ohlcv)
        vals = result["rsi_14"].dropna().values
        # After z-score, mean ≈ 0, std ≈ 1
        assert abs(np.mean(vals)) < 1.0
        assert abs(np.std(vals) - 1.0) < 0.1

    def test_normalization_minmax(self, ohlcv):
        """Min-max normalisation produces values in [0, 1]."""
        pipeline = FeaturePipeline()
        pipeline.register(
            "rsi_14",
            rsi,
            params={"period": 14},
            normalization="minmax",
        )
        result = pipeline.compute(ohlcv)
        vals = result["rsi_14"].dropna().values
        assert np.min(vals) >= 0.0 and np.max(vals) <= 1.0

    def test_disabled_feature(self, ohlcv):
        """Disabled features are not computed."""
        pipeline = FeaturePipeline()
        pipeline.register("sma_20", sma, params={"period": 20}, enabled=True)
        pipeline.register("sma_50", sma, params={"period": 50}, enabled=False)

        result = pipeline.compute(ohlcv)
        assert "sma_20" in result.columns
        assert "sma_50" not in result.columns

    def test_missing_columns_raises(self, ohlcv):
        """Missing required columns raise ValueError."""
        bad_df = ohlcv.drop(columns=["volume"])
        pipeline = FeaturePipeline()
        pipeline.register("sma_20", sma, params={"period": 20})

        with pytest.raises(ValueError, match="volume"):
            pipeline.compute(bad_df)

    def test_feature_history(self, ohlcv):
        """Compute twice with lookback-dependent features works."""
        pipeline = FeaturePipeline(lookback_buffer=200)
        pipeline.register("rsi_14", rsi, params={"period": 14})

        # First chunk
        first = ohlcv.iloc[:100]
        r1 = pipeline.compute(first)
        assert len(r1) == 100

        # Second chunk — should use history to compute correctly
        second = ohlcv.iloc[100:]
        r2 = pipeline.compute(second)
        assert len(r2) == 100

        # The first valid RSI value of the second chunk should use history
        # (this just checks it doesn't crash and produces finite values)
        assert not np.isnan(r2["rsi_14"].iloc[13])

    def test_clear_history(self, ohlcv):
        """Clearing history resets the buffer."""
        pipeline = FeaturePipeline()
        pipeline.register("sma_10", sma, params={"period": 10})
        pipeline.compute(ohlcv.iloc[:50])
        pipeline.clear_history()
        assert pipeline._feature_history.empty

    def test_registered_features_list(self, ohlcv):
        """registered_features() returns current names."""
        pipeline = FeaturePipeline()
        pipeline.register("a", sma, params={"period": 5})
        pipeline.register("b", sma, params={"period": 10})
        assert set(pipeline.registered_features()) == {"a", "b"}

    def test_multi_output_feature(self, ohlcv):
        """Multi-output indicators (e.g. Bollinger Bands) need separate
        registration via lambdas."""
        pipeline = FeaturePipeline()
        pipeline.register(
            "bb_upper",
            lambda close, period, stddev: bollinger_bands(close, period, stddev)[0],
            params={"period": 20, "stddev": 2},
        )
        pipeline.register(
            "bb_middle",
            lambda close, period, stddev: bollinger_bands(close, period, stddev)[1],
            params={"period": 20, "stddev": 2},
        )
        pipeline.register(
            "bb_lower",
            lambda close, period, stddev: bollinger_bands(close, period, stddev)[2],
            params={"period": 20, "stddev": 2},
        )

        result = pipeline.compute(ohlcv)
        for col in ["bb_upper", "bb_middle", "bb_lower"]:
            assert col in result.columns
        # NaN-safe comparison
        valid = result["bb_middle"].notna()
        assert (result.loc[valid, "bb_upper"] >= result.loc[valid, "bb_middle"]).all()
        assert (result.loc[valid, "bb_middle"] >= result.loc[valid, "bb_lower"]).all()

    def test_normalization_none(self, ohlcv):
        """'none' normalization leaves raw values unchanged."""
        pipeline = FeaturePipeline()
        pipeline.register("sma_20", sma, params={"period": 20}, normalization="none")
        result = pipeline.compute(ohlcv)
        # Manually compute SMA to compare
        manual = ohlcv["close"].rolling(20).mean()
        assert np.allclose(result["sma_20"].iloc[19:].values, manual.iloc[19:].values, equal_nan=True)
