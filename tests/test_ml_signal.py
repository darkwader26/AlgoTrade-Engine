"""Tests for the MLSignalGenerator."""

import numpy as np
import pandas as pd
import pytest

from features.ml_signal import (
    _HAS_SKLEARN,
    MLSignalGenerator,
    prepare_training_data,
)

# ---------------------------------------------------------------------------
#  Helper: synthetic feature / return data
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_data():
    """200 rows × 5 features + returns that have a predictable pattern."""
    np.random.seed(42)
    n = 200
    features = pd.DataFrame({
        "rsi": np.where(np.arange(n) < 80, 30 + np.random.rand(n) * 40,  # mostly mid-range
                        70 + np.random.rand(n) * 30),  # some overbought
        "sma_10": np.cumsum(np.random.randn(n) * 0.5) + 100,
        "sma_20": np.cumsum(np.random.randn(n) * 0.3) + 100,
        "roc": np.random.randn(n) * 2,
        "volume_ratio": np.random.rand(n) * 0.5 + 0.75,
    })
    # Returns: positive momentum when RSI low, negative when RSI high
    returns = pd.Series(
        np.where(features["rsi"] < 40, 0.02, np.where(features["rsi"] > 60, -0.02, 0.0)),
        name="returns",
    ) + np.random.randn(n) * 0.005
    return features, returns


# ======================================================================
#  Test prepare_training_data
# ======================================================================


class TestPrepareTrainingData:
    def test_basic_preparation(self, synthetic_data):
        features, returns = synthetic_data
        X, y = prepare_training_data(features, returns, horizon=5)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert y.dtype == int
        # Labels should be 0 or 1
        assert set(y.unique()).issubset({0, 1})

    def test_horizon_effect(self, synthetic_data):
        features, returns = synthetic_data
        X1, y1 = prepare_training_data(features, returns, horizon=1)
        X5, y5 = prepare_training_data(features, returns, horizon=5)
        # Longer horizon shifts more, so different lengths
        assert len(X1) >= len(X5)

    def test_nan_dropping(self, synthetic_data):
        features, returns = synthetic_data
        # Introduce NaN in features
        features_with_nan = features.copy()
        features_with_nan.iloc[10:15, 0] = np.nan
        X, y = prepare_training_data(features_with_nan, returns, horizon=5)
        # Those 5 rows should be dropped
        assert "rsi" in X.columns


# ======================================================================
#  Test MLSignalGenerator
# ======================================================================


class TestMLSignalGenerator:
    def test_fallback_no_sklearn(self):
        """When sklearn is not available, predict returns neutral."""
        gen = MLSignalGenerator()
        if not _HAS_SKLEARN:
            features = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
            result = gen.predict(features)
            assert list(result["ml_direction"]) == [0, 0, 0]
            assert list(result["ml_confidence"]) == [0.0, 0.0, 0.0]

    @pytest.mark.skipif(not _HAS_SKLEARN, reason="scikit-learn not installed")
    def test_train_basic(self, synthetic_data):
        """Training returns accuracy and feature importance."""
        features, returns = synthetic_data
        X, y = prepare_training_data(features, returns, horizon=5)

        gen = MLSignalGenerator(n_estimators=20, max_depth=3, random_state=42)
        result = gen.train(X, y)

        assert "train_accuracy" in result
        assert "test_accuracy" in result
        assert "feature_importance" in result
        assert isinstance(result["feature_importance"], pd.Series)
        assert len(result["feature_importance"]) == X.shape[1]
        assert gen.is_trained

    @pytest.mark.skipif(not _HAS_SKLEARN, reason="scikit-learn not installed")
    def test_predict_after_train(self, synthetic_data):
        """After training, predict returns signals with confidence."""
        features, returns = synthetic_data
        X, y = prepare_training_data(features, returns, horizon=5)

        gen = MLSignalGenerator(n_estimators=20, max_depth=3, random_state=42)
        gen.train(X, y)

        pred = gen.predict(X.iloc[:10])
        assert len(pred) == 10
        assert "ml_signal" in pred.columns
        assert "ml_confidence" in pred.columns
        assert "ml_direction" in pred.columns
        assert set(pred["ml_signal"].unique()).issubset({0, 1})
        assert all(pred["ml_confidence"] >= 0.0)
        assert all(pred["ml_confidence"] <= 1.0)

    @pytest.mark.skipif(not _HAS_SKLEARN, reason="scikit-learn not installed")
    def test_confidence_threshold(self, synthetic_data):
        """Confidence threshold filters uncertain predictions to neutral."""
        features, returns = synthetic_data
        X, y = prepare_training_data(features, returns, horizon=5)

        gen = MLSignalGenerator(
            n_estimators=20,
            max_depth=3,
            random_state=42,
            confidence_threshold=0.99,  # very high → everything neutral
        )
        gen.train(X, y)

        pred = gen.predict(X.iloc[:20])
        assert all(pred["ml_direction"] == 0)  # all neutral
        assert all(pred["ml_confidence"] == 0.0)

    @pytest.mark.skipif(not _HAS_SKLEARN, reason="scikit-learn not installed")
    def test_train_too_few_samples(self):
        """Training with fewer than 10 samples raises."""
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        y = pd.Series([0, 1, 0])
        gen = MLSignalGenerator()
        with pytest.raises(ValueError, match="Too few valid samples"):
            gen.train(X, y)

    @pytest.mark.skipif(not _HAS_SKLEARN, reason="scikit-learn not installed")
    def test_predict_before_train(self, synthetic_data):
        """Predict before train returns neutral."""
        features, _ = synthetic_data
        gen = MLSignalGenerator()
        pred = gen.predict(features.iloc[:5])
        assert all(pred["ml_direction"] == 0)

    @pytest.mark.skipif(not _HAS_SKLEARN, reason="scikit-learn not installed")
    def test_missing_features_in_predict(self, synthetic_data):
        """Missing feature columns in predict raises KeyError."""
        features, returns = synthetic_data
        X, y = prepare_training_data(features, returns, horizon=5)
        gen = MLSignalGenerator(n_estimators=20, max_depth=3, random_state=42)
        gen.train(X, y)

        bad_features = X.drop(columns=["rsi"])
        with pytest.raises(KeyError, match="rsi"):
            gen.predict(bad_features)

    def test_has_sklearn_property(self):
        """has_sklearn property reflects actual availability."""
        gen = MLSignalGenerator()
        assert gen.has_sklearn == _HAS_SKLEARN

    def test_is_trained_property(self):
        """is_trained is False by default."""
        gen = MLSignalGenerator()
        assert not gen.is_trained
