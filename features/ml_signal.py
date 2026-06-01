"""ML Signal Generation module.

Provides :class:`MLSignalGenerator` that trains a Random Forest classifier
on feature-engineered data to produce buy/sell signals with confidence
scores.  Falls back gracefully when scikit-learn is not installed.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
#  Conditional sklearn import
# ---------------------------------------------------------------------------

_HAS_SKLEARN = False
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
#  Helper: prepare training data
# ---------------------------------------------------------------------------


def prepare_training_data(
    feature_df: pd.DataFrame,
    forward_returns: pd.Series,
    horizon: int = 5,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare aligned feature matrix and labels for supervised learning.

    The label is 1 if the forward return (over *horizon* periods) is
    positive, 0 otherwise.  Rows where the forward return cannot be
    computed (end of series) are dropped.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Feature matrix (rows = timestamps, columns = features).
    forward_returns : pd.Series
        Price returns (can be raw percentages or log returns).  Must be
        aligned with ``feature_df`` index.
    horizon : int, default 5
        Number of periods to look ahead.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with NaN rows dropped, aligned with ``y``.
    y : pd.Series
        Binary labels (1 = positive forward return, 0 = otherwise).
    """
    # Compute forward returns
    fwd = forward_returns.shift(-horizon)

    # Create binary labels
    labels = pd.Series(
        np.where(fwd > 0, 1, 0),
        index=forward_returns.index,
        dtype=int,
    )

    # Drop NaN rows (both in features and labels)
    valid = labels.notna() & (~feature_df.isna().any(axis=1))
    X = feature_df.loc[valid].copy()
    y = labels.loc[valid].copy()

    return X, y


# ---------------------------------------------------------------------------
#  MLSignalGenerator
# ---------------------------------------------------------------------------


class MLSignalGenerator:
    """Generate trading signals using a Random Forest classifier.

    Falls back gracefully (returns neutral signals) when scikit-learn is
    not installed.

    Parameters
    ----------
    n_estimators : int, default 100
        Number of trees in the Random Forest.
    max_depth : int or None, default 5
        Maximum tree depth (``None`` = unlimited).
    random_state : int, default 42
        Random seed for reproducibility.
    confidence_threshold : float, default 0.6
        Minimum predicted probability to emit a non-neutral signal.
        Below this threshold the signal is marked neutral.
    test_size : float, default 0.2
        Fraction of data held out for validation during ``train()``.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 5,
        random_state: int = 42,
        confidence_threshold: float = 0.6,
        test_size: float = 0.2,
    ) -> None:
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._random_state = random_state
        self._confidence_threshold = confidence_threshold
        self._test_size = test_size

        self._model: Any = None
        self._feature_names: Optional[List[str]] = None
        self._is_trained = False
        self._train_accuracy: Optional[float] = None
        self._test_accuracy: Optional[float] = None
        self._feature_importance: Optional[pd.Series] = None

    # ------------------------------------------------------------------
    #  Training
    # ------------------------------------------------------------------

    def train(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        *,
        ret: bool = True,
    ) -> Dict[str, Any]:
        """Train the model on provided features and labels.

        Parameters
        ----------
        features : pd.DataFrame
            Training feature matrix.
        labels : pd.Series
            Training labels (0/1).
        ret : bool, default True
            If True, return a results dict.

        Returns
        -------
        dict
            ``{"accuracy": float, "test_accuracy": float,
            "feature_importance": pd.Series}``.
            Returns an error dict if sklearn is unavailable.
        """
        if not _HAS_SKLEARN:
            msg = (
                "scikit-learn is not installed.  Install with: "
                "pip install scikit-learn>=1.2"
            )
            warnings.warn(msg)
            self._is_trained = False
            return {"error": msg}

        # Drop rows with NaN
        valid = labels.notna() & (~features.isna().any(axis=1))
        X = features.loc[valid].values
        y = labels.loc[valid].values

        if len(X) < 10:
            raise ValueError(
                f"Too few valid samples ({len(X)}) to train. "
                "Need at least 10."
            )

        self._feature_names = list(features.columns)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self._test_size,
            random_state=self._random_state,
            stratify=y,
        )

        # Train
        self._model = RandomForestClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            random_state=self._random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
        self._model.fit(X_train, y_train)

        # Evaluate
        y_train_pred = self._model.predict(X_train)
        y_test_pred = self._model.predict(X_test)

        self._train_accuracy = float(accuracy_score(y_train, y_train_pred))
        self._test_accuracy = float(accuracy_score(y_test, y_test_pred))
        self._feature_importance = pd.Series(
            self._model.feature_importances_,
            index=self._feature_names,
            name="importance",
        ).sort_values(ascending=False)
        self._is_trained = True

        if ret:
            return {
                "train_accuracy": self._train_accuracy,
                "test_accuracy": self._test_accuracy,
                "feature_importance": self._feature_importance,
            }
        return {}

    # ------------------------------------------------------------------
    #  Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate trading signals with confidence scores.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix (same columns as training).

        Returns
        -------
        pd.DataFrame
            Columns:
            - ``ml_signal``: predicted class (1 = bullish, 0 = bearish/neutral)
            - ``ml_confidence``: probability of the predicted class
            - ``ml_direction``: BULLISH (1), NEUTRAL (0), or BEARISH (-1)
        """
        index = features.index

        # Fallback if sklearn unavailable or model not trained
        if not _HAS_SKLEARN or not self._is_trained or self._model is None:
            result = pd.DataFrame(index=index)
            result["ml_signal"] = 0
            result["ml_confidence"] = 0.0
            result["ml_direction"] = 0
            return result

        # Ensure feature columns match
        if self._feature_names is not None:
            missing = set(self._feature_names) - set(features.columns)
            if missing:
                raise KeyError(
                    f"Features missing for prediction: {missing}"
                )
            X = features[self._feature_names].values
        else:
            X = features.values

        # Handle NaN: predict_proba can't handle NaN, so we fill with 0
        X_clean = np.nan_to_num(X, nan=0.0)

        preds = self._model.predict(X_clean)
        probs = self._model.predict_proba(X_clean)

        # Probability of class 1 (bullish)
        if self._model.classes_[1] == 1:
            bullish_prob = probs[:, 1]
        else:
            bullish_prob = probs[:, 0]

        # Apply confidence threshold
        direction = np.where(
            bullish_prob >= self._confidence_threshold,
            1,  # BULLISH
            np.where(
                1 - bullish_prob >= self._confidence_threshold,
                -1,  # BEARISH
                0,  # NEUTRAL
            ),
        )

        result = pd.DataFrame(index=index)
        result["ml_signal"] = preds
        result["ml_confidence"] = np.where(
            direction != 0,
            np.maximum(bullish_prob, 1 - bullish_prob),
            0.0,
        )
        result["ml_direction"] = direction
        return result

    # ------------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def has_sklearn(self) -> bool:
        return _HAS_SKLEARN

    @property
    def model(self) -> Any:
        return self._model

    @property
    def train_accuracy(self) -> Optional[float]:
        return self._train_accuracy

    @property
    def test_accuracy(self) -> Optional[float]:
        return self._test_accuracy

    @property
    def feature_importance(self) -> Optional[pd.Series]:
        return self._feature_importance
