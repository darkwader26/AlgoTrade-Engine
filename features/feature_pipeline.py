"""Feature engineering pipeline for algorithmic trading.

The :class:`FeaturePipeline` allows registering indicator functions with
names and parameters, then computing them all at once over raw OHLCV data.
It supports z-score and min-max normalisation, and maintains a feature
history for lookback-dependent calculations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd


class Normalization(str, Enum):
    """Supported normalisation modes."""
    NONE = "none"
    ZSCORE = "zscore"
    MINMAX = "minmax"


@dataclass
class FeatureSpec:
    """Describes a single registered feature."""
    name: str
    func: Callable[..., np.ndarray]
    func_name: str  # name of the indicator function
    params: Dict[str, Any] = field(default_factory=dict)
    input_map: Dict[str, str] = field(default_factory=dict)
    # input_map maps argument names of *func* to OHLCV column names
    # e.g. {"close": "close", "high": "high", ...}
    normalization: Normalization = Normalization.NONE
    enabled: bool = True


class FeaturePipeline:
    """Pipeline that holds a set of registered feature computations.

    Parameters
    ----------
    lookback_buffer : int, default 500
        Number of past rows kept in ``feature_history`` for indicators that
        need historical context beyond the current batch.
    """

    def __init__(self, lookback_buffer: int = 500) -> None:
        self._specs: Dict[str, FeatureSpec] = {}
        self._lookback_buffer = lookback_buffer
        self._feature_history: pd.DataFrame = pd.DataFrame()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        func: Callable[..., np.ndarray],
        *,
        params: Optional[Dict[str, Any]] = None,
        input_map: Optional[Dict[str, str]] = None,
        normalization: Union[str, Normalization] = Normalization.NONE,
        enabled: bool = True,
    ) -> "FeaturePipeline":
        """Register a feature function with its configuration.

        Parameters
        ----------
        name : str
            Unique feature name (e.g. ``"rsi_14"``).
        func : callable
            One of the indicator functions from :mod:`features.indicators`.
        params : dict or None
            Keyword arguments passed to *func* (e.g. ``{"period": 14}``).
        input_map : dict or None
            Maps *func* parameter names to OHLCV DataFrame columns.  Defaults
            to ``{"close": "close", "high": "high", "low": "low",
            "volume": "volume"}`` adjusted for the function signature.
        normalization : str or Normalization, default "none"
            Normalisation to apply after computing the feature.
        enabled : bool, default True
            Whether the feature is active.
        """
        if name in self._specs:
            raise KeyError(f"Feature '{name}' is already registered.")

        params = params or {}
        if input_map is None:
            input_map = self._infer_input_map(func)

        self._specs[name] = FeatureSpec(
            name=name,
            func=func,
            func_name=func.__name__,
            params=params,
            input_map=input_map,
            normalization=Normalization(normalization),
            enabled=enabled,
        )
        return self

    def unregister(self, name: str) -> None:
        """Remove a previously registered feature."""
        self._specs.pop(name, None)

    def registered_features(self) -> List[str]:
        """Return list of registered feature names."""
        return list(self._specs.keys())

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute(self, raw_ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Compute all registered (enabled) features on *raw_ohlcv*.

        Parameters
        ----------
        raw_ohlcv : pd.DataFrame
            Must contain columns matching the registered ``input_map`` values
            (typically ``open``, ``high``, ``low``, ``close``, ``volume``).

        Returns
        -------
        pd.DataFrame
            Original OHLCV data augmented with computed feature columns.
            Feature columns are named ``<registered_name>``.
        """
        self._validate_columns(raw_ohlcv)

        # Merge history so lookback-dependent indicators work correctly
        if not self._feature_history.empty:
            combined = pd.concat(
                [self._feature_history, raw_ohlcv],
                axis=0,
                ignore_index=True,
            )
        else:
            combined = raw_ohlcv.copy()

        result = raw_ohlcv.copy()

        for spec in self._specs.values():
            if not spec.enabled:
                continue

            feature = self._compute_spec(spec, combined)
            # Align to the length of the original (current) slice
            result[spec.name] = feature.iloc[-len(raw_ohlcv):].values

        # Update history
        self._feature_history = combined.iloc[
            max(0, len(combined) - self._lookback_buffer):
        ].reset_index(drop=True)

        return result

    def compute_single(
        self,
        feature_name: str,
        raw_ohlcv: pd.DataFrame,
    ) -> pd.Series:
        """Compute a single named feature on *raw_ohlcv*.

        Parameters
        ----------
        feature_name : str
            Name of a previously registered feature.
        raw_ohlcv : pd.DataFrame
            OHLCV data.

        Returns
        -------
        pd.Series
            Feature values.
        """
        spec = self._specs.get(feature_name)
        if spec is None:
            raise KeyError(
                f"Feature '{feature_name}' is not registered. "
                f"Registered: {list(self._specs.keys())}"
            )
        self._validate_columns(raw_ohlcv)

        if not self._feature_history.empty:
            combined = pd.concat(
                [self._feature_history, raw_ohlcv],
                axis=0,
                ignore_index=True,
            )
        else:
            combined = raw_ohlcv.copy()

        feature = self._compute_spec(spec, combined)
        return feature.iloc[-len(raw_ohlcv):]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_spec(self, spec: FeatureSpec, data: pd.DataFrame) -> pd.Series:
        """Execute one feature specification and return a Series."""
        # Build the keyword arguments for the indicator function
        kwargs: Dict[str, Any] = {}
        for func_param, col in spec.input_map.items():
            kwargs[func_param] = data[col].values.astype(np.float64)
        kwargs.update(spec.params)

        raw = spec.func(**kwargs)

        # Apply normalisation
        if spec.normalization == Normalization.ZSCORE:
            raw = self._zscore(raw)
        elif spec.normalization == Normalization.MINMAX:
            raw = self._minmax(raw)

        return pd.Series(raw, index=data.index, name=spec.name)

    @staticmethod
    def _infer_input_map(func: Callable) -> Dict[str, str]:
        """Guess the column mapping from the function's parameter names."""
        import inspect

        sig = inspect.signature(func)
        known_columns = {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
        mapping: Dict[str, str] = {}
        for param in sig.parameters:
            if param in known_columns:
                mapping[param] = known_columns[param]
        # Fallback: map 'close' by default if nothing was inferred
        if not mapping and "close" in sig.parameters:
            mapping["close"] = "close"
        return mapping

    @staticmethod
    def _validate_columns(df: pd.DataFrame) -> None:
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns.str.lower())
        if missing:
            raise ValueError(
                f"DataFrame must contain columns: {required}. "
                f"Missing: {missing}"
            )

    @staticmethod
    def _zscore(arr: np.ndarray) -> np.ndarray:
        """In-place z-score normalisation (ignores NaN)."""
        mean = np.nanmean(arr)
        std = np.nanstd(arr)
        if std == 0:
            return arr - mean
        return (arr - mean) / std

    @staticmethod
    def _minmax(arr: np.ndarray) -> np.ndarray:
        """In-place min-max normalisation to [0, 1] (ignores NaN)."""
        min_val = np.nanmin(arr)
        max_val = np.nanmax(arr)
        if max_val == min_val:
            return arr - min_val
        return (arr - min_val) / (max_val - min_val)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def clear_history(self) -> None:
        """Reset the internal feature history buffer."""
        self._feature_history = pd.DataFrame()
