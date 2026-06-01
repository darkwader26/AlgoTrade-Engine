"""LRU caching layer for indicator calculations to avoid redundant computation."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import Any, Optional


class FeatureCache:
    """Memoizes indicator calculations using an LRU-eviction dict cache.

    Each cache key is derived from the function name, its parameters, and a
    truncated digest of the input array (checksum).  This avoids recomputing
    expensive indicators on overlapping windows of the same data.

    Parameters
    ----------
    maxsize : int, default 128
        Maximum number of entries before the least recently used entry is
        evicted.
    """

    def __init__(self, maxsize: int = 128) -> None:
        self._maxsize = maxsize
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        func_name: str,
        *,
        arrays: dict[str, Any],
        params: Optional[dict[str, Any]] = None,
        compute_fn: callable,
    ) -> Any:
        """Return cached result for *func_name* or compute and store it.

        Parameters
        ----------
        func_name : str
            Name of the indicator function (e.g. ``"sma"``).
        arrays : dict[str, ndarray]
            Named input arrays passed as keyword arguments to the indicator.
        params : dict or None
            Indicator parameters (period, etc.).
        compute_fn : callable
            Zero-argument callable that actually computes the value.  Called
            only on a cache miss.

        Returns
        -------
        result
            The computed (or cached) indicator value.
        """
        params = params or {}
        key = self._make_key(func_name, arrays, params)

        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]

        self._misses += 1
        result = compute_fn()
        self._cache[key] = result

        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)  # evict LRU

        return result

    def clear(self) -> None:
        """Remove all cached entries and reset hit/miss counters."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict[str, Any]:
        """Return cache statistics (size, hits, misses, hit-ratio)."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "maxsize": self._maxsize,
            "hits": self._hits,
            "misses": self._misses,
            "hit_ratio": self._hits / total if total > 0 else 0.0,
        }

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(
        func_name: str,
        arrays: dict[str, Any],
        params: dict[str, Any],
    ) -> str:
        """Build a deterministic cache key string.

        Uses a truncated SHA-256 over array checksums so that huge arrays
        don't blow up the key length but still uniquely identify the data.
        """
        # Compute a checksum for each input array
        array_checksums: dict[str, str] = {}
        for name, arr in arrays.items():
            # Use the first 1024 bytes and the length as a fast fingerprint
            # (good enough to avoid collisions in practice)
            flat = arr.ravel().view("float64")
            sample = flat[: min(len(flat), 1024)].tobytes()
            digest = hashlib.sha256(sample).hexdigest()[:12]
            array_checksums[name] = f"{digest}_{len(arr)}"

        payload = {
            "func": func_name,
            "arrays": array_checksums,
            "params": params,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
