"""Tests for FeatureCache (LRU memoization layer)."""

import numpy as np

from features.cache import FeatureCache


def _dummy_compute() -> np.ndarray:
    return np.array([1.0, 2.0, 3.0])


def _compute_with_state(state) -> callable:
    """Return a compute function that tracks invocation count."""
    def _inner():
        state["calls"] += 1
        return np.array([1.0, 2.0, 3.0])
    return _inner


class TestFeatureCache:
    def test_basic_get_and_hit(self):
        """A value computed once is returned from cache on second call."""
        cache = FeatureCache(maxsize=64)
        arr = np.array([10.0, 20.0, 30.0])
        state = {"calls": 0}

        result1 = cache.get(
            "sma",
            arrays={"close": arr},
            params={"period": 3},
            compute_fn=_compute_with_state(state),
        )
        assert state["calls"] == 1
        assert np.allclose(result1, [1.0, 2.0, 3.0])

        result2 = cache.get(
            "sma",
            arrays={"close": arr},
            params={"period": 3},
            compute_fn=_compute_with_state(state),
        )
        assert state["calls"] == 1  # not incremented
        assert np.allclose(result2, [1.0, 2.0, 3.0])

        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_ratio"] == 0.5

    def test_different_params_produce_different_cache_entries(self):
        """Different parameters (e.g. period) → separate cache slots."""
        cache = FeatureCache(maxsize=64)
        arr = np.array([10.0, 20.0, 30.0])

        state_a = {"calls": 0}
        state_b = {"calls": 0}

        cache.get("sma", arrays={"close": arr}, params={"period": 3}, compute_fn=_compute_with_state(state_a))
        cache.get("sma", arrays={"close": arr}, params={"period": 5}, compute_fn=_compute_with_state(state_b))

        assert state_a["calls"] == 1
        assert state_b["calls"] == 1
        assert cache.stats()["misses"] == 2

    def test_lru_eviction(self):
        """Oldest entries are evicted when maxsize is exceeded."""
        cache = FeatureCache(maxsize=3)

        for i in range(5):
            arr = np.array([float(i)])
            cache.get(
                f"feat_{i}",
                arrays={"arr": arr},
                params={},
                compute_fn=lambda arr=arr: arr.copy(),
            )

        assert len(cache) == 3  # only last 3 survive

        # First two should be evicted
        evicted = cache.get(
            "feat_0",
            arrays={"arr": np.array([0.0])},
            params={},
            compute_fn=lambda: np.array([999.0]),
        )
        # Since feat_0 was evicted, it got recomputed
        assert evicted[0] == 999.0

    def test_clear_resets_everything(self):
        """Clear removes all entries and resets counters."""
        cache = FeatureCache(maxsize=16)
        arr = np.array([1.0, 2.0])

        cache.get("ema", arrays={"close": arr}, params={"period": 3}, compute_fn=_dummy_compute)
        cache.clear()

        assert len(cache) == 0
        assert cache.stats()["hits"] == 0
        assert cache.stats()["misses"] == 0

    def test_different_arrays_produce_different_keys(self):
        """Different input data → separate cache entries."""
        cache = FeatureCache(maxsize=64)
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([4.0, 5.0, 6.0])

        state1 = {"calls": 0}
        state2 = {"calls": 0}

        cache.get("sma", arrays={"close": arr1}, params={"period": 3}, compute_fn=_compute_with_state(state1))
        cache.get("sma", arrays={"close": arr2}, params={"period": 3}, compute_fn=_compute_with_state(state2))

        assert state1["calls"] == 1
        assert state2["calls"] == 1

    def test_contains(self):
        """__contains__ works correctly."""
        cache = FeatureCache(maxsize=16)
        arr = np.array([1.0])
        key = cache._make_key("test", {"arr": arr}, {})

        assert key not in cache
        cache.get("test", arrays={"arr": arr}, params={}, compute_fn=_dummy_compute)
        assert key in cache

    def test_empty_array_handling(self):
        """Empty arrays should not crash the cache."""
        cache = FeatureCache(maxsize=16)
        arr = np.array([])

        result = cache.get("test", arrays={"close": arr}, params={}, compute_fn=lambda: np.array([]))
        assert len(result) == 0
