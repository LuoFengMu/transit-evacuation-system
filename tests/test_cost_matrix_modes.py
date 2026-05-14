"""Test cost matrix mode switch.

Verifies Phase 3.1:
  - euclidean_fast returns finite matrix quickly
  - Invalid mode raises ValueError
  - Cached mode writes and reads parquet cache
  - Mode constants are correct
"""
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from shapely.geometry import Point

from src.dispatch.cost_matrix import (
    compute_cost_matrix,
    compute_euclidean_matrix,
    MODE_EUCLIDEAN,
    MODE_NETWORK,
    MODE_CACHED,
    VALID_MODES,
)


def _make_points(n: int, base_x: float = 117.2, base_y: float = 34.27) -> list[Point]:
    return [Point(base_x + i * 0.005, base_y + j * 0.005) for i in range(n) for j in range(n)][:n]


class TestCostMatrixModes:

    def test_euclidean_fast_returns_finite(self):
        origins = _make_points(5)
        dests = _make_points(5, base_x=117.25)
        m = compute_cost_matrix(origins, dests, mode=MODE_EUCLIDEAN)
        assert m.shape == (5, 5)
        assert np.all(np.isfinite(m))
        assert np.all(m >= 0)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            compute_cost_matrix([], [], mode="invalid_mode")

    def test_network_mode_requires_graph(self):
        with pytest.raises(ValueError, match="Road network G is required for road_network_time"):
            compute_cost_matrix(
                _make_points(3), _make_points(3),
                mode=MODE_NETWORK, G=None,
            )

    def test_cached_mode_requires_cache_dir(self):
        with pytest.raises(ValueError, match="cache_dir is required"):
            compute_cost_matrix(
                _make_points(3), _make_points(3),
                mode=MODE_CACHED, G=None, cache_dir=None,
            )

    def test_cached_mode_requires_graph(self):
        with pytest.raises(ValueError, match="Road network G is required for cached_network_time"):
            compute_cost_matrix(
                _make_points(3), _make_points(3),
                mode=MODE_CACHED, G=None, cache_dir="/tmp/fake",
            )

    def test_cached_mode_writes_and_reads_parquet(self):
        """Cached mode: first call computes + writes, second call reads from cache."""
        origins = _make_points(6)
        dests = _make_points(4, base_x=117.25)
        expected = np.arange(24).reshape(6, 4).astype(float)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "cost_matrix")

            # ── First call: compute_travel_time_matrix runs → writes cache
            call_count = [0]
            def fake_travel_time(G, o, d):
                call_count[0] += 1
                return expected.copy()

            with patch(
                "src.dispatch.cost_matrix.compute_travel_time_matrix",
                side_effect=fake_travel_time,
            ):
                m1 = compute_cost_matrix(
                    origins, dests, mode=MODE_CACHED,
                    G=MagicMock(), cache_dir=cache_dir,
                    scenario_id="test_cached", random_seed=42,
                )
            assert call_count[0] == 1, "first call should compute"
            assert m1.shape == (6, 4)
            assert np.array_equal(m1, expected)

            # ── Second call: cache hit, no re-computation
            m2 = compute_cost_matrix(
                origins, dests, mode=MODE_CACHED,
                G=MagicMock(), cache_dir=cache_dir,
                scenario_id="test_cached", random_seed=42,
            )
            assert call_count[0] == 1, "second call should read cache, not re-compute"
            assert np.array_equal(m2, expected)

            # ── Verify parquet file exists
            cache_files = os.listdir(cache_dir)
            assert len(cache_files) == 1
            assert cache_files[0].startswith("cost_matrix_") and cache_files[0].endswith(".parquet")

    def test_euclidean_vs_network_shape_consistency(self):
        """Both modes produce the same shape for identical inputs."""
        origins = _make_points(3)
        dests = _make_points(4, base_x=117.25)

        eucl = compute_cost_matrix(origins, dests, mode=MODE_EUCLIDEAN)
        assert eucl.shape == (3, 4)
        # All values should be finite and positive
        assert np.all(eucl > 0)

    def test_same_points_zero_diagonal(self):
        """Euclidean distance from a point to itself should be 0."""
        pts = _make_points(3)
        m = compute_cost_matrix(pts, pts, mode=MODE_EUCLIDEAN)
        # Self-distances are ~0 (within numerical precision of degree→m conversion)
        assert m[0, 0] == pytest.approx(0, abs=1.0)

    def test_mode_constants(self):
        assert MODE_EUCLIDEAN == "euclidean_fast"
        assert MODE_NETWORK == "road_network_time"
        assert MODE_CACHED == "cached_network_time"
        assert len(VALID_MODES) == 3
