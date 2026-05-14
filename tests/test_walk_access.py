"""Test walking access modes.

Verifies Phase 3.2:
  - euclidean_fast returns finite walk times
  - walk_network requires graph
  - Both modes produce same output structure
  - snap_to_nearest_network with mock graph
"""
import pytest
from unittest.mock import MagicMock, patch
import geopandas as gpd
from shapely.geometry import Point
import networkx as nx

from src.walking.access import (
    snap_to_nearest,
    snap_to_nearest_network,
    compute_access_matrix,
    walking_time_m,
    WALK_EUCLIDEAN,
    WALK_NETWORK,
)


def _make_demand_gdf(n: int = 3) -> gpd.GeoDataFrame:
    pts = [Point(117.205 + i * 0.003, 34.268 + i * 0.002) for i in range(n)]
    return gpd.GeoDataFrame({
        "demand_id": [f"D{i}" for i in range(n)],
        "demand_name": [f"Demand {i}" for i in range(n)],
        "geometry": pts,
    }, crs="EPSG:4326")


def _make_rail_gdf(n: int = 2) -> gpd.GeoDataFrame:
    pts = [Point(117.210 + i * 0.005, 34.270 + i * 0.003) for i in range(n)]
    return gpd.GeoDataFrame({
        "station_id": [f"S{i}" for i in range(n)],
        "station_name": [f"Station {i}" for i in range(n)],
        "geometry": pts,
    }, crs="EPSG:4326")


def _make_shelter_gdf(n: int = 2) -> gpd.GeoDataFrame:
    pts = [Point(117.200, 34.260 + i * 0.005) for i in range(n)]
    return gpd.GeoDataFrame({
        "shelter_id": [f"SH{i}" for i in range(n)],
        "shelter_name": [f"Shelter {i}" for i in range(n)],
        "geometry": pts,
    }, crs="EPSG:4326")


class TestWalkAccessEuclidean:

    def test_euclidean_computes_finite_times(self):
        demand = _make_demand_gdf(3)
        rail = _make_rail_gdf(2)
        shelters = _make_shelter_gdf(2)
        result = compute_access_matrix(demand, rail, shelters, mode=WALK_EUCLIDEAN)

        assert len(result["to_rail"]) == 3
        assert len(result["to_shelter"]) == 3
        for a in result["to_rail"]:
            assert a["distance_m"] > 0
            assert a["walk_time_s"] > 0
            assert "origin_id" in a
            assert "target_id" in a

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown walk mode"):
            compute_access_matrix(
                _make_demand_gdf(1), _make_rail_gdf(1), _make_shelter_gdf(1),
                mode="invalid",
            )

    def test_walk_network_requires_graph(self):
        with pytest.raises(ValueError, match="walk_G is required"):
            compute_access_matrix(
                _make_demand_gdf(1), _make_rail_gdf(1), _make_shelter_gdf(1),
                mode=WALK_NETWORK, walk_G=None,
            )


class TestSnapToNearestNetwork:

    def test_network_mode_with_mock_graph(self):
        """snap_to_nearest_network returns correct structure with a mock graph."""
        G = nx.MultiDiGraph()
        nodes = {0: (117.205, 34.268), 1: (117.210, 34.270), 2: (117.215, 34.272)}
        for nid, (x, y) in nodes.items():
            G.add_node(nid, x=x, y=y)
        # Bidirectional edges with lengths in meters
        G.add_edge(0, 1, length=800)
        G.add_edge(1, 0, length=800)
        G.add_edge(1, 2, length=600)
        G.add_edge(2, 1, length=600)

        demand = _make_demand_gdf(1)   # near node 0
        rail = _make_rail_gdf(1)       # near node 1

        # side_effect: 1st call = target pre-map (rail → node 1),
        #              2nd call = origin (demand → node 0)
        with patch("src.walking.access._nearest_node",
                   side_effect=[1, 0]):
            results = snap_to_nearest_network(G, demand, rail)

        assert len(results) == 1
        assert results[0]["distance_m"] == pytest.approx(800, abs=50)
        assert results[0]["walk_time_s"] > 0


class TestWalkingTime:

    def test_normal_speed(self):
        t = walking_time_m(139)  # ~100m at 5km/h
        assert t == pytest.approx(100, abs=5)

    def test_crowded_slower(self):
        t_normal = walking_time_m(100, crowded=False)
        t_crowded = walking_time_m(100, crowded=True)
        assert t_crowded > t_normal

    def test_zero_distance(self):
        assert walking_time_m(0) == 0.0
