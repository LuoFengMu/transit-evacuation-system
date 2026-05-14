"""Test SUMO trip departure times are deterministic.

Verifies Phase 1.4 P0 fix:
  - hash(vid) % 60 → vehicle_index % 60
  - Same inputs produce identical departure times across runs
"""
import os
import tempfile
import xml.etree.ElementTree as ET
from unittest import mock
from dataclasses import dataclass, field

import pytest

from src.simulation.route_builder import dispatch_to_sumo_trips


@dataclass
class MockDispatchResult:
    vehicle_routes: dict = field(default_factory=dict)
    sub_demand_quantities: list = field(default_factory=list)
    unserved_demand: list = field(default_factory=list)
    split_origin_map: dict = field(default_factory=dict)


@dataclass
class MockDepot:
    lon: float
    lat: float


def _parse_depart_times(trip_path: str) -> dict[str, float]:
    """Extract {trip_id: depart_time} from a SUMO trip XML file."""
    tree = ET.parse(trip_path)
    departures = {}
    for trip in tree.findall(".//trip"):
        tid = trip.get("id", "")
        depart = float(trip.get("depart", "0"))
        departures[tid] = depart
    return departures


class TestRouteBuilderDeterminism:

    def test_departure_times_deterministic_across_runs(self):
        """Two calls with identical inputs produce identical depart times."""
        # Build a deterministic dispatch result with 5 vehicles
        dispatch = MockDispatchResult(vehicle_routes={
            "bus_00": [("depot", "depot_00", 0), ("pickup", 0, 300)],
            "bus_01": [("depot", "depot_00", 0), ("pickup", 1, 300)],
            "bus_02": [("depot", "depot_00", 0), ("pickup", 2, 300)],
            "bus_03": [("depot", "depot_00", 0), ("pickup", 3, 300)],
            "bus_04": [("depot", "depot_00", 0), ("pickup", 4, 300)],
        }, split_origin_map={0: 0, 1: 1, 2: 2, 3: 3, 4: 4})

        depots = [MockDepot(lon=117.205, lat=34.268)]
        from shapely.geometry import Point
        demand_gdf = type("GDF", (), {"geometry": [Point(117.210, 34.275)] * 5})()

        # Mock _find_nearest_edge to return stable fake edge IDs
        def fake_edge(lon, lat, network_path):
            return f"edge_{(lon * 1000 + lat * 1000) % 1000:.0f}"
        # But we need deterministic edge IDs. Use fixed ones:
        call_count = [0]
        def fake_edge_stable(lon, lat, network_path):
            call_count[0] += 1
            return f"edge_{call_count[0]:04d}"

        with tempfile.TemporaryDirectory() as tmpdir:
            net_path = os.path.join(tmpdir, "fake.net.xml")
            # Create minimal SUMO-like XML
            with open(net_path, "w") as f:
                f.write('<?xml version="1.0"?><net><location netOffset="0,0" projParameter="+proj=utm +zone=50"/></net>')

            with mock.patch("src.simulation.route_builder._find_nearest_edge", side_effect=fake_edge_stable):
                trip1, _ = dispatch_to_sumo_trips(
                    dispatch, vehicles=None, depots=depots,
                    demand_gdf=demand_gdf, output_dir=tmpdir,
                    network_path=net_path, max_rounds=1,
                )

            depart1 = _parse_depart_times(trip1)
            assert len(depart1) == 5, f"Expected 5 trips, got {len(depart1)}"

            # Reset counter and run again
            call_count[0] = 0
            with mock.patch("src.simulation.route_builder._find_nearest_edge", side_effect=fake_edge_stable):
                trip2, _ = dispatch_to_sumo_trips(
                    dispatch, vehicles=None, depots=depots,
                    demand_gdf=demand_gdf, output_dir=tmpdir,
                    network_path=net_path, max_rounds=1,
                )

            depart2 = _parse_depart_times(trip2)

            # Departure times should be identical
            for tid in depart1:
                assert tid in depart2, f"Trip {tid} missing in second run"
                assert depart1[tid] == depart2[tid], \
                    f"Trip {tid}: depart {depart1[tid]} vs {depart2[tid]}"

    def test_departure_is_vehicle_index_mod_60(self):
        """First vehicle depart = 0.0, second = 1.0, ..., 60th = 0.0."""
        n_vehicles = 65
        routes = {}
        for i in range(n_vehicles):
            routes[f"bus_{i:02d}"] = [("depot", "depot_00", 0), ("pickup", 0, 300)]

        dispatch = MockDispatchResult(
            vehicle_routes=routes,
            split_origin_map={0: 0},
        )
        depots = [MockDepot(lon=117.205, lat=34.268)]
        from shapely.geometry import Point
        demand_gdf = type("GDF", (), {"geometry": [Point(117.210, 34.275)]})()

        counter = [0]
        def fake_edge(lon, lat, network_path):
            counter[0] += 1
            return f"edge_{counter[0]:04d}"

        with tempfile.TemporaryDirectory() as tmpdir:
            net_path = os.path.join(tmpdir, "fake.net.xml")
            with open(net_path, "w") as f:
                f.write('<?xml version="1.0"?><net><location netOffset="0,0" projParameter="+proj=utm +zone=50"/></net>')

            with mock.patch("src.simulation.route_builder._find_nearest_edge", side_effect=fake_edge):
                trip_path, _ = dispatch_to_sumo_trips(
                    dispatch, vehicles=None, depots=depots,
                    demand_gdf=demand_gdf, output_dir=tmpdir,
                    network_path=net_path, max_rounds=1,
                )

            departures = _parse_depart_times(trip_path)

            # Check depart times for specific vehicles
            assert departures["bus_00_r0_leg0"] == 0.0
            assert departures["bus_01_r0_leg0"] == 1.0
            assert departures["bus_59_r0_leg0"] == 59.0
            assert departures["bus_60_r0_leg0"] == 0.0  # wraps
            assert departures["bus_64_r0_leg0"] == 4.0

    def test_same_inputs_same_departures_regardless_of_hash_seed(self):
        """Even if PYTHONHASHSEED changes, departures stay the same."""
        dispatch = MockDispatchResult(vehicle_routes={
            f"bus_{i:02d}": [("depot", "depot_00", 0), ("pickup", 0, 300)]
            for i in range(10)
        }, split_origin_map={0: 0})
        depots = [MockDepot(lon=117.205, lat=34.268)]
        from shapely.geometry import Point
        demand_gdf = type("GDF", (), {"geometry": [Point(117.210, 34.275)]})()

        # Run with fixed hash seed
        import subprocess
        import sys

        # We test in-process: departure should NOT depend on hash(vid)
        # The old code used hash(vid) % 60; new code uses vehicle_index % 60
        # Verify by computing expected departures: vehicle_index % 60
        counter = [0]
        def fake_edge(lon, lat, network_path):
            counter[0] += 1
            return f"edge_{counter[0]:04d}"

        with tempfile.TemporaryDirectory() as tmpdir:
            net_path = os.path.join(tmpdir, "fake.net.xml")
            with open(net_path, "w") as f:
                f.write('<?xml version="1.0"?><net><location netOffset="0,0" projParameter="+proj=utm +zone=50"/></net>')

            with mock.patch("src.simulation.route_builder._find_nearest_edge", side_effect=fake_edge):
                trip_path, _ = dispatch_to_sumo_trips(
                    dispatch, vehicles=None, depots=depots,
                    demand_gdf=demand_gdf, output_dir=tmpdir,
                    network_path=net_path, max_rounds=1,
                )

            departures = _parse_depart_times(trip_path)
            # Each vehicle has exactly 1 leg, so depart = vehicle_index % 60
            for i in range(10):
                tid = f"bus_{i:02d}_r0_leg0"
                expected = float(i % 60)
                assert departures[tid] == expected, \
                    f"Vehicle {i}: expected depart {expected}, got {departures[tid]}"
