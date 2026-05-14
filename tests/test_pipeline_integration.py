"""Integration test for run_analysis() pipeline.

Verifies Phase 4.2.1:
  - Pipeline returns expected result keys
  - Output directory contains 7 artifact files
  - Same params produce deterministic results
  - Config is reproducible

Mock policy: only mock I/O and heavy computation (OSMnx, SUMO, file system).
Core logic (OR-Tools dispatch, metrics, rail allocation) runs unmocked.
"""
import os
import tempfile
from unittest.mock import patch, MagicMock

import yaml
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point

from src.app.pipeline import run_analysis


def _build_test_graph() -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:4326"
    for i in range(10):
        G.add_node(i, x=117.205 + i * 0.003, y=34.268 + i * 0.002)
    for u in G.nodes:
        for v in G.nodes:
            if u != v:
                ux, uy = G.nodes[u]["x"], G.nodes[u]["y"]
                vx, vy = G.nodes[v]["x"], G.nodes[v]["y"]
                d = ((ux - vx)**2 + (uy - vy)**2)**0.5 * 111320
                G.add_edge(u, v, length=d, speed_kph=40.0, travel_time=d / 11.11)
    return G


# ── Synthetic test data ────────────────────────────────────────

def _make_demand_gdf(n: int = 5) -> gpd.GeoDataFrame:
    pts = [Point(117.205 + i * 0.005, 34.268 + i * 0.003) for i in range(n)]
    return gpd.GeoDataFrame({
        "demand_id": [f"D{i}" for i in range(n)],
        "demand_name": [f"Demand {i}" for i in range(n)],
        "geometry": pts,
        "lon": [p.x for p in pts],
        "lat": [p.y for p in pts],
        "people_count": [100 + i * 50 for i in range(n)],
        "priority": [1] * n,
        "population_type": ["居民"] * n,
    }, crs="EPSG:4326")


def _make_shelter_gdf(n: int = 3) -> gpd.GeoDataFrame:
    pts = [Point(117.180, 34.240 + i * 0.02) for i in range(n)]
    return gpd.GeoDataFrame({
        "shelter_id": [f"SH{i}" for i in range(n)],
        "shelter_name": [f"Shelter {i}" for i in range(n)],
        "geometry": pts,
        "capacity": [5000] * n,
    }, crs="EPSG:4326")


def _fake_paths(G, demand_gdf, shelters_gdf, *args, **kwargs):
    """Return fake path objects for each demand point."""
    paths = []
    for _, row in demand_gdf.iterrows():
        p = MagicMock()
        p.node_path = [0, 1]
        p.demand_id = row["demand_id"]
        p.candidates = []
        p.best_idx = 0
        paths.append(p)
    return paths


# ── Tests ───────────────────────────────────────────────────────

class TestPipelineIntegration:

    def _make_patches(self, tmpdir: str) -> list:
        """Return patches for data loading, paths, and file output."""
        G = _build_test_graph()
        demand = _make_demand_gdf()
        shelters = _make_shelter_gdf()
        nodes_gdf = gpd.GeoDataFrame({"x": [117.205], "y": [34.268]}, geometry=[Point(117.205, 34.268)], crs="EPSG:4326")
        edges_gdf = gpd.GeoDataFrame({"u": [0], "v": [1], "length": [100.0], "speed_kph": [40.0],
                                       "geometry": [Point(117.205, 34.268)]}, crs="EPSG:4326")

        # Mock event
        fake_event = MagicMock()
        fake_event.center = Point(117.205, 34.268)
        fake_event.radius_m = 1500

        return [
            patch("src.app.pipeline.load_road_network",
                  return_value=(G, nodes_gdf, edges_gdf)),
            patch("src.app.pipeline.load_demand", return_value=demand),
            patch("src.app.pipeline.load_shelter_data", return_value=shelters),
            patch("src.app.pipeline.load_bus_stops", return_value=None),
            patch("src.app.pipeline.load_stations", return_value=[]),
            patch("src.app.pipeline.compute_evacuation_paths",
                  side_effect=_fake_paths),
            patch("src.app.pipeline.create_event_from_yaml", return_value=fake_event),
            patch("src.app.pipeline.get_affected_roads", return_value=[]),
            patch("src.app.pipeline.summarize_shelters",
                  return_value={"total_points": len(shelters), "total_capacity": 15000}),
            patch("src.app.pipeline.summarize_demand",
                  return_value={"total_points": len(demand), "total_people": sum(demand["people_count"])}),
            patch("src.app.pipeline.RUNS_DIR", tmpdir),
        ]

    def _make_params(self, scenario_path: str) -> dict:
        return {
            "scenario_path": scenario_path,
            "event_location": "彭城广场",
            "radius_m": 1500,
            "actual_demand": 3000,
            "random_seed": 42,
            "enable_perturbation": False,
            "enable_bus": True,
            "bus_params": {"n_buses": 10, "bus_capacity": 50, "boarding_rate": 2.0, "time_limit": 5},
            "cost_matrix_mode": "euclidean_fast",
            "enable_sumo": False,
            "enable_crop": False, "enable_traci": False,
            "enable_rail": False,
            "walk_self_min": 20, "walk_rail_min": 10, "pressure_limit": 1.1,
            "walk_mode": "euclidean_fast", "cap_factor": 1.0,
            "enable_sensitivity": False,
            "enable_snap": False, "enable_water_filter": False,
        }

    def _run(self, tmpdir: str, params: dict) -> dict:
        scenario_path = os.path.join(tmpdir, "test_scenario.yaml")
        with open(scenario_path, "w") as f:
            yaml.dump({"scenario_id": "test", "scenario_name": "Test", "event": {"type": "crowd"}}, f)
        params["scenario_path"] = scenario_path

        patches = self._make_patches(tmpdir)
        for p in patches:
            p.start()
        try:
            return run_analysis(params)
        finally:
            for p in patches:
                p.stop()

    def test_pipeline_returns_expected_result_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            R = self._run(tmpdir, self._make_params(os.path.join(tmpdir, "s.yaml")))

            expected = [
                "log_lines", "run_id", "run_output_dir",
                "demand_gdf", "shelters_gdf", "shelters_all",
                "demand_summary", "shelter_summary",
                "event", "affected", "paths", "path_elapsed",
                "dispatch_result", "vehicles", "depots", "depot_locations",
                "bus_routes", "sumo_result", "sumo_bus_routes",
                "allocation_result", "rail_stations", "station_pressures",
                "evac_metrics", "comparison", "walking_paths",
                "bus_params", "actual_demand", "radius_m", "event_location",
            ]
            for key in expected:
                assert key in R, f"Missing result key: {key}"

    def test_pipeline_writes_all_7_output_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            R = self._run(tmpdir, self._make_params(os.path.join(tmpdir, "s.yaml")))
            run_dir = R["run_output_dir"]

            for fname in [
                "config.yaml", "scenario.yaml", "metrics.json",
                "run_meta.json", "report.txt",
                "station_pressure.csv", "dispatch_summary.csv",
            ]:
                assert os.path.exists(os.path.join(run_dir, fname)), f"Missing: {fname}"

    def test_pipeline_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            params = self._make_params(os.path.join(tmpdir, "s.yaml"))
            R1 = self._run(tmpdir, params)
            R2 = self._run(tmpdir, params)

            d1, d2 = R1["dispatch_result"], R2["dispatch_result"]
            assert d1 is not None and d2 is not None
            assert d1.total_cost == d2.total_cost
            assert d1.solver_status == d2.solver_status

    def test_config_yaml_is_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            R = self._run(tmpdir, self._make_params(os.path.join(tmpdir, "s.yaml")))
            config_path = os.path.join(R["run_output_dir"], "config.yaml")
            with open(config_path) as f:
                saved = yaml.safe_load(f)
            assert saved["run_id"] == R["run_id"]
            assert saved["random_seed"] == 42
            assert saved["enable_sumo"] is False

    def test_bus_dispatch_produces_routes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            R = self._run(tmpdir, self._make_params(os.path.join(tmpdir, "s.yaml")))
            dr = R["dispatch_result"]
            assert dr is not None
            assert dr.solver_status in ("optimal", "feasible")
            assert len(dr.vehicle_routes) > 0
