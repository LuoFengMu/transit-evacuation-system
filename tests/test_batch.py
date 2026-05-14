"""Test batch experiment runner.

Verifies Phase 4.4-4.5:
  - Scenario path resolution
  - Metrics extraction from run result
  - summary.csv and manifest.json generation
"""
import os
import json
import tempfile
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.app.batch import (
    build_parser, _scenario_path, _extract_metrics, main, EXPERIMENTS_DIR,
)


class TestScenarioPath:

    def test_A_maps_to_A_bus_direct(self):
        path = _scenario_path("A")
        assert "A_bus_direct.yaml" in path
        assert os.path.isfile(path)

    def test_B_maps_to_B_rail_priority(self):
        path = _scenario_path("B")
        assert "B_rail_priority.yaml" in path

    def test_C_maps_to_C_hybrid_cooperative(self):
        path = _scenario_path("C")
        assert "C_hybrid_cooperative.yaml" in path

    def test_unknown_raises(self):
        with pytest.raises(FileNotFoundError):
            _scenario_path("nonexistent")


class TestExtractMetrics:

    def test_extracts_dispatch_status(self):
        result = {
            "dispatch_result": MagicMock(solver_status="optimal",
                                         vehicle_routes={"v1": [("depot", "d0", 0), ("pickup", 0, 300)]}),
            "comparison": None,
            "evac_metrics": None,
        }
        row = _extract_metrics(result, 12.3)
        assert row["dispatch_status"] == "optimal"
        assert row["dispatch_vehicles_used"] == 1
        assert row["runtime_s"] == 12.3

    def test_extracts_evac_metrics(self):
        em = MagicMock()
        em.completion_rate = 0.95
        em.rail_share = 0.72
        em.unserved = 150
        em.overloaded_stations = 2
        result = {
            "dispatch_result": None,
            "comparison": None,
            "evac_metrics": em,
        }
        row = _extract_metrics(result, 5.0)
        assert row["completion_rate"] == 0.95
        assert row["rail_share"] == 0.72
        assert row["unserved"] == 150
        assert row["overloaded_stations"] == 2

    def test_disabled_dispatch(self):
        result = {"dispatch_result": None, "comparison": None, "evac_metrics": None}
        row = _extract_metrics(result, 1.0)
        assert row["dispatch_status"] == "disabled"
        assert row["dispatch_vehicles_used"] == 0


class TestBatchMain:

    def test_writes_summary_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.app.batch.EXPERIMENTS_DIR", tmpdir), \
                 patch("src.app.batch.run_analysis") as mock_run:

                # Mock one run
                fake_em = MagicMock()
                fake_em.completion_rate = 0.88
                fake_em.rail_share = 0.65
                fake_em.unserved = 200
                fake_em.overloaded_stations = 1
                mock_run.return_value = {
                    "run_id": "test_run_001", "run_output_dir": tmpdir,
                    "log_lines": [], "dispatch_result": None,
                    "comparison": None, "evac_metrics": fake_em,
                }

                main(["--scenarios", "A", "--seeds", "42", "--capacity-factors", "1.0",
                      "--no-sumo", "--no-rail", "--experiment-id", "test_batch"])

            exp_dir = os.path.join(tmpdir, "test_batch")
            assert os.path.isdir(exp_dir)

            # Check summary.csv
            summary_path = os.path.join(exp_dir, "summary.csv")
            assert os.path.exists(summary_path)
            df = pd.read_csv(summary_path)
            assert len(df) == 1
            assert df.iloc[0]["scenario"] == "A"
            assert df.iloc[0]["seed"] == 42
            assert df.iloc[0]["capacity_factor"] == 1.0

            # Check manifest.json
            manifest_path = os.path.join(exp_dir, "manifest.json")
            assert os.path.exists(manifest_path)
            with open(manifest_path) as f:
                manifest = json.load(f)
            assert manifest["experiment_id"] == "test_batch"
            assert len(manifest["runs"]) == 1
            assert manifest["runs"][0]["run_id"] == "test_run_001"

    def test_multiple_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.app.batch.EXPERIMENTS_DIR", tmpdir), \
                 patch("src.app.batch.run_analysis") as mock_run:

                fake_em = MagicMock()
                fake_em.completion_rate = 0.9
                fake_em.rail_share = 0.5
                fake_em.unserved = 0
                fake_em.overloaded_stations = 0
                mock_run.return_value = {
                    "run_id": "run_X", "run_output_dir": tmpdir,
                    "log_lines": [], "dispatch_result": None,
                    "comparison": None, "evac_metrics": fake_em,
                }

                main(["--scenarios", "A", "B", "--seeds", "42", "99",
                      "--capacity-factors", "1.0", "--no-sumo", "--no-rail",
                      "--experiment-id", "multi"])

            df = pd.read_csv(os.path.join(tmpdir, "multi", "summary.csv"))
            # 2 scenarios × 2 seeds × 1 factor = 4 runs
            assert len(df) == 4
            assert set(df["scenario"]) == {"A", "B"}
            assert set(df["seed"]) == {42, 99}
