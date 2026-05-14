"""Test CLI argument parsing and pipeline invocation.

Verifies Phase 4.2:
  - Argument parser produces correct defaults
  - Scenario path resolution
  - main() builds correct params dict
"""
import os
import tempfile
from unittest.mock import patch

import pytest
import yaml

from src.app.cli import build_parser, resolve_scenario_path, main


class TestParser:

    def test_required_scenario(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["--scenario", "test.yaml"])
        assert args.seed is None
        assert args.demand is None
        assert args.cost_mode is None
        assert args.no_bus is False
        assert args.no_rail is False
        assert args.no_sumo is False

    def test_full_overrides(self):
        parser = build_parser()
        args = parser.parse_args([
            "--scenario", "C_hybrid_cooperative.yaml",
            "--seed", "123",
            "--demand", "50000",
            "--cost-mode", "cached_network_time",
            "--no-bus", "--no-rail", "--no-sumo",
            "--capacity-factor", "0.7",
        ])
        assert args.seed == 123
        assert args.demand == 50000
        assert args.cost_mode == "cached_network_time"
        assert args.no_bus is True
        assert args.no_rail is True
        assert args.no_sumo is True
        assert args.capacity_factor == 0.7


class TestResolveScenario:

    def test_full_path(self):
        path = resolve_scenario_path(
            os.path.abspath("configs/scenarios/C_hybrid_cooperative.yaml"))
        assert path.endswith("C_hybrid_cooperative.yaml")

    def test_short_name(self):
        path = resolve_scenario_path("C_hybrid_cooperative.yaml")
        assert path.endswith("C_hybrid_cooperative.yaml")

    def test_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            resolve_scenario_path("nonexistent_file.yaml")


class TestCLIMain:

    def test_main_calls_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write minimal scenario YAML
            scenario_path = os.path.join(tmpdir, "test.yaml")
            with open(scenario_path, "w") as f:
                yaml.dump({
                    "scenario_id": "test",
                    "scenario_name": "Test",
                    "event": {"type": "crowd", "location": "彭城广场", "radius_m": 1000},
                }, f)

            with patch("src.app.cli.run_analysis") as mock_run:
                mock_run.return_value = {
                    "run_id": "test_run_001", "run_output_dir": tmpdir,
                    "log_lines": ["test log"], "dispatch_result": None,
                    "evac_metrics": None,
                }
                result = main(["--scenario", scenario_path, "--seed", "99"])

            mock_run.assert_called_once()
            params = mock_run.call_args[0][0]
            assert params["random_seed"] == 99
            assert params["scenario_path"] == scenario_path
            assert params["enable_bus"] is True
            assert params["enable_rail"] is True
            assert params["enable_sumo"] is True
            assert result["run_id"] == "test_run_001"

    def test_main_with_disabled_modes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scenario_path = os.path.join(tmpdir, "test.yaml")
            with open(scenario_path, "w") as f:
                yaml.dump({
                    "scenario_id": "test",
                    "scenario_name": "Test",
                    "event": {"type": "crowd"},
                }, f)

            with patch("src.app.cli.run_analysis") as mock_run:
                mock_run.return_value = {
                    "run_id": "r2", "run_output_dir": tmpdir,
                    "log_lines": [], "dispatch_result": None, "evac_metrics": None,
                }
                main(["--scenario", scenario_path, "--no-bus", "--no-rail", "--no-sumo"])

            params = mock_run.call_args[0][0]
            assert params["enable_bus"] is False
            assert params["enable_rail"] is False
            assert params["enable_sumo"] is False

    def test_main_passes_custom_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scenario_path = os.path.join(tmpdir, "test.yaml")
            output_dir = os.path.join(tmpdir, "custom_run")
            with open(scenario_path, "w") as f:
                yaml.dump({
                    "scenario_id": "test",
                    "scenario_name": "Test",
                    "event": {"type": "crowd"},
                }, f)

            with patch("src.app.cli.run_analysis") as mock_run:
                mock_run.return_value = {
                    "run_id": "r3", "run_output_dir": output_dir,
                    "log_lines": [], "dispatch_result": None, "evac_metrics": None,
                }
                main(["--scenario", scenario_path, "--output", output_dir])

            params = mock_run.call_args[0][0]
            assert params["output_dir"] == output_dir
