"""Test evaluation metrics compute correctly by people count.

Verifies Phase 1.3 P0 fixes:
  - rail_share = (walk_rail + bus_rail) / total_demand
  - total_demand = total_evacuated + unserved
  - Mode shares use people counts, not demand point counts
"""
import pytest
from dataclasses import dataclass, field
import pandas as pd

from src.evaluation.metrics import compute_evacuation_metrics, EvacuationMetrics


@dataclass
class MockAllocationResult:
    mode_people: dict = field(default_factory=dict)
    unassigned_people: int = 0
    unassigned: list = field(default_factory=list)
    destination_type: dict = field(default_factory=dict)
    station_pressures: list = field(default_factory=list)


def _make_demand_gdf(people_counts: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"people_count": people_counts})


class TestComputeEvacuationMetrics:
    """Verify P0 metric fixes."""

    def test_rail_share_is_walk_rail_plus_bus_rail_over_total(self):
        """rail_share should count people in walk_rail + bus_rail, not demand points."""
        demand = _make_demand_gdf([100, 200, 300, 50])  # total = 650
        alloc = MockAllocationResult(mode_people={
            "walk_self": 50,
            "walk_rail": 100,
            "bus_rail": 200,
            "bus_periphery": 250,
        }, unassigned_people=50)

        m = compute_evacuation_metrics(demand, alloc)

        expected_rail = (100 + 200) / 650  # walk_rail + bus_rail
        assert m.rail_share == pytest.approx(expected_rail)

    def test_total_demand_equals_evacuated_plus_unserved(self):
        """Conservation: total_demand == total_evacuated + unserved."""
        demand = _make_demand_gdf([1000, 500, 300])  # total = 1800
        alloc = MockAllocationResult(mode_people={
            "walk_self": 400,
            "walk_rail": 300,
            "bus_rail": 200,
            "bus_periphery": 600,
        }, unassigned_people=300)  # 1800 - 1500

        m = compute_evacuation_metrics(demand, alloc)

        assert m.total_demand == 1800
        assert m.total_evacuated == 1500
        assert m.unserved == 300
        assert m.total_demand == m.total_evacuated + m.unserved

    def test_mode_shares_are_by_people_not_demand_points(self):
        """A single large demand point should dominate, not be equal-weight."""
        # 5 demand points: one big (5000), four small (10 each) = 5040 total
        demand = _make_demand_gdf([5000, 10, 10, 10, 10])
        alloc = MockAllocationResult(mode_people={
            "walk_self": 0,
            "walk_rail": 5000,  # the big one goes to rail
            "bus_rail": 10,      # one small one
            "bus_periphery": 20,  # two small ones
        }, unassigned_people=10)  # one small unserved

        m = compute_evacuation_metrics(demand, alloc)

        # rail_share should be ~5010/5040 ≈ 0.994, not 2/4 = 0.5
        assert m.rail_share == pytest.approx(5010 / 5040)
        assert m.rail_share > 0.9

    def test_walk_direct_share_is_walk_self_plus_walk_rail(self):
        demand = _make_demand_gdf([100, 200])
        alloc = MockAllocationResult(mode_people={
            "walk_self": 50,
            "walk_rail": 100,
            "bus_rail": 50,
            "bus_periphery": 50,
        }, unassigned_people=50)

        m = compute_evacuation_metrics(demand, alloc)

        expected_walk = (50 + 100) / 300
        assert m.walk_direct_share == pytest.approx(expected_walk)

    def test_bus_direct_share_is_bus_rail_plus_bus_periphery(self):
        demand = _make_demand_gdf([100, 200])
        alloc = MockAllocationResult(mode_people={
            "walk_self": 0,
            "walk_rail": 0,
            "bus_rail": 100,
            "bus_periphery": 150,
        }, unassigned_people=50)

        m = compute_evacuation_metrics(demand, alloc)

        expected_bus = (100 + 150) / 300
        assert m.bus_direct_share == pytest.approx(expected_bus)

    def test_empty_allocation_all_unserved(self):
        """When no allocation done, all demand is unserved."""
        demand = _make_demand_gdf([100])
        alloc = MockAllocationResult()

        m = compute_evacuation_metrics(demand, alloc)

        assert m.rail_share == 0.0
        assert m.total_evacuated == 0
        assert m.unserved == 100
        assert m.total_demand == m.total_evacuated + m.unserved

    def test_all_served_no_unassigned(self):
        demand = _make_demand_gdf([500, 500])
        alloc = MockAllocationResult(mode_people={
            "walk_self": 200,
            "walk_rail": 300,
            "bus_rail": 200,
            "bus_periphery": 300,
        }, unassigned_people=0)

        m = compute_evacuation_metrics(demand, alloc)

        assert m.total_demand == 1000
        assert m.total_evacuated == 1000
        assert m.unserved == 0
        assert m.completion_rate == 1.0
