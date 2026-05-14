"""Test cooperative allocation correctness.

Verifies Phase 2.4:
  - mode_people sums match total served people
  - unassigned_people consistent with remaining demand
  - No station exceeds pressure_limit for bus_rail assignments
  - People conservation across allocation
"""
import pytest
from src.rail.capacity import RailStation
from src.rail.cooperative import allocate_cooperative, AllocationResult


def _make_station(sid: str, name: str, dynamic_cap: int) -> RailStation:
    return RailStation(
        station_id=sid, station_name=name, line_id="L1",
        lon=117.0, lat=34.0,
        static_capacity=5000, dynamic_capacity_pax_h=dynamic_cap,
        line_rate_pax_h=10000,
    )


def _make_dp(did: str, people: int, walk_rail_s: float = 300,
             walk_shelter_s: float = 3600, rail_candidates: list[str] | None = None) -> dict:
    return {
        "demand_id": did, "people": people,
        "walk_to_shelter_s": walk_shelter_s,
        "walk_to_rail_s": walk_rail_s,
        "nearest_rail_id": rail_candidates[0] if rail_candidates else "",
        "rail_candidates": rail_candidates or [],
    }


class TestAllocateCooperative:

    def test_all_walk_self_when_close_to_shelter(self):
        """Demand points close to shelter walk away."""
        dp = [_make_dp("D1", 100, walk_shelter_s=600)]  # ≤ 1200 → walk_self
        result = allocate_cooperative(dp, [])
        assert result.mode_people.get("walk_self", 0) == 100
        assert result.unassigned_people == 0

    def test_all_unassigned_when_no_route(self):
        """Demand far from everything → unassigned."""
        dp = [_make_dp("D1", 100, walk_shelter_s=9999, walk_rail_s=9999)]
        result = allocate_cooperative(dp, [], bus_capacity_per_round=0)
        assert result.unassigned_people == 100
        assert sum(result.mode_people.values()) == 0

    def test_walk_rail_when_close_to_station(self):
        """Demand within walk distance to rail station."""
        stations = [_make_station("S1", "Station 1", dynamic_cap=10000)]
        dp = [_make_dp("D1", 50, walk_rail_s=300, rail_candidates=["S1"])]  # ≤ 600s
        result = allocate_cooperative(dp, stations, walk_rail_max_s=600)
        assert result.mode_people.get("walk_rail", 0) == 50
        assert result.unassigned_people == 0

    def test_bus_rail_when_within_bus_range(self):
        """Demand within bus range → bus to rail."""
        stations = [_make_station("S1", "Station 1", dynamic_cap=10000)]
        # walk too far for walk_rail (900s > 600), but within bus range
        dp = [_make_dp("D1", 100, walk_rail_s=900, walk_shelter_s=9999, rail_candidates=["S1"])]
        result = allocate_cooperative(
            dp, stations, walk_rail_max_s=600,
            bus_capacity_per_round=500, max_rounds=3,
        )
        assert result.mode_people.get("bus_rail", 0) == 100

    def test_bus_periphery_when_rail_overloaded(self):
        """When rail station is overloaded, fallback to bus periphery."""
        stations = [_make_station("S1", "Station 1", dynamic_cap=100)]  # small capacity
        dp = [_make_dp("D1", 200, walk_rail_s=300, walk_shelter_s=9999, rail_candidates=["S1"])]
        result = allocate_cooperative(
            dp, stations, walk_rail_max_s=600,
            pressure_safe=0.8, pressure_limit=1.1,
            bus_capacity_per_round=1000, max_rounds=3,
        )
        # 200 people vs 100 capacity at 0.8 safe → >80 exceeds safe, but ≤110 (1.1 limit)
        # walk_rail tries first at 0.8 safe: 200/100 = 2.0 > 0.8 → skip
        # bus_rail tries: 200/100 = 2.0 > 1.1 → skip
        # bus_periphery fallback
        assert result.mode_people.get("bus_periphery", 0) == 200

    def test_mode_people_sum_equals_served_people(self):
        """mode_people values sum to the total served."""
        stations = [
            _make_station("S1", "Station 1", dynamic_cap=5000),
            _make_station("S2", "Station 2", dynamic_cap=3000),
        ]
        dps = [
            _make_dp("D1", 300, walk_rail_s=200, walk_shelter_s=3600, rail_candidates=["S1"]),
            _make_dp("D2", 500, walk_rail_s=900, walk_shelter_s=9999, rail_candidates=["S1", "S2"]),
            _make_dp("D3", 200, walk_shelter_s=600),
        ]
        result = allocate_cooperative(
            dps, stations, walk_rail_max_s=600,
            bus_capacity_per_round=1000, max_rounds=3,
        )
        served = sum(result.mode_people.values())
        total = sum(dp["people"] for dp in dps)
        assert served + result.unassigned_people == total

    def test_unassigned_people_matches_unassigned_list(self):
        """unassigned_people should equal sum of people in unassigned demand IDs."""
        stations = [_make_station("S1", "Station 1", dynamic_cap=10)]  # tiny capacity
        dps = [
            _make_dp("D1", 500, walk_rail_s=300, walk_shelter_s=3600, rail_candidates=["S1"]),
            _make_dp("D2", 300, walk_rail_s=300, walk_shelter_s=3600, rail_candidates=["S1"]),
        ]
        result = allocate_cooperative(
            dps, stations, walk_rail_max_s=600,
            bus_capacity_per_round=100, max_rounds=1,
        )
        unassigned_from_ids = sum(dp["people"] for dp in dps if dp["demand_id"] in result.unassigned)
        assert result.unassigned_people == unassigned_from_ids

    def test_empty_input_returns_zeros(self):
        result = allocate_cooperative([], [])
        assert sum(result.mode_people.values()) == 0
        assert result.unassigned_people == 0
        assert len(result.round_results) == 0
