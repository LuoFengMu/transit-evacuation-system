"""Test rail capacity sensitivity analysis.

Verifies Phase 3.4:
  - capacity_factor applies to dynamic_capacity_pax_h
  - conservative (0.7) allocates fewer people to rail than optimistic (1.2)
  - baseline (1.0) produces intermediate results
"""
import pytest
from src.rail.capacity import RailStation, compute_pressure
from src.rail.cooperative import allocate_cooperative


def _make_station(sid: str, name: str, dynamic_cap: int) -> RailStation:
    return RailStation(
        station_id=sid, station_name=name, line_id="L1",
        lon=117.0, lat=34.0,
        static_capacity=5000, dynamic_capacity_pax_h=dynamic_cap,
        line_rate_pax_h=10000,
    )


def _make_dp(did: str, people: int, walk_rail_s: float = 900,
             walk_shelter_s: float = 9999) -> dict:
    return {
        "demand_id": did, "people": people,
        "walk_to_shelter_s": walk_shelter_s,
        "walk_to_rail_s": walk_rail_s,
        "nearest_rail_id": "S1",
        "rail_candidates": ["S1"],
    }


class TestCapacitySensitivity:

    def test_conservative_allocates_less_rail_than_baseline(self):
        """Lower capacity factor → station can handle fewer people → less rail allocation."""
        station = _make_station("S1", "Station 1", dynamic_cap=300)
        dp = _make_dp("D1", 200, walk_rail_s=300)  # walk_rail mode

        cons = allocate_cooperative(
            [dp], [station], walk_rail_max_s=600,
            capacity_factor=0.5,
        )
        base = allocate_cooperative(
            [dp], [station], walk_rail_max_s=600,
            capacity_factor=1.0,
        )

        rail_cons = cons.mode_people.get("walk_rail", 0)
        rail_base = base.mode_people.get("walk_rail", 0)
        # Conservative may block walk_rail; baseline should allow it
        assert rail_cons <= rail_base
        assert rail_base == 200  # baseline: 200/(300*1.0)=0.67 ≤ 0.8 safe

    def test_optimistic_allocates_more_rail_than_baseline(self):
        """Higher capacity factor → station can handle more people → more bus-to-rail allocation."""
        station = _make_station("S1", "Station 1", dynamic_cap=150)
        dp = _make_dp("D1", 200, walk_rail_s=900)

        opti = allocate_cooperative(
            [dp], [station], walk_rail_max_s=600,
            bus_capacity_per_round=500, max_rounds=3,
            capacity_factor=1.2,
        )
        base = allocate_cooperative(
            [dp], [station], walk_rail_max_s=600,
            bus_capacity_per_round=500, max_rounds=3,
            capacity_factor=1.0,
        )

        rail_opti = opti.mode_people.get("bus_rail", 0)
        rail_base = base.mode_people.get("bus_rail", 0)
        assert rail_opti >= rail_base

    def test_capacity_factor_scales_effective_capacity(self):
        """With factor=0.5, effective capacity is halved.

        Station has dynamic_cap=150. At factor=1.0: cap=150, 90/150=0.6 ≤ 0.8 safe → walk_rail=90.
        At factor=0.5: cap=75, 90/75=1.2 > 1.1 limit → walk_rail blocked → 0.
        """
        station = _make_station("S1", "Station 1", dynamic_cap=150)
        dp = _make_dp("D1", 90, walk_rail_s=300)

        base = allocate_cooperative(
            [dp], [station], walk_rail_max_s=600,
            capacity_factor=1.0,
        )
        half = allocate_cooperative(
            [dp], [station], walk_rail_max_s=600,
            bus_capacity_per_round=500, max_rounds=3,
            capacity_factor=0.5,
        )

        walk_rail_base = base.mode_people.get("walk_rail", 0)
        walk_rail_half = half.mode_people.get("walk_rail", 0)
        assert walk_rail_base == 90
        assert walk_rail_half == 0

    def test_default_factor_is_one(self):
        station = _make_station("S1", "Station 1", dynamic_cap=1000)
        dp = _make_dp("D1", 100, walk_rail_s=300)
        result = allocate_cooperative([dp], [station], walk_rail_max_s=600)
        assert result.mode_people.get("walk_rail", 0) == 100

    def test_pressure_higher_with_lower_capacity_factor(self):
        """compute_pressure: same arrivals → higher pressure with lower factor.

        P_s = arrivals / (Q_s × capacity_factor × Δt)
        Lower factor → smaller denominator → higher pressure.
        Test compute_pressure directly with identical arrivals.
        """
        station = _make_station("S1", "Station 1", dynamic_cap=100)
        arrivals = {"S1": 80}  # 80 people allocated to station

        p_cons = compute_pressure(
            [station], arrivals, time_window_h=1.0, capacity_factor=0.7,
        )[0]
        p_base = compute_pressure(
            [station], arrivals, time_window_h=1.0, capacity_factor=1.0,
        )[0]
        p_opti = compute_pressure(
            [station], arrivals, time_window_h=1.0, capacity_factor=1.2,
        )[0]

        # Conservative: 80/(100*0.7) = 80/70 = 1.143 → overloaded
        # Baseline:    80/(100*1.0) = 80/100 = 0.8 → normal/saturated boundary
        # Optimistic:  80/(100*1.2) = 80/120 = 0.667 → normal
        assert p_cons.pressure > p_base.pressure, \
            f"cons {p_cons.pressure} > base {p_base.pressure}"
        assert p_base.pressure > p_opti.pressure, \
            f"base {p_base.pressure} > opti {p_opti.pressure}"
        assert p_cons.level == "overloaded"
        assert p_opti.level == "normal"

    def test_mode_people_still_sums_correctly_with_factor(self):
        """Conservation holds regardless of capacity factor."""
        station = _make_station("S1", "Station 1", dynamic_cap=200)
        dps = [
            _make_dp("D1", 100, walk_rail_s=300),
            _make_dp("D2", 200, walk_rail_s=900),
            _make_dp("D3", 150, walk_shelter_s=600),
        ]
        total = sum(dp["people"] for dp in dps)

        for cf in [0.7, 1.0, 1.2]:
            result = allocate_cooperative(
                dps, [station], walk_rail_max_s=600,
                bus_capacity_per_round=500, max_rounds=3,
                capacity_factor=cf,
            )
            served = sum(result.mode_people.values())
            assert served + result.unassigned_people == total, \
                f"Factor {cf}: served={served} + unassigned={result.unassigned_people} != total={total}"
