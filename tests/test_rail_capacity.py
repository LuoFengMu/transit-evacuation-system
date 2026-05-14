"""Test rail station capacity and pressure model.

Verifies Phase 2.4:
  - Pressure level classification (normal/saturated/overloaded/severe)
  - compute_pressure works correctly with allocations
"""
import pytest
from src.rail.capacity import (
    RailStation, StationPressure, compute_pressure,
)


def _make_station(sid: str, name: str, dynamic_cap: int) -> RailStation:
    return RailStation(
        station_id=sid, station_name=name, line_id="L1",
        lon=117.0, lat=34.0,
        static_capacity=5000, dynamic_capacity_pax_h=dynamic_cap,
        line_rate_pax_h=10000,
    )


class TestComputePressure:

    def test_normal_pressure(self):
        """Allocation well below capacity → normal."""
        s = _make_station("S1", "Station 1", 1000)
        results = compute_pressure([s], {"S1": 500}, time_window_h=1.0)
        assert results[0].pressure == pytest.approx(0.5)
        assert results[0].level == "normal"

    def test_saturated_pressure(self):
        """Allocation at 80-100% → saturated."""
        s = _make_station("S1", "Station 1", 1000)
        results = compute_pressure([s], {"S1": 850}, time_window_h=1.0)
        assert results[0].level == "saturated"

    def test_overloaded_pressure(self):
        """Allocation at 100-120% → overloaded."""
        s = _make_station("S1", "Station 1", 1000)
        results = compute_pressure([s], {"S1": 1100}, time_window_h=1.0)
        assert results[0].level == "overloaded"

    def test_severe_pressure(self):
        """Allocation >120% → severe."""
        s = _make_station("S1", "Station 1", 1000)
        results = compute_pressure([s], {"S1": 1300}, time_window_h=1.0)
        assert results[0].level == "severe"

    def test_time_window_scales_capacity(self):
        """Longer time window increases effective capacity, reducing pressure."""
        s = _make_station("S1", "Station 1", 1000)
        # 1000 pax allocated over 2h → 1000 / (1000 * 2) = 0.5
        results = compute_pressure([s], {"S1": 1000}, time_window_h=2.0)
        assert results[0].pressure == pytest.approx(0.5)
        assert results[0].level == "normal"
        assert results[0].capacity_used == 2000

    def test_zero_allocation_still_returns_result(self):
        s = _make_station("S1", "Station 1", 1000)
        results = compute_pressure([s], {}, time_window_h=1.0)
        assert results[0].arrivals == 0
        assert results[0].pressure == 0.0

    def test_zero_capacity_is_infinite_pressure(self):
        s = _make_station("S1", "Station 1", 0)
        results = compute_pressure([s], {"S1": 100}, time_window_h=1.0)
        assert results[0].pressure == float("inf")
        assert results[0].level == "severe"

    def test_multiple_stations(self):
        stations = [
            _make_station("S1", "Station 1", 1000),
            _make_station("S2", "Station 2", 2000),
            _make_station("S3", "Station 3", 500),
        ]
        results = compute_pressure(
            stations,
            {"S1": 200, "S2": 2300, "S3": 450},
            time_window_h=1.0,
        )
        pressures = {r.station_id: r for r in results}
        assert pressures["S1"].level == "normal"   # 200/1000=0.2
        assert pressures["S2"].level == "overloaded"  # 2300/2000=1.15
        assert pressures["S3"].level == "saturated"   # 450/500=0.9

    def test_background_flow_adds_to_arrivals(self):
        s = _make_station("S1", "Station 1", 1000)
        # 600 allocated + 300 background = 900 → 0.9 pressure
        results = compute_pressure(
            [s], {"S1": 600}, time_window_h=1.0,
            background_flow={"S1": 300},
        )
        assert results[0].arrivals == 900
        assert results[0].pressure == pytest.approx(0.9)
