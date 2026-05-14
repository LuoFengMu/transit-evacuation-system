"""Evacuation evaluation metrics."""
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class EvacuationMetrics:
    total_demand: int = 0
    total_evacuated: int = 0
    completion_rate: float = 0.0
    unserved: int = 0
    avg_walk_distance_m: float = 0.0
    avg_walk_time_s: float = 0.0
    avg_bus_time_s: float = 0.0
    avg_rail_time_s: float = 0.0
    bus_utilization: float = 0.0
    rail_share: float = 0.0       # fraction using rail
    bus_direct_share: float = 0.0  # fraction bus-to-shelter
    walk_direct_share: float = 0.0 # fraction walking only
    max_station_pressure: float = 0.0
    overloaded_stations: int = 0
    extra: dict = field(default_factory=dict)


def compute_evacuation_metrics(
    demand_gdf,
    allocation_result,       # AllocationResult
    dispatch_result=None,     # DispatchResult (optional)
    station_pressures=None,   # list[StationPressure]
    walking_access=None,      # access matrix
    bus_time_s: float = 0,
    rail_time_s: float = 0,
) -> EvacuationMetrics:
    m = EvacuationMetrics()
    m.total_demand = int(demand_gdf["people_count"].sum())

    # ── Mode shares by people count ──────────────────────────
    mode_people = getattr(allocation_result, "mode_people", {})
    walk_rail = mode_people.get("walk_rail", 0)
    bus_rail = mode_people.get("bus_rail", 0)
    walk_self = mode_people.get("walk_self", 0)
    bus_periphery = mode_people.get("bus_periphery", 0)

    m.rail_share = (walk_rail + bus_rail) / m.total_demand if m.total_demand > 0 else 0
    m.bus_direct_share = (bus_rail + bus_periphery) / m.total_demand if m.total_demand > 0 else 0
    m.walk_direct_share = (walk_self + walk_rail) / m.total_demand if m.total_demand > 0 else 0

    m.total_evacuated = walk_rail + bus_rail + walk_self + bus_periphery
    # If dispatch_result is richer, use it (only for bus-dispatched people)
    if dispatch_result and dispatch_result.vehicle_routes:
        sub_qty = dispatch_result.sub_demand_quantities
        dispatched = sum(sub_qty[i] for i in range(len(sub_qty)) if i not in dispatch_result.unserved_demand)
        # Use the larger of the two: allocation covers all modes, dispatch covers bus
        if dispatched > (walk_rail + bus_rail + walk_self + bus_periphery):
            m.total_evacuated = dispatched

    m.unserved = max(
        m.total_demand - m.total_evacuated,
        getattr(allocation_result, "unassigned_people", 0),
    )
    m.completion_rate = m.total_evacuated / m.total_demand if m.total_demand > 0 else 0

    # Walk distances
    if walking_access:
        rail_dist = [a["distance_m"] for a in walking_access.get("to_rail", [])]
        shelter_dist = [a["distance_m"] for a in walking_access.get("to_shelter", [])]
        all_dist = rail_dist + shelter_dist
        m.avg_walk_distance_m = sum(all_dist) / len(all_dist) if all_dist else 0
        walk_times = [a["walk_time_s"] for a in walking_access.get("to_rail", [])]
        walk_times += [a["walk_time_s"] for a in walking_access.get("to_shelter", [])]
        m.avg_walk_time_s = sum(walk_times) / len(walk_times) if walk_times else 0

    # Station pressure
    if station_pressures:
        pressures = [p.pressure for p in station_pressures if p.arrivals > 0]
        m.max_station_pressure = max(pressures) if pressures else 0
        m.overloaded_stations = sum(1 for p in station_pressures if p.level in ("overloaded", "severe"))

    m.avg_bus_time_s = bus_time_s
    m.avg_rail_time_s = rail_time_s

    return m


def metrics_to_dict(m: EvacuationMetrics) -> dict:
    return {
        "total_demand": m.total_demand,
        "total_evacuated": m.total_evacuated,
        "completion_rate": round(m.completion_rate, 3),
        "unserved": m.unserved,
        "avg_walk_distance_m": round(m.avg_walk_distance_m, 1),
        "avg_walk_time_s": round(m.avg_walk_time_s, 1),
        "bus_utilization": round(m.bus_utilization, 3),
        "rail_share": round(m.rail_share, 3),
        "bus_direct_share": round(m.bus_direct_share, 3),
        "walk_direct_share": round(m.walk_direct_share, 3),
        "max_station_pressure": round(m.max_station_pressure, 3),
        "overloaded_stations": m.overloaded_stations,
    }
