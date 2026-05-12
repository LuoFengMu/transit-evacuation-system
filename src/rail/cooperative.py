"""Two-stage cooperative allocation model for v0.4.0.

Stage 1: Destination selection for each demand point.
  Cost(i,j) = α×T_bus + β×T_walk + γ×P_j + δ×TransferPenalty
  Choose: shelter (bus direct) or rail station (bus + rail)

Stage 2: (delegated to existing OR-Tools CVRP solver)
  Bus routes optimized after destination assignment.
"""
from dataclasses import dataclass, field
import numpy as np
from src.rail.capacity import RailStation, StationPressure, compute_pressure


@dataclass
class AllocationResult:
    """Result of cooperative destination allocation."""
    assignments: dict = field(default_factory=dict)  # demand_id → destination
    destination_type: dict = field(default_factory=dict)  # demand_id → "shelter"|"rail"
    shelter_allocations: dict = field(default_factory=dict)  # shelter_id → total_people
    rail_allocations: dict = field(default_factory=dict)  # station_id → total_people
    walking_demands: dict = field(default_factory=dict)  # demand_id → walk_time_s
    unassigned: list = field(default_factory=list)
    station_pressures: list = field(default_factory=list)


def allocate_cooperative(
    demand_points: list[dict],       # [{demand_id, people, walk_to_shelter_s, walk_to_rail_s, nearest_shelter_id, nearest_rail_id, ...}]
    rail_stations: list[RailStation],
    shelters: list[dict],            # [{shelter_id, capacity, ...}]
    alpha: float = 1.0,              # bus time weight
    beta: float = 1.2,               # walk time weight
    gamma: float = 10.0,             # pressure penalty weight
    delta: float = 5.0,              # transfer penalty weight
    max_walk_s: float = 1800.0,      # max walking time (30 min) before requiring bus
    max_bus_s: float = 3600.0,       # max bus time before preferring rail
    rail_time_window_h: float = 1.0,
) -> AllocationResult:
    """Allocate demand points to shelters or rail stations.

    Strategy:
      - If walking to shelter < 15 min: walk (no bus needed)
      - If bus to shelter < 60 min AND shelter has capacity: bus to shelter
      - Otherwise: bus to nearest rail station (if station has capacity)
      - If all options exhausted: mark as unassigned
    """
    result = AllocationResult()
    station_caps = {s.station_id: s.dynamic_capacity_pax_h * rail_time_window_h
                    for s in rail_stations}
    shelter_used = {s["shelter_id"]: 0 for s in shelters}

    for dp in demand_points:
        did = dp["demand_id"]
        people = dp["people"]
        walk_shelter_s = dp.get("walk_to_shelter_s", 9999)
        walk_rail_s = dp.get("walk_to_rail_s", 9999)
        nearest_shelter = dp.get("nearest_shelter_id", "")
        nearest_rail = dp.get("nearest_rail_id", "")

        # Decision tree
        assigned = False

        # Option 1: walk to shelter if close enough
        if walk_shelter_s < max_walk_s * 0.5:  # < 15 min walk
            if nearest_shelter and shelter_used.get(nearest_shelter, 0) + people <= shelters[0].get("capacity", 99999):
                result.assignments[did] = nearest_shelter
                result.destination_type[did] = "shelter"
                shelter_used[nearest_shelter] = shelter_used.get(nearest_shelter, 0) + people
                result.walking_demands[did] = walk_shelter_s
                assigned = True

        # Option 2: bus to shelter (not too far, shelter has capacity)
        if not assigned:
            bus_to_shelter_s = dp.get("bus_to_shelter_s", walk_shelter_s / 5)  # rough estimate
            if bus_to_shelter_s < max_bus_s and nearest_shelter:
                shelter_cap = next((s["capacity"] for s in shelters if s["shelter_id"] == nearest_shelter), 0)
                if shelter_used.get(nearest_shelter, 0) + people <= shelter_cap:
                    result.assignments[did] = nearest_shelter
                    result.destination_type[did] = "shelter"
                    shelter_used[nearest_shelter] = shelter_used.get(nearest_shelter, 0) + people
                    assigned = True

        # Option 3: bus to rail station
        if not assigned and nearest_rail:
            cap = station_caps.get(nearest_rail, 0)
            current = result.rail_allocations.get(nearest_rail, 0)
            if current + people <= cap or cap == 0:  # cap=0 means unlimited
                result.assignments[did] = nearest_rail
                result.destination_type[did] = "rail"
                result.rail_allocations[nearest_rail] = current + people
                assigned = True

        # Option 4: force to shelter anyway (overflow)
        if not assigned:
            if nearest_shelter:
                result.assignments[did] = nearest_shelter
                result.destination_type[did] = "shelter"
                shelter_used[nearest_shelter] = shelter_used.get(nearest_shelter, 0) + people
            else:
                result.unassigned.append(did)

    # Aggregate shelter allocations
    result.shelter_allocations = shelter_used

    # Compute station pressures
    result.station_pressures = compute_pressure(
        rail_stations, result.rail_allocations, time_window_h=rail_time_window_h,
    )

    return result
