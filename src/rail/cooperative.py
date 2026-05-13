"""Cooperative evacuation allocation — v0.4.0 revised.

Five evacuation modes with rule priority + capacity constraints + cost comparison:

  Mode 1: Walk to shelter     (T_walk ≤ 5min, shelter capacity OK)
  Mode 2: Walk to rail        (T_walk ≤ 15min pref, ≤ 30max, P_s < 0.8)
  Mode 3: Bus to rail         (P_s < 1.0, cost < bus-to-shelter + tolerance)
  Mode 4: Bus to shelter      (rail overloaded / unavailable, or cost lower)
  Mode 5: Unserved / wait     (all capacities exhausted)

All bus paths: demand → walk → bus stop → bus → destination
"""
from dataclasses import dataclass, field
from src.rail.capacity import RailStation, compute_pressure


@dataclass
class RoundResult:
    round_id: int
    served_people: int
    remaining_people: int
    bus_utilization: float
    rail_assigned: int
    shelter_assigned: int
    walk_assigned: int
    unserved: int


@dataclass
class AllocationResult:
    assignments: dict = field(default_factory=dict)
    destination_type: dict = field(default_factory=dict)
    shelter_allocations: dict = field(default_factory=dict)
    rail_allocations: dict = field(default_factory=dict)
    walking_demands: dict = field(default_factory=dict)
    unassigned: list = field(default_factory=list)
    station_pressures: list = field(default_factory=list)
    round_results: list = field(default_factory=list)


def allocate_cooperative(
    demand_points: list[dict],
    rail_stations: list[RailStation],
    shelters: list[dict],
    # ── Thresholds (all parameterized) ──────────────────────
    walk_shelter_max_s: float = 300.0,       # 5 min
    walk_rail_preferred_s: float = 900.0,    # 15 min
    walk_rail_max_s: float = 1800.0,         # 30 min
    pressure_safe: float = 0.8,              # below = safe for walk-in
    pressure_limit: float = 1.0,             # below = ok for bus-to-rail
    bus_rail_cost_tolerance: float = 1.2,    # bus-to-rail allowed to be X times bus-to-shelter
    rail_time_window_h: float = 1.0,
    bus_capacity_per_round: int = 1500,      # total bus capacity per round
    max_rounds: int = 3,
    reachable_rail: set = None,              # station IDs reachable after road closure
    reachable_shelters: set = None,           # shelter IDs reachable
) -> AllocationResult:
    """Multi-round cooperative evacuation allocation."""
    result = AllocationResult()

    station_caps = {s.station_id: max(1, int(s.dynamic_capacity_pax_h * rail_time_window_h))
                    for s in rail_stations}
    shelter_caps = {s["shelter_id"]: s.get("capacity", 99999) for s in shelters}
    shelter_used = {s["shelter_id"]: 0 for s in shelters}
    rail_used = {s.station_id: 0 for s in rail_stations}

    pending = list(demand_points)  # demand points not yet assigned
    total_people = sum(dp["people"] for dp in demand_points)
    cumulative_served = 0

    for round_id in range(1, max_rounds + 1):
        if not pending:
            break

        round_bus_used = 0
        round_rail = 0
        round_shelter = 0
        round_walk = 0

        # Sort pending by urgency (far from any safe destination first)
        pending.sort(key=lambda dp: min(dp.get("walk_to_shelter_s", 9999),
                                         dp.get("walk_to_rail_s", 9999)), reverse=True)

        next_pending = []
        for dp in pending:
            did = dp["demand_id"]
            people = dp["people"]
            w_shelter = dp.get("walk_to_shelter_s", 9999)
            w_rail = dp.get("walk_to_rail_s", 9999)
            near_shelter = dp.get("nearest_shelter_id", "")
            near_rail = dp.get("nearest_rail_id", "")
            assigned = False

            # Check reachability
            shelter_ok = (reachable_shelters is None or near_shelter in reachable_shelters)
            rail_ok = (reachable_rail is None or near_rail in reachable_rail)

            # ── Mode 1: Walk to shelter ─────────────────────
            if (not assigned and w_shelter <= walk_shelter_max_s
                    and near_shelter and shelter_ok
                    and shelter_used.get(near_shelter, 0) + people <= shelter_caps.get(near_shelter, 0)):
                result.assignments[did] = near_shelter
                result.destination_type[did] = "walk_shelter"
                result.walking_demands[did] = w_shelter
                shelter_used[near_shelter] = shelter_used.get(near_shelter, 0) + people
                round_walk += people
                assigned = True

            # ── Mode 2: Walk to rail ────────────────────────
            if (not assigned and w_rail <= walk_rail_max_s
                    and near_rail and rail_ok):
                rcap = station_caps.get(near_rail, 0)
                rused = rail_used.get(near_rail, 0)
                pressure = (rused + people) / rcap if rcap > 0 else 999
                if pressure <= pressure_safe or w_rail <= walk_rail_preferred_s:
                    result.assignments[did] = near_rail
                    result.destination_type[did] = "walk_rail"
                    result.walking_demands[did] = w_rail
                    rail_used[near_rail] = rused + people
                    round_rail += people
                    assigned = True

            # ── Mode 3+4: Bus required ──────────────────────
            if not assigned and round_bus_used + people <= bus_capacity_per_round:
                # Compute costs
                bus_to_shelter_s = dp.get("bus_to_shelter_s", w_shelter / 3)
                bus_to_rail_s = dp.get("bus_to_rail_s", w_rail / 3)

                rail_full = False
                if near_rail and rail_ok:
                    rcap = station_caps.get(near_rail, 0)
                    rused = rail_used.get(near_rail, 0)
                    if rcap > 0 and (rused + people) / rcap >= pressure_limit:
                        rail_full = True

                # Mode 3: Bus to rail (if not full and cost-competitive)
                if (not assigned and not rail_full and near_rail and rail_ok
                        and bus_to_rail_s <= bus_to_shelter_s * bus_rail_cost_tolerance):
                    result.assignments[did] = near_rail
                    result.destination_type[did] = "bus_rail"
                    rail_used[near_rail] = rail_used.get(near_rail, 0) + people
                    round_rail += people
                    round_bus_used += people
                    assigned = True

                # Mode 4: Bus to shelter
                if not assigned and near_shelter and shelter_ok:
                    result.assignments[did] = near_shelter
                    result.destination_type[did] = "bus_shelter"
                    shelter_used[near_shelter] = shelter_used.get(near_shelter, 0) + people
                    round_shelter += people
                    round_bus_used += people
                    assigned = True

            if not assigned:
                next_pending.append(dp)

        # Record round result
        cumulative_served += round_walk + round_rail + round_shelter
        result.round_results.append(RoundResult(
            round_id=round_id,
            served_people=round_walk + round_rail + round_shelter,
            remaining_people=total_people - cumulative_served,
            bus_utilization=round_bus_used / bus_capacity_per_round if bus_capacity_per_round > 0 else 0,
            rail_assigned=round_rail,
            shelter_assigned=round_shelter,
            walk_assigned=round_walk,
            unserved=total_people - cumulative_served,
        ))

        pending = next_pending
        if round_bus_used < bus_capacity_per_round * 0.1:
            break  # bus capacity not the bottleneck, stop iterating

    result.unassigned = [dp["demand_id"] for dp in pending]
    result.shelter_allocations = shelter_used
    result.rail_allocations = rail_used
    result.station_pressures = compute_pressure(
        rail_stations, rail_used, time_window_h=rail_time_window_h,
    )

    return result
