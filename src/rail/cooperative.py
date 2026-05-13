"""Cooperative evacuation allocation — large crowd scenario (no shelters)."""
from dataclasses import dataclass, field
from typing import Optional
from src.rail.capacity import RailStation, compute_pressure


@dataclass
class RoundResult:
    round_id: int
    served_people: int
    remaining_people: int
    bus_utilization: float
    rail_assigned: int
    bus_periphery: int
    walk_self: int
    walk_rail: int
    unserved: int


@dataclass
class AllocationResult:
    assignments: dict = field(default_factory=dict)
    destination_type: dict = field(default_factory=dict)
    rail_allocations: dict = field(default_factory=dict)
    bus_periphery_count: int = 0
    walk_self_count: int = 0
    unassigned: list = field(default_factory=list)
    station_pressures: list = field(default_factory=list)
    round_results: list = field(default_factory=list)


def allocate_cooperative(
    demand_points: list[dict],
    rail_stations: list[RailStation],
    shelters: Optional[list[dict]] = None,  # backward compat
    # ── Thresholds ──────────────────────────────────────────
    walk_self_max_s: float = 1200.0,       # 20 min (~1.7km)
    walk_rail_max_s: float = 600.0,        # 10 min (~800m)
    pressure_safe: float = 0.8,
    pressure_limit: float = 1.1,
    rail_time_window_h: float = 1.0,
    bus_capacity_per_round: int = 1500,
    max_rounds: int = 3,
    reachable_rail: Optional[set] = None,
) -> AllocationResult:
    """Multi-round crowd evacuation: rail is the primary organized channel."""
    result = AllocationResult()

    station_caps = {s.station_id: max(1, int(s.dynamic_capacity_pax_h * rail_time_window_h))
                    for s in rail_stations}
    rail_used = {s.station_id: 0 for s in rail_stations}

    pending = list(demand_points)
    total_people = sum(dp["people"] for dp in demand_points)
    cumulative_served = 0

    for round_id in range(1, max_rounds + 1):
        if not pending:
            break

        round_bus_used = 0
        round_rail = 0
        round_periphery = 0
        round_walk_self = 0
        round_walk_rail = 0

        pending.sort(key=lambda dp: min(dp.get("walk_to_shelter_s", 9999),
                                         dp.get("walk_to_rail_s", 9999)), reverse=True)

        next_pending = []
        for dp in pending:
            did = dp["demand_id"]
            people = dp["people"]
            w_rail = dp.get("walk_to_rail_s", 9999)
            w_self = dp.get("walk_to_shelter_s", 9999)  # walk-away time
            # Get top-3 nearest rail stations (for load balancing)
            rail_candidates = dp.get("rail_candidates", [])
            if not rail_candidates and dp.get("nearest_rail_id"):
                rail_candidates = [dp["nearest_rail_id"]]
            assigned = False

            # ── Mode 1: Walk away / self-evacuate ────────────
            if not assigned and w_self <= walk_self_max_s:
                result.assignments[did] = "self_evac"
                result.destination_type[did] = "walk_self"
                round_walk_self += people
                assigned = True

            # ── Mode 2: Walk to nearest rail ────────────────
            if not assigned and w_rail <= walk_rail_max_s and rail_candidates:
                for rid in rail_candidates:
                    rcap = station_caps.get(rid, 0)
                    rused = rail_used.get(rid, 0)
                    if rcap > 0 and (rused + people) / rcap <= pressure_safe:
                        result.assignments[did] = rid
                        result.destination_type[did] = "walk_rail"
                        rail_used[rid] = rused + people
                        round_walk_rail += people
                        assigned = True
                        break

            # ── Mode 3: Bus to rail (pick least-loaded candidate) ──
            if not assigned and round_bus_used + people <= bus_capacity_per_round:
                # Find best rail candidate: lowest pressure among those under limit
                best_rail = None
                best_pressure = float("inf")
                for rid in rail_candidates:
                    rcap = station_caps.get(rid, 0)
                    rused = rail_used.get(rid, 0)
                    if rcap > 0:
                        p = (rused + people) / rcap
                        if p < pressure_limit and p < best_pressure:
                            best_pressure = p
                            best_rail = rid

                if best_rail:
                    result.assignments[did] = best_rail
                    result.destination_type[did] = "bus_rail"
                    rail_used[best_rail] = rail_used.get(best_rail, 0) + people
                    round_rail += people
                    round_bus_used += people
                    assigned = True

                # Mode 4: Bus to periphery (rail full → bus out of area)
                if not assigned:
                    result.assignments[did] = "periphery"
                    result.destination_type[did] = "bus_periphery"
                    round_periphery += people
                    round_bus_used += people
                    assigned = True

            if not assigned:
                next_pending.append(dp)

        cumulative_served += round_walk_self + round_walk_rail + round_rail + round_periphery
        result.round_results.append(RoundResult(
            round_id=round_id,
            served_people=round_walk_self + round_walk_rail + round_rail + round_periphery,
            remaining_people=total_people - cumulative_served,
            bus_utilization=round_bus_used / bus_capacity_per_round if bus_capacity_per_round > 0 else 0,
            rail_assigned=round_rail,
            bus_periphery=round_periphery,
            walk_self=round_walk_self,
            walk_rail=round_walk_rail,
            unserved=total_people - cumulative_served,
        ))

        pending = next_pending
        if round_bus_used < bus_capacity_per_round * 0.1:
            break

    result.unassigned = [dp["demand_id"] for dp in pending]
    result.rail_allocations = rail_used
    result.bus_periphery_count = sum(1 for v in result.destination_type.values() if v == "bus_periphery")
    result.walk_self_count = sum(1 for v in result.destination_type.values() if v == "walk_self")
    result.station_pressures = compute_pressure(
        rail_stations, rail_used, time_window_h=rail_time_window_h,
    )

    return result
