"""Simplified discrete-event simulation for bus evacuation validation.

Before v0.3.0 (SUMO), this provides a lightweight way to estimate
bus travel times, passenger boarding, and total evacuation duration.
"""
from dataclasses import dataclass, field
from typing import Optional
import heapq
import numpy as np
from shapely.geometry import Point

from src.dispatch.vehicle import BusVehicle, BusDepot
from src.dispatch.solver import DispatchResult


@dataclass
class SimulationEvent:
    """A discrete event in the simulation."""
    time_s: float
    event_type: str          # "depart", "arrive_pickup", "board_complete", "arrive_shelter", "done"
    vehicle_id: str
    location_id: str = ""
    passengers: int = 0
    note: str = ""

    def __lt__(self, other):
        return self.time_s < other.time_s


@dataclass
class SimpleSimResult:
    total_evacuation_time_s: float = 0.0
    evacuated_people: int = 0
    unserved_people: int = 0
    completion_rate: float = 0.0
    vehicle_utilization: dict = field(default_factory=dict)
    vehicle_logs: list[dict] = field(default_factory=list)
    timeline: list[tuple[float, str]] = field(default_factory=list)


def run_simple_simulation(
    dispatch_result: DispatchResult,
    vehicles: list[BusVehicle],
    travel_times: dict,              # (from_id, to_id) → seconds
    demand_people: dict,             # point_id → count
    boarding_rate_pax_s: float = 2.0,  # passengers boarded per second
) -> SimpleSimResult:
    """Run a simplified discrete-event simulation of the dispatch plan.

    The simulation models:
      - Buses depart from depot at time 0 (or their available time)
      - Travel to each demand point in sequence (fixed travel time)
      - Boarding takes time proportional to passengers
      - Buses carry passengers to the final destination
      - Capacity constraint already enforced by OR-Tools assignment
    """
    event_queue: list[SimulationEvent] = []
    vehicle_state: dict[str, dict] = {}  # vehicle_id → current state

    # Initialize vehicles
    vehicle_map = {v.vehicle_id: v for v in vehicles}
    for v in vehicles:
        vehicle_state[v.vehicle_id] = {
            "status": "idle",
            "passengers": 0,
            "route_index": 0,
            "current_location": f"depot_{v.depot_id}",
        }
        # Schedule first departure
        heapq.heappush(event_queue, SimulationEvent(
            time_s=v.available_time,
            event_type="depart",
            vehicle_id=v.vehicle_id,
            location_id=f"depot_{v.depot_id}",
            note=f"Depart from depot (capacity={v.capacity})",
        ))

    sim_result = SimpleSimResult()
    max_time = 0.0
    total_evacuated = 0

    while event_queue:
        evt = heapq.heappop(event_queue)
        max_time = max(max_time, evt.time_s)
        sim_result.timeline.append((evt.time_s, f"{evt.vehicle_id}: {evt.event_type} {evt.note}"))

        state = vehicle_state[evt.vehicle_id]
        vehicle = vehicle_map.get(evt.vehicle_id)

        if evt.event_type == "depart":
            # Get next stop from dispatch route
            route = dispatch_result.vehicle_routes.get(evt.vehicle_id, [])
            idx = state["route_index"]
            if idx < len(route):
                stop_type, stop_id, _ = route[idx]
                # Travel to next stop
                from_id = state["current_location"]
                to_id = stop_id
                travel_time = travel_times.get((from_id, to_id), 300)
                heapq.heappush(event_queue, SimulationEvent(
                    time_s=evt.time_s + travel_time,
                    event_type="arrive_pickup",
                    vehicle_id=evt.vehicle_id,
                    location_id=stop_id,
                    note=f"Arrive at {stop_id}",
                ))
                state["status"] = "traveling"

        elif evt.event_type == "arrive_pickup":
            route = dispatch_result.vehicle_routes.get(evt.vehicle_id, [])
            idx = state["route_index"]
            if idx < len(route):
                _, stop_id, _ = route[idx]
                people = demand_people.get(stop_id, 0)
                boarding_time = people / boarding_rate_pax_s
                heapq.heappush(event_queue, SimulationEvent(
                    time_s=evt.time_s + boarding_time,
                    event_type="board_complete",
                    vehicle_id=evt.vehicle_id,
                    location_id=stop_id,
                    passengers=people,
                    note=f"Boarded {people} pax at {stop_id}",
                ))
                state["status"] = "boarding"

        elif evt.event_type == "board_complete":
            state["passengers"] += evt.passengers
            total_evacuated += evt.passengers
            state["route_index"] += 1

            route = dispatch_result.vehicle_routes.get(evt.vehicle_id, [])
            idx = state["route_index"]
            if idx < len(route):
                # Head to next pickup or depot
                heapq.heappush(event_queue, SimulationEvent(
                    time_s=evt.time_s + 60,  # brief turnaround
                    event_type="depart",
                    vehicle_id=evt.vehicle_id,
                    note=f"Continue route (loaded {state['passengers']} pax)",
                ))
            else:
                # Route complete, return to depot
                heapq.heappush(event_queue, SimulationEvent(
                    time_s=evt.time_s + 300,  # return trip
                    event_type="done",
                    vehicle_id=evt.vehicle_id,
                    note=f"Route complete. Evacuated {state['passengers']} people.",
                ))
                state["status"] = "returning"

        elif evt.event_type == "done":
            state["status"] = "done"
            sim_result.vehicle_logs.append({
                "vehicle_id": evt.vehicle_id,
                "total_passengers": state["passengers"],
                "completion_time": evt.time_s,
            })
            if vehicle:
                sim_result.vehicle_utilization[evt.vehicle_id] = (
                    state["passengers"] / vehicle.capacity
                )

    # ── Aggregate results ───────────────────────────────────
    total_demand = sum(demand_people.values())
    sim_result.total_evacuation_time_s = max_time
    sim_result.evacuated_people = total_evacuated
    sim_result.unserved_people = max(0, total_demand - total_evacuated)
    sim_result.completion_rate = total_evacuated / total_demand if total_demand > 0 else 0.0

    return sim_result
