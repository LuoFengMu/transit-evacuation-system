"""OR-Tools CVRP solver for bus evacuation dispatch."""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from src.dispatch.vehicle import BusVehicle, BusDepot
from src.dispatch.cost_matrix import compute_euclidean_matrix
from shapely.geometry import Point


@dataclass
class DispatchResult:
    """Result of a dispatch optimization run."""
    vehicle_routes: dict = field(default_factory=dict)  # vehicle_id → [(stop_type, point_id, arrival_time)]
    assignments: dict = field(default_factory=dict)     # demand_id → vehicle_id
    total_cost: float = 0.0
    unserved_demand: list = field(default_factory=list)
    objective_value: float = 0.0
    solver_status: str = "unknown"
    runtime_s: float = 0.0
    sub_demand_quantities: list = field(default_factory=list)  # people per sub-demand
    n_original_demands: int = 0


def build_and_solve_cvrp(
    cost_matrix: np.ndarray,
    demand_quantities: list[int],
    vehicle_capacities: list[int],
    depot_indices: list[int],
    time_limit_s: float = 30.0,
) -> tuple[object, object, object]:
    """Build and solve a CVRP model with OR-Tools.

    Args:
        cost_matrix: (n_locations × n_locations) travel cost matrix.
                      Location 0..(n_depots-1) are depots, rest are demand points.
        demand_quantities: Number of people at each demand point (excluding depots).
        vehicle_capacities: Capacity of each bus.
        depot_indices: Which location index each vehicle starts from.

    Returns:
        (solution, routing, manager) or raises if no solution found.
    """
    n_locations = cost_matrix.shape[0]
    n_vehicles = len(vehicle_capacities)

    manager = pywrapcp.RoutingIndexManager(n_locations, n_vehicles, depot_indices, depot_indices)
    routing = pywrapcp.RoutingModel(manager)

    # ── Cost callback ───────────────────────────────────────
    def distance_callback(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)
        to_node = manager.IndexToNode(to_idx)
        return int(cost_matrix[from_node, to_node])

    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    # ── Capacity constraint ─────────────────────────────────
    def demand_callback(node_idx):
        node = manager.IndexToNode(node_idx)
        n_depots = len(set(depot_indices))
        if node < n_depots:
            return 0
        demand_idx = node - n_depots
        if demand_idx < len(demand_quantities):
            return demand_quantities[demand_idx]
        return 0

    demand_idx_cb = routing.RegisterUnaryTransitCallback(demand_callback)
    capacity_dim = routing.AddDimensionWithVehicleCapacity(
        demand_idx_cb, 0, vehicle_capacities, True, "Capacity",
    )

    # ── Allow skipping demand points (disjunction) ──────────
    # Penalty for not visiting a demand point. High enough to
    # prioritize service, but allows infeasible-capacity cases.
    penalty = 1_000_000
    n_depots = len(set(depot_indices))
    for node in range(n_depots, n_locations):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # ── Search parameters ───────────────────────────────────
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_params.time_limit.seconds = int(time_limit_s)

    solution = routing.SolveWithParameters(search_params)
    return solution, routing, manager


def solve_evacuation_dispatch(
    depots: list[BusDepot],
    vehicles: list[BusVehicle],
    demand_points: list[Point],
    demand_quantities: list[int],
    cost_matrix: Optional[np.ndarray] = None,
    time_limit_s: float = 30.0,
) -> DispatchResult:
    """Solve the bus evacuation dispatch problem.

    Automatically splits demand points into sub-demands no larger
    than the smallest vehicle capacity, so vehicles can make
    partial pickups at each location.

    Args:
        depots: List of bus depots (start locations).
        vehicles: List of available bus vehicles.
        demand_points: Demand point locations (each is a pickup).
        demand_quantities: Number of people at each demand point.
        cost_matrix: Optional pre-computed travel time matrix.
                     If None, uses Euclidean approximation.
        time_limit_s: Solver time limit in seconds.

    Returns:
        DispatchResult with vehicle routes and assignments.
    """
    import time
    t0 = time.perf_counter()

    min_cap = min(v.capacity for v in vehicles)

    # ── Split demand points into sub-demands ────────────────
    # Each sub-demand ≤ min vehicle capacity so at least one vehicle can serve it.
    split_points: list[Point] = []
    split_quantities: list[int] = []
    split_origin_map: dict[int, int] = {}  # sub_idx → original_idx

    for orig_idx, (pt, qty) in enumerate(zip(demand_points, demand_quantities)):
        remaining = qty
        while remaining > 0:
            chunk = min(remaining, min_cap)
            split_points.append(pt)
            split_quantities.append(chunk)
            split_origin_map[len(split_points) - 1] = orig_idx
            remaining -= chunk

    n_depots = len(depots)
    n_demands = len(split_points)
    n_vehicles = len(vehicles)

    depot_points = [d.location for d in depots]
    all_points = depot_points + split_points

    if cost_matrix is not None:
        # Rebuild cost matrix for split points (duplicate rows/cols)
        orig_n = cost_matrix.shape[0]
        if orig_n == n_depots + len(demand_points):
            new_cost = np.zeros((n_depots + n_demands, n_depots + n_demands))
            for i in range(n_depots + n_demands):
                orig_i = i if i < n_depots else n_depots + split_origin_map[i - n_depots]
                for j in range(n_depots + n_demands):
                    orig_j = j if j < n_depots else n_depots + split_origin_map[j - n_depots]
                    new_cost[i, j] = cost_matrix[orig_i, orig_j]
            cost_matrix = new_cost
    else:
        cost_matrix = compute_euclidean_matrix(all_points, all_points)

    if cost_matrix.shape != (n_depots + n_demands, n_depots + n_demands):
        raise ValueError(
            f"Cost matrix shape {cost_matrix.shape} doesn't match "
            f"{n_depots + n_demands} locations"
        )

    # Each vehicle starts from its depot
    depot_indices = []
    for v in vehicles:
        for di, depot in enumerate(depots):
            if v.depot_id == depot.depot_id:
                depot_indices.append(di)
                break
        else:
            depot_indices.append(0)  # fallback: first depot

    vehicle_capacities = [v.capacity for v in vehicles]

    try:
        solution, routing, manager = build_and_solve_cvrp(
            cost_matrix, split_quantities, vehicle_capacities,
            depot_indices, time_limit_s,
        )
    except Exception as e:
        return DispatchResult(
            solver_status=f"error: {e}",
            runtime_s=time.perf_counter() - t0,
        )

    if not solution:
        return DispatchResult(
            solver_status="no_solution",
            runtime_s=time.perf_counter() - t0,
        )

    # ── Extract results ─────────────────────────────────────
    result = DispatchResult(
        solver_status="optimal" if routing.status() == 1 else "feasible",
        objective_value=solution.ObjectiveValue(),
        runtime_s=time.perf_counter() - t0,
        sub_demand_quantities=split_quantities,
        n_original_demands=len(demand_quantities),
    )

    serviced_demands: set[int] = set()

    for vi in range(n_vehicles):
        vehicle_id = vehicles[vi].vehicle_id
        route = []
        idx = routing.Start(vi)
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node < n_depots:
                route.append(("depot", f"depot_{node:02d}", 0))
            else:
                demand_idx = node - n_depots
                if demand_idx < n_demands:
                    serviced_demands.add(demand_idx)
                    route.append(("pickup", demand_idx, -1))  # time filled later
                    result.assignments[f"demand_{demand_idx:03d}"] = vehicle_id
            idx = solution.Value(routing.NextVar(idx))
        result.vehicle_routes[vehicle_id] = route

    result.unserved_demand = [
        i for i in range(n_demands) if i not in serviced_demands
    ]
    result.total_cost = solution.ObjectiveValue()

    return result
