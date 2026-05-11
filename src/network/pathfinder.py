"""Shortest path computation for evacuation routing.

Supports both serial and parallel (multi-core CPU) execution.
"""
from dataclasses import dataclass, field
from typing import Optional
import multiprocessing as mp
import networkx as nx
import osmnx as ox
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point


@dataclass
class PathResult:
    origin_id: str
    destination_id: str
    origin_name: str
    destination_name: str
    path_geometry: Optional[LineString] = None
    distance_m: float = 0.0
    travel_time_s: float = 0.0
    node_path: list = field(default_factory=list)


# ── Shared state for multiprocessing workers ──────────────────
_worker_graph: Optional[nx.MultiDiGraph] = None


def _init_worker(graph: nx.MultiDiGraph):
    """Initializer for worker processes. One graph copy per process."""
    global _worker_graph
    _worker_graph = _prepare_graph(graph)


def _prepare_graph(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Ensure all edges have speed_kph and travel_time attributes."""
    G = G.copy()
    for u, v, k, data in G.edges(keys=True, data=True):
        if "speed_kph" not in data:
            data["speed_kph"] = data.get("speed_kmh", 40.0)
        speed_ms = max(data["speed_kph"] * 1000 / 3600, 1.0)
        data["travel_time"] = data.get("length", 0) / speed_ms
    return G


def _get_nearest_node(G: nx.MultiDiGraph, point: Point) -> int:
    return ox.nearest_nodes(G, point.x, point.y)


def _shortest_path_core(
    G: nx.MultiDiGraph,
    origin: Point,
    destination: Point,
    origin_id: str,
    destination_id: str,
    origin_name: str,
    destination_name: str,
) -> PathResult:
    """Core path computation shared by serial and parallel paths."""
    try:
        orig_node = _get_nearest_node(G, origin)
        dest_node = _get_nearest_node(G, destination)
        node_path = nx.shortest_path(G, orig_node, dest_node, weight="travel_time")

        coords = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in node_path]
        if len(coords) < 2:
            coords = [coords[0], (coords[0][0] + 0.0001, coords[0][1] + 0.0001)]
        geometry = LineString(coords)

        distance_m = 0.0
        travel_time_s = 0.0
        for i in range(len(node_path) - 1):
            u, v = node_path[i], node_path[i + 1]
            edge_data = G.get_edge_data(u, v)
            if edge_data:
                best = min(edge_data.values(), key=lambda d: d.get("length", float("inf")))
                distance_m += best.get("length", 0)
                travel_time_s += best.get("travel_time", 0)

        return PathResult(
            origin_id=origin_id,
            destination_id=destination_id,
            origin_name=origin_name,
            destination_name=destination_name,
            path_geometry=geometry,
            distance_m=distance_m,
            travel_time_s=travel_time_s,
            node_path=node_path,
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        dist_m = origin.distance(destination) * 111320
        return PathResult(
            origin_id=origin_id,
            destination_id=destination_id,
            origin_name=origin_name,
            destination_name=destination_name,
            path_geometry=LineString([origin, destination]),
            distance_m=dist_m,
            travel_time_s=dist_m / (40.0 * 1000 / 3600),
            node_path=[],
        )


def shortest_path(
    G: nx.MultiDiGraph,
    origin: Point,
    destination: Point,
    origin_id: str = "",
    destination_id: str = "",
    origin_name: str = "",
    destination_name: str = "",
) -> PathResult:
    """Compute the shortest path between two points on the network.
    Public API — prepares the graph and delegates to core computation.
    """
    G = _prepare_graph(G)
    return _shortest_path_core(
        G, origin, destination, origin_id, destination_id, origin_name, destination_name,
    )


# ── Task descriptor for parallel dispatch ─────────────────────

class _PathTask:
    """Lightweight task descriptor (avoids passing GeoDataFrame rows across processes)."""
    __slots__ = ("origin_id", "destination_id", "origin_name", "destination_name",
                 "origin_lon", "origin_lat", "dest_lon", "dest_lat")

    def __init__(self, origin_id, destination_id, origin_name, destination_name,
                 origin_lon, origin_lat, dest_lon, dest_lat):
        self.origin_id = origin_id
        self.destination_id = destination_id
        self.origin_name = origin_name
        self.destination_name = destination_name
        self.origin_lon = origin_lon
        self.origin_lat = origin_lat
        self.dest_lon = dest_lon
        self.dest_lat = dest_lat


def _worker_compute_path(task: _PathTask) -> PathResult:
    """Worker function: compute one path using the process-local graph copy."""
    global _worker_graph
    return _shortest_path_core(
        _worker_graph,
        Point(task.origin_lon, task.origin_lat),
        Point(task.dest_lon, task.dest_lat),
        task.origin_id,
        task.destination_id,
        task.origin_name,
        task.destination_name,
    )


# ── Public API ────────────────────────────────────────────────

def compute_evacuation_paths(
    G: nx.MultiDiGraph,
    demand_gdf: gpd.GeoDataFrame,
    shelters_gdf: gpd.GeoDataFrame,
    max_shelters_per_demand: int = 3,
    parallel: bool = True,
    n_workers: Optional[int] = None,
) -> list[PathResult]:
    """Compute shortest paths from each demand point to its nearest shelters.

    Args:
        G: Road network graph.
        demand_gdf: Demand points with 'demand_id', 'demand_name', 'geometry'.
        shelters_gdf: Shelter points with 'shelter_id', 'shelter_name', 'geometry'.
        max_shelters_per_demand: Number of nearest shelters to route to per demand.
        parallel: Use multiprocessing if True.
        n_workers: Number of worker processes. Defaults to CPU count - 1.

    Returns:
        List of PathResult for each demand-shelter pair.
    """
    # Build task list (lightweight, no GeoDataFrame references)
    tasks: list[_PathTask] = []
    for _, demand in demand_gdf.iterrows():
        distances = []
        for _, shelter in shelters_gdf.iterrows():
            dist = demand.geometry.distance(shelter.geometry) * 111320
            distances.append((shelter, dist))
        distances.sort(key=lambda x: x[1])

        for shelter, _ in distances[:max_shelters_per_demand]:
            tasks.append(_PathTask(
                origin_id=str(demand.get("demand_id", "")),
                destination_id=str(shelter.get("shelter_id", "")),
                origin_name=str(demand.get("demand_name", "")),
                destination_name=str(shelter.get("shelter_name", "")),
                origin_lon=demand.geometry.x,
                origin_lat=demand.geometry.y,
                dest_lon=shelter.geometry.x,
                dest_lat=shelter.geometry.y,
            ))

    if not parallel or len(tasks) < 50:
        # Serial path — for small workloads, serial is faster (no spawn/pickle overhead)
        G = _prepare_graph(G)
        results = []
        for task in tasks:
            results.append(_shortest_path_core(
                G,
                Point(task.origin_lon, task.origin_lat),
                Point(task.dest_lon, task.dest_lat),
                task.origin_id,
                task.destination_id,
                task.origin_name,
                task.destination_name,
            ))
        return results

    # Parallel path
    G_prepared = _prepare_graph(G)
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers, initializer=_init_worker, initargs=(G_prepared,)) as pool:
        results = pool.map(_worker_compute_path, tasks)

    return results


def get_cpu_count() -> int:
    """Return the number of available CPU cores."""
    return mp.cpu_count()
