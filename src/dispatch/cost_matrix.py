"""Cost matrix computation from road network.

Supports three modes:
  - euclidean_fast: Euclidean distance × speed (fast, no road network needed).
  - road_network_time: Dijkstra on OSM road network (accurate, slow).
  - cached_network_time: Dijkstra with disk cache in cache/cost_matrix/.
"""
import os
import hashlib
import json
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point
import osmnx as ox


# ── Mode constants ─────────────────────────────────────────────
MODE_EUCLIDEAN = "euclidean_fast"
MODE_NETWORK = "road_network_time"
MODE_CACHED = "cached_network_time"
VALID_MODES = {MODE_EUCLIDEAN, MODE_NETWORK, MODE_CACHED}


def _get_nearest_node(G: nx.MultiDiGraph, point: Point) -> int:
    return ox.nearest_nodes(G, point.x, point.y)


def _cache_key(all_points: list[Point], scenario_id: str, seed: int) -> str:
    """Stable cache key from point coordinates and scenario."""
    coords = [(round(p.x, 5), round(p.y, 5)) for p in all_points]
    h = hashlib.md5(json.dumps([coords, scenario_id, seed]).encode()).hexdigest()[:12]
    return h


def compute_travel_time_matrix(
    G: nx.MultiDiGraph,
    origins: list[Point],
    destinations: list[Point],
) -> np.ndarray:
    """Compute travel-time cost matrix between origins and destinations.

    Uses shortest-path on the road network. Returns a 2D array
    with shape (len(origins), len(destinations)) in seconds.
    """
    n_orig = len(origins)
    n_dest = len(destinations)
    matrix = np.full((n_orig, n_dest), np.inf)

    # Pre-compute nearest nodes
    orig_nodes = [_get_nearest_node(G, p) for p in origins]
    dest_nodes = [_get_nearest_node(G, p) for p in destinations]

    # Ensure travel_time edge attribute
    for u, v, k, data in G.edges(keys=True, data=True):
        if "speed_kph" not in data:
            data["speed_kph"] = data.get("speed_kmh", 40.0)
        speed_ms = max(data["speed_kph"] * 1000 / 3600, 1.0)
        data["travel_time"] = data.get("length", 0) / speed_ms

    for i, o_node in enumerate(orig_nodes):
        try:
            lengths, paths = nx.single_source_dijkstra(
                G, o_node, weight="travel_time", cutoff=7200,
            )
            for j, d_node in enumerate(dest_nodes):
                if d_node in lengths:
                    matrix[i, j] = lengths[d_node]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

    return matrix


def compute_euclidean_matrix(
    origins: list[Point],
    destinations: list[Point],
    default_speed_kmh: float = 40.0,
) -> np.ndarray:
    """Fast fallback: Euclidean distance matrix converted to travel time."""
    n_orig = len(origins)
    n_dest = len(destinations)
    matrix = np.zeros((n_orig, n_dest))
    speed_ms = default_speed_kmh * 1000 / 3600

    for i, o in enumerate(origins):
        for j, d in enumerate(destinations):
            dist_m = o.distance(d) * 111320
            matrix[i, j] = dist_m / speed_ms

    return matrix


def compute_cost_matrix(
    origins: list[Point],
    destinations: list[Point],
    mode: str = MODE_EUCLIDEAN,
    G: nx.MultiDiGraph | None = None,
    cache_dir: str | None = None,
    scenario_id: str = "",
    random_seed: int = 42,
) -> np.ndarray:
    """Unified cost matrix entry point with mode selection.

    Args:
        origins: Origin points.
        destinations: Destination points.
        mode: One of euclidean_fast / road_network_time / cached_network_time.
        G: Road network graph (required for network modes).
        cache_dir: Directory for cached matrices (required for cached mode).
        scenario_id: Scenario ID for cache key.
        random_seed: Seed for cache key stability.

    Returns:
        n × m cost matrix in seconds.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode '{mode}', expected one of {VALID_MODES}")

    if mode == MODE_EUCLIDEAN:
        return compute_euclidean_matrix(origins, destinations)

    if mode == MODE_NETWORK:
        if G is None:
            raise ValueError("Road network G is required for road_network_time mode")
        return compute_travel_time_matrix(G, origins, destinations)

    # MODE_CACHED: try disk cache, compute + save on miss
    if mode == MODE_CACHED:
        if not cache_dir:
            raise ValueError("cache_dir is required for cached_network_time mode")
        if G is None:
            raise ValueError("Road network G is required for cached_network_time mode")

        all_points = origins + destinations
        ck = _cache_key(all_points, scenario_id, random_seed)
        cache_path = os.path.join(cache_dir, f"cost_matrix_{ck}.parquet")

        if os.path.exists(cache_path):
            cached = pd.read_parquet(cache_path).values
            if cached.shape == (len(origins), len(destinations)):
                return cached

        # Compute and cache
        matrix = compute_travel_time_matrix(G, origins, destinations)
        os.makedirs(cache_dir, exist_ok=True)
        pd.DataFrame(matrix).to_parquet(cache_path, index=False)
        return matrix

    return compute_euclidean_matrix(origins, destinations)  # unreachable fallback
