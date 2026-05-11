"""Cost matrix computation from road network."""
import networkx as nx
import numpy as np
from shapely.geometry import Point
import osmnx as ox


def _get_nearest_node(G: nx.MultiDiGraph, point: Point) -> int:
    return ox.nearest_nodes(G, point.x, point.y)


def compute_travel_time_matrix(
    G: nx.MultiDiGraph,
    origins: list[Point],
    destinations: list[Point],
) -> np.ndarray:
    """Compute travel-time cost matrix between origins and destinations.

    Uses shortest-path on the road network. Returns a 2D array
    with shape (len(origins), len(destinations)) in seconds.

    For large matrices, this is the computational bottleneck;
    consider using the parallel pathfinder for batches.
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
                G, o_node, weight="travel_time", cutoff=7200,  # max 2 hours
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
            dist_m = o.distance(d) * 111320  # approximate degrees → meters
            matrix[i, j] = dist_m / speed_ms

    return matrix
