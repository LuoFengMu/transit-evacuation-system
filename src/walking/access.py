"""Walking access computation.

Supports two modes:
  - euclidean_fast: Euclidean (crow-flies) distance × speed (default, fast).
  - walk_network: OSM walk-network shortest path (accurate, requires graph).
"""
from typing import Optional
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
import osmnx as ox

# Walking speed: 5 km/h (normal), 3 km/h (crowded)
WALK_SPEED_MS = 5.0 * 1000 / 3600  # 1.389 m/s
WALK_SPEED_CROWDED_MS = 3.0 * 1000 / 3600

# Mode constants
WALK_EUCLIDEAN = "euclidean_fast"
WALK_NETWORK = "walk_network"
WALK_VALID_MODES = {WALK_EUCLIDEAN, WALK_NETWORK}


def walking_time_m(dist_m: float, crowded: bool = False) -> float:
    """Walking time in seconds for a given distance."""
    speed = WALK_SPEED_CROWDED_MS if crowded else WALK_SPEED_MS
    return dist_m / speed


def _nearest_node(G: nx.MultiDiGraph, point: Point) -> int:
    return ox.nearest_nodes(G, point.x, point.y)


def snap_to_nearest(
    origins: gpd.GeoDataFrame,
    targets: gpd.GeoDataFrame,
    origin_id_col: str = "demand_id",
    target_id_col: str = "station_id",
    target_name_col: str = "station_name",
) -> list[dict]:
    """For each origin, find the nearest target and compute walking distance/time
    using Euclidean (crow-flies) distance × 111320 m/deg.

    Returns list of dicts with: origin_id, target_id, target_name,
    distance_m, walk_time_s, origin_lon, origin_lat, target_lon, target_lat.
    """
    results = []
    for _, orig in origins.iterrows():
        best_dist = float("inf")
        best_target = None
        for _, tgt in targets.iterrows():
            d = orig.geometry.distance(tgt.geometry) * 111320  # approximate meters
            if d < best_dist:
                best_dist = d
                best_target = tgt
        if best_target is not None:
            results.append({
                "origin_id": str(orig.get(origin_id_col, "")),
                "target_id": str(best_target.get(target_id_col, "")),
                "target_name": str(best_target.get(target_name_col, "")),
                "distance_m": round(best_dist, 1),
                "walk_time_s": round(walking_time_m(best_dist), 1),
                "origin_lon": orig.geometry.x, "origin_lat": orig.geometry.y,
                "target_lon": best_target.geometry.x, "target_lat": best_target.geometry.y,
            })
    return results


def snap_to_nearest_network(
    G: nx.MultiDiGraph,
    origins: gpd.GeoDataFrame,
    targets: gpd.GeoDataFrame,
    origin_id_col: str = "demand_id",
    target_id_col: str = "station_id",
    target_name_col: str = "station_name",
    cutoff_s: float = 3600.0,
) -> list[dict]:
    """For each origin, find the nearest target via walk-network shortest path.

    Args:
        G: OSMnx walk-network graph (network_type='walk'), must have
           'length' edge attribute in meters.
        origins: Demand point GeoDataFrame.
        targets: Target GeoDataFrame (rail stations or shelters).
        cutoff_s: Maximum walk time in seconds (default 1 hour).

    Returns:
        List of dicts with distance_m, walk_time_s, origin/target coords.
    """
    results = []
    # Pre-map targets to nearest graph nodes
    target_nodes = [_nearest_node(G, tgt.geometry) for _, tgt in targets.iterrows()]

    for _, orig in origins.iterrows():
        try:
            o_node = _nearest_node(G, orig.geometry)
        except Exception:
            continue

        # Single-source Dijkstra to all reachable nodes within cutoff distance
        cutoff_m = cutoff_s * WALK_SPEED_MS
        try:
            lengths, _ = nx.single_source_dijkstra(G, o_node, weight="length", cutoff=cutoff_m)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            lengths = {}

        best_dist = float("inf")
        best_idx = -1
        for j, t_node in enumerate(target_nodes):
            if t_node in lengths:
                d = lengths[t_node]
                if d < best_dist:
                    best_dist = d
                    best_idx = j

        if best_idx >= 0:
            best_target = targets.iloc[best_idx]
            results.append({
                "origin_id": str(orig.get(origin_id_col, "")),
                "target_id": str(best_target.get(target_id_col, "")),
                "target_name": str(best_target.get(target_name_col, "")),
                "distance_m": round(best_dist, 1),
                "walk_time_s": round(walking_time_m(best_dist), 1),
                "origin_lon": orig.geometry.x, "origin_lat": orig.geometry.y,
                "target_lon": best_target.geometry.x, "target_lat": best_target.geometry.y,
            })
    return results


def compute_access_matrix(
    demand_gdf: gpd.GeoDataFrame,
    rail_stations_gdf: gpd.GeoDataFrame,
    shelters_gdf: gpd.GeoDataFrame,
    bus_stops_gdf: Optional[gpd.GeoDataFrame] = None,
    mode: str = WALK_EUCLIDEAN,
    walk_G: Optional[nx.MultiDiGraph] = None,
) -> dict:
    """Compute walking access from each demand point to all candidate destinations.

    Args:
        demand_gdf: Demand points.
        rail_stations_gdf: Rail station locations.
        shelters_gdf: Shelter locations.
        bus_stops_gdf: Optional bus stop locations.
        mode: 'euclidean_fast' (default) or 'walk_network'.
        walk_G: Walk network graph (required for walk_network mode).

    Returns:
        dict with keys 'to_rail', 'to_shelter', 'to_bus_stop'.
    """
    if mode == WALK_NETWORK:
        if walk_G is None:
            raise ValueError("walk_G is required for walk_network mode")
        def _network_snap(o, t, origin_id_col, target_id_col, target_name_col):
            return snap_to_nearest_network(
                walk_G, o, t, origin_id_col, target_id_col, target_name_col)
        snap_fn = _network_snap
    elif mode == WALK_EUCLIDEAN:
        snap_fn = snap_to_nearest
    else:
        raise ValueError(f"Unknown walk mode '{mode}', expected {WALK_VALID_MODES}")

    result = {
        "to_rail": snap_fn(demand_gdf, rail_stations_gdf,
                           origin_id_col="demand_id",
                           target_id_col="station_id",
                           target_name_col="station_name"),
        "to_shelter": snap_fn(demand_gdf, shelters_gdf,
                              origin_id_col="demand_id",
                              target_id_col="shelter_id",
                              target_name_col="shelter_name"),
    }
    if bus_stops_gdf is not None and len(bus_stops_gdf) > 0:
        result["to_bus_stop"] = snap_fn(demand_gdf, bus_stops_gdf,
                                        origin_id_col="demand_id",
                                        target_id_col="stop_id",
                                        target_name_col="stop_name")
    return result
