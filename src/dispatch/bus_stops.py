"""Map demand points to nearest road-network boarding points.

Buses can only stop at road-accessible points, not at arbitrary
demand locations. This module snaps demand points to the nearest
road network node, which serves as the bus boarding/alighting point.
"""
import geopandas as gpd
import networkx as nx
import osmnx as ox
import numpy as np
from shapely.geometry import Point


def snap_demands_to_network(
    G: nx.MultiDiGraph,
    demand_gdf: gpd.GeoDataFrame,
    max_walk_m: float = 500.0,
) -> gpd.GeoDataFrame:
    """Snap each demand point to the nearest road network node.

    The snapped point becomes the bus boarding location.
    People are assumed to walk from the original demand point
    to this boarding point.

    Args:
        G: Road network graph.
        demand_gdf: Demand points GeoDataFrame.
        max_walk_m: Maximum acceptable walking distance in meters.
                    Points beyond this get a warning flag.

    Returns:
        GeoDataFrame with added columns:
        - board_lon, board_lat: snapped boarding point coordinates
        - walk_distance_m: walking distance from demand to boarding point
    """
    result = demand_gdf.copy()

    board_lons = []
    board_lats = []
    walk_dists = []

    for _, row in result.iterrows():
        pt = row.geometry
        try:
            nearest = ox.nearest_nodes(G, pt.x, pt.y)
            node_x = G.nodes[nearest]["x"]
            node_y = G.nodes[nearest]["y"]
            board_lons.append(node_x)
            board_lats.append(node_y)
            dist_m = pt.distance(Point(node_x, node_y)) * 111320
            walk_dists.append(round(dist_m, 1))
        except Exception:
            board_lons.append(pt.x)
            board_lats.append(pt.y)
            walk_dists.append(0.0)

    result["board_lon"] = board_lons
    result["board_lat"] = board_lats
    result["board_geometry"] = [Point(x, y) for x, y in zip(board_lons, board_lats)]
    result["walk_distance_m"] = walk_dists

    n_far = (result["walk_distance_m"] > max_walk_m).sum()
    if n_far > 0:
        import warnings
        warnings.warn(f"{n_far} demand points > {max_walk_m}m from nearest road")

    return result


def get_board_points(demand_gdf: gpd.GeoDataFrame) -> list[Point]:
    """Get snapped boarding point coordinates as a list of Points."""
    if "board_geometry" in demand_gdf.columns:
        return [g for g in demand_gdf["board_geometry"]]
    return [g for g in demand_gdf.geometry]
