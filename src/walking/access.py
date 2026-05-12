"""Walking access computation."""
from typing import Optional
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

# Walking speed: 5 km/h (normal), 3 km/h (crowded)
WALK_SPEED_MS = 5.0 * 1000 / 3600  # 1.389 m/s
WALK_SPEED_CROWDED_MS = 3.0 * 1000 / 3600


def walking_time_m(dist_m: float, crowded: bool = False) -> float:
    """Walking time in seconds for a given distance."""
    speed = WALK_SPEED_CROWDED_MS if crowded else WALK_SPEED_MS
    return dist_m / speed


def snap_to_nearest(
    origins: gpd.GeoDataFrame,
    targets: gpd.GeoDataFrame,
    origin_id_col: str = "demand_id",
    target_id_col: str = "station_id",
    target_name_col: str = "station_name",
) -> list[dict]:
    """For each origin, find the nearest target and compute walking distance/time.

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


def compute_access_matrix(
    demand_gdf: gpd.GeoDataFrame,
    rail_stations_gdf: gpd.GeoDataFrame,
    shelters_gdf: gpd.GeoDataFrame,
    bus_stops_gdf: Optional[gpd.GeoDataFrame] = None,
) -> dict:
    """Compute walking access from each demand point to all candidate destinations.

    Returns dict with keys: 'to_rail', 'to_shelter', 'to_bus_stop'
    Each value is a list of dicts from snap_to_nearest.
    """
    result = {
        "to_rail": snap_to_nearest(demand_gdf, rail_stations_gdf,
                                   origin_id_col="demand_id",
                                   target_id_col="station_id",
                                   target_name_col="station_name"),
        "to_shelter": snap_to_nearest(demand_gdf, shelters_gdf,
                                      origin_id_col="demand_id",
                                      target_id_col="shelter_id",
                                      target_name_col="shelter_name"),
    }
    if bus_stops_gdf is not None and len(bus_stops_gdf) > 0:
        result["to_bus_stop"] = snap_to_nearest(demand_gdf, bus_stops_gdf,
                                                origin_id_col="demand_id",
                                                target_id_col="stop_id",
                                                target_name_col="stop_name")
    return result
