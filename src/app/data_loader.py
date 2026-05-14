"""Centralized data loading for the evacuation simulation.

Extracted from streamlit_app.py — loads and caches road network,
demand points, shelters, bus stops, rail stations, and SUMO files.
"""
import os
import geopandas as gpd
from shapely.geometry import Point

from src.app.config import (
    GRAPHML_PATH, DEMAND_PATH, SHELTERS_PATH, BUS_STOPS_PATH,
    RAIL_STATIONS_PATH, EVENT_CENTERS,
)
from src.network.osm_loader import load_network_from_graphml, network_to_geodataframes
from src.demand.generator import load_demand_points, snap_to_road_nodes, filter_water_bodies
from src.demand.shelter import load_shelters
from src.rail.capacity import load_stations


# ── Module-level caches ────────────────────────────────────────
_network_cache: dict[str, tuple] = {}


def load_road_network(path: str | None = None):
    """Load OSM road network. Cached per path."""
    path = path or GRAPHML_PATH
    if path not in _network_cache:
        G = load_network_from_graphml(path)
        nodes_gdf, edges_gdf = network_to_geodataframes(G)
        _network_cache[path] = (G, nodes_gdf, edges_gdf)
    return _network_cache[path]


def load_demand(path: str | None = None):
    """Load demand points GeoDataFrame."""
    return load_demand_points(path or DEMAND_PATH)


def load_shelter_data(path: str | None = None):
    """Load shelters GeoDataFrame."""
    return load_shelters(path or SHELTERS_PATH)


def load_bus_stops(path: str | None = None):
    """Load bus stops GeoDataFrame. Returns None if file missing."""
    p = path or BUS_STOPS_PATH
    if os.path.exists(p):
        return gpd.read_file(p)
    return None


def load_rail_stations(stations_path: str | None = None):
    """Load rail stations as list[RailStation] and GeoDataFrame."""
    sp = stations_path or RAIL_STATIONS_PATH
    stations = load_stations(sp)
    gdf = gpd.read_file(sp)
    return stations, gdf


def shift_demand_to_event(
    demand_gdf: gpd.GeoDataFrame,
    event_location: str,
) -> gpd.GeoDataFrame:
    """Translate demand points from default center (彭城广场) to event location."""
    default_center = (117.205, 34.268)
    new_center = EVENT_CENTERS.get(event_location, default_center)
    shift_lon = new_center[0] - default_center[0]
    shift_lat = new_center[1] - default_center[1]
    if shift_lon != 0 or shift_lat != 0:
        demand_gdf = demand_gdf.copy()
        demand_gdf["geometry"] = demand_gdf["geometry"].apply(
            lambda g: Point(g.x + shift_lon, g.y + shift_lat))
        demand_gdf["lon"] = demand_gdf["geometry"].apply(lambda g: g.x)
        demand_gdf["lat"] = demand_gdf["geometry"].apply(lambda g: g.y)
    return demand_gdf


def preprocess_demand(
    demand_gdf: gpd.GeoDataFrame,
    G,
    event_location: str,
    actual_demand: int,
    enable_snap: bool = True,
    enable_water_filter: bool = False,
    cache_dir: str | None = None,
) -> tuple[gpd.GeoDataFrame, list[str]]:
    """Apply demand point translation, snapping, water filtering, and scaling.

    Returns (processed_gdf, log_messages).
    """
    logs = []
    demand_gdf = shift_demand_to_event(demand_gdf, event_location)

    if enable_snap:
        demand_gdf, snapped = snap_to_road_nodes(G, demand_gdf)
        logs.append(f"需求点道路吸附: {snapped}/{len(demand_gdf)}个移至最近道路节点")

    if enable_water_filter:
        demand_gdf, relocated = filter_water_bodies(
            demand_gdf, G, water_gdf=None, cache_dir=cache_dir,
        )
        if relocated > 0:
            logs.append(f"水体过滤: {relocated}个需求点从水域移至道路节点")

    base_total = demand_gdf["people_count"].sum()
    if base_total > 0:
        scale = actual_demand / base_total
        demand_gdf["people_count"] = (demand_gdf["people_count"] * scale).astype(int)
    logs.append(f"需求量级: {actual_demand:,}人 (基准{base_total:,}缩放{scale:.2f}x)")

    return demand_gdf, logs
