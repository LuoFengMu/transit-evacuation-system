"""Load and generate evacuation demand points."""
import os
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point


def load_demand_points(path: str) -> gpd.GeoDataFrame:
    """Load demand points from a GeoJSON file."""
    gdf = gpd.read_file(path)
    required_cols = ["demand_id", "demand_name", "people_count", "priority", "geometry"]
    for col in required_cols:
        if col not in gdf.columns:
            raise ValueError(f"Demand points file missing required column: {col}")
    return gdf


def generate_demand_points(
    bbox: tuple[float, float, float, float],
    n: int = 30,
    seed: int = 42,
) -> gpd.GeoDataFrame:
    """Generate random demand points within a bounding box.

    bbox: (min_lon, min_lat, max_lon, max_lat)
    """
    rng = np.random.default_rng(seed)
    lons = rng.uniform(bbox[0], bbox[2], n)
    lats = rng.uniform(bbox[1], bbox[3], n)
    people = rng.integers(100, 800, n)
    priorities = rng.choice([1, 2, 3], n, p=[0.3, 0.5, 0.2])
    types = rng.choice(["居民", "学生", "游客"], n)

    data = {
        "demand_id": [f"demand_{i:03d}" for i in range(n)],
        "demand_name": [f"疏散需求点_{i:02d}" for i in range(n)],
        "lon": lons,
        "lat": lats,
        "geometry": [Point(lon, lat) for lon, lat in zip(lons, lats)],
        "people_count": people,
        "priority": priorities,
        "population_type": types,
    }
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


def summarize_demand(gdf: gpd.GeoDataFrame) -> dict:
    """Return summary statistics for demand points."""
    return {
        "total_points": len(gdf),
        "total_people": int(gdf["people_count"].sum()),
        "priority_counts": gdf["priority"].value_counts().to_dict(),
        "type_counts": gdf.get("population_type", pd.Series(dtype=str))
        .value_counts().to_dict(),
    }


def snap_to_road_nodes(
    G: nx.MultiDiGraph,
    demand_gdf: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, int]:
    """Snap each demand point to its nearest road network node.

    Args:
        G: OSMnx road network graph.
        demand_gdf: Demand point GeoDataFrame.

    Returns:
        (snapped_gdf, snapped_count) — a new GeoDataFrame with demand points
        relocated to nearest road nodes, and the count of points moved >10m.
    """
    import osmnx as ox
    result = demand_gdf.copy()
    snapped = 0
    for i, row in result.iterrows():
        try:
            node_id = ox.nearest_nodes(G, row.geometry.x, row.geometry.y)
            nx_x, nx_y = G.nodes[node_id]["x"], G.nodes[node_id]["y"]
            new_pt = Point(nx_x, nx_y)
            dist = row.geometry.distance(new_pt) * 111320
            if dist > 10:
                snapped += 1
            result.at[i, "geometry"] = new_pt
            result.at[i, "lon"] = nx_x
            result.at[i, "lat"] = nx_y
        except Exception:
            pass
    return result, snapped


def _nearest_non_water_node(
    G: nx.MultiDiGraph,
    x: float, y: float,
    water_gdf: gpd.GeoDataFrame,
    max_candidates: int = 20,
) -> tuple[int, float, float]:
    """Find the nearest road node that is not inside any water polygon.

    Searches up to max_candidates nearest nodes by Euclidean distance;
    returns the first one whose coordinates are not within any water polygon.
    Falls back to the absolute nearest node if all candidates are in water.
    """
    # Build sorted list of (node_id, x, y, distance) by distance from (x, y)
    candidates = []
    for nid, data in G.nodes(data=True):
        nx_v, ny_v = data.get('x', 0), data.get('y', 0)
        d2 = (nx_v - x) ** 2 + (ny_v - y) ** 2
        candidates.append((d2, nid, nx_v, ny_v))
    candidates.sort(key=lambda t: t[0])

    # Unify water geometries for fast containment check
    water_union = water_gdf.union_all() if hasattr(water_gdf, 'union_all') else water_gdf.unary_union

    for _, nid, nx_v, ny_v in candidates[:max_candidates]:
        pt = Point(nx_v, ny_v)
        if not water_union.contains(pt):
            return nid, nx_v, ny_v

    # All candidates in water — fall back to nearest
    _, nid, nx_v, ny_v = candidates[0]
    return nid, nx_v, ny_v


def filter_water_bodies(
    demand_gdf: gpd.GeoDataFrame,
    G: nx.MultiDiGraph,
    water_gdf: gpd.GeoDataFrame | None = None,
    cache_dir: str | None = None,
) -> tuple[gpd.GeoDataFrame, int]:
    """Relocate demand points that fall on water bodies to the nearest
    road node that is NOT inside any water polygon.

    If water_gdf is None, attempts to load OSM water polygons via osmnx
    (cached to cache_dir/water_polygons.geojson for reuse).

    Args:
        demand_gdf: Demand point GeoDataFrame.
        G: Road network graph for re-snapping.
        water_gdf: Optional pre-loaded water polygon GeoDataFrame.
        cache_dir: Directory to cache downloaded water data.

    Returns:
        (filtered_gdf, relocated_count).
    """
    import osmnx as ox

    # Load or download water polygons
    if water_gdf is None:
        cache_path = os.path.join(cache_dir, "water_polygons.geojson") if cache_dir else None
        if cache_path and os.path.exists(cache_path):
            water_gdf = gpd.read_file(cache_path)
        else:
            try:
                water_gdf = ox.geometries_from_place(
                    '徐州市, 江苏省, China',
                    tags={'natural': ['water', 'bay'], 'waterway': ['river', 'canal', 'stream']},
                )
                # Keep only polygon/multipolygon geometries
                water_gdf = water_gdf[water_gdf.geometry.type.isin(
                    ['Polygon', 'MultiPolygon'])].copy()
                if cache_path:
                    os.makedirs(cache_dir, exist_ok=True)
                    water_gdf.to_file(cache_path, driver='GeoJSON')
            except Exception:
                return demand_gdf.copy(), 0

    if water_gdf is None or len(water_gdf) == 0:
        return demand_gdf.copy(), 0

    # Unify to single CRS
    if water_gdf.crs != demand_gdf.crs:
        water_gdf = water_gdf.to_crs(demand_gdf.crs)

    # Spatial join: find demand points inside water polygons
    joined = gpd.sjoin(demand_gdf, water_gdf, how='left', predicate='within')
    water_mask = joined['index_right'].notna()

    if not water_mask.any():
        return demand_gdf.copy(), 0

    result = demand_gdf.copy()
    relocated = 0
    for i in result.index[water_mask]:
        try:
            px, py = result.at[i, 'geometry'].x, result.at[i, 'geometry'].y
            nid, nx_v, ny_v = _nearest_non_water_node(G, px, py, water_gdf)
            result.at[i, 'geometry'] = Point(nx_v, ny_v)
            result.at[i, 'lon'] = nx_v
            result.at[i, 'lat'] = ny_v
            relocated += 1
        except Exception:
            pass

    return result, relocated
