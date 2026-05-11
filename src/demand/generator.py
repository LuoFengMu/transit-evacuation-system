"""Load and generate evacuation demand points."""
import geopandas as gpd
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
