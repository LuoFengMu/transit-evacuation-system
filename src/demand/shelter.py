"""Load and query shelter data."""
import geopandas as gpd
import numpy as np
import pandas as pd


def load_shelters(path: str) -> gpd.GeoDataFrame:
    """Load shelter points from a GeoJSON file."""
    gdf = gpd.read_file(path)
    required_cols = ["shelter_id", "shelter_name", "capacity", "geometry"]
    for col in required_cols:
        if col not in gdf.columns:
            raise ValueError(f"Shelters file missing required column: {col}")
    return gdf


def get_nearest_shelters(
    demand_gdf: gpd.GeoDataFrame,
    shelters_gdf: gpd.GeoDataFrame,
    n: int = 3,
) -> pd.DataFrame:
    """For each demand point, find the n nearest shelters with distances."""
    rows = []
    for _, demand in demand_gdf.iterrows():
        distances = []
        for _, shelter in shelters_gdf.iterrows():
            dist_deg = demand.geometry.distance(shelter.geometry)
            dist_m = dist_deg * 111320  # approximate
            distances.append((shelter["shelter_id"], shelter["shelter_name"], dist_m))

        distances.sort(key=lambda x: x[2])
        for rank, (sid, sname, dist) in enumerate(distances[:n]):
            rows.append({
                "demand_id": demand["demand_id"],
                "demand_name": demand.get("demand_name", ""),
                "shelter_id": sid,
                "shelter_name": sname,
                "rank": rank + 1,
                "distance_m": round(dist, 1),
            })
    return pd.DataFrame(rows)


def summarize_shelters(gdf: gpd.GeoDataFrame) -> dict:
    """Return summary statistics for shelters."""
    return {
        "total_shelters": len(gdf),
        "total_capacity": int(gdf["capacity"].sum()),
        "type_counts": gdf.get("shelter_type", pd.Series(dtype=str))
        .value_counts().to_dict() if "shelter_type" in gdf.columns else {},
    }
