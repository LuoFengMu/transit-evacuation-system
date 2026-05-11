"""Generate demand points near the event center (realistic evacuation scenario)."""
import os
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMAND_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "demand_points_v0.1.geojson")

# Xuzhou city center (彭城广场)
CENTER_LON, CENTER_LAT = 117.205, 34.268
EVENT_RADIUS_M = 1500  # inner danger zone
DEMAND_RADIUS_M = 3000  # demand points spread up to 3km from center

rng = np.random.default_rng(42)
n = 30

# Generate points with radial distribution (more near center, fewer at edge)
angles = rng.uniform(0, 2 * np.pi, n)
# Use sqrt to make distribution denser near center
radii_m = rng.uniform(EVENT_RADIUS_M * 0.3, DEMAND_RADIUS_M, n)

# Convert meters to degrees (approximate)
lons = CENTER_LON + (radii_m * np.cos(angles)) / 111320
lats = CENTER_LAT + (radii_m * np.sin(angles)) / 111320

people = rng.integers(100, 600, n)
priorities = rng.choice([1, 2, 3], n, p=[0.3, 0.5, 0.2])

data = {
    "demand_id": [f"demand_{i:03d}" for i in range(n)],
    "demand_name": [f"疏散点_{i:02d}" for i in range(n)],
    "lon": lons,
    "lat": lats,
    "geometry": [Point(lon, lat) for lon, lat in zip(lons, lats)],
    "people_count": people,
    "priority": priorities,
    "population_type": rng.choice(["居民", "商户", "游客"], n),
}
gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
gdf.to_file(DEMAND_PATH, driver="GeoJSON")
print(f"Generated {len(gdf)} demand points around event center")
print(f"Total people: {gdf['people_count'].sum()}")
print(f"Radius range: {radii_m.min():.0f}m - {radii_m.max():.0f}m from center")
