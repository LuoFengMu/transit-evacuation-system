"""
Download and prepare Xuzhou base data for the transit-evacuation system.
Outputs: road network, metro stations, shelters, sample demand points.
"""
import json
import os
from datetime import datetime

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import Point, Polygon

# ── Config ──────────────────────────────────────────────────
CITY = "Xuzhou, Jiangsu, China"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _create_manual_metro_stations():
    """Create metro stations from known Xuzhou Metro data."""
    known_stations = [
        ("metro_001", "徐州东站", 117.312, 34.267, "line_1"),
        ("metro_002", "庆丰路", 117.290, 34.262, "line_1"),
        ("metro_003", "黄山垅", 117.275, 34.258, "line_1"),
        ("metro_004", "彭城广场", 117.205, 34.268, "line_1"),
        ("metro_005", "苏堤路", 117.188, 34.264, "line_1"),
        ("metro_006", "工农路", 117.170, 34.260, "line_1"),
        ("metro_007", "韩山", 117.145, 34.252, "line_1"),
        ("metro_008", "杏山子", 117.122, 34.245, "line_1"),
        ("metro_009", "路窝", 117.095, 34.235, "line_1"),
        ("metro_010", "客运北站", 117.220, 34.310, "line_2"),
        ("metro_011", "奔腾大道", 117.215, 34.295, "line_2"),
        ("metro_012", "徐州火车站", 117.210, 34.275, "line_2"),
        ("metro_013", "青年路", 117.200, 34.265, "line_2"),
        ("metro_014", "市中心医院", 117.195, 34.252, "line_2"),
        ("metro_015", "淮塔", 117.198, 34.240, "line_2"),
        ("metro_016", "科技城", 117.205, 34.225, "line_2"),
        ("metro_017", "新城区东", 117.260, 34.210, "line_2"),
        ("metro_018", "马场湖", 117.180, 34.290, "line_3"),
        ("metro_019", "铜山副中心", 117.195, 34.230, "line_3"),
        ("metro_020", "高新区南", 117.200, 34.195, "line_3"),
    ]
    gdf = gpd.GeoDataFrame(
        {
            "station_id": [s[0] for s in known_stations],
            "station_name": [s[1] for s in known_stations],
            "lon": [s[2] for s in known_stations],
            "lat": [s[3] for s in known_stations],
            "geometry": [Point(s[2], s[3]) for s in known_stations],
            "line_id": [s[4] for s in known_stations],
            "station_capacity": 5000,
            "platform_capacity": 1500,
            "entrance_count": 4,
        },
        crs="EPSG:4326",
    )
    path = os.path.join(DATA_PROCESSED, "metro_stations_v0.1.geojson")
    gdf.to_file(path, driver="GeoJSON")
    print(f"  {len(gdf)} metro stations (manual) → {path}")
    return gdf


DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
DATA_OSM = os.path.join(PROJECT_ROOT, "data", "osm")
DATA_MANIFEST = os.path.join(PROJECT_ROOT, "data", "manifest")

os.makedirs(DATA_RAW, exist_ok=True)
os.makedirs(DATA_PROCESSED, exist_ok=True)
os.makedirs(DATA_OSM, exist_ok=True)
os.makedirs(DATA_MANIFEST, exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")
print(f"Target city: {CITY}")

# ── 1. Download road network ─────────────────────────────────
print("\n[1/6] Downloading road network from OSM...")
G = ox.graph_from_place(CITY, network_type="drive", simplify=True)
print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

# Save raw graphml
graphml_path = os.path.join(DATA_OSM, "xuzhou_road_network.graphml")
ox.save_graphml(G, graphml_path)
print(f"  Saved: {graphml_path}")

# ── 2. Extract road edges ────────────────────────────────────
print("\n[2/6] Extracting road edges...")
edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
edges_gdf = edges_gdf.reset_index()

# Select and rename columns for the project schema
edge_cols = {
    "u": "from_node",
    "v": "to_node",
    "osmid": "road_id",
    "geometry": "geometry",
    "length": "length_m",
    "speed_kph": "speed_kmh",
    "highway": "road_type",
    "lanes": "lanes",
    "maxspeed": "maxspeed",
    "oneway": "direction",
    "name": "name",
}
edges_out = edges_gdf[[c for c in edge_cols if c in edges_gdf.columns]].rename(
    columns={k: v for k, v in edge_cols.items() if k in edges_gdf.columns}
)
# Ensure speed_kmh is numeric
if "speed_kmh" in edges_out.columns:
    edges_out["speed_kmh"] = pd.to_numeric(edges_out["speed_kmh"], errors="coerce").fillna(40.0)
else:
    edges_out["speed_kmh"] = 40.0
edges_out["road_id"] = edges_out["road_id"].astype(str)
edges_out["from_node"] = edges_out["from_node"].astype(str)
edges_out["to_node"] = edges_out["to_node"].astype(str)

edges_path = os.path.join(DATA_PROCESSED, "road_edges_v0.1.geojson")
edges_out.to_file(edges_path, driver="GeoJSON")
print(f"  {len(edges_out)} edges → {edges_path}")

# ── 3. Extract road nodes ────────────────────────────────────
print("\n[3/6] Extracting road nodes...")
nodes_gdf = ox.graph_to_gdfs(G, nodes=True, edges=False)
nodes_gdf = nodes_gdf.reset_index()

node_cols = {
    "osmid": "node_id",
    "geometry": "geometry",
    "x": "lon",
    "y": "lat",
    "highway": "node_type",
    "street_count": "connected_road_count",
}
nodes_out = nodes_gdf[[c for c in node_cols if c in nodes_gdf.columns]].rename(
    columns={k: v for k, v in node_cols.items() if k in nodes_gdf.columns}
)
nodes_out["node_id"] = nodes_out["node_id"].astype(str)

nodes_path = os.path.join(DATA_PROCESSED, "road_nodes_v0.1.geojson")
nodes_out.to_file(nodes_path, driver="GeoJSON")
print(f"  {len(nodes_out)} nodes → {nodes_path}")

# ── 4. Extract metro stations from OSM ───────────────────────
print("\n[4/6] Extracting metro stations and rail data...")

try:
    # Metro stations
    metro_tags = {"railway": "station", "station": "subway"}
    metro_gdf = ox.features_from_place(CITY, tags=metro_tags)
    if len(metro_gdf) == 0:
        # Try broader query
        metro_tags = {"railway": "station"}
        metro_gdf = ox.features_from_place(CITY, tags=metro_tags)

    if len(metro_gdf) > 0:
        metro_gdf = metro_gdf.reset_index()
        metro_gdf = metro_gdf[metro_gdf.geometry.type == "Point"].copy()
        metro_gdf["lon"] = metro_gdf.geometry.x
        metro_gdf["lat"] = metro_gdf.geometry.y

        # Safe column extraction
        names = metro_gdf["name"] if "name" in metro_gdf.columns else pd.Series(["未知站点"] * len(metro_gdf))
        lines = metro_gdf["line"] if "line" in metro_gdf.columns else pd.Series([""] * len(metro_gdf))

        metro_out = gpd.GeoDataFrame(
            {
                "station_id": [f"metro_{i:03d}" for i in range(len(metro_gdf))],
                "station_name": names.fillna("未知站点"),
                "lon": metro_gdf["lon"],
                "lat": metro_gdf["lat"],
                "geometry": metro_gdf.geometry,
                "line_id": lines.fillna(""),
                "station_capacity": 5000,
                "platform_capacity": 1500,
                "entrance_count": 4,
            }
        )
        metro_path = os.path.join(DATA_PROCESSED, "metro_stations_v0.1.geojson")
        metro_out.to_file(metro_path, driver="GeoJSON")
        print(f"  {len(metro_out)} metro stations → {metro_path}")
    else:
        _create_manual_metro_stations()

except Exception as e:
    print(f"  Metro extraction failed: {e}")
    _create_manual_metro_stations()

# ── 5. Extract shelters from OSM ──────────────────────────────
print("\n[5/6] Extracting potential shelters...")

shelter_queries = {
    "park": {"leisure": "park"},
    "stadium": {"leisure": "stadium"},
    "school": {"amenity": "school"},
    "university": {"amenity": "university"},
    "square": {"place": "square"},
    "sports_centre": {"leisure": "sports_centre"},
    "hospital": {"amenity": "hospital"},
}

shelter_list = []
for shelter_type, tags in shelter_queries.items():
    try:
        gdf = ox.features_from_place(CITY, tags=tags)
        if len(gdf) > 0:
            gdf = gdf.reset_index()
            gdf["centroid"] = gdf.geometry.centroid
            gdf["shelter_type"] = shelter_type
            gdf["name_raw"] = gdf.get("name", "")
            if "name_raw" in gdf.columns:
                gdf["name_raw"] = gdf["name_raw"].fillna(f"{shelter_type}_unnamed")
            else:
                gdf["name_raw"] = f"{shelter_type}_unnamed"
            # Estimate capacity based on type and area
            if "area" not in gdf.columns:
                gdf["area_m2"] = gdf.geometry.area
            gdf["capacity"] = (gdf["area_m2"] / 5).clip(upper=50000).fillna(1000).astype(int)

            for _, row in gdf.iterrows():
                shelter_list.append({
                    "shelter_id": f"shelter_{shelter_type}_{len(shelter_list):03d}",
                    "shelter_name": row.get("name_raw", f"{shelter_type}"),
                    "lon": row["centroid"].x,
                    "lat": row["centroid"].y,
                    "geometry": row["centroid"],
                    "capacity": row["capacity"],
                    "shelter_type": shelter_type,
                })
    except Exception:
        pass

if shelter_list:
    shelters_out = gpd.GeoDataFrame(shelter_list, crs="EPSG:4326")
else:
    # Generate placeholder shelters around city center
    print("  No shelters found, generating placeholders...")
    center = (117.205, 34.268)  # Xuzhou city center
    shelter_list = []
    for i in range(10):
        lon = center[0] + (i % 5 - 2) * 0.03
        lat = center[1] + (i // 5 - 1) * 0.03
        shelter_list.append({
            "shelter_id": f"shelter_{i:03d}",
            "shelter_name": f"避难场所_{i:02d}",
            "lon": lon, "lat": lat,
            "geometry": Point(lon, lat),
            "capacity": (i + 1) * 2000,
            "shelter_type": "generic",
        })
    shelters_out = gpd.GeoDataFrame(shelter_list, crs="EPSG:4326")

shelters_path = os.path.join(DATA_PROCESSED, "shelters_v0.1.geojson")
shelters_out.to_file(shelters_path, driver="GeoJSON")
print(f"  {len(shelters_out)} shelters → {shelters_path}")

# ── 6. Generate sample demand points ──────────────────────────
print("\n[6/6] Generating sample demand points...")

# Generate points within Xuzhou bounding box
bbox = nodes_out.total_bounds  # [minx, miny, maxx, maxy] from road nodes
np.random.seed(42)

n_demand = 30
demand_list = []
for i in range(n_demand):
    lon = float(np.random.uniform(bbox[0], bbox[2]))
    lat = float(np.random.uniform(bbox[1], bbox[3]))
    demand_list.append({
        "demand_id": f"demand_{i:03d}",
        "demand_name": f"疏散需求点_{i:02d}",
        "lon": lon,
        "lat": lat,
        "geometry": Point(lon, lat),
        "people_count": int(np.random.randint(100, 800)),
        "priority": int(np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])),
        "population_type": np.random.choice(["居民", "学生", "游客"]),
    })

demand_out = gpd.GeoDataFrame(demand_list, crs="EPSG:4326")
demand_path = os.path.join(DATA_PROCESSED, "demand_points_v0.1.geojson")
demand_out.to_file(demand_path, driver="GeoJSON")
total_demand = demand_out["people_count"].sum()
print(f"  {len(demand_out)} demand points ({total_demand} total people) → {demand_path}")

# ── Create data manifest ──────────────────────────────────────
print("\nCreating data manifest...")
manifest = {
    "manifest_schema_version": "1.0",
    "data_version": "data_v0.1",
    "created": datetime.now().strftime("%Y-%m-%d"),
    "description": "Xuzhou base data for minimum viable prototype (v0.1.0)",
    "files": {
        "osm_network": os.path.relpath(graphml_path, PROJECT_ROOT),
        "road_edges": os.path.relpath(edges_path, PROJECT_ROOT),
        "road_nodes": os.path.relpath(nodes_path, PROJECT_ROOT),
        "metro_stations": os.path.relpath(metro_path, PROJECT_ROOT),
        "shelters": os.path.relpath(shelters_path, PROJECT_ROOT),
        "demand_points": os.path.relpath(demand_path, PROJECT_ROOT),
    },
    "stats": {
        "road_edges_count": len(edges_out),
        "road_nodes_count": len(nodes_out),
        "metro_stations_count": len(metro_out),
        "shelters_count": len(shelters_out),
        "demand_points_count": len(demand_out),
        "total_demand_people": int(total_demand),
    },
}

manifest_path = os.path.join(DATA_MANIFEST, "data_manifest_v0.1.json")
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
print(f"  Manifest → {manifest_path}")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 56)
print("Download complete! Data summary:")
print(f"  Road edges:     {manifest['stats']['road_edges_count']}")
print(f"  Road nodes:     {manifest['stats']['road_nodes_count']}")
print(f"  Metro stations: {manifest['stats']['metro_stations_count']}")
print(f"  Shelters:       {manifest['stats']['shelters_count']}")
print(f"  Demand points:  {manifest['stats']['demand_points_count']}")
print(f"  Total demand:   {manifest['stats']['total_demand_people']} people")
print("=" * 56)
