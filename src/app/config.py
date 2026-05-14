"""Project-wide paths, constants, and configuration."""
import os

VERSION = "0.5.0"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SCENARIOS_DIR = os.path.join(PROJECT_ROOT, "configs", "scenarios")

# Input data paths
GRAPHML_PATH = os.path.join(DATA_DIR, "osm", "xuzhou_road_network.graphml")
DEMAND_PATH = os.path.join(DATA_DIR, "processed", "demand_points_v0.1.geojson")
SHELTERS_PATH = os.path.join(DATA_DIR, "processed", "shelters_v0.1.geojson")
BUS_STOPS_PATH = os.path.join(DATA_DIR, "processed", "bus_stops_v0.3.geojson")
RAIL_STATIONS_PATH = os.path.join(DATA_DIR, "processed", "rail_stations_v0.4.geojson")
RAIL_LINES_PATH = os.path.join(DATA_DIR, "processed", "rail_lines_v0.4.csv")

# Output paths
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
SUMO_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "sumo")
RUNS_DIR = os.path.join(OUTPUTS_DIR, "runs")

# SUMO paths
SUMO_DIR = os.path.join(PROJECT_ROOT, "sumo")
SUMO_NETWORKS_DIR = os.path.join(SUMO_DIR, "networks")
SUMO_ROUTES_DIR = os.path.join(SUMO_DIR, "routes")
SUMO_CONFIGS_DIR = os.path.join(SUMO_DIR, "configs")
SUMO_NET_FULL = os.path.join(SUMO_NETWORKS_DIR, "xuzhou_full_v2.net.xml")
SUMO_NET_FALLBACK = os.path.join(SUMO_NETWORKS_DIR, "xuzhou_full.net.xml")
SUMO_NET_CROPPED = os.path.join(SUMO_NETWORKS_DIR, "xuzhou_cropped.net.xml")

# Cache directory
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
COST_MATRIX_CACHE = os.path.join(CACHE_DIR, "cost_matrix")

# Event location coordinates (lon, lat)
EVENT_CENTERS = {
    "彭城广场": (117.205, 34.268),
    "徐州奥体中心": (117.283, 34.251),
    "徐州音乐厅": (117.172, 34.243),
    "徐州火车站": (117.210, 34.275),
    "徐州东站": (117.312, 34.267),
    "云龙湖广场": (117.155, 34.245),
}

EVENT_LOCATION_OPTIONS = list(EVENT_CENTERS.keys())
DEFAULT_CENTER = (117.205, 34.268)  # 彭城广场

# Demand scale presets
DEMAND_SCALE_OPTIONS = [30000, 50000, 80000, 100000, 150000, 200000]
DEFAULT_DEMAND_SCALE = 30000

# Cost matrix mode labels
COST_MODE_LABELS = {
    "euclidean_fast": "欧氏距离 (快速)",
    "road_network_time": "路网时间 (精确)",
    "cached_network_time": "缓存路网 (论文)",
}

# Capacity factor labels
CAPACITY_FACTOR_LABELS = {0.7: "保守 (×0.7)", 1.0: "基准 (×1.0)", 1.2: "乐观 (×1.2)"}

# Real bus depots/terminals in Xuzhou (name, lon, lat)
XUZHOU_DEPOTS = [
    ("徐州老火车站", 117.1989, 34.2689),
    ("徐州汽车总站", 117.1998, 34.2637),
    ("铜山汽车站", 117.2000, 34.2623),
    ("宣武客运站", 117.1945, 34.2588),
    ("徐州汽车客运西站", 117.1334, 34.2565),
    ("徐州东站公交枢纽", 117.313, 34.268),
    ("徐州站北广场", 117.211, 34.278),
]


def list_scenario_files() -> list[str]:
    """List available scenario YAML files, A/B/C first."""
    files = sorted([f for f in os.listdir(SCENARIOS_DIR) if f.endswith('.yaml')])
    priority = [f for f in files if f.startswith(('A_', 'B_', 'C_'))]
    others = [f for f in files if f not in priority]
    return priority + others
