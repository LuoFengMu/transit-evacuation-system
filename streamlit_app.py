"""
v0.1.0 — 城市大客流与突发事件的公交-轨道协同疏散仿真系统
最小可运行原型：路网加载 → 事件设置 → 需求点 → 最短路径 → 地图展示
"""
import os
import sys
import time
import yaml
import streamlit as st
import geopandas as gpd

from src.network.osm_loader import load_network_from_graphml, network_to_geodataframes
from src.network.event import create_event_from_yaml, get_affected_roads
from src.network.pathfinder import compute_evacuation_paths, get_cpu_count
from src.demand.generator import load_demand_points, summarize_demand
from src.demand.shelter import load_shelters, summarize_shelters
from src.visualization.map_view import render_map, render_path_table, render_metrics


# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="公交-轨道协同疏散仿真系统",
    page_icon="🚌",
    layout="wide",
)

# ── Paths ─────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")

GRAPHML_PATH = os.path.join(DATA_DIR, "osm", "xuzhou_road_network.graphml")
DEMAND_PATH = os.path.join(DATA_DIR, "processed", "demand_points_v0.1.geojson")
SHELTERS_PATH = os.path.join(DATA_DIR, "processed", "shelters_v0.1.geojson")
SCENARIO_PATH = os.path.join(CONFIG_DIR, "scenarios", "scenario_v0.1_demo.yaml")


# ── Load scenario config ──────────────────────────────────────
@st.cache_data
def load_scenario(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_network(path: str):
    G = load_network_from_graphml(path)
    nodes_gdf, edges_gdf = network_to_geodataframes(G)
    return G, nodes_gdf, edges_gdf


# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("公交-轨道协同疏散仿真系统")
st.sidebar.caption("v0.1.0 — 最小可运行原型")

scenario = load_scenario(SCENARIO_PATH)

st.sidebar.header("场景配置")
scenario_name = st.sidebar.text_input("场景名称", value=scenario.get("scenario_name", ""))
event_type = st.sidebar.selectbox(
    "事件类型",
    options=["flood", "earthquake", "fire", "accident", "crowd"],
    index=0,
    format_func=lambda x: {"flood": "暴雨", "earthquake": "地震", "fire": "火灾",
                           "accident": "事故", "crowd": "大客流"}.get(x, x),
)
radius_m = st.sidebar.slider("影响半径 (m)", 200, 5000, 1500, step=100)
run_btn = st.sidebar.button("运行分析", type="primary", use_container_width=True)

# Update scenario with user selections
scenario["event"]["type"] = event_type
scenario["event"]["radius_m"] = radius_m

st.sidebar.divider()
with st.sidebar.expander("数据文件"):
    st.caption(f"路网: {os.path.basename(GRAPHML_PATH)}")
    st.caption(f"需求点: {os.path.basename(DEMAND_PATH)}")
    st.caption(f"避难点: {os.path.basename(SHELTERS_PATH)}")


# ── Main content ──────────────────────────────────────────────
st.title("城市大客流与突发事件的公交-轨道协同疏散仿真")
st.caption("v0.1.0 — 路网加载 · 事件模拟 · 最短路径 · 基础可视化")

if not run_btn:
    st.info("请在侧边栏配置场景参数，然后点击「运行分析」")
    # Show overview of data availability
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("场景", scenario.get("scenario_name", ""))
    with col_b:
        st.metric("事件类型", event_type)
    st.stop()


# ── Run analysis ──────────────────────────────────────────────
with st.spinner("加载路网数据..."):
    G, nodes_gdf, edges_gdf = load_network(GRAPHML_PATH)
    st.success(f"路网加载完成：{G.number_of_nodes():,} 节点, {G.number_of_edges():,} 边")

with st.spinner("加载疏散需求点和避难点..."):
    demand_gdf = load_demand_points(DEMAND_PATH)
    shelters_all = load_shelters(SHELTERS_PATH)

with st.spinner("解析突发事件..."):
    event = create_event_from_yaml(scenario)
    affected = get_affected_roads(event, edges_gdf)

    # ── Evacuation logic ──────────────────────────────────────
    # Danger zone = event radius. Shelters inside danger zone are excluded.
    # Demand points are already generated near the event (within ~3km).
    danger_radius_deg = event.radius_m / 111320
    shelter_distances = shelters_all.geometry.apply(
        lambda g: g.distance(event.center)
    )
    safe_mask = shelter_distances > danger_radius_deg
    shelters_gdf = shelters_all[safe_mask].copy()

    demand_summary = summarize_demand(demand_gdf)
    shelter_summary = summarize_shelters(shelters_gdf)

    st.success(
        f"事件: {event.event_type} | 中心: 彭城广场 | 危险半径: {event.radius_m}m\n\n"
        f"需求点: {len(demand_gdf)} 个 ({demand_summary['total_people']:,} 人) 分布于事件周边\n\n"
        f"可用避难点: {len(shelters_gdf)}/{len(shelters_all)} 个 (危险区内 {len(shelters_all) - len(shelters_gdf)} 个已排除)"
    )

with st.spinner(f"计算最短疏散路径..."):
    t0 = time.perf_counter()
    paths = compute_evacuation_paths(
        G, demand_gdf, shelters_gdf,
        max_shelters_per_demand=1,  # 1 path per demand = faster
        parallel=False,             # serial is faster for small workloads on macOS
    )
    elapsed = time.perf_counter() - t0
    n_valid = sum(1 for p in paths if p.node_path)
    st.success(f"路径计算完成：{n_valid}/{len(paths)} 条有效路径，耗时 {elapsed:.1f}s")


# ── Display results ───────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["地图", "路径详情", "数据概览"])

with tab1:
    st.subheader("疏散路径地图")
    render_map(
        edges_gdf=edges_gdf,
        event=event,
        demand_gdf=demand_gdf,
        shelters_gdf=shelters_gdf,
        paths=paths,
        affected_roads=affected,
    )
    render_metrics(paths, demand_gdf)

with tab2:
    st.subheader("疏散路径详情")
    render_path_table(paths)

with tab3:
    st.subheader("数据概览")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**疏散需求**")
        st.json(demand_summary)
    with c2:
        st.write("**避难点**")
        st.json(shelter_summary)
    st.write("**突发事件**")
    st.json({
        "event_id": event.event_id,
        "event_type": event.event_type,
        "center": [event.center.x, event.center.y],
        "radius_m": event.radius_m,
        "affected_roads": len(event.affected_road_ids),
    })
