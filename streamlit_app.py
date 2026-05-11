"""
v0.2.0 — 公交-轨道协同疏散仿真系统
公交调度优化 + 简化仿真验证
"""
import os
import time
import yaml
import streamlit as st
from shapely.geometry import Point

from src.network.osm_loader import load_network_from_graphml, network_to_geodataframes
from src.network.event import create_event_from_yaml, get_affected_roads
from src.network.pathfinder import compute_evacuation_paths, get_cpu_count, _prepare_graph, _shortest_path_core
from src.demand.generator import load_demand_points, summarize_demand
from src.demand.shelter import load_shelters, summarize_shelters
from src.visualization.map_view import render_map, render_path_table, render_metrics
from src.dispatch.vehicle import BusDepot, BusVehicle
from src.dispatch.cost_matrix import compute_euclidean_matrix
from src.dispatch.solver import solve_evacuation_dispatch
from src.visualization.dispatch_view import render_dispatch_results, render_dispatch_params


# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="公交-轨道协同疏散仿真系统",
    page_icon="🚌",
    layout="wide",
)

# ── Paths ─────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
GRAPHML_PATH = os.path.join(DATA_DIR, "osm", "xuzhou_road_network.graphml")
DEMAND_PATH = os.path.join(DATA_DIR, "processed", "demand_points_v0.1.geojson")
SHELTERS_PATH = os.path.join(DATA_DIR, "processed", "shelters_v0.1.geojson")
SCENARIO_PATH = os.path.join(PROJECT_ROOT, "configs", "scenarios", "scenario_v0.1_demo.yaml")


# ── Cached loaders ────────────────────────────────────────────
@st.cache_data
def load_scenario(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_network(path: str):
    G = load_network_from_graphml(path)
    nodes_gdf, edges_gdf = network_to_geodataframes(G)
    return G, nodes_gdf, edges_gdf


# ── Sidebar — scenario config ────────────────────────────────
st.sidebar.title("公交-轨道协同疏散仿真")
st.sidebar.caption("v0.2.0 — 公交调度 + 简化仿真")

scenario = load_scenario(SCENARIO_PATH)

st.sidebar.header("突发事件")
event_type = st.sidebar.selectbox(
    "事件类型",
    options=["flood", "earthquake", "fire", "accident", "crowd"],
    index=0,
    format_func=lambda x: {"flood": "暴雨", "earthquake": "地震", "fire": "火灾",
                           "accident": "事故", "crowd": "大客流"}.get(x, x),
)
radius_m = st.sidebar.slider("影响半径 (m)", 200, 5000, 1500, step=100)

scenario["event"]["type"] = event_type
scenario["event"]["radius_m"] = radius_m

# ── Sidebar — dispatch toggle ─────────────────────────────────
enable_bus = st.sidebar.checkbox("启用公交调度", value=True,
    help="勾选后使用 OR-Tools 优化公交车辆路径规划")

if enable_bus:
    bus_params = render_dispatch_params()
else:
    bus_params = None

run_btn = st.sidebar.button("运行分析", type="primary", use_container_width=True)


# ── Main content ──────────────────────────────────────────────
st.title("公交-轨道协同疏散仿真系统")
st.caption("v0.2.0 — 路网加载 · 事件模拟 · 公交调度优化 · 简化仿真验证")

if not run_btn:
    st.info("请在侧边栏配置场景参数，然后点击「运行分析」")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("场景", scenario.get("scenario_name", ""))
    col_b.metric("事件类型", event_type)
    col_c.metric("公交调度", "启用" if enable_bus else "关闭")
    st.stop()


# ═══════════════════════════════════════════════════════════════
# ── Run analysis ──────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════

with st.spinner("加载路网数据..."):
    G, nodes_gdf, edges_gdf = load_network(GRAPHML_PATH)
    st.success(f"路网加载完成：{G.number_of_nodes():,} 节点, {G.number_of_edges():,} 边")

with st.spinner("加载疏散需求点和避难点..."):
    demand_gdf = load_demand_points(DEMAND_PATH)
    shelters_all = load_shelters(SHELTERS_PATH)

with st.spinner("解析突发事件..."):
    event = create_event_from_yaml(scenario)
    affected = get_affected_roads(event, edges_gdf)

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
        f"需求点: {len(demand_gdf)} 个 ({demand_summary['total_people']:,} 人)\n\n"
        f"可用避难点: {len(shelters_gdf)}/{len(shelters_all)} 个"
    )

# ── Path computation (always run for baseline) ──────────────
with st.spinner("计算行人疏散路径 (基线)..."):
    t0 = time.perf_counter()
    paths = compute_evacuation_paths(
        G, demand_gdf, shelters_gdf,
        max_shelters_per_demand=1,
        parallel=False,
    )
    path_elapsed = time.perf_counter() - t0
    n_valid = sum(1 for p in paths if p.node_path)
    st.success(f"行人路径计算完成：{n_valid}/{len(paths)} 条，耗时 {path_elapsed:.1f}s")

# ── Bus dispatch (optional) ───────────────────────────────────
dispatch_result = None
sim_result = None
if enable_bus and bus_params:
    with st.spinner("生成公交集结区和车辆数据..."):
        # Place depots at fixed locations near the event center (outside danger zone)
        offset = radius_m * 1.5 / 111320  # 1.5× event radius away
        depot_defs = [
            ("depot_00", "公交集结区_东", event.center.x + offset, event.center.y),
            ("depot_01", "公交集结区_西", event.center.x - offset, event.center.y),
            ("depot_02", "公交集结区_南", event.center.x, event.center.y - offset),
        ]
        depots = [
            BusDepot(did, dname, dlon, dlat, 20, Point(dlon, dlat))
            for did, dname, dlon, dlat in depot_defs
        ]
        vehicles = []
        for i, depot in enumerate(depots):
            n_buses_per = bus_params["n_buses"] // len(depots)
            for j in range(n_buses_per):
                vehicles.append(BusVehicle(
                    vehicle_id=f"{depot.depot_id}_bus_{j:02d}",
                    depot_id=depot.depot_id,
                    capacity=bus_params["bus_capacity"],
                ))
        st.success(f"生成 {len(depots)} 个集结区, {len(vehicles)} 辆车 (总运力 {sum(v.capacity for v in vehicles):,})")

    with st.spinner("构建调度优化模型 (OR-Tools CVRP)..."):
        demand_points = [g for g in demand_gdf.geometry]
        demand_quantities = demand_gdf["people_count"].tolist()

        # Use Euclidean cost matrix (fast, adequate for planning)
        all_points = [d.location for d in depots] + demand_points
        cost = compute_euclidean_matrix(all_points, all_points)

        dispatch_result = solve_evacuation_dispatch(
            depots=depots,
            vehicles=vehicles,
            demand_points=demand_points,
            demand_quantities=demand_quantities,
            cost_matrix=cost,
            time_limit_s=bus_params["time_limit"],
        )

        if dispatch_result.solver_status in ("optimal", "feasible"):
            n_used = sum(1 for r in dispatch_result.vehicle_routes.values() if len(r) > 1)
            n_sub = len(dispatch_result.sub_demand_quantities)
            n_unserved = len(dispatch_result.unserved_demand)
            st.success(
                f"调度求解完成: {dispatch_result.solver_status} "
                f"({dispatch_result.runtime_s:.1f}s)\n\n"
                f"使用车辆: {n_used}/{len(vehicles)} | "
                f"服务子需求: {n_sub - n_unserved}/{n_sub} | "
                f"未服务: {n_unserved}"
            )
            sim_result = None
        else:
            st.warning(f"调度求解失败: {dispatch_result.solver_status}")
            sim_result = None


# ═══════════════════════════════════════════════════════════════
# ── Build bus route paths for map (before tabs) ──────────────
# ═══════════════════════════════════════════════════════════════
bus_routes = []
depot_locations = []
if dispatch_result and vehicles and dispatch_result.solver_status in ("optimal", "feasible"):
    with st.spinner("计算公交行驶路线..."):
        G_prepared = _prepare_graph(G)
        depot_locations = [d.location for d in depots]
        demand_pts_list = [g for g in demand_gdf.geometry]

        for vid, route in dispatch_result.vehicle_routes.items():
            seq_pts: list[Point] = []
            for stop_type, stop_id, _ in route:
                if stop_type == "depot":
                    for di, d in enumerate(depots):
                        if stop_id == f"depot_{di:02d}":
                            seq_pts.append(d.location)
                            break
                elif stop_type == "pickup" and isinstance(stop_id, int) and stop_id < len(demand_gdf):
                    seq_pts.append(demand_pts_list[stop_id])

            if len(seq_pts) < 2:
                continue

            all_coords: list[tuple] = []
            for k in range(len(seq_pts) - 1):
                path_result = _shortest_path_core(
                    G_prepared, seq_pts[k], seq_pts[k + 1], "", "", "", "",
                )
                if path_result.path_geometry and path_result.path_geometry.geom_type == "LineString":
                    all_coords.extend(list(path_result.path_geometry.coords))
                else:
                    all_coords.append((seq_pts[k].x, seq_pts[k].y))
                    all_coords.append((seq_pts[k + 1].x, seq_pts[k + 1].y))

            if len(all_coords) >= 2:
                n_stops = len([s for s in route if s[0] == "pickup"])
                bus_routes.append({"vehicle_id": vid, "coords": all_coords, "n_stops": n_stops})

        if bus_routes:
            st.success(f"公交路线计算完成: {len(bus_routes)} 条")
        else:
            st.info("所有公交车未被分配路线（需求超出运力）")

# ═══════════════════════════════════════════════════════════════
# ── Display results ──────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════
has_bus = bool(dispatch_result and dispatch_result.solver_status in ("optimal", "feasible"))

if has_bus:
    t_map, t_dispatch, t_paths, t_data = st.tabs(["地图", "公交调度", "路径详情", "数据概览"])
else:
    t_map, t_paths, t_data = st.tabs(["地图", "路径详情", "数据概览"])

# Tab: Map
with t_map:
    st.subheader("疏散仿真地图")
    render_map(
        edges_gdf=edges_gdf,
        event=event,
        demand_gdf=demand_gdf,
        shelters_gdf=shelters_gdf,
        paths=paths,
        affected_roads=affected,
        bus_routes=bus_routes if bus_routes else None,
        depot_locations=depot_locations if depot_locations else None,
    )
    st.caption("黑色: 行人步行路径 | 蓝色: 公交行驶路线 | 蓝色方块: 公交集结区")
    render_metrics(paths, demand_gdf)

# Tab: Dispatch
if has_bus:
    with t_dispatch:
        render_dispatch_results(
            dispatch_result, vehicles, depots,
            pedestrian_paths=paths,
            total_demand_people=int(demand_gdf["people_count"].sum()),
        )

# Tab: Path details
with t_paths:
    st.subheader("行人疏散路径")
    render_path_table(paths)

# Tab: Data overview
with t_data:
    st.subheader("数据概览")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**疏散需求**")
        st.json(demand_summary)
    with c2:
        st.write("**避难点**")
        st.json(shelter_summary)
    if enable_bus and vehicles:
        st.write("**公交运力**")
        st.json({
            "集结区": len(depots),
            "公交车数": len(vehicles),
            "总运力": sum(v.capacity for v in vehicles),
        })
