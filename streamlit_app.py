"""
v0.4.0 — 公交-轨道协同疏散仿真系统
公交-轨道-步行多方式协同疏散
"""
import os
import time
import yaml
import streamlit as st
import pandas as pd
import geopandas as gpd
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
from src.rail.capacity import load_stations, compute_pressure
from src.walking.access import compute_access_matrix
from src.rail.cooperative import allocate_cooperative
from src.evaluation.metrics import compute_evacuation_metrics, metrics_to_dict
from src.evaluation.comparison import ComparisonResult


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
st.sidebar.caption("v0.4.0 — 公交-轨道协同疏散")

scenario = load_scenario(SCENARIO_PATH)

st.sidebar.header("突发事件")
event_type = st.sidebar.selectbox(
    "事件类型",
    options=["crowd"],
    index=0,
    format_func=lambda x: {"crowd": "大客流"}.get(x, x),
)
radius_m = st.sidebar.slider("影响半径 (m)", 200, 5000, 1500, step=100)

demand_scale = st.sidebar.selectbox(
    "疏散人数量级",
    options=[10000, 30000, 50000, 100000],
    index=0,
    format_func=lambda x: f"{x//10000}万人" if x >= 10000 else f"{x}人",
)
st.session_state["demand_scale"] = demand_scale

scenario["event"]["type"] = event_type
scenario["event"]["radius_m"] = radius_m

# ── Sidebar — dispatch toggle ─────────────────────────────────
enable_bus = st.sidebar.checkbox("启用公交调度", value=True,
    help="勾选后使用 OR-Tools 优化公交车辆路径规划")

if enable_bus:
    bus_params = render_dispatch_params()
else:
    bus_params = None

enable_sumo = st.sidebar.checkbox("启用 SUMO 仿真", value=True,
    help="勾选后运行 SUMO 动态交通仿真。关闭则使用 OSMnx 路网计算公交路线。")

enable_rail = st.sidebar.checkbox("启用轨道协同", value=True,
    help="引入轨道交通作为大容量中长距离疏散通道")

if enable_rail:
    with st.sidebar.expander("轨道协同参数"):
        walk_shelter_min = st.slider("步行到避难点上限(min)", 3, 15, 5, 1)
        walk_rail_min = st.slider("步行到轨道站上限(min)", 10, 40, 30, 5)
        pressure_limit = st.slider("轨道站压力上限", 0.5, 2.0, 1.0, 0.1)

enable_traci = st.sidebar.checkbox("TraCI 道路封闭", value=False,
    help="在SUMO仿真中实时关闭事件影响范围内的道路")

run_btn = st.sidebar.button("运行分析", type="primary", use_container_width=True)


# ── Main content ──────────────────────────────────────────────
st.title("公交-轨道协同疏散仿真系统")
st.caption("v0.4.0 — 公交-轨道-步行协同疏散仿真系统")

if not run_btn:
    st.info("请在侧边栏配置场景参数，然后点击「运行分析」")
    evt_label = "大客流"
    st.markdown(f"**场景**: {scenario.get('scenario_name', '')}　|　**事件**: {evt_label}，半径 {radius_m}m　|　**需求**: {demand_scale//10000}万人　|　**公交**: {'启用' if enable_bus else '关闭'}")
    if enable_bus and bus_params:
        st.markdown(f"**运力**: {bus_params['n_buses']}辆 × {bus_params['bus_capacity']}人 = {bus_params['n_buses'] * bus_params['bus_capacity']}人")
    if enable_sumo:
        st.caption("SUMO 仿真将在调度完成后自动运行")
    st.stop()


# ═══════════════════════════════════════════════════════════════
# ── Run analysis ──────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════
log_lines: list[str] = []
depot_locations = []

def _log(msg: str):
    log_lines.append(msg)

with st.spinner("加载路网数据..."):
    G, nodes_gdf, edges_gdf = load_network(GRAPHML_PATH)
    _log(f"路网加载完成：{G.number_of_nodes():,} 节点, {G.number_of_edges():,} 边")

BUS_STOPS_PATH = os.path.join(DATA_DIR, "processed", "bus_stops_v0.3.geojson")

with st.spinner("加载路网数据..."):
    G, nodes_gdf, edges_gdf = load_network(GRAPHML_PATH)
    _log(f"路网加载完成：{G.number_of_nodes():,} 节点, {G.number_of_edges():,} 边")

with st.spinner("加载疏散需求点和避难点..."):
    demand_gdf = load_demand_points(DEMAND_PATH)
    shelters_all = load_shelters(SHELTERS_PATH)

# Load bus stops if available
bus_stops_gdf = None
if os.path.exists(BUS_STOPS_PATH):
    bus_stops_gdf = gpd.read_file(BUS_STOPS_PATH)

    # Scale demand to selected magnitude
    base_total = demand_gdf["people_count"].sum()
    if base_total > 0:
        scale = demand_scale / base_total
        demand_gdf["people_count"] = (demand_gdf["people_count"] * scale).astype(int)
    _log(f"需求量级: {demand_scale:,}人")

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

    _log(f"事件: {event.event_type} | 中心: 彭城广场 | 危险半径: {event.radius_m}m")
    _log(f"需求点: {len(demand_gdf)} 个 ({demand_summary['total_people']:,} 人)")
    _log(f"可用避难点: {len(shelters_gdf)}/{len(shelters_all)} 个")
    if bus_stops_gdf is not None:
        _log(f"公交站: {len(bus_stops_gdf)} 个")

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
    _log(f"行人路径计算完成：{n_valid}/{len(paths)} 条，耗时 {path_elapsed:.1f}s")

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
        _log(f"生成 {len(depots)} 个集结区, {len(vehicles)} 辆车 (总运力 {sum(v.capacity for v in vehicles):,})")
        depot_locations = [d.location for d in depots]

    with st.spinner("构建调度优化模型 (OR-Tools CVRP)..."):
        # Snap demand points to nearest bus stops (or road nodes as fallback)
        if bus_stops_gdf is not None and len(bus_stops_gdf) > 0:
            # Snap each demand point to the nearest bus stop
            board_pts = []
            walk_dists = []
            for _, demand in demand_gdf.iterrows():
                best_dist = float("inf")
                best_pt = None
                for _, stop in bus_stops_gdf.iterrows():
                    d = demand.geometry.distance(stop.geometry) * 111320
                    if d < best_dist:
                        best_dist = d
                        best_pt = stop.geometry
                board_pts.append(best_pt if best_pt else demand.geometry)
                walk_dists.append(round(best_dist, 1))
            n_far = sum(1 for d in walk_dists if d > 500)
            _log(f"需求点→公交站: 平均 {sum(walk_dists)/len(walk_dists):.0f}m"
                 + (f", {n_far}个超500m" if n_far > 0 else ""))
        else:
            from src.dispatch.bus_stops import snap_demands_to_network, get_board_points
            demand_snapped = snap_demands_to_network(G, demand_gdf)
            board_pts = get_board_points(demand_snapped)
            _log("公交站数据未加载，使用路网节点作为乘降点")

        demand_quantities = demand_gdf["people_count"].tolist()
        all_points = [d.location for d in depots] + board_pts
        cost = compute_euclidean_matrix(all_points, all_points)

        # Build a GeoDataFrame with boarding point geometries for SUMO trip gen
        board_gdf = demand_gdf.copy()
        board_gdf["geometry"] = board_pts

        dispatch_result = solve_evacuation_dispatch(
            depots=depots,
            vehicles=vehicles,
            demand_points=board_pts,
            demand_quantities=demand_quantities,
            cost_matrix=cost,
            time_limit_s=bus_params["time_limit"],
        )

        if dispatch_result.solver_status in ("optimal", "feasible"):
            n_used = sum(1 for r in dispatch_result.vehicle_routes.values() if len(r) > 1)
            sub_qty = dispatch_result.sub_demand_quantities
            total_assigned = sum(sub_qty[i] for i in range(len(sub_qty)) if i not in dispatch_result.unserved_demand)
            _log(
                f"调度: {dispatch_result.solver_status} "
                f"({dispatch_result.runtime_s:.1f}s) | "
                f"用车: {n_used}/{len(vehicles)}辆 | "
                f"单趟接走: {total_assigned}人 / 总需求 {sum(sub_qty)}人"
            )
        else:
            st.warning(f"调度求解失败: {dispatch_result.solver_status}")

# ── SUMO simulation (optional) ────────────────────────────────
sumo_result = None
sumo_bus_routes = []
if enable_sumo and dispatch_result and dispatch_result.solver_status in ("optimal", "feasible"):
    import os as _os
    SUMO_NET = _os.path.join(PROJECT_ROOT, "sumo", "networks", "xuzhou_full.net.xml")
    SUMO_OUTPUT_DIR = _os.path.join(PROJECT_ROOT, "outputs", "sumo")

    if not _os.path.exists(SUMO_NET):
        st.warning("SUMO 路网未生成。请先运行 OSM→SUMO 转换。")
    else:
        with st.spinner("将调度方案转换为 SUMO 路径..."):
            from src.simulation.route_builder import dispatch_to_sumo_trips
            SUMO_ROUTES_DIR = _os.path.join(PROJECT_ROOT, "sumo", "routes")
            try:
                from src.simulation.route_builder import compute_rounds_needed
                n_rounds = compute_rounds_needed(dispatch_result, vehicles, demand_gdf)
                trip_path, route_path = dispatch_to_sumo_trips(
                    dispatch_result, vehicles, depots, board_gdf,
                    SUMO_ROUTES_DIR, SUMO_NET,
                    max_rounds=n_rounds,
                )
                _log(f"SUMO 路径转换完成 ({n_rounds} 轮循环)")

                with st.spinner("运行 SUMO 仿真 (191k 路段, 需要 1-3 分钟)..."):
                    from src.simulation.sumo_runner import (create_sumocfg, run_sumo_headless,
                        run_sumo_with_traci, find_edges_near_event)
                    sumocfg_path = _os.path.join(PROJECT_ROOT, "sumo", "configs", "simulation_v0.3.sumocfg")
                    create_sumocfg(SUMO_NET, route_path, sumocfg_path)
                    sumo_output_dir = _os.path.join(PROJECT_ROOT, "outputs", "sumo")

                    if enable_traci:
                        closure_edges = find_edges_near_event(
                            SUMO_NET,
                            (event.center.x, event.center.y),
                            event.radius_m * 1.2,
                        )
                        _log(f"TraCI 道路封闭: {len(closure_edges)} 条路段")
                        sumo_result = run_sumo_with_traci(
                            sumocfg_path, sumo_output_dir,
                            road_closure_edges=closure_edges,
                            closure_time_s=300,
                        )
                    else:
                        sumo_result = run_sumo_headless(sumocfg_path, sumo_output_dir)

                    if sumo_result.success:
                        n_logs = len(sumo_result.vehicle_logs)
                        n_finished = sumo_result.vehicles_arrived
                        _log(
                            f"SUMO 仿真: {n_finished}/{n_logs} 趟行程, "
                            f"平均 {sumo_result.avg_duration_s:.0f}s, "
                            f"{sumo_result.avg_speed_ms * 3.6:.1f} km/h"
                        )

                        # Extract trajectories for map overlay
                        from src.simulation.sumo_runner import extract_vehicle_trajectories
                        vehroute_path = _os.path.join(SUMO_OUTPUT_DIR, "vehroute.xml")
                        if _os.path.exists(vehroute_path):
                            sumo_trajectories = extract_vehicle_trajectories(vehroute_path, SUMO_NET)
                            # Merge legs into per-vehicle trajectories
                            merged: dict[str, list] = {}
                            for t in sumo_trajectories:
                                vid = t["vehicle_id"]
                                if vid not in merged:
                                    merged[vid] = {"vehicle_id": vid, "coords": [], "depart": t["depart"], "arrival": t["arrival"]}
                                merged[vid]["coords"].extend(t["coords"])
                                merged[vid]["arrival"] = max(merged[vid]["arrival"], t["arrival"])
                            sumo_bus_routes = [v for v in merged.values() if len(v["coords"]) >= 2]
                            _log(f"SUMO 轨迹提取完成: {len(sumo_bus_routes)} 辆车")
                    else:
                        st.warning(f"SUMO 仿真异常: {sumo_result.error}")
            except Exception as e:
                st.error(f"SUMO 流程失败: {e}")

# ── Rail cooperative allocation (v0.4.0) ──────────────────────
allocation_result = None
rail_stations = None
station_pressures = None
evac_metrics = None
comparison = ComparisonResult()

RAIL_STATIONS_PATH = os.path.join(DATA_DIR, "processed", "rail_stations_v0.4.geojson")
RAIL_LINES_PATH = os.path.join(DATA_DIR, "processed", "rail_lines_v0.4.csv")

if enable_rail and os.path.exists(RAIL_STATIONS_PATH):
    with st.spinner("加载轨道数据 + 步行接入计算..."):
        rail_stations = load_stations(RAIL_STATIONS_PATH)
        rail_gdf = gpd.read_file(RAIL_STATIONS_PATH)
        _log(f"轨道站: {len(rail_stations)} 个 (1/2/3号线)")

        # Walking access: demand → rail / shelter
        access = compute_access_matrix(demand_gdf, rail_gdf, shelters_gdf)
        avg_rail_walk = sum(a["walk_time_s"] for a in access["to_rail"]) / len(access["to_rail"])
        avg_shelter_walk = sum(a["walk_time_s"] for a in access["to_shelter"]) / len(access["to_shelter"])
        _log(f"步行接入: 轨道站 {avg_rail_walk/60:.1f}min, 避难点 {avg_shelter_walk/60:.1f}min")

    with st.spinner("协同分配 (需求点→避难点/轨道站)..."):
        dp_list = []
        for i, (_, demand) in enumerate(demand_gdf.iterrows()):
            ra = access["to_rail"][i] if i < len(access["to_rail"]) else {}
            sa = access["to_shelter"][i] if i < len(access["to_shelter"]) else {}
            # Estimate bus travel times: ~1/3 of walking time as rough approx
            dp_list.append({
                "demand_id": demand["demand_id"], "people": int(demand["people_count"]),
                "walk_to_shelter_s": sa.get("walk_time_s", 9999),
                "walk_to_rail_s": ra.get("walk_time_s", 9999),
                "bus_to_shelter_s": sa.get("walk_time_s", 9999) / 3,
                "bus_to_rail_s": ra.get("walk_time_s", 9999) / 3,
                "nearest_shelter_id": sa.get("target_id", ""),
                "nearest_rail_id": ra.get("target_id", ""),
            })
        shelter_dicts = [{"shelter_id": r["shelter_id"], "capacity": int(r["capacity"])}
                         for _, r in shelters_gdf.iterrows()]
        bus_cap = (bus_params["n_buses"] * bus_params["bus_capacity"]) if bus_params else 1500

        allocation_result = allocate_cooperative(
            dp_list, rail_stations, shelter_dicts,
            walk_shelter_max_s=walk_shelter_min * 60,
            walk_rail_max_s=walk_rail_min * 60,
            pressure_limit=pressure_limit,
            bus_capacity_per_round=bus_cap,
            max_rounds=3,
        )
        station_pressures = allocation_result.station_pressures

        # Count by mode
        mode_counts = {}
        for v in allocation_result.destination_type.values():
            mode_counts[v] = mode_counts.get(v, 0) + 1
        _log(f"协同: 步行避难点{mode_counts.get('walk_shelter',0)} 步行轨道{mode_counts.get('walk_rail',0)} "
             f"公交轨道{mode_counts.get('bus_rail',0)} 公交避难点{mode_counts.get('bus_shelter',0)} "
             f"未分配{len(allocation_result.unassigned)}")

        # Compute metrics
        evac_metrics = compute_evacuation_metrics(
            demand_gdf, allocation_result,
            station_pressures=station_pressures,
            walking_access=access,
        )
        comparison.add("方案C: 混合协同", evac_metrics)

        # Baseline A: bus-to-shelter only (no rail)
        baseline_result = allocate_cooperative(
            dp_list, [], shelter_dicts,
            walk_shelter_max_s=walk_shelter_min * 60,
            bus_capacity_per_round=bus_cap, max_rounds=3,
        )
        baseline_metrics = compute_evacuation_metrics(demand_gdf, baseline_result, walking_access=access)
        comparison.add("方案A: 公交直达", baseline_metrics)

        # Baseline B: rail priority (force rail for non-walk)
        dp_rail_bias = []
        for dp in dp_list:
            d = dict(dp)
            d["bus_to_rail_s"] = d.get("bus_to_rail_s", 9999) * 0.5
            d["bus_to_shelter_s"] = d.get("bus_to_shelter_s", 9999) * 2.0
            dp_rail_bias.append(d)
        rail_priority_result = allocate_cooperative(
            dp_rail_bias, rail_stations, shelter_dicts,
            walk_shelter_max_s=walk_shelter_min * 60,
            walk_rail_max_s=walk_rail_min * 60,
            pressure_limit=pressure_limit,
            bus_capacity_per_round=bus_cap, max_rounds=3,
        )
        rail_priority_metrics = compute_evacuation_metrics(demand_gdf, rail_priority_result, walking_access=access)
        comparison.add("方案B: 轨道优先", rail_priority_metrics)

    _log(f"轨道协同完成: {len(allocation_result.assignments)}个需求点已分配")

# ═══════════════════════════════════════════════════════════════
# ── Build bus route paths for map (before tabs) ──────────────
# ═══════════════════════════════════════════════════════════════
bus_routes = []
if dispatch_result and vehicles and dispatch_result.solver_status in ("optimal", "feasible") and not sumo_bus_routes:
    with st.spinner("计算公交行驶路线..."):
        G_prepared = _prepare_graph(G)
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
            _log(f"公交路线计算完成: {len(bus_routes)} 条")
        else:
            _log("所有公交车未被分配路线（需求超出运力）")

# ═══════════════════════════════════════════════════════════════
# ── Display results ──────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════
has_bus = bool(
    (dispatch_result and dispatch_result.solver_status in ("optimal", "feasible"))
    or st.session_state.get("dispatch_result")
)
has_sumo = bool(sumo_result and sumo_result.success)
has_rail = bool(allocation_result is not None)

tab_names = ["地图"]
if has_bus:
    tab_names.append("公交调度")
if has_rail:
    tab_names.append("轨道协同")
    tab_names.append("站点压力")
    tab_names.append("方案对比")
if has_sumo:
    tab_names.append("SUMO 仿真")
tab_names += ["路径详情", "数据概览"]

all_tabs = st.tabs(tab_names)
t_map = all_tabs[0]
tab_idx = 1
t_dispatch = all_tabs[tab_idx] if has_bus else None; tab_idx += 1 if has_bus else 0
t_rail = all_tabs[tab_idx] if has_rail else None; tab_idx += 1 if has_rail else 0
t_pressure = all_tabs[tab_idx] if has_rail else None; tab_idx += 1 if has_rail else 0
t_comparison = all_tabs[tab_idx] if has_rail else None; tab_idx += 1 if has_rail else 0
t_sumo = all_tabs[tab_idx] if has_sumo else None; tab_idx += 1 if has_sumo else 0
t_paths = all_tabs[tab_idx]; tab_idx += 1
t_data = all_tabs[tab_idx]

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
        bus_routes=sumo_bus_routes if sumo_bus_routes else (bus_routes if bus_routes else None),
        depot_locations=depot_locations if depot_locations else None,
        rail_stations=rail_stations if rail_stations else None,
        rail_pressures=station_pressures if station_pressures else None,
    )
    if sumo_bus_routes:
        st.success(f"SUMO 仿真已运行 — {len(sumo_bus_routes)} 条公交轨迹 (紫色)")
    elif enable_sumo and dispatch_result:
        st.warning("SUMO 未产生轨迹，使用 OSMnx 路网路线 (蓝色) 作为替代。请展开底部「运行日志」查看详情。")
    elif dispatch_result:
        st.info("SUMO 未启用，显示 OSMnx 路网公交路线 (蓝色)")
    st.caption("橙色: 行人路径 | 紫色/蓝色: 公交轨迹 | 点击图例切换图层")
    render_metrics(paths, demand_gdf)

# Tab: Dispatch
if has_bus and t_dispatch:
    with t_dispatch:
        render_dispatch_results(
            dispatch_result, vehicles, depots,
            pedestrian_paths=paths,
            total_demand_people=int(demand_gdf["people_count"].sum()),
            n_rounds=3,
        )

# Tab: Rail cooperative
if has_rail and t_rail:
    with t_rail:
        st.subheader("协同分配结果 (5 种方式)")

        mode_labels = {
            "walk_shelter": "步行→避难点", "walk_rail": "步行→轨道站",
            "bus_rail": "公交→轨道站", "bus_shelter": "公交→避难点",
        }
        dest_types = allocation_result.destination_type
        mode_counts = {}
        for v in dest_types.values():
            mode_counts[v] = mode_counts.get(v, 0) + 1

        cols = st.columns(5)
        for i, (mode, label) in enumerate(mode_labels.items()):
            cols[i].metric(label, mode_counts.get(mode, 0))
        cols[4].metric("未分配", len(allocation_result.unassigned))

        # Round results
        if allocation_result.round_results:
            st.subheader("多轮调度追踪")
            rr_rows = []
            for rr in allocation_result.round_results:
                rr_rows.append({
                    "轮次": rr.round_id,
                    "本轮到": rr.served_people,
                    "剩余": rr.remaining_people,
                    "步行": rr.walk_assigned,
                    "轨道": rr.rail_assigned,
                    "避难点": rr.shelter_assigned,
                    "未疏散": rr.unserved,
                })
            st.dataframe(pd.DataFrame(rr_rows), use_container_width=True, hide_index=True)

        # Allocation detail table
        st.subheader("需求点分配明细")
        rows = []
        for i, (_, demand) in enumerate(demand_gdf.iterrows()):
            did = demand["demand_id"]
            dest = allocation_result.assignments.get(did, "—")
            dtype = allocation_result.destination_type.get(did, "—")
            ra = access["to_rail"][i] if i < len(access["to_rail"]) else {}
            sa = access["to_shelter"][i] if i < len(access["to_shelter"]) else {}
            rows.append({
                "需求点": demand.get("demand_name", did),
                "人数": int(demand["people_count"]),
                "方式": mode_labels.get(dtype, dtype),
                "目的地": dest[:30],
                "步行到轨道(min)": round(ra.get("walk_time_s", 0) / 60, 1),
                "步行到避难点(min)": round(sa.get("walk_time_s", 0) / 60, 1),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# Tab: Station pressure
if has_rail and t_pressure and station_pressures:
    with t_pressure:
        st.subheader("轨道站点压力评估")
        import plotly.express as px

        press_rows = []
        for p in station_pressures:
            color = {"normal": "#27ae60", "saturated": "#f39c12",
                     "overloaded": "#e67e22", "severe": "#e74c3c"}.get(p.level, "#95a5a6")
            press_rows.append({
                "站点": p.station_name,
                "到达人数": p.arrivals,
                "处理能力": p.capacity_used,
                "压力指数": p.pressure,
                "状态": p.level,
                "颜色": color,
            })
        df_press = pd.DataFrame(press_rows).sort_values("压力指数", ascending=False)

        # Horizontal bar chart
        fig = px.bar(
            df_press, x="压力指数", y="站点", orientation="h",
            color="状态",
            color_discrete_map={"normal": "#27ae60", "saturated": "#f39c12",
                                "overloaded": "#e67e22", "severe": "#e74c3c"},
            title="站点压力 (绿色正常/黄色饱和/橙色过载/红色严重)",
        )
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            df_press[["站点", "到达人数", "处理能力", "压力指数", "状态"]],
            use_container_width=True, hide_index=True,
        )

# Tab: Comparison
if has_rail and t_comparison:
    with t_comparison:
        st.subheader("方案对比: 公交直达 vs 混合协同")
        comparison.render_chart()
        comparison.render_table()

# Tab: SUMO
if has_sumo and t_sumo:
    with t_sumo:
        st.subheader("SUMO 动态仿真结果")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("车辆发出", sumo_result.vehicles_inserted)
        c2.metric("到达目的地", sumo_result.vehicles_arrived)
        c3.metric("平均行程时间", f"{sumo_result.avg_duration_s:.0f}s")
        c4.metric("平均速度", f"{sumo_result.avg_speed_ms * 3.6:.1f} km/h")

        if sumo_result.vehicle_logs:
            import pandas as pd
            df_log = pd.DataFrame(sumo_result.vehicle_logs)
            st.dataframe(
                df_log[["id", "depart", "arrival", "duration", "routeLength", "waitingTime"]],
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "上述为 SUMO 对每辆公交车的仿真结果。routeLength 为实际行驶距离(m), "
                "waitingTime 为拥堵等待时间(s)。"
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

# ── Log expander (at bottom of page) ──────────────────────────
if log_lines:
    with st.expander("运行日志", expanded=False):
        st.code("\n".join(log_lines), language=None)
