"""Core analysis pipeline: data loading → dispatch → SUMO → rail → evaluate.

Returns a results dict consumed by the tabs renderer.
"""
import os
import time
from datetime import datetime

import yaml
import geopandas as gpd
from shapely.geometry import Point

from src.app.config import (
    GRAPHML_PATH, DEMAND_PATH, SHELTERS_PATH,
    BUS_STOPS_PATH, RAIL_STATIONS_PATH,
    SUMO_NET_FULL, SUMO_NET_FALLBACK, SUMO_NET_CROPPED,
    SUMO_ROUTES_DIR, SUMO_OUTPUT_DIR, SUMO_CONFIGS_DIR,
    RUNS_DIR, CACHE_DIR, COST_MATRIX_CACHE,
    EVENT_CENTERS, XUZHOU_DEPOTS,
)
from src.app.outputs import save_run_artifacts
from src.network.event import create_event_from_yaml, get_affected_roads
from src.network.pathfinder import compute_evacuation_paths, _prepare_graph, _shortest_path_core
from src.demand.generator import summarize_demand
from src.demand.shelter import summarize_shelters
from src.app.data_loader import (
    load_road_network, load_demand, load_shelter_data, load_bus_stops, preprocess_demand,
)
from src.dispatch.vehicle import BusDepot, BusVehicle
from src.dispatch.cost_matrix import (
    compute_euclidean_matrix, compute_cost_matrix, MODE_EUCLIDEAN,
)
from src.dispatch.solver import solve_evacuation_dispatch
from src.rail.capacity import load_stations
from src.walking.access import compute_access_matrix, WALK_EUCLIDEAN, WALK_NETWORK
from src.rail.cooperative import allocate_cooperative
from src.evaluation.metrics import compute_evacuation_metrics
from src.evaluation.comparison import ComparisonResult


def _load_scenario(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_analysis(params: dict) -> dict:
    """Execute the full evacuation analysis pipeline.

    Args:
        params: Dict with all sidebar/user configuration. Key fields:
            scenario_path, event_location, radius_m, actual_demand,
            random_seed, enable_perturbation, enable_bus, bus_params,
            cost_matrix_mode, enable_sumo, enable_crop, enable_traci,
            enable_rail, walk_self_min, walk_rail_min, pressure_limit,
            walk_mode, cap_factor, enable_sensitivity,
            enable_snap, enable_water_filter,

    Returns:
        Dict with all results needed for display and saving.
    """
    log_lines: list[str] = []
    def _log(msg: str):
        log_lines.append(msg)

    # ── Unpack params ─────────────────────────────────────────
    scenario_path = params["scenario_path"]
    event_location = params["event_location"]
    radius_m = params["radius_m"]
    actual_demand = params["actual_demand"]
    random_seed = params["random_seed"]
    enable_perturbation = params.get("enable_perturbation", False)
    enable_bus = params.get("enable_bus", True)
    bus_params = params.get("bus_params")
    cost_matrix_mode = params.get("cost_matrix_mode", MODE_EUCLIDEAN)
    enable_sumo = params.get("enable_sumo", True)
    enable_crop = params.get("enable_crop", True)
    enable_traci = params.get("enable_traci", False)
    enable_rail = params.get("enable_rail", True)
    walk_self_min = params.get("walk_self_min", 20)
    walk_rail_min = params.get("walk_rail_min", 10)
    pressure_limit = params.get("pressure_limit", 1.1)
    walk_mode = params.get("walk_mode", WALK_EUCLIDEAN)
    cap_factor = params.get("cap_factor", 1.0)
    enable_sensitivity = params.get("enable_sensitivity", False)
    enable_snap = params.get("enable_snap", True)
    enable_water_filter = params.get("enable_water_filter", False)

    scenario = _load_scenario(scenario_path)
    selected_scenario_file = os.path.basename(scenario_path)
    _scenario_id = scenario.get("scenario_id", selected_scenario_file.replace(".yaml", ""))

    # ── Run ID & output directory ─────────────────────────────
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_scenario_id}_seed{random_seed}"
    run_output_dir = params.get("output_dir") or os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_output_dir, exist_ok=True)

    _log(f"[run] {run_id}")

    # ═══════════════════════════════════════════════════════════
    # DATA LOADING (delegated to data_loader)
    # ═══════════════════════════════════════════════════════════
    G, nodes_gdf, edges_gdf = load_road_network(GRAPHML_PATH)
    _log(f"路网加载完成：{G.number_of_nodes():,} 节点, {G.number_of_edges():,} 边")

    demand_gdf = load_demand(DEMAND_PATH)
    shelters_all = load_shelter_data(SHELTERS_PATH)
    bus_stops_gdf = load_bus_stops(BUS_STOPS_PATH)

    demand_gdf, demand_logs = preprocess_demand(
        demand_gdf, G, event_location, actual_demand,
        enable_snap=enable_snap, enable_water_filter=enable_water_filter,
        cache_dir=CACHE_DIR,
    )
    for msg in demand_logs:
        _log(msg)

    # Event & shelters
    event = create_event_from_yaml(scenario)
    affected = get_affected_roads(event, edges_gdf)
    danger_radius_deg = event.radius_m / 111320
    shelter_distances = shelters_all.geometry.apply(lambda g: g.distance(event.center))
    safe_mask = shelter_distances > danger_radius_deg
    shelters_gdf = shelters_all[safe_mask].copy()
    demand_summary = summarize_demand(demand_gdf)
    shelter_summary = summarize_shelters(shelters_gdf)

    _log(f"事件: 大客流 | 地点: {event_location} | 聚集半径: {event.radius_m}m")
    _log(f"需求点: {len(demand_gdf)} 个 ({demand_summary['total_people']:,} 人)")
    _log(f"安全区域: {len(shelters_gdf)}/{len(shelters_all)} 处")
    if bus_stops_gdf is not None:
        _log(f"公交站: {len(bus_stops_gdf)} 个")

    # ═══════════════════════════════════════════════════════════
    # PEDESTRIAN PATHS (baseline)
    # ═══════════════════════════════════════════════════════════
    t0 = time.perf_counter()
    paths = compute_evacuation_paths(G, demand_gdf, shelters_gdf, max_shelters_per_demand=1, parallel=False)
    path_elapsed = time.perf_counter() - t0
    n_valid = sum(1 for p in paths if p.node_path)
    _log(f"行人路径计算完成：{n_valid}/{len(paths)} 条，耗时 {path_elapsed:.1f}s")

    # ═══════════════════════════════════════════════════════════
    # BUS DISPATCH
    # ═══════════════════════════════════════════════════════════
    dispatch_result = None
    vehicles = []
    depots = []
    depot_locations = []
    board_gdf = None
    n_rounds_display = 1

    if enable_bus and bus_params:
        # Sort depots by distance to event center, pick closest 3
        cx, cy = EVENT_CENTERS[event_location]
        sorted_depots = sorted(XUZHOU_DEPOTS, key=lambda x: (x[1]-cx)**2 + (x[2]-cy)**2)
        depot_defs = [(f"depot_{i:02d}", name, lon, lat) for i, (name, lon, lat) in enumerate(sorted_depots[:3])]
        depots = [BusDepot(did, dname, dlon, dlat, 20, Point(dlon, dlat)) for did, dname, dlon, dlat in depot_defs]
        vehicles = []
        for i, depot in enumerate(depots):
            n_buses_per = bus_params["n_buses"] // len(depots)
            for j in range(n_buses_per):
                vehicles.append(BusVehicle(
                    vehicle_id=f"{depot.depot_id}_bus_{j:02d}",
                    depot_id=depot.depot_id, capacity=bus_params["bus_capacity"],
                ))
        _log(f"生成 {len(depots)} 个集结区, {len(vehicles)} 辆车 (总运力 {sum(v.capacity for v in vehicles):,})")
        depot_locations = [d.location for d in depots]

        # Board points
        if bus_stops_gdf is not None and len(bus_stops_gdf) > 0:
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
        all_points = depot_locations + board_pts
        if cost_matrix_mode == MODE_EUCLIDEAN:
            cost = compute_euclidean_matrix(all_points, all_points)
        else:
            cost = compute_cost_matrix(
                all_points, all_points, mode=cost_matrix_mode,
                G=G, cache_dir=COST_MATRIX_CACHE,
                scenario_id=_scenario_id, random_seed=random_seed,
            )

        board_gdf = demand_gdf.copy()
        board_gdf["geometry"] = board_pts

        dispatch_result = solve_evacuation_dispatch(
            depots=depots, vehicles=vehicles, demand_points=board_pts,
            demand_quantities=demand_quantities, cost_matrix=cost,
            time_limit_s=bus_params["time_limit"],
        )

        if dispatch_result.solver_status in ("optimal", "feasible"):
            n_used = sum(1 for r in dispatch_result.vehicle_routes.values() if len(r) > 1)
            sub_qty = dispatch_result.sub_demand_quantities
            total_assigned = sum(sub_qty[i] for i in range(len(sub_qty)) if i not in dispatch_result.unserved_demand)
            _log(f"调度: {dispatch_result.solver_status} ({dispatch_result.runtime_s:.1f}s) | "
                 f"用车: {n_used}/{len(vehicles)}辆 | 单趟接走: {total_assigned}人 / 总需求 {sum(sub_qty)}人")
        else:
            _log(f"调度求解失败: {dispatch_result.solver_status}")

    # ═══════════════════════════════════════════════════════════
    # SUMO SIMULATION
    # ═══════════════════════════════════════════════════════════
    sumo_result = None
    sumo_bus_routes = []
    sumo_net_actual = None

    if enable_sumo and dispatch_result and dispatch_result.solver_status in ("optimal", "feasible"):
        sumo_net = SUMO_NET_FULL if os.path.exists(SUMO_NET_FULL) else (SUMO_NET_FALLBACK if os.path.exists(SUMO_NET_FALLBACK) else None)
        if sumo_net is None:
            _log("SUMO 路网未生成，跳过仿真")
        else:
            sumo_net_actual = sumo_net
            if enable_crop:
                if not os.path.exists(SUMO_NET_CROPPED) or True:
                    from src.simulation.osm_to_sumo import crop_network
                    try:
                        sumo_net_actual = crop_network(
                            sumo_net, SUMO_NET_CROPPED,
                            EVENT_CENTERS[event_location][0],
                            EVENT_CENTERS[event_location][1], radius_m=8000,
                        )
                        import xml.etree.ElementTree as ET
                        ct = ET.parse(sumo_net_actual)
                        n_edges = len(ct.findall(".//edge"))
                        _log(f"SUMO 子网裁剪: {n_edges} 边 (8km半径)")
                    except Exception as e:
                        _log(f"裁剪失败, 使用全网: {e}")
                        sumo_net_actual = sumo_net

            from src.simulation.route_builder import dispatch_to_sumo_trips
            try:
                total_cap = sum(v.capacity for v in vehicles)
                total_demand_val = int(demand_gdf["people_count"].sum())
                n_rounds = max(1, min((total_demand_val // max(total_cap, 1)) + 1, 20))
                n_rounds_display = n_rounds
                _log(f"循环轮次: {n_rounds} (需求{total_demand_val:,} / 运力{total_cap:,})")
                trip_path, route_path = dispatch_to_sumo_trips(
                    dispatch_result, vehicles, depots, board_gdf,
                    SUMO_ROUTES_DIR, sumo_net_actual, max_rounds=n_rounds,
                )
                _log(f"SUMO 路径转换完成 ({n_rounds} 轮循环)")

                from src.simulation.sumo_runner import (
                    create_sumocfg, run_sumo_headless, run_sumo_with_traci,
                    find_edges_near_event, extract_vehicle_trajectories,
                )
                sumocfg_path = os.path.join(SUMO_CONFIGS_DIR, "simulation_v0.3.sumocfg")
                create_sumocfg(sumo_net_actual, route_path, sumocfg_path)

                if enable_traci:
                    closure_edges = find_edges_near_event(
                        sumo_net_actual, (event.center.x, event.center.y), event.radius_m * 1.2,
                    )
                    _log(f"TraCI 道路封闭: {len(closure_edges)} 条路段")
                    sumo_result = run_sumo_with_traci(
                        sumocfg_path, SUMO_OUTPUT_DIR,
                        road_closure_edges=closure_edges, closure_time_s=300,
                    )
                else:
                    sumo_result = run_sumo_headless(sumocfg_path, SUMO_OUTPUT_DIR)

                if sumo_result.success:
                    n_logs = len(sumo_result.vehicle_logs)
                    n_finished = sumo_result.vehicles_arrived
                    _log(f"SUMO 仿真: {n_finished}/{n_logs} 趟行程, "
                         f"平均 {sumo_result.avg_duration_s:.0f}s, "
                         f"{sumo_result.avg_speed_ms * 3.6:.1f} km/h")

                    vehroute_path = os.path.join(SUMO_OUTPUT_DIR, "vehroute.xml")
                    if os.path.exists(vehroute_path):
                        sumo_trajectories = extract_vehicle_trajectories(vehroute_path, sumo_net_actual)
                        merged: dict[str, dict] = {}
                        for t in sumo_trajectories:
                            vid = t.get("vehicle_id", "")
                            if vid not in merged:
                                merged[vid] = dict(vehicle_id=vid, coords=[], depart=t.get("depart", 0),
                                                   arrival=t.get("arrival", 0), edge_count=0, shape_pt_count=0, missing_edges=0)
                            merged[vid]["coords"].extend(t.get("coords", []))
                            merged[vid]["arrival"] = max(merged[vid]["arrival"], t.get("arrival", 0))
                            merged[vid]["edge_count"] += t.get("edge_count", 0)
                            merged[vid]["shape_pt_count"] += t.get("shape_pt_count", 0)
                            merged[vid]["missing_edges"] += t.get("missing_edges", 0)
                        sumo_bus_routes = [v for v in merged.values() if len(v["coords"]) >= 2]
                        total_edges = sum(v.get("edge_count", 0) for v in sumo_bus_routes)
                        total_pts = sum(v.get("shape_pt_count", 0) for v in sumo_bus_routes)
                        _log(f"SUMO轨迹: {len(sumo_bus_routes)}车, 均{total_edges//max(len(sumo_bus_routes),1)}边/车, {total_pts}shape点")
                else:
                    _log(f"SUMO 仿真异常: {sumo_result.error}")
            except Exception as e:
                _log(f"SUMO 流程失败: {e}")

    # ═══════════════════════════════════════════════════════════
    # RAIL COOPERATIVE ALLOCATION
    # ═══════════════════════════════════════════════════════════
    allocation_result = None
    rail_stations = None
    station_pressures = None
    evac_metrics = None
    comparison = ComparisonResult()
    walking_paths = []
    dp_list = []
    access = None
    sensitivity_results = []

    if enable_rail and os.path.exists(RAIL_STATIONS_PATH):
        rail_stations = load_stations(RAIL_STATIONS_PATH)
        rail_gdf = gpd.read_file(RAIL_STATIONS_PATH)
        _log(f"轨道站: {len(rail_stations)} 个 (1/2/3号线)")

        # Walking access
        walk_G = None
        if walk_mode == WALK_NETWORK:
            import osmnx as ox
            walk_G = ox.graph_from_place('徐州市, 江苏省, China', network_type='walk', simplify=True)
            _log(f"步行网络: {walk_G.number_of_nodes():,} 节点, {walk_G.number_of_edges():,} 边")
        access = compute_access_matrix(demand_gdf, rail_gdf, shelters_gdf,
                                       mode=walk_mode, walk_G=walk_G)
        avg_rail_walk = sum(a["walk_time_s"] for a in access["to_rail"]) / len(access["to_rail"])
        avg_shelter_walk = sum(a["walk_time_s"] for a in access["to_shelter"]) / len(access["to_shelter"])
        _log(f"步行接入: 轨道站 {avg_rail_walk/60:.1f}min, 步行离开 {avg_shelter_walk/60:.1f}min")

        # Build walking path lines for map
        for a in access["to_rail"]:
            walking_paths.append({
                "coords": [(a["origin_lon"], a["origin_lat"]), (a["target_lon"], a["target_lat"])],
            })

        # Build demand point list
        for i, (_, demand) in enumerate(demand_gdf.iterrows()):
            ra = access["to_rail"][i] if i < len(access["to_rail"]) else {}
            sa = access["to_shelter"][i] if i < len(access["to_shelter"]) else {}
            rail_dists = []
            for j, (_, station) in enumerate(rail_gdf.iterrows()):
                d = demand.geometry.distance(station.geometry) * 111320
                rail_dists.append((station["station_id"], d))
            rail_dists.sort(key=lambda x: x[1])
            rail_candidates = [rid for rid, _ in rail_dists[:3]]
            dp_list.append({
                "demand_id": demand["demand_id"], "people": int(demand["people_count"]),
                "walk_to_shelter_s": sa.get("walk_time_s", 9999),
                "walk_to_rail_s": ra.get("walk_time_s", 9999),
                "nearest_rail_id": ra.get("target_id", ""),
                "rail_candidates": rail_candidates,
            })
        bus_cap = (bus_params["n_buses"] * bus_params["bus_capacity"]) if bus_params else 1500

        # Scheme C: hybrid cooperative
        allocation_result = allocate_cooperative(
            dp_list, rail_stations,
            walk_self_max_s=walk_self_min * 60,
            walk_rail_max_s=walk_rail_min * 60,
            pressure_limit=pressure_limit,
            bus_capacity_per_round=bus_cap, max_rounds=3,
            capacity_factor=cap_factor,
        )
        station_pressures = allocation_result.station_pressures

        mode_counts = {}
        for v in allocation_result.destination_type.values():
            mode_counts[v] = mode_counts.get(v, 0) + 1
        _log(f"协同: 步行离开{mode_counts.get('walk_self',0)} 步行轨道{mode_counts.get('walk_rail',0)} "
             f"公交轨道{mode_counts.get('bus_rail',0)} 公交外围{mode_counts.get('bus_periphery',0)} "
             f"未分配{len(allocation_result.unassigned)} (容量因子×{cap_factor})")

        evac_metrics = compute_evacuation_metrics(
            demand_gdf, allocation_result,
            station_pressures=station_pressures, walking_access=access,
        )
        comparison.add("方案C: 混合协同", evac_metrics)

        # Scheme A: bus only
        baseline_result = allocate_cooperative(
            dp_list, [], walk_self_max_s=walk_self_min * 60,
            bus_capacity_per_round=bus_cap, max_rounds=3,
        )
        baseline_metrics = compute_evacuation_metrics(demand_gdf, baseline_result, walking_access=access)
        comparison.add("方案A: 纯公交", baseline_metrics)

        # Scheme B: rail priority
        rail_priority_result = allocate_cooperative(
            dp_list, rail_stations,
            walk_self_max_s=walk_self_min * 60,
            walk_rail_max_s=walk_rail_min * 60,
            pressure_limit=pressure_limit * 0.8,
            bus_capacity_per_round=bus_cap, max_rounds=3,
            capacity_factor=cap_factor,
        )
        rail_priority_metrics = compute_evacuation_metrics(demand_gdf, rail_priority_result, walking_access=access)
        comparison.add("方案B: 轨道优先", rail_priority_metrics)

        # Sensitivity analysis
        if enable_sensitivity:
            cap_factors = [("保守 ×0.7", 0.7), ("基准 ×1.0", 1.0), ("乐观 ×1.2", 1.2)]
            for label, cf in cap_factors:
                sens_result = allocate_cooperative(
                    dp_list, rail_stations,
                    walk_self_max_s=walk_self_min * 60,
                    walk_rail_max_s=walk_rail_min * 60,
                    pressure_limit=pressure_limit,
                    bus_capacity_per_round=bus_cap, max_rounds=3,
                    capacity_factor=cf,
                )
                sens_metrics = compute_evacuation_metrics(
                    demand_gdf, sens_result,
                    station_pressures=sens_result.station_pressures,
                    walking_access=access,
                )
                sensitivity_results.append({
                    "情景": label, "疏散完成率": sens_metrics.completion_rate,
                    "轨道分担率": sens_metrics.rail_share,
                    "未服务人数": sens_metrics.unserved,
                    "过载站数": sens_metrics.overloaded_stations,
                })
            _log(f"容量敏感性: 3组完成")

        _log(f"轨道协同完成: {len(allocation_result.assignments)}个需求点已分配")

    # ═══════════════════════════════════════════════════════════
    # BUS ROUTE PATHS (for map when no SUMO)
    # ═══════════════════════════════════════════════════════════
    bus_routes = []
    if dispatch_result and vehicles and dispatch_result.solver_status in ("optimal", "feasible") and not sumo_bus_routes:
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

    # ═══════════════════════════════════════════════════════════
    # SAVE OUTPUT ARTIFACTS
    # ═══════════════════════════════════════════════════════════
    # Build run config for output
    run_config = {
        "run_id": run_id, "scenario_file": selected_scenario_file,
        "scenario_id": _scenario_id, "event_location": event_location,
        "event_radius_m": radius_m, "demand_scale": actual_demand,
        "random_seed": random_seed, "enable_perturbation": enable_perturbation,
        "cost_matrix_mode": cost_matrix_mode,
        "enable_bus": enable_bus, "enable_sumo": enable_sumo,
        "enable_rail": enable_rail, "enable_crop": enable_crop,
        "enable_traci": enable_traci,
        "enable_snap_to_road": enable_snap,
        "enable_water_filter": enable_water_filter,
        "timestamp": datetime.now().isoformat(),
    }
    if enable_rail:
        run_config["rail"] = {"walk_self_min": walk_self_min, "walk_rail_min": walk_rail_min,
                              "pressure_limit": pressure_limit, "walk_mode": walk_mode,
                              "capacity_factor": cap_factor, "enable_sensitivity": enable_sensitivity}
    if enable_bus and bus_params:
        run_config["bus"] = bus_params

    try:
        save_msg = save_run_artifacts(
            run_output_dir, run_id, scenario_path, random_seed,
            selected_scenario_file, _scenario_id,
            comparison, station_pressures, dispatch_result,
            enable_bus, enable_rail, enable_sumo, enable_perturbation,
            event_location, radius_m, actual_demand,
            GRAPHML_PATH, DEMAND_PATH, SHELTERS_PATH, RAIL_STATIONS_PATH,
            run_config=run_config,
        )
        _log(f"[output] {save_msg}")
    except Exception as e:
        _log(f"[output] 保存输出失败: {e}")

    # ═══════════════════════════════════════════════════════════
    # RETURN RESULTS
    # ═══════════════════════════════════════════════════════════
    return {
        "log_lines": log_lines, "run_id": run_id, "run_output_dir": run_output_dir,
        "G": G, "nodes_gdf": nodes_gdf, "edges_gdf": edges_gdf,
        "demand_gdf": demand_gdf, "shelters_gdf": shelters_gdf, "shelters_all": shelters_all,
        "demand_summary": demand_summary, "shelter_summary": shelter_summary,
        "event": event, "affected": affected, "paths": paths, "path_elapsed": path_elapsed,
        "dispatch_result": dispatch_result, "vehicles": vehicles, "depots": depots,
        "depot_locations": depot_locations, "bus_routes": bus_routes,
        "sumo_result": sumo_result, "sumo_bus_routes": sumo_bus_routes,
        "sumo_net_actual": sumo_net_actual, "n_rounds_display": n_rounds_display,
        "allocation_result": allocation_result, "rail_stations": rail_stations,
        "station_pressures": station_pressures, "evac_metrics": evac_metrics,
        "comparison": comparison, "walking_paths": walking_paths,
        "dp_list": dp_list, "access": access,
        "bus_stops_gdf": bus_stops_gdf,
        "enable_bus": enable_bus, "enable_sumo": enable_sumo,
        "enable_rail": enable_rail, "enable_crop": enable_crop,
        "enable_traci": enable_traci,
        "enable_sensitivity": enable_sensitivity,
        "sensitivity_results": sensitivity_results,
        "bus_params": bus_params,
        "actual_demand": actual_demand, "radius_m": radius_m,
        "event_location": event_location,
    }
