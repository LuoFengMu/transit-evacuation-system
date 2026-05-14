"""
v0.5.0 — 公交-轨道协同疏散仿真系统
公交-轨道-步行多方式协同疏散 — 可复现实验平台
"""
import os
import random
import yaml
import streamlit as st
import pandas as pd

from src.app.config import (
    VERSION, SCENARIOS_DIR,
    SUMO_NET_FULL, SUMO_NET_FALLBACK,
    EVENT_CENTERS, EVENT_LOCATION_OPTIONS,
    DEMAND_SCALE_OPTIONS, DEFAULT_DEMAND_SCALE,
    COST_MODE_LABELS, CAPACITY_FACTOR_LABELS,
    list_scenario_files,
)
from src.app.pipeline import run_analysis
from src.dispatch.cost_matrix import MODE_EUCLIDEAN
from src.walking.access import WALK_EUCLIDEAN, WALK_NETWORK
from src.visualization.map_view import render_map, render_path_table, render_metrics
from src.visualization.dispatch_view import render_dispatch_results, render_dispatch_params
from src.evaluation.report import render_report
from src.evaluation.audit import render_audit_tab

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="公交-轨道协同疏散仿真系统",
    page_icon="🚌",
    layout="wide",
)


# ── Sidebar — scenario selection ──────────────────────────────
st.sidebar.title("公交-轨道协同疏散仿真")
st.sidebar.caption(f"v{VERSION} — 可复现实验平台")

scenario_files = list_scenario_files()
default_idx = next((i for i, f in enumerate(scenario_files) if "C_hybrid" in f), 0)
selected_scenario_file = st.sidebar.selectbox(
    "场景配置", options=scenario_files, index=default_idx,
    format_func=lambda f: f.replace(".yaml", "").replace("_", " "),
    help="选择预设场景配置，侧边栏参数将以此为默认值",
)
scenario_path = os.path.join(SCENARIOS_DIR, selected_scenario_file)
with open(scenario_path, "r", encoding="utf-8") as f:
    scenario = yaml.safe_load(f)

# ── Sidebar — event ───────────────────────────────────────────
st.sidebar.header("大客流突发事件")
event_type = "crowd"

yaml_loc = scenario.get("event", {}).get("location", "彭城广场")
event_loc_default = next((i for i, o in enumerate(EVENT_LOCATION_OPTIONS) if o == yaml_loc), 0)
event_location = st.sidebar.selectbox("事件地点", options=EVENT_LOCATION_OPTIONS, index=event_loc_default)
scenario["event"]["center"] = list(EVENT_CENTERS[event_location])
scenario["event"]["type"] = event_type

yaml_radius = scenario.get("event", {}).get("radius_m", 1500)
radius_m = st.sidebar.slider("人群聚集半径 (m)", 200, 5000, yaml_radius, step=100)

# ── Sidebar — demand ──────────────────────────────────────────
yaml_demand = scenario.get("demand", {})
yaml_scale = yaml_demand.get("scale", DEFAULT_DEMAND_SCALE)
demand_scale_idx = next((i for i, v in enumerate(DEMAND_SCALE_OPTIONS) if v == yaml_scale), 0)
demand_scale = st.sidebar.selectbox(
    "疏散人数量级", options=DEMAND_SCALE_OPTIONS, index=demand_scale_idx,
    format_func=lambda x: f"约{x//10000}万人" if x >= 10000 else f"{x}人",
)

yaml_seed = scenario.get("simulation", {}).get("random_seed", 42)
yaml_perturb = yaml_demand.get("random_variation", False)
random_seed = st.sidebar.number_input("随机种子", value=yaml_seed, min_value=0, max_value=99999, step=1)
enable_perturbation = st.sidebar.checkbox("需求随机扰动 (±15%)", value=yaml_perturb)
random.seed(random_seed)
actual_demand = int(demand_scale * random.uniform(0.85, 1.15)) if enable_perturbation else demand_scale

# ── Sidebar — bus dispatch ────────────────────────────────────
yaml_bus = scenario.get("bus_enabled", True)
enable_bus = st.sidebar.checkbox("启用公交调度", value=yaml_bus)
yaml_cost = scenario.get("simulation", {}).get("cost_matrix_mode", MODE_EUCLIDEAN)
cost_mode_options = list(COST_MODE_LABELS.keys())
cost_idx = cost_mode_options.index(yaml_cost) if yaml_cost in cost_mode_options else 0
if enable_bus:
    bus_params = render_dispatch_params(scenario.get("bus"))
    cost_matrix_mode = st.sidebar.selectbox(
        "成本矩阵模式", options=cost_mode_options, index=cost_idx,
        format_func=lambda m: COST_MODE_LABELS[m],
    )
else:
    bus_params = None
    cost_matrix_mode = MODE_EUCLIDEAN

# ── Sidebar — SUMO ────────────────────────────────────────────
sumo_cfg = scenario.get("sumo", {}) or {}
yaml_sumo = sumo_cfg.get("enabled", True) if isinstance(sumo_cfg, dict) else True
enable_sumo = st.sidebar.checkbox("启用 SUMO 仿真", value=yaml_sumo)
yaml_crop = sumo_cfg.get("crop_network", True) if isinstance(sumo_cfg, dict) else True
enable_crop = st.sidebar.checkbox("SUMO 子网裁剪", value=yaml_crop)
yaml_traci = sumo_cfg.get("traci_closure", False) if isinstance(sumo_cfg, dict) else False
enable_traci = st.sidebar.checkbox("TraCI 道路封闭", value=yaml_traci)

# ── Sidebar — rail ────────────────────────────────────────────
yaml_rail = scenario.get("rail_enabled", True)
enable_rail = st.sidebar.checkbox("启用轨道协同", value=yaml_rail)
if enable_rail:
    with st.sidebar.expander("轨道协同参数"):
        rail_cfg = scenario.get("rail", {}) or {}
        walk_self_min = st.slider("步行自行离开上限(min)", 5, 30, rail_cfg.get("walk_self_min", 20), 1)
        walk_rail_min = st.slider("步行到轨道站上限(min)", 3, 20, rail_cfg.get("walk_rail_min", 10), 1)
        pressure_limit = st.slider("轨道站压力上限", 0.5, 2.0, rail_cfg.get("pressure_limit", 1.1), 0.1)

        yaml_walk_mode = rail_cfg.get("walk_mode", WALK_EUCLIDEAN)
        walk_mode_idx = 1 if yaml_walk_mode == WALK_NETWORK else 0
        walk_mode = st.radio("步行距离模式", options=[WALK_EUCLIDEAN, WALK_NETWORK],
                             index=walk_mode_idx,
                             format_func=lambda m: "欧氏距离 (快速)" if m == WALK_EUCLIDEAN else "步行网络 (精确)")

        cap_factor = st.select_slider("轨道容量情景", options=[0.7, 1.0, 1.2],
                                      value=rail_cfg.get("capacity_factor", 1.0),
                                      format_func=lambda v: CAPACITY_FACTOR_LABELS[v])
        enable_sensitivity = st.checkbox("容量敏感性分析", value=False)
else:
    walk_self_min, walk_rail_min, pressure_limit = 20, 10, 1.1
    walk_mode, cap_factor, enable_sensitivity = WALK_EUCLIDEAN, 1.0, False

# ── Sidebar — demand preprocessing ────────────────────────────
with st.sidebar.expander("需求点预处理"):
    enable_snap = st.checkbox("吸附到道路节点", value=yaml_demand.get("snap_to_road", True))
    enable_water_filter = st.checkbox("水体过滤", value=yaml_demand.get("filter_water", False))

# ── Sidebar — run button ──────────────────────────────────────
run_btn = st.sidebar.button("运行分析", type="primary", use_container_width=True)


# ── Main content ──────────────────────────────────────────────
st.title("公交-轨道协同疏散仿真系统")
st.caption(f"v{VERSION} — 公交-轨道-步行协同疏散 · 可复现实验平台")

if not run_btn:
    st.info("请在侧边栏配置场景参数，然后点击「运行分析」")
    st.markdown(f"**地点**: {event_location}　|　**事件**: 大客流，聚集半径 {radius_m}m　|　"
                f"**需求**: 约{actual_demand//10000}万人　|　**公交**: {'启用' if enable_bus else '关闭'}")
    if enable_bus and bus_params:
        st.markdown(f"**运力**: {bus_params['n_buses']}辆 × {bus_params['bus_capacity']}人 = "
                    f"{bus_params['n_buses'] * bus_params['bus_capacity']}人")
    st.stop()


# ═══════════════════════════════════════════════════════════════
# RUN PIPELINE
# ═══════════════════════════════════════════════════════════════
params = {
    "scenario_path": scenario_path, "event_location": event_location,
    "radius_m": radius_m, "actual_demand": actual_demand,
    "random_seed": random_seed, "enable_perturbation": enable_perturbation,
    "enable_bus": enable_bus, "bus_params": bus_params,
    "cost_matrix_mode": cost_matrix_mode,
    "enable_sumo": enable_sumo, "enable_crop": enable_crop,
    "enable_traci": enable_traci,
    "enable_rail": enable_rail,
    "walk_self_min": walk_self_min, "walk_rail_min": walk_rail_min,
    "pressure_limit": pressure_limit,
    "walk_mode": walk_mode, "cap_factor": cap_factor,
    "enable_sensitivity": enable_sensitivity,
    "enable_snap": enable_snap, "enable_water_filter": enable_water_filter,
}

with st.spinner("运行分析..."):
    R = run_analysis(params)

log_lines = R["log_lines"]


# ═══════════════════════════════════════════════════════════════
# DISPLAY RESULTS
# ═══════════════════════════════════════════════════════════════
has_bus = bool(R["dispatch_result"] and R["dispatch_result"].solver_status in ("optimal", "feasible"))
has_sumo = bool(R["sumo_result"] and R["sumo_result"].success)
has_rail = bool(R["allocation_result"] is not None)
has_data = has_bus or has_rail

tab_names = ["地图"]
if has_bus:
    tab_names.append("公交调度")
if has_rail:
    tab_names += ["轨道协同", "站点压力", "方案对比"]
if has_sumo:
    tab_names.append("SUMO 仿真")
tab_names += ["路径详情", "路径审计", "总结报告", "数据概览"]

all_tabs = st.tabs(tab_names)
t_map = all_tabs[0]; ti = 1
t_dispatch = all_tabs[ti] if has_bus else None; ti += 1 if has_bus else 0
t_rail = all_tabs[ti] if has_rail else None; ti += 1 if has_rail else 0
t_pressure = all_tabs[ti] if has_rail else None; ti += 1 if has_rail else 0
t_comparison = all_tabs[ti] if has_rail else None; ti += 1 if has_rail else 0
t_sumo = all_tabs[ti] if has_sumo else None; ti += 1 if has_sumo else 0
t_paths = all_tabs[ti]; ti += 1
t_audit = all_tabs[ti]; ti += 1
t_report = all_tabs[ti]; ti += 1
t_data = all_tabs[ti]

# ── Tab: 地图 ─────────────────────────────────
with t_map:
    st.subheader("疏散仿真地图")
    render_map(
        edges_gdf=R["edges_gdf"], event=R["event"],
        demand_gdf=R["demand_gdf"], shelters_gdf=R["shelters_gdf"],
        paths=R["paths"], affected_roads=R["affected"],
        bus_routes=R["sumo_bus_routes"] if R["sumo_bus_routes"] else (R["bus_routes"] if R["bus_routes"] else None),
        depot_locations=R["depot_locations"] if R["depot_locations"] else None,
        rail_stations=R["rail_stations"] if R["rail_stations"] else None,
        rail_pressures=R["station_pressures"] if R["station_pressures"] else None,
        show_shelters=False,
        walking_paths=R["walking_paths"] if R["walking_paths"] else None,
    )
    if R["dispatch_result"]:
        dispatched = sum(1 for r in R["dispatch_result"].vehicle_routes.values() if len(r) > 1)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("配置车辆", len(R["vehicles"]))
        c2.metric("调度派出", dispatched)
        c3.metric("SUMO轨迹", len(R["sumo_bus_routes"]))
        c4.metric("未派出", len(R["vehicles"]) - dispatched)
    render_metrics(R["paths"], R["demand_gdf"])

# ── Tab: 公交调度 ──────────────────────────────
if has_bus and t_dispatch:
    with t_dispatch:
        render_dispatch_results(
            R["dispatch_result"], R["vehicles"], R["depots"],
            pedestrian_paths=R["paths"],
            total_demand_people=int(R["demand_gdf"]["people_count"].sum()),
            n_rounds=R["n_rounds_display"],
        )

# ── Tab: 轨道协同 ──────────────────────────────
if has_rail and t_rail and (ar := R["allocation_result"]):
    with t_rail:
        st.subheader("协同分配结果 (5 种方式)")
        mode_labels = {"walk_self": "步行自行离开", "walk_rail": "步行→轨道站",
                       "bus_rail": "公交→轨道站", "bus_periphery": "公交→外围疏散"}
        mode_counts = {}
        for v in ar.destination_type.values():
            mode_counts[v] = mode_counts.get(v, 0) + 1
        cols = st.columns(5)
        for i, (mode, label) in enumerate(mode_labels.items()):
            cols[i].metric(label, mode_counts.get(mode, 0))
        cols[4].metric("未分配", len(ar.unassigned))

        # Mode share chart
        import plotly.express as px
        mode_people = {}
        for did, dtype in ar.destination_type.items():
            for dp in R["dp_list"]:
                if dp["demand_id"] == did:
                    mode_people[dtype] = mode_people.get(dtype, 0) + dp["people"]
                    break
        if mode_people:
            share_rows = [{"方式": mode_labels.get(m, m), "人数": p} for m, p in mode_people.items()]
            df_share = pd.DataFrame(share_rows)
            fig = px.bar(df_share, x="方式", y="人数", color="方式", text="人数",
                         title="疏散方式分担",
                         color_discrete_sequence=["#27ae60", "#3498db", "#9b59b6", "#e67e22", "#e74c3c"])
            fig.update_traces(texttemplate="%{text:,}", textposition="outside")
            fig.update_layout(height=300, showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True, key="mode_share_chart")

        # Round results table
        if ar.round_results:
            st.subheader("多轮调度追踪")
            rr_rows = [{"轮次": rr.round_id, "本轮到": rr.served_people, "剩余": rr.remaining_people,
                        "步行离开": rr.walk_self, "轨道": rr.rail_assigned,
                        "外围公交": rr.bus_periphery, "未疏散": rr.unserved} for rr in ar.round_results]
            st.dataframe(pd.DataFrame(rr_rows), use_container_width=True, hide_index=True)

        # Demand detail table
        st.subheader("需求点分配明细")
        rows = []
        for i, (_, demand) in enumerate(R["demand_gdf"].iterrows()):
            did = demand["demand_id"]
            dtype = ar.destination_type.get(did, "—")
            ra = R["access"]["to_rail"][i] if R["access"] and i < len(R["access"]["to_rail"]) else {}
            sa = R["access"]["to_shelter"][i] if R["access"] and i < len(R["access"]["to_shelter"]) else {}
            rows.append({
                "需求点": demand.get("demand_name", did), "人数": int(demand["people_count"]),
                "方式": mode_labels.get(dtype, dtype), "目的地": ar.assignments.get(did, "—")[:30],
                "步行到轨道(min)": round(ra.get("walk_time_s", 0) / 60, 1),
                "步行到避难点(min)": round(sa.get("walk_time_s", 0) / 60, 1),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Tab: 站点压力 ──────────────────────────────
if has_rail and t_pressure and R["station_pressures"]:
    with t_pressure:
        st.subheader("轨道站点压力评估")
        import plotly.express as px
        press_rows = []
        for p in R["station_pressures"]:
            if p.arrivals > 0:
                press_rows.append({
                    "站点": p.station_name, "到达人数": p.arrivals,
                    "处理能力": p.capacity_used, "压力指数": p.pressure,
                    "状态": p.level,
                })
        if press_rows:
            df_press = pd.DataFrame(press_rows).sort_values("压力指数", ascending=False)
            fig = px.bar(df_press, x="压力指数", y="站点", orientation="h", color="状态",
                         color_discrete_map={"normal": "#27ae60", "saturated": "#f39c12",
                                            "overloaded": "#e67e22", "severe": "#e74c3c"},
                         title="站点压力")
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_press[["站点", "到达人数", "处理能力", "压力指数", "状态"]],
                        use_container_width=True, hide_index=True)

# ── Sensitivity table (before comparison) ─────
if enable_sensitivity and R["sensitivity_results"]:
    st.subheader("容量敏感性分析")
    st.caption("三组容量假设下方案C（混合协同）的指标对比")
    df_sens = pd.DataFrame(R["sensitivity_results"])
    styled = df_sens.style.format({"疏散完成率": "{:.1%}", "轨道分担率": "{:.1%}",
                                    "未服务人数": "{:,}", "过载站数": "{:.0f}"})
    st.dataframe(styled, use_container_width=True, hide_index=True)

# ── Tab: 方案对比 ──────────────────────────────
if has_rail and t_comparison:
    with t_comparison:
        st.subheader("方案对比: 公交直达 vs 混合协同")
        R["comparison"].render_chart(key="tab")
        R["comparison"].render_table()

# ── Tab: SUMO 仿真 ─────────────────────────────
if has_sumo and t_sumo:
    with t_sumo:
        st.subheader("SUMO 动态仿真结果")
        sr = R["sumo_result"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("车辆发出", sr.vehicles_inserted)
        c2.metric("到达目的地", sr.vehicles_arrived)
        c3.metric("平均行程时间", f"{sr.avg_duration_s:.0f}s")
        c4.metric("平均速度", f"{sr.avg_speed_ms * 3.6:.1f} km/h")
        if sr.vehicle_logs:
            audit_rows = [{
                "车辆": v.get("id", "").split("_leg")[0], "出发(s)": v.get("depart", 0),
                "到达(s)": v.get("arrival", 0), "行程(s)": v.get("duration", 0),
                "距离(m)": v.get("routeLength", 0), "等待(s)": v.get("waitingTime", 0),
                "状态": "到达" if not v.get("vaporized") else "未到达",
            } for v in sr.vehicle_logs]
            st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)

# ── Tab: 路径详情 ──────────────────────────────
with t_paths:
    st.subheader("行人疏散路径")
    render_path_table(R["paths"])

# ── Tab: 路径审计 ──────────────────────────────
with t_audit:
    sumo_net = R.get("sumo_net_actual") or (SUMO_NET_FULL if os.path.exists(SUMO_NET_FULL) else SUMO_NET_FALLBACK)
    render_audit_tab(sumo_net, R["sumo_bus_routes"])

# ── Tab: 总结报告 ──────────────────────────────
with t_report:
    if has_data:
        render_report(
            demand_summary=R["demand_summary"], shelter_summary=R["shelter_summary"],
            dispatch_result=R["dispatch_result"], allocation_result=R["allocation_result"],
            station_pressures=R["station_pressures"], comparison=R["comparison"],
            sumo_result=R["sumo_result"],
            path_time_s=R["path_elapsed"],
            dispatch_time_s=R["dispatch_result"].runtime_s if R["dispatch_result"] else 0,
            sumo_time_s=R["sumo_result"].avg_duration_s if (R["sumo_result"] and R["sumo_result"].success) else None,
            bus_params=bus_params, demand_scale=actual_demand,
            event_type=event_type, radius_m=radius_m,
        )
    else:
        st.info("请先运行分析")

# ── Tab: 数据概览 ──────────────────────────────
with t_data:
    st.subheader("数据概览")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**疏散需求**")
        st.json(R["demand_summary"])
    with c2:
        st.write("**避难点**")
        st.json(R["shelter_summary"])
    if enable_bus and R["vehicles"]:
        st.write("**公交运力**")
        st.json({"集结区": len(R["depots"]), "公交车数": len(R["vehicles"]),
                 "总运力": sum(v.capacity for v in R["vehicles"])})

# ── Log expander ───────────────────────────────────────────────
if log_lines:
    with st.expander("运行日志", expanded=False):
        st.code("\n".join(log_lines), language=None)
