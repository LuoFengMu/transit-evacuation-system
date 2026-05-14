"""Streamlit visualization for dispatch results."""
import io
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.dispatch.vehicle import BusVehicle, BusDepot
from src.dispatch.solver import DispatchResult


def render_dispatch_results(
    dispatch_result: DispatchResult,
    vehicles: list[BusVehicle],
    depots: list[BusDepot],
    pedestrian_paths: list = None,
    total_demand_people: int = 0,
    n_rounds: int = 1,
) -> None:
    """Display dispatch optimization results."""
    if dispatch_result.solver_status.startswith("error"):
        st.error(f"调度求解失败: {dispatch_result.solver_status}")
        return
    if dispatch_result.solver_status == "no_solution":
        st.warning("当前约束下无可行调度方案，请增加车辆或减少需求")
        return

    # ── Summary metrics ────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("求解状态", dispatch_result.solver_status)
    col2.metric("求解耗时", f"{dispatch_result.runtime_s:.1f}s")
    col3.metric("目标函数值", f"{dispatch_result.objective_value:.0f}")
    col4.metric("未服务需求点", len(dispatch_result.unserved_demand))

    # ── Vehicle routes ─────────────────────────────────────
    st.subheader("车辆调度方案")
    vehicle_map = {v.vehicle_id: v for v in vehicles}
    rows = []
    for vid, route in dispatch_result.vehicle_routes.items():
        v = vehicle_map.get(vid)
        stop_labels = [
            s[1] if isinstance(s[1], str) else f"需求点_{s[1]:02d}"
            for s in route
        ]
        rows.append({
            "车辆": vid,
            "容量": v.capacity if v else "?",
            "停靠点": " → ".join(stop_labels),
            "停靠数": len([s for s in route if s[0] == "pickup"]),
        })
    if rows:
        df_routes = pd.DataFrame(rows)
        st.dataframe(df_routes, use_container_width=True, hide_index=True)

        # CSV download
        csv = df_routes.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="导出车辆路径 CSV",
            data=csv,
            file_name="vehicle_routes_v0.2.csv",
            mime="text/csv",
        )

    # ── Vehicle utilization ────────────────────────────────
    if dispatch_result.vehicle_routes:
        st.subheader("车辆装载与利用率")
        sub_qty = dispatch_result.sub_demand_quantities

        util_rows = []
        for vid, route in dispatch_result.vehicle_routes.items():
            v = vehicle_map.get(vid)
            if v is None:
                continue
            assigned = 0
            for stop_type, stop_id, _ in route:
                if stop_type == "pickup" and isinstance(stop_id, int) and stop_id < len(sub_qty):
                    assigned += sub_qty[stop_id]
            util_rows.append({
                "车辆": vid.replace("depot_0", "D").replace("_bus_", "🚌"),
                "容量": v.capacity,
                "分配人数": assigned,
                "利用率": assigned / v.capacity if v.capacity > 0 else 0,
                "剩余": v.capacity - assigned,
                "趟次": n_rounds,
                "集结区": vid.split("_")[0] + "_" + vid.split("_")[1] if "_" in vid else "",
            })

        if util_rows:
            df_util = pd.DataFrame(util_rows)
            n_total = len(df_util)
            page_size = 10
            total_pages = max(1, (n_total + page_size - 1) // page_size)

            show_n = min(15, n_total)
            df_page = df_util.head(show_n)

            # Horizontal stacked bar
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=df_page["车辆"], x=df_page["分配人数"],
                name="分配人数", orientation="h",
                marker=dict(color=df_page["利用率"].apply(
                    lambda r: "#27ae60" if r > 0.7 else ("#f39c12" if r > 0.3 else "#e74c3c")
                )),
                text=df_page.apply(lambda r: f"{r['分配人数']}人 ({r['利用率']:.0%})", axis=1),
                textposition="inside", insidetextanchor="middle",
                hovertemplate="%{y}<br>分配: %{x}人<br>利用率: %{customdata:.0%}<extra></extra>",
                customdata=df_page["利用率"],
            ))
            fig.add_trace(go.Bar(
                y=df_page["车辆"], x=df_page["剩余"],
                name="空余", orientation="h",
                marker=dict(color="#ecf0f1", line=dict(color="#bdc3c7", width=1)),
                hovertemplate="%{y}<br>空余: %{x}人<extra></extra>",
            ))
            fig.update_layout(
                barmode="stack",
                height=max(180, 35 * len(df_page) + 30),
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                xaxis_title="人数",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Summary
            used_buses = [r for r in util_rows if r["分配人数"] > 0]
            avg_util = sum(r["利用率"] for r in used_buses) / len(used_buses) if used_buses else 0
            idle = n_total - len(used_buses)
            total_cap = sum(r["容量"] for r in util_rows)
            total_assigned = sum(r["分配人数"] for r in util_rows)
            if n_total > show_n:
                st.caption(f"显示前 {show_n} 辆，共 {n_total} 辆")
            st.caption(
                f"{len(used_buses)}/{n_total} 辆车被分配任务"
                + (f", {idle} 辆闲置" if idle > 0 else "")
                + f" | 平均利用率 {avg_util:.0%}"
                + f" | 单趟运力 {total_cap:,} → 分配 {total_assigned:,}人"
                + (f" | 循环 {n_rounds} 趟" if n_rounds > 1 else "")
            )

    # ── Multi-trip estimation ────────────────────────────────
    if dispatch_result.vehicle_routes and dispatch_result.sub_demand_quantities:
        st.subheader("循环摆渡估算")
        sub_qty = dispatch_result.sub_demand_quantities
        total_served = sum(
            sub_qty[i] for i in range(len(sub_qty))
            if i not in dispatch_result.unserved_demand
        )
        total_demand = sum(sub_qty)
        n_buses_used = sum(1 for r in dispatch_result.vehicle_routes.values() if len(r) > 1)

        if n_buses_used > 0 and total_served > 0:
            avg_trip_s = dispatch_result.objective_value / max(n_buses_used, 1)
            trips_needed = total_demand / total_served
            total_time_min = (trips_needed * avg_trip_s) / 60

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("单趟服务人数", f"{total_served:,}")
            c2.metric("待疏散总人数", f"{total_demand:,}")
            c3.metric("需要循环轮次", f"{trips_needed:.1f}")
            c4.metric("估计总耗时", f"{total_time_min:.0f} min")
            st.caption(
                "公交车完成第一趟后返回集结区继续下一趟，直到所有人疏散完毕。"
                "精确时间仿真将在 v0.3.0 (SUMO) 中完成。"
            )

    # ── Comparison: pedestrian vs bus ─────────────────────────
    if pedestrian_paths is not None and total_demand_people > 0:
        st.subheader("方案对比：行人步行 vs 公交调度")

        valid_paths = [p for p in pedestrian_paths if p.node_path]
        avg_dist_m = sum(p.distance_m for p in valid_paths) / len(valid_paths) if valid_paths else 0
        avg_time_min = sum(p.travel_time_s for p in valid_paths) / len(valid_paths) / 60 if valid_paths else 0
        slowest_min = max((p.travel_time_s for p in valid_paths), default=0) / 60

        # Bus metrics
        sub_qty = dispatch_result.sub_demand_quantities
        bus_served = sum(
            sub_qty[i] for i in range(len(sub_qty))
            if i not in dispatch_result.unserved_demand
        )
        n_buses = sum(1 for r in dispatch_result.vehicle_routes.values() if len(r) > 1)
        bus_rounds = total_demand_people / max(bus_served, 1)
        bus_time_min = (bus_rounds * dispatch_result.objective_value / max(n_buses, 1)) / 60

        comp_data = {
            "指标": [
                "覆盖人数", "平均耗时", "最慢耗时",
                "依赖道路", "依赖车辆", "适合距离",
            ],
            "行人步行": [
                f"{total_demand_people:,}人 (全部可行走)",
                f"{avg_time_min:.0f} min",
                f"{slowest_min:.0f} min",
                "需要步行道",
                "否",
                "< 3 km 合适",
            ],
            "公交调度": [
                f"{bus_served:,}人 (单趟)",
                f"{bus_time_min:.0f} min ({bus_rounds:.1f}轮)",
                f"{bus_time_min:.0f} min",
                "需要道路畅通",
                f"是 ({n_buses}辆)",
                "> 1 km 合适",
            ],
        }
        df_comp = pd.DataFrame(comp_data)
        st.dataframe(df_comp.set_index("指标"), use_container_width=True)
        st.caption(
            "行人适合短距离（< 2-3km），公交适合中长距离。"
            "实际疏散中两者协同：近的走路，远的坐车。"
        )


def render_dispatch_params(bus_yaml: dict | None = None) -> dict:
    """Render dispatch parameter controls and return user selections.

    Args:
        bus_yaml: Optional bus section from scenario YAML for default values.
    """
    bus_yaml = bus_yaml or {}
    st.sidebar.header("公交调度参数")
    st.sidebar.caption("公交从集结区出发，前往需求点接人送往安全区")
    ds = st.session_state.get("demand_scale", 30000)
    yaml_n_buses = bus_yaml.get("num_buses", max(30, ds // 400))
    yaml_capacity = bus_yaml.get("capacity_per_bus", 50)
    n_buses = st.sidebar.slider("公交车数量", 5, 300, min(yaml_n_buses, 300), step=5)
    bus_capacity = st.sidebar.slider("单车容量 (人)", 30, 150, yaml_capacity, step=10)
    boarding_rate = st.sidebar.slider("上车速率 (人/秒)", 1.0, 5.0, 2.0, step=0.5)
    time_limit = st.sidebar.slider("求解时间上限 (秒)", 5, 60, 30, step=5)

    return {
        "n_buses": n_buses,
        "bus_capacity": bus_capacity,
        "boarding_rate": boarding_rate,
        "time_limit": time_limit,
    }
