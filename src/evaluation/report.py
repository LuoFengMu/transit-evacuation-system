"""Template-based evacuation simulation report generator."""
from typing import Optional
import streamlit as st
import pandas as pd
from datetime import datetime

from src.evaluation.metrics import EvacuationMetrics, metrics_to_dict


def render_report(
    demand_summary: dict,
    shelter_summary: Optional[dict],
    dispatch_result,
    allocation_result,
    station_pressures: list,
    comparison,  # ComparisonResult
    sumo_result,
    path_time_s: float,
    dispatch_time_s: float,
    sumo_time_s: Optional[float] = None,
    bus_params: Optional[dict] = None,
    demand_scale: int = 10000,
    event_type: str = "crowd",
    radius_m: int = 1500,
) -> None:
    """Render a comprehensive evacuation simulation report."""

    st.header("疏散仿真总结报告")
    st.caption(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── 1. Scenario overview ───────────────────────────────
    st.subheader("1. 场景概览")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("事件类型", {"crowd": "大客流"}.get(event_type, event_type))
    c2.metric("影响半径", f"{radius_m}m")
    c3.metric("疏散人数", f"{demand_scale:,}")
    c4.metric("需求点数", demand_summary.get("total_points", 0))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("轨道站", "20个 (1/2/3号线)")
    c2.metric("公交站", "95个")
    if bus_params:
        c4.metric("公交运力", f"{bus_params['n_buses']}辆×{bus_params['bus_capacity']}人")

    # ── 2. Time breakdown ──────────────────────────────────
    st.subheader("2. 计算耗时")
    c1, c2, c3 = st.columns(3)
    c1.metric("行人路径", f"{path_time_s:.1f}s")
    c2.metric("公交调度", f"{dispatch_time_s:.1f}s")
    if sumo_time_s:
        c3.metric("SUMO仿真", f"{sumo_time_s:.1f}s")
    else:
        c3.metric("SUMO仿真", "未运行")

    # ── 3. Mode distribution ───────────────────────────────
    if allocation_result:
        st.subheader("3. 疏散方式分布")
        mode_labels = {"walk_self": "步行自行离开", "walk_rail": "步行→轨道站",
                       "bus_rail": "公交→轨道站", "bus_periphery": "公交→外围疏散"}
        mode_counts = {}
        for v in allocation_result.destination_type.values():
            mode_counts[v] = mode_counts.get(v, 0) + 1

        mode_rows = []
        for mode, label in mode_labels.items():
            mode_rows.append({"方式": label, "需求点数": mode_counts.get(mode, 0)})
        mode_rows.append({"方式": "未分配", "需求点数": len(allocation_result.unassigned)})
        st.dataframe(pd.DataFrame(mode_rows).set_index("方式"), use_container_width=True)

        # Round results
        if allocation_result.round_results:
            st.subheader("4. 多轮调度")
            rr = allocation_result.round_results
            rr_rows = [{"轮次": r.round_id, "本轮到(人)": r.served_people,
                        "剩余(人)": r.remaining_people, "未疏散(人)": r.unserved}
                       for r in rr]
            st.dataframe(pd.DataFrame(rr_rows), use_container_width=True, hide_index=True)

    # ── 5. Station pressure ────────────────────────────────
    if station_pressures:
        st.subheader("5. 轨道站点压力")
        press_rows = []
        for p in sorted(station_pressures, key=lambda x: -x.pressure):
            if p.arrivals > 0:
                press_rows.append({
                    "站点": p.station_name, "到达人数": p.arrivals,
                    "处理能力": p.capacity_used, "压力指数": f"{p.pressure:.2f}",
                    "状态": p.level,
                })
        if press_rows:
            st.dataframe(pd.DataFrame(press_rows), use_container_width=True, hide_index=True)

    # ── 6. SUMO results ─────────────────────────────────────
    if sumo_result and sumo_result.success:
        st.subheader("6. SUMO 仿真")
        c1, c2, c3 = st.columns(3)
        c1.metric("发出车辆", sumo_result.vehicles_inserted)
        c2.metric("到达车辆", sumo_result.vehicles_arrived)
        c3.metric("平均速度", f"{sumo_result.avg_speed_ms * 3.6:.1f} km/h")

    # ── 7. Comparison ───────────────────────────────────────
    if comparison and len(comparison.metrics) >= 2:
        st.subheader("7. 方案对比")
        comparison.render_chart(key="report")
        comparison.render_table()

    # ── Export ──────────────────────────────────────────────
    st.subheader("8. 导出")
    report_text = _build_text_report(
        demand_summary=demand_summary, shelter_summary=shelter_summary,
        dispatch_result=dispatch_result, allocation_result=allocation_result,
        station_pressures=station_pressures, comparison=comparison,
        sumo_result=sumo_result, path_time_s=path_time_s,
        dispatch_time_s=dispatch_time_s, sumo_time_s=sumo_time_s,
        bus_params=bus_params, demand_scale=demand_scale,
        event_type=event_type, radius_m=radius_m,
    )
    st.download_button(
        label="导出文字报告 (TXT)",
        data=report_text,
        file_name=f"evacuation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
    )


def _build_text_report(**kwargs) -> str:
    """Build a plain-text evacuation report."""
    lines = []
    lines.append("=" * 60)
    lines.append("  城市大客流公交-轨道协同疏散仿真报告")
    lines.append("=" * 60)
    lines.append(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  需求量级: {kwargs['demand_scale']:,}人")
    lines.append(f"  事件类型: {kwargs['event_type']}, 半径: {kwargs['radius_m']}m")
    lines.append(f"  公交站: 95个")
    lines.append(f"  轨道站: 20个(1/2/3号线), 公交站: 95个")
    if kwargs.get('bus_params'):
        bp = kwargs['bus_params']
        lines.append(f"  公交运力: {bp['n_buses']}辆×{bp['bus_capacity']}人 = {bp['n_buses']*bp['bus_capacity']}人")
    lines.append("")
    lines.append(f"  计算耗时: 行人路径{kwargs['path_time_s']:.1f}s, 调度{kwargs['dispatch_time_s']:.1f}s")
    if kwargs.get('sumo_time_s'):
        lines.append(f"  SUMO仿真: {kwargs['sumo_time_s']:.1f}s")
    lines.append("")

    ar = kwargs.get('allocation_result')
    if ar:
        lines.append("  疏散方式分布:")
        mode_labels = {"walk_self": "步行自行离开", "walk_rail": "步行→轨道站",
                       "bus_rail": "公交→轨道站", "bus_periphery": "公交→外围疏散"}
        for v in ar.destination_type.values():
            pass  # skip counting here, use round results instead
        if ar.round_results:
            lines.append("  多轮调度:")
            for r in ar.round_results:
                lines.append(f"    第{r.round_id}轮: 到{r.served_people}人, 剩{r.remaining_people}人, 未{r.unserved}人")
        lines.append(f"    未分配需求点: {len(ar.unassigned)}个")

    sp = kwargs.get('station_pressures')
    if sp:
        lines.append("")
        lines.append("  轨道站点压力:")
        for p in sorted(sp, key=lambda x: -x.pressure):
            if p.arrivals > 0:
                lines.append(f"    {p.station_name}: P={p.pressure:.2f} ({p.level}), {p.arrivals}人")

    sr = kwargs.get('sumo_result')
    if sr and sr.success:
        lines.append("")
        lines.append(f"  SUMO仿真: {sr.vehicles_inserted}发出, {sr.vehicles_arrived}到达, "
                     f"均速{sr.avg_speed_ms*3.6:.1f}km/h")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)
