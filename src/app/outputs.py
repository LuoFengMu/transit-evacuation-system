"""Save run artifacts to outputs/runs/{run_id}/.

Extracted from streamlit_app.py — always writes all 7 file types,
with empty data when analysis didn't produce results.
"""
import os
import json
import sys
import shutil
from datetime import datetime

import pandas as pd

from src.app.config import PROJECT_ROOT, VERSION
from src.evaluation.metrics import metrics_to_dict


def save_run_artifacts(
    run_output_dir: str,
    run_id: str,
    scenario_path: str,
    random_seed: int,
    selected_scenario_file: str,
    scenario_id: str,
    comparison,            # ComparisonResult | None
    station_pressures,     # list[StationPressure] | None
    dispatch_result,       # DispatchResult | None
    enable_bus: bool,
    enable_rail: bool,
    enable_sumo: bool,
    enable_perturbation: bool,
    event_location: str,
    radius_m: int,
    actual_demand: int,
    graphml_path: str,
    demand_path: str,
    shelters_path: str,
    rail_stations_path: str,
    run_config: dict | None = None,
) -> str:
    """Save all output artifacts. Returns human-readable summary string.

    Always writes 7 files: config.yaml, scenario.yaml, metrics.json,
    run_meta.json, report.txt, station_pressure.csv, dispatch_summary.csv.
    """
    import yaml
    saved = []

    # config.yaml — write from run_config
    if run_config:
        with open(os.path.join(run_output_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(run_config, f, allow_unicode=True, default_flow_style=False)
    else:
        shutil.copy(scenario_path, os.path.join(run_output_dir, "config.yaml"))
    saved.append("config.yaml")

    # scenario.yaml — copy
    shutil.copy(scenario_path, os.path.join(run_output_dir, "scenario.yaml"))
    saved.append("scenario.yaml")

    # metrics.json
    metrics_records = {}
    if comparison and comparison.metrics:
        for label, m in zip(comparison.scenario_labels, comparison.metrics):
            metrics_records[label] = metrics_to_dict(m)
    with open(os.path.join(run_output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_records, f, ensure_ascii=False, indent=2)
    saved.append("metrics.json")

    # station_pressure.csv
    press_rows = []
    if station_pressures:
        for p in station_pressures:
            press_rows.append({
                "station": p.station_name, "arrivals": p.arrivals,
                "capacity_used": p.capacity_used, "pressure": p.pressure,
                "level": p.level,
            })
    pd.DataFrame(press_rows).to_csv(
        os.path.join(run_output_dir, "station_pressure.csv"), index=False)
    saved.append("station_pressure.csv")

    # dispatch_summary.csv
    disp_rows = []
    if dispatch_result and dispatch_result.solver_status in ("optimal", "feasible"):
        for vid, route in dispatch_result.vehicle_routes.items():
            disp_rows.append({
                "vehicle_id": vid,
                "n_stops": len([s for s in route if s[0] == "pickup"]),
                "route": str(route),
            })
    pd.DataFrame(disp_rows).to_csv(
        os.path.join(run_output_dir, "dispatch_summary.csv"), index=False)
    saved.append("dispatch_summary.csv")

    # run_meta.json
    git_commit = _git_commit()
    sumo_version = _sumo_version()
    run_meta = {
        "run_id": run_id,
        "version": VERSION,
        "git_commit": git_commit,
        "python_version": sys.version,
        "sumo_version": sumo_version,
        "random_seed": random_seed,
        "scenario_file": selected_scenario_file,
        "scenario_id": scenario_id,
        "data_paths": {
            "graphml": graphml_path, "demand": demand_path,
            "shelters": shelters_path, "rail_stations": rail_stations_path,
        },
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(run_output_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)
    saved.append("run_meta.json")

    # report.txt
    report_path = os.path.join(run_output_dir, "report.txt")
    report_lines = _build_report(
        run_id, event_location, radius_m, actual_demand, random_seed,
        enable_perturbation, enable_bus, enable_rail, enable_sumo,
        comparison, station_pressures,
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    saved.append("report.txt")

    return f"已保存 {len(saved)} 个文件: {', '.join(saved)}"


# ── Helpers ────────────────────────────────────────────────────

def _git_commit() -> str:
    try:
        import subprocess
        r = subprocess.run(["git", "-C", PROJECT_ROOT, "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _sumo_version() -> str:
    try:
        import subprocess
        r = subprocess.run(
            [os.path.expanduser("~/Library/Python/3.9/bin/sumo"), "--version"],
            capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return "unknown"


def _build_report(
    run_id, event_location, radius_m, actual_demand, random_seed,
    enable_perturbation, enable_bus, enable_rail, enable_sumo,
    comparison, station_pressures,
) -> list[str]:
    lines = [
        "疏散仿真总结报告",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"运行ID: {run_id}", "",
        "=== 场景概览 ===",
        f"事件类型: 大客流", f"事件地点: {event_location}",
        f"影响半径: {radius_m}m", f"疏散人数: {actual_demand:,}",
        f"随机种子: {random_seed}",
        f"需求扰动: {'开启' if enable_perturbation else '关闭'}",
        f"公交调度: {'启用' if enable_bus else '关闭'}",
        f"轨道协同: {'启用' if enable_rail else '关闭'}",
        f"SUMO仿真: {'启用' if enable_sumo else '关闭'}",
        "",
    ]
    if comparison and comparison.metrics:
        lines.append("=== 方案对比 ===")
        for label, m in zip(comparison.scenario_labels, comparison.metrics):
            d = metrics_to_dict(m)
            lines.append(f"{label}: 完成率={d['completion_rate']:.3f}, "
                        f"轨道分担={d['rail_share']:.3f}, 未服务={d['unserved']:,}")
    else:
        lines.append("=== 方案对比 ===\n(未运行轨道协同分析)")
    if station_pressures:
        lines.append("\n=== 站点压力 ===")
        for p in sorted(station_pressures, key=lambda x: -x.pressure):
            if p.arrivals > 0:
                lines.append(f"{p.station_name}: 压力={p.pressure:.3f} ({p.level}) "
                           f"到达={p.arrivals} 能力={p.capacity_used}")
    return lines
