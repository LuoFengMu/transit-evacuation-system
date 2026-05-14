"""Batch experiment runner — runs multiple scenarios × seeds × capacity factors.

Usage:
    python -m src.app.batch --scenarios A B C --seeds 42 123 456
    python -m src.app.batch --scenarios C --seeds 42 --capacity-factors 0.7 1.0 1.2
    python -m src.app.batch --scenarios A B C --seeds 42 99 --no-sumo

Output:
    outputs/experiments/{experiment_id}/
    ├── summary.csv       # one row per run, key metrics
    └── manifest.json     # full metadata for every run
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import pandas as pd

from src.app.config import SCENARIOS_DIR, RUNS_DIR
from src.app.pipeline import run_analysis
from src.dispatch.cost_matrix import MODE_EUCLIDEAN


EXPERIMENTS_DIR = os.path.join(os.path.dirname(RUNS_DIR), "experiments")


def _scenario_path(scenario_id: str) -> str:
    mapping = {
        "A": "A_bus_direct.yaml",
        "B": "B_rail_priority.yaml",
        "C": "C_hybrid_cooperative.yaml",
        "A_bus_direct": "A_bus_direct.yaml",
        "B_rail_priority": "B_rail_priority.yaml",
        "C_hybrid_cooperative": "C_hybrid_cooperative.yaml",
    }
    fname = mapping.get(scenario_id, scenario_id)
    if not fname.endswith(".yaml"):
        fname += ".yaml"
    candidate = os.path.join(SCENARIOS_DIR, fname)
    if os.path.isfile(candidate):
        return candidate
    if os.path.isfile(scenario_id):
        return scenario_id
    raise FileNotFoundError(f"场景文件未找到: {scenario_id}")


def _extract_metrics(result: dict, elapsed_s: float) -> dict:
    """Extract key metrics from a run_analysis result dict."""
    row = {"runtime_s": round(elapsed_s, 1)}

    dr = result.get("dispatch_result")
    if dr:
        row["dispatch_status"] = dr.solver_status
        row["dispatch_vehicles_used"] = sum(1 for r in dr.vehicle_routes.values() if len(r) > 1)
    else:
        row["dispatch_status"] = "disabled"
        row["dispatch_vehicles_used"] = 0

    comp = result.get("comparison")
    if comp and comp.metrics:
        from src.evaluation.metrics import metrics_to_dict
        for label, m in zip(comp.scenario_labels, comp.metrics):
            d = metrics_to_dict(m)
            prefix = label.replace("方案", "").replace(":", "").replace(" ", "_")
            row[f"{prefix}_completion_rate"] = d["completion_rate"]
            row[f"{prefix}_rail_share"] = d["rail_share"]
            row[f"{prefix}_unserved"] = d["unserved"]

    em = result.get("evac_metrics")
    if em:
        row["completion_rate"] = em.completion_rate
        row["rail_share"] = em.rail_share
        row["unserved"] = em.unserved
        row["overloaded_stations"] = em.overloaded_stations

    return row


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="批量疏散仿真实验 — 多场景 × 多种子 × 多容量因子",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --scenarios A B C --seeds 42 123 456
  %(prog)s --scenarios C --seeds 42 --capacity-factors 0.7 1.0 1.2
  %(prog)s --scenarios A B C --seeds 42 --no-sumo --no-rail
        """,
    )
    p.add_argument("--scenarios", "-s", nargs="+", required=True,
                   help="场景标识: A, B, C 或完整文件名")
    p.add_argument("--seeds", nargs="+", type=int, default=[42],
                   help="随机种子列表 (默认: 42)")
    p.add_argument("--capacity-factors", nargs="+", type=float,
                   default=[1.0], choices=[0.7, 1.0, 1.2],
                   help="轨道容量因子 (默认: 1.0)")
    p.add_argument("--demand", type=int, default=30000,
                   help="疏散人数量级 (默认: 30000)")
    p.add_argument("--experiment-id", "-e", default=None,
                   help="实验 ID (默认: 自动生成 YYYYMMDD_HHMMSS)")
    p.add_argument("--no-sumo", action="store_true", help="禁用 SUMO")
    p.add_argument("--no-rail", action="store_true", help="禁用轨道协同")
    p.add_argument("--no-bus", action="store_true", help="禁用公交调度")
    return p


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)

    exp_id = args.experiment_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(EXPERIMENTS_DIR, exp_id)
    os.makedirs(exp_dir, exist_ok=True)

    # Build run list
    runs = []
    for sid in args.scenarios:
        for seed in args.seeds:
            for cf in args.capacity_factors:
                runs.append((sid, seed, cf))

    print(f"实验: {exp_id}")
    print(f"场景: {args.scenarios}  种子: {args.seeds}  容量因子: {args.capacity_factors}")
    print(f"总运行次数: {len(runs)}")
    print(f"输出: {exp_dir}")
    print("-" * 60)

    summary_rows = []
    manifest = {"experiment_id": exp_id, "args": vars(args), "runs": []}

    for idx, (sid, seed, cf) in enumerate(runs):
        label = f"[{idx+1}/{len(runs)}] {sid} seed={seed} cf={cf}"
        print(f"{label} ...", end=" ", flush=True)

        try:
            scenario_path = _scenario_path(sid)
            params = {
                "scenario_path": scenario_path,
                "event_location": "彭城广场",
                "radius_m": 1500,
                "actual_demand": args.demand,
                "random_seed": seed,
                "enable_perturbation": False,
                "enable_bus": not args.no_bus,
                "bus_params": {"n_buses": 30, "bus_capacity": 50, "boarding_rate": 2.0, "time_limit": 30}
                              if not args.no_bus else None,
                "cost_matrix_mode": MODE_EUCLIDEAN,
                "enable_sumo": not args.no_sumo,
                "enable_crop": True,
                "enable_traci": False,
                "enable_rail": not args.no_rail,
                "walk_self_min": 20, "walk_rail_min": 10, "pressure_limit": 1.1,
                "walk_mode": "euclidean_fast",
                "cap_factor": cf,
                "enable_sensitivity": False,
                "enable_snap": True,
                "enable_water_filter": False,
            }

            t0 = time.perf_counter()
            result = run_analysis(params)
            elapsed = time.perf_counter() - t0

            metrics = _extract_metrics(result, elapsed)
            metrics["experiment_id"] = exp_id
            metrics["run_id"] = result["run_id"]
            metrics["scenario"] = sid
            metrics["seed"] = seed
            metrics["capacity_factor"] = cf
            summary_rows.append(metrics)

            manifest["runs"].append({
                "run_id": result["run_id"],
                "scenario": sid,
                "seed": seed,
                "capacity_factor": cf,
                "runtime_s": round(elapsed, 1),
                "output_dir": result["run_output_dir"],
                "dispatch_status": metrics.get("dispatch_status", "unknown"),
                "completion_rate": metrics.get("completion_rate", 0),
                "rail_share": metrics.get("rail_share", 0),
                "unserved": metrics.get("unserved", 0),
            })

            status = f"✓ {elapsed:.0f}s"
            if metrics.get("completion_rate"):
                status += f" 完成率={metrics['completion_rate']:.1%}"
            print(status)

        except Exception as e:
            print(f"✗ {e}")
            manifest["runs"].append({
                "scenario": sid, "seed": seed, "capacity_factor": cf,
                "error": str(e),
            })

    # Write outputs
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(exp_dir, "summary.csv")
        df.to_csv(summary_path, index=False)
        print(f"\nsummary → {summary_path} ({len(df)} runs)")

        # Quick stats
        if "completion_rate" in df.columns:
            print(f"  完成率: mean={df['completion_rate'].mean():.1%}  "
                  f"min={df['completion_rate'].min():.1%}  max={df['completion_rate'].max():.1%}")

    manifest_path = os.path.join(exp_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
    print(f"manifest → {manifest_path}")

    print(f"\n实验完成: {exp_dir}")


if __name__ == "__main__":
    main()
