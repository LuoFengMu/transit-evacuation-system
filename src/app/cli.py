"""CLI entry point for single-scenario evacuation simulation.

Usage:
    python -m src.app.cli --scenario configs/scenarios/C_hybrid_cooperative.yaml
    python -m src.app.cli --scenario configs/scenarios/A_bus_direct.yaml --seed 123 --demand 50000
    python -m src.app.cli --scenario configs/scenarios/B_rail_priority.yaml --no-rail --no-sumo
"""
import argparse
import os
import sys
import yaml

from src.app.config import SCENARIOS_DIR, DEMAND_SCALE_OPTIONS, DEFAULT_DEMAND_SCALE
from src.app.pipeline import run_analysis
from src.dispatch.cost_matrix import MODE_EUCLIDEAN


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="公交-轨道协同疏散仿真 — 命令行实验入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --scenario C_hybrid_cooperative.yaml
  %(prog)s --scenario A_bus_direct.yaml --seed 123 --demand 50000 --no-rail
  %(prog)s --scenario B_rail_priority.yaml --cost-mode cached_network_time
        """,
    )
    p.add_argument("--scenario", "-s", required=True,
                   help="场景 YAML 文件名或完整路径 (相对于 configs/scenarios/)")
    p.add_argument("--seed", type=int, default=None,
                   help="随机种子 (默认: YAML 配置值或 42)")
    p.add_argument("--demand", type=int, default=None,
                   choices=DEMAND_SCALE_OPTIONS,
                   help=f"疏散人数量级 (默认: YAML 配置值或 {DEFAULT_DEMAND_SCALE})")
    p.add_argument("--output", "-o", default=None,
                   help="自定义输出目录 (默认: outputs/runs/<run_id>)")
    p.add_argument("--cost-mode", default=None,
                   choices=["euclidean_fast", "road_network_time", "cached_network_time"],
                   help="成本矩阵模式 (默认: YAML 配置值或 euclidean_fast)")
    p.add_argument("--no-bus", action="store_true",
                   help="禁用公交调度")
    p.add_argument("--no-rail", action="store_true",
                   help="禁用轨道协同")
    p.add_argument("--no-sumo", action="store_true",
                   help="禁用 SUMO 仿真")
    p.add_argument("--capacity-factor", type=float, default=None,
                   choices=[0.7, 1.0, 1.2],
                   help="轨道容量因子: 0.7 保守 / 1.0 基准 / 1.2 乐观")
    return p


def resolve_scenario_path(name: str) -> str:
    """Resolve scenario file from short name or full path."""
    if os.path.isfile(name):
        return name
    candidate = os.path.join(SCENARIOS_DIR, name)
    if os.path.isfile(candidate):
        return candidate
    if not name.endswith(".yaml"):
        candidate = os.path.join(SCENARIOS_DIR, name + ".yaml")
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"场景文件未找到: {name}")


def main(argv: list[str] | None = None) -> dict:
    """Parse args, run analysis, print summary. Returns result dict."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Resolve scenario
    scenario_path = resolve_scenario_path(args.scenario)
    with open(scenario_path, "r", encoding="utf-8") as f:
        scenario = yaml.safe_load(f)

    # Build params — CLI overrides > YAML defaults > hardcoded defaults
    yaml_sim = scenario.get("simulation", {}) or {}
    yaml_demand = scenario.get("demand", {}) or {}
    yaml_rail = scenario.get("rail", {}) or {}

    params = {
        "scenario_path": scenario_path,
        "event_location": scenario.get("event", {}).get("location", "彭城广场"),
        "radius_m": scenario.get("event", {}).get("radius_m", 1500),
        "actual_demand": args.demand or yaml_demand.get("scale", DEFAULT_DEMAND_SCALE),
        "random_seed": args.seed if args.seed is not None else yaml_sim.get("random_seed", 42),
        "enable_perturbation": yaml_demand.get("random_variation", False),
        "enable_bus": not args.no_bus and scenario.get("bus_enabled", True),
        "bus_params": {"n_buses": 30, "bus_capacity": 50, "boarding_rate": 2.0, "time_limit": 30}
                      if not args.no_bus else None,
        "cost_matrix_mode": args.cost_mode or yaml_sim.get("cost_matrix_mode", MODE_EUCLIDEAN),
        "enable_sumo": not args.no_sumo,
        "enable_crop": True,
        "enable_traci": False,
        "enable_rail": not args.no_rail and scenario.get("rail_enabled", True),
        "walk_self_min": yaml_rail.get("walk_self_min", 20),
        "walk_rail_min": yaml_rail.get("walk_rail_min", 10),
        "pressure_limit": yaml_rail.get("pressure_limit", 1.1),
        "walk_mode": yaml_rail.get("walk_mode", "euclidean_fast"),
        "cap_factor": args.capacity_factor or yaml_rail.get("capacity_factor", 1.0),
        "enable_sensitivity": False,
        "enable_snap": yaml_demand.get("snap_to_road", True),
        "enable_water_filter": yaml_demand.get("filter_water", False),
    }
    if args.output:
        params["output_dir"] = args.output

    # Override bus params from YAML if available
    yaml_bus = scenario.get("bus", {}) or {}
    if params["enable_bus"] and params["bus_params"]:
        if yaml_bus.get("num_buses"):
            params["bus_params"]["n_buses"] = yaml_bus["num_buses"]
        if yaml_bus.get("capacity_per_bus"):
            params["bus_params"]["bus_capacity"] = yaml_bus["capacity_per_bus"]

    print(f"场景: {os.path.basename(scenario_path)}")
    print(f"地点: {params['event_location']}  需求: {params['actual_demand']:,}人  seed: {params['random_seed']}")
    print(f"公交: {'启用' if params['enable_bus'] else '关闭'}  "
          f"轨道: {'启用' if params['enable_rail'] else '关闭'}  "
          f"SUMO: {'启用' if params['enable_sumo'] else '关闭'}")
    print(f"成本矩阵: {params['cost_matrix_mode']}  "
          f"容量因子: ×{params['cap_factor']}")
    print("-" * 60)

    result = run_analysis(params)

    print("-" * 60)
    print(f"run_id: {result['run_id']}")
    print(f"输出目录: {result['run_output_dir']}")

    dr = result.get("dispatch_result")
    if dr:
        n_used = sum(1 for r in dr.vehicle_routes.values() if len(r) > 1)
        print(f"调度: {dr.solver_status}  用车: {n_used}辆")
    if result.get("evac_metrics"):
        m = result["evac_metrics"]
        print(f"完成率: {m.completion_rate:.1%}  轨道分担: {m.rail_share:.1%}  未服务: {m.unserved:,}")

    # Print log
    for line in result.get("log_lines", []):
        print(f"  {line}")

    return result


if __name__ == "__main__":
    main()
