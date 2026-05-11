"""Build SUMO trip and route files from dispatch plans.

Supports both single-trip and multi-trip (cycling) generation.
"""
import os
import subprocess


def dispatch_to_sumo_trips(
    dispatch_result,
    vehicles,
    depots,
    demand_gdf,
    output_dir: str,
    network_path: str,
    bus_type: str = "bus_evac",
    max_rounds: int = 1,
    round_trip_s: float = 900.0,
) -> tuple[str, str]:
    """Convert dispatch plan to SUMO trip + route files.

    Args:
        dispatch_result: DispatchResult from solver.
        vehicles: List of BusVehicle.
        depots: List of BusDepot.
        demand_gdf: Demand points GeoDataFrame.
        output_dir: Output directory for trip/route files.
        network_path: Path to SUMO .net.xml.
        bus_type: vType ID for buses.
        max_rounds: Number of round trips per bus (1 = single trip).
        round_trip_s: Estimated round-trip time for staggering re-departures.

    Returns:
        (trip_path, route_path) tuple.
    """
    os.makedirs(output_dir, exist_ok=True)
    demand_pts = [g for g in demand_gdf.geometry]

    # ── 1. Write trips XML ──────────────────────────────────
    trip_path = os.path.join(output_dir, "bus_trips.trips.xml")
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<routes>']
    lines.append(
        f'  <vType id="{bus_type}" accel="1.5" decel="4.0" '
        f'length="12.0" maxSpeed="13.89" guiShape="bus"/>'
    )

    for vid, route in dispatch_result.vehicle_routes.items():
        # Build the geographic point sequence for this route
        seq_pts = []
        for stop_type, stop_id, _ in route:
            if stop_type == "depot":
                for di, d in enumerate(depots):
                    if stop_id == f"depot_{di:02d}":
                        seq_pts.append((d.lon, d.lat))
                        break
            elif stop_type == "pickup" and isinstance(stop_id, int):
                if stop_id < len(demand_pts):
                    pt = demand_pts[stop_id]
                    seq_pts.append((pt.x, pt.y))

        if len(seq_pts) < 2:
            continue

        # Each vehicle starts at a staggered time, then repeats
        vehicle_depart = float(hash(vid) % 60)  # stagger by up to 60s

        for round_num in range(max_rounds):
            for k in range(len(seq_pts) - 1):
                from_lon, from_lat = seq_pts[k]
                to_lon, to_lat = seq_pts[k + 1]
                trip_id = f"{vid}_r{round_num}_leg{k}"
                lines.append(
                    f'  <trip id="{trip_id}" type="{bus_type}" '
                    f'depart="{vehicle_depart:.1f}" '
                    f'fromLonLat="{from_lon},{from_lat}" toLonLat="{to_lon},{to_lat}"/>'
                )
                vehicle_depart += 5.0  # gap between legs
            vehicle_depart += round_trip_s  # gap between rounds

    lines.append("</routes>")
    with open(trip_path, "w") as f:
        f.write("\n".join(lines))

    # ── 2. Run duarouter ────────────────────────────────────
    route_path = os.path.join(output_dir, "bus_routes.rou.xml")
    result = subprocess.run(
        [
            "duarouter",
            "--net-file", network_path,
            "--route-files", trip_path,
            "--output-file", route_path,
            "--begin", "0",
            "--end", "28800",  # 8 hours max
            "--ignore-errors",
            "--no-warnings",
        ],
        capture_output=True, text=True, timeout=120,
    )

    if not os.path.exists(route_path):
        raise RuntimeError(f"duarouter failed: {result.stderr}")

    return trip_path, route_path


def compute_rounds_needed(
    dispatch_result,
    vehicles,
    demand_gdf,
) -> int:
    """Estimate how many round trips are needed to serve all demand."""
    sub_qty = dispatch_result.sub_demand_quantities
    total_served_per_round = sum(
        sub_qty[i] for i in range(len(sub_qty))
        if i not in dispatch_result.unserved_demand
    )
    total_demand = sum(sub_qty)
    if total_served_per_round <= 0:
        return 1
    rounds = int(total_demand / max(total_served_per_round, 1)) + 1
    return max(1, min(rounds, 3))  # cap at 3 rounds for performance
