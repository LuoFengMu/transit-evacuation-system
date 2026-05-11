"""SUMO simulation runner for evacuation scenarios."""
import os
import subprocess
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SumoSimResult:
    """Results from a SUMO simulation run."""
    total_duration_s: float = 0.0
    vehicles_inserted: int = 0
    vehicles_arrived: int = 0
    avg_speed_ms: float = 0.0
    avg_duration_s: float = 0.0
    avg_waiting_s: float = 0.0
    vehicle_logs: list[dict] = field(default_factory=list)
    success: bool = False
    error: str = ""


def create_sumocfg(net_path: str, route_path: str, output_path: str,
                   begin_s: float = 0, end_s: float = 7200) -> str:
    """Create a SUMO configuration file. Paths are converted to absolute."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    net_abs = os.path.abspath(net_path)
    route_abs = os.path.abspath(route_path)
    # Auto-extend end time for multi-round simulations
    actual_end = max(end_s, begin_s + 28800)  # up to 8 hours
    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
  <input>
    <net-file value="{net_abs}"/>
    <route-files value="{route_abs}"/>
  </input>
  <time>
    <begin value="{begin_s}"/>
    <end value="{actual_end}"/>
  </time>
</configuration>"""
    with open(output_path, "w") as f:
        f.write(content)
    return output_path


def run_sumo_headless(
    sumocfg_path: str,
    output_dir: str,
) -> SumoSimResult:
    """Run SUMO in headless mode and collect tripinfo results."""
    os.makedirs(output_dir, exist_ok=True)
    tripinfo_path = os.path.join(output_dir, "tripinfo.xml")

    vehroute_path = os.path.join(output_dir, "vehroute.xml")
    cmd = [
        "sumo",
        "-c", sumocfg_path,
        "--tripinfo-output", tripinfo_path,
        "--tripinfo-output.write-unfinished", "true",
        "--vehroute-output", vehroute_path,
        "--vehroute-output.exit-times", "true",
        "--no-step-log",
        "--duration-log.statistics",
        "--no-warnings",
    ]

    result = SumoSimResult()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if proc.returncode != 0:
            result.error = proc.stderr
            return result

        if os.path.exists(tripinfo_path):
            result.vehicle_logs = _parse_tripinfo(tripinfo_path)
            if result.vehicle_logs:
                durations = [v.get("duration", 0) for v in result.vehicle_logs]
                waitings = [v.get("waitingTime", 0) for v in result.vehicle_logs]
                speeds = [v.get("routeLength", 0) / max(v.get("duration", 1), 1)
                          for v in result.vehicle_logs]
                result.vehicles_arrived = sum(
                    1 for v in result.vehicle_logs if v.get("vaporized", "") == ""
                )
                result.vehicles_inserted = len(result.vehicle_logs)
                if durations:
                    result.avg_duration_s = sum(durations) / len(durations)
                    result.avg_waiting_s = sum(waitings) / len(waitings)
                    result.avg_speed_ms = sum(speeds) / len(speeds)

        result.success = True
    except subprocess.TimeoutExpired:
        result.error = "Simulation timed out (180s)"
    except Exception as e:
        result.error = str(e)

    return result


def run_sumo_with_traci(
    sumocfg_path: str,
    output_dir: str,
    road_closures: Optional[list[str]] = None,
    closure_time_s: float = 0.0,
) -> SumoSimResult:
    """Run SUMO with TraCI control for real-time road closures."""
    import sys
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home and sumo_home not in sys.path:
        sys.path.insert(0, os.path.join(sumo_home, "tools"))
    import traci

    os.makedirs(output_dir, exist_ok=True)
    tripinfo_path = os.path.join(output_dir, "tripinfo_traci.xml")

    result = SumoSimResult()
    try:
        traci.start(
            ["sumo", "-c", sumocfg_path,
             "--tripinfo-output", tripinfo_path,
             "--tripinfo-output.write-unfinished", "true",
             "--no-step-log", "--no-warnings"],
        )

        closures_applied = False
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

            if (road_closures and not closures_applied
                    and traci.simulation.getTime() >= closure_time_s):
                for edge_id in road_closures:
                    try:
                        traci.edge.setDisallowed(edge_id, ["all"])
                    except traci.exceptions.TraCIException:
                        pass
                closures_applied = True

        traci.close()
        result.success = True

        if os.path.exists(tripinfo_path):
            result.vehicle_logs = _parse_tripinfo(tripinfo_path)
            if result.vehicle_logs:
                durations = [v.get("duration", 0) for v in result.vehicle_logs]
                result.avg_duration_s = sum(durations) / len(durations)
                result.vehicles_arrived = sum(
                    1 for v in result.vehicle_logs if v.get("vaporized", "") == ""
                )

    except Exception as e:
        result.error = str(e)
        try:
            traci.close()
        except Exception:
            pass

    return result


def _parse_tripinfo(path: str) -> list[dict]:
    """Parse SUMO tripinfo XML into a list of dicts."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(path)
    trips = []
    for trip in tree.findall(".//tripinfo"):
        trips.append({
            "id": trip.get("id", ""),
            "depart": float(trip.get("depart", 0)),
            "arrival": float(trip.get("arrival", 0)),
            "duration": float(trip.get("duration", 0)),
            "routeLength": float(trip.get("routeLength", 0)),
            "waitingTime": float(trip.get("waitingTime", 0)),
            "timeLoss": float(trip.get("timeLoss", 0)),
            "vaporized": trip.get("vaporized", ""),
        })
    return trips


def extract_vehicle_trajectories(
    vehroute_path: str,
    network_path: str,
) -> list[dict]:
    """Extract vehicle trajectories from SUMO output for map display.

    Returns list of dicts with 'vehicle_id', 'coords', 'depart', 'arrival'.
    """
    import xml.etree.ElementTree as ET

    # Build edge geometry lookup and get coordinate transform
    net_tree = ET.parse(network_path)
    location = net_tree.find(".//location")
    net_offset_x = float(location.get("netOffset", "0,0").split(",")[0]) if location is not None else 0
    net_offset_y = float(location.get("netOffset", "0,0").split(",")[1]) if location is not None else 0
    proj_str = location.get("projParameter", "") if location is not None else ""

    from pyproj import Transformer
    try:
        to_wgs84 = Transformer.from_crs(proj_str, "EPSG:4326", always_xy=True)
    except Exception:
        # Fallback: try UTM zone from proj string
        to_wgs84 = Transformer.from_crs("EPSG:32650", "EPSG:4326", always_xy=True)

    edge_geoms = {}
    for edge in net_tree.findall(".//edge"):
        eid = edge.get("id", "")
        shape_str = edge.get("shape", "")
        if shape_str:
            coords = []
            for pt in shape_str.split():
                parts = pt.split(",")
                if len(parts) == 2:
                    # Convert SUMO internal → UTM → WGS84
                    sumo_x = float(parts[0])
                    sumo_y = float(parts[1])
                    utm_x = sumo_x - net_offset_x
                    utm_y = sumo_y - net_offset_y
                    lon, lat = to_wgs84.transform(utm_x, utm_y)
                    coords.append((lon, lat))
            if coords:
                edge_geoms[eid] = coords

    # Parse vehicle routes
    route_tree = ET.parse(vehroute_path)
    trajectories = []
    for vehicle in route_tree.findall(".//vehicle"):
        vid = vehicle.get("id", "")
        depart = float(vehicle.get("depart", 0))
        arrival = float(vehicle.get("arrival", 0))
        route = vehicle.find("route")
        if route is None:
            continue
        edges_str = route.get("edges", "")
        if not edges_str:
            continue

        edge_ids = edges_str.split()
        all_coords: list = []
        for eid in edge_ids:
            geom = edge_geoms.get(eid)
            if geom is None:
                # Try without lane suffix
                base = eid.split("#")[0] if "#" in eid else eid
                geom = edge_geoms.get(base)
            if geom:
                if all_coords and all_coords[-1] == geom[0]:
                    all_coords.extend(geom[1:])
                else:
                    all_coords.extend(geom)

        if len(all_coords) >= 2:
            trajectories.append({
                "vehicle_id": vid.split("_leg")[0] if "_leg" in vid else vid,
                "coords": all_coords,
                "depart": depart,
                "arrival": arrival,
            })

    return trajectories
