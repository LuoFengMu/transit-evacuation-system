"""SUMO simulation runner for evacuation scenarios."""
import os
import subprocess
from dataclasses import dataclass, field
from typing import Optional


def _get_sumo_bin_dir() -> str:
    """Find the directory containing SUMO binaries (sumo, netconvert, duarouter)."""
    candidates = [
        os.path.expanduser("~/Library/Python/3.9/bin"),
        os.path.expanduser("~/Library/Python/3.10/bin"),
        os.path.expanduser("~/Library/Python/3.11/bin"),
    ]
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        candidates.insert(0, os.path.join(sumo_home, "bin"))
    for c in candidates:
        if os.path.isfile(os.path.join(c, "sumo")):
            return c
    return ""


def _find_exe(name: str) -> str:
    """Find a SUMO binary, autodetecting if not on PATH."""
    bin_dir = _get_sumo_bin_dir()
    if bin_dir:
        path = os.path.join(bin_dir, name)
        if os.path.isfile(path):
            return path
    return name  # fallback: hope it's on PATH


_SUMO_BIN = _find_exe("sumo")
_NETCONVERT_BIN = _find_exe("netconvert")
_DUAROUTER_BIN = _find_exe("duarouter")


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
    actual_end = max(end_s, begin_s + 28800)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    net_abs = os.path.abspath(net_path)
    route_abs = os.path.abspath(route_path)
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


def run_sumo_headless(sumocfg_path: str, output_dir: str) -> SumoSimResult:
    os.makedirs(output_dir, exist_ok=True)
    tripinfo_path = os.path.join(output_dir, "tripinfo.xml")
    vehroute_path = os.path.join(output_dir, "vehroute.xml")
    cmd = [_SUMO_BIN, "-c", sumocfg_path,
           "--tripinfo-output", tripinfo_path,
           "--tripinfo-output.write-unfinished", "true",
           "--vehroute-output", vehroute_path,
           "--vehroute-output.exit-times", "true",
           "--no-step-log", "--duration-log.statistics", "--no-warnings"]
    result = SumoSimResult()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            result.error = proc.stderr
            return result
        result = _fill_result(result, tripinfo_path)
        result.success = True
    except subprocess.TimeoutExpired:
        result.error = "Simulation timed out"
    except Exception as e:
        result.error = str(e)
    return result


def run_sumo_with_traci(
    sumocfg_path: str, output_dir: str,
    road_closure_edges: Optional[list[str]] = None,
    closure_time_s: float = 300.0,
) -> SumoSimResult:
    import sys
    # Auto-detect SUMO_HOME from common install locations
    sumo_home = os.environ.get("SUMO_HOME", "")
    if not sumo_home:
        candidates = [
            os.path.expanduser("~/Library/Python/3.9/lib/python/site-packages/sumo"),
            os.path.expanduser("~/Library/Python/3.10/lib/python/site-packages/sumo"),
            os.path.expanduser("~/Library/Python/3.11/lib/python/site-packages/sumo"),
            "/opt/homebrew/opt/sumo/share/sumo",
            "/usr/local/opt/sumo/share/sumo",
        ]
        for c in candidates:
            if os.path.isdir(c):
                sumo_home = c
                break
    if not sumo_home:
        raise RuntimeError("SUMO_HOME not set and sumo not found in common locations")
    if sumo_home not in sys.path:
        sys.path.insert(0, os.path.join(sumo_home, "tools"))
    import traci

    os.makedirs(output_dir, exist_ok=True)
    tripinfo_path = os.path.join(output_dir, "tripinfo_traci.xml")
    vehroute_path = os.path.join(output_dir, "vehroute_traci.xml")

    result = SumoSimResult()
    try:
        traci.start([_SUMO_BIN, "-c", sumocfg_path,
                      "--tripinfo-output", tripinfo_path,
                      "--tripinfo-output.write-unfinished", "true",
                      "--vehroute-output", vehroute_path,
                      "--vehroute-output.exit-times", "true",
                      "--no-step-log", "--no-warnings"])
        applied = False
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            if road_closure_edges and not applied and traci.simulation.getTime() >= closure_time_s:
                for eid in road_closure_edges:
                    try:
                        traci.edge.setDisallowed(eid, ["all"])
                    except traci.exceptions.TraCIException:
                        pass
                applied = True
        traci.close()
        result.success = True
        result = _fill_result(result, tripinfo_path)
    except Exception as e:
        result.error = str(e)
        try:
            traci.close()
        except Exception:
            pass
    return result


def find_edges_near_event(
    network_path: str,
    event_center: tuple[float, float],
    radius_m: float,
) -> list[str]:
    """Find SUMO edge IDs within a radius of an event center (WGS84)."""
    import xml.etree.ElementTree as ET
    from pyproj import Transformer

    net_tree = ET.parse(network_path)
    loc = net_tree.find(".//location")
    if loc is None:
        return []
    net_ox = float(loc.get("netOffset", "0,0").split(",")[0])
    net_oy = float(loc.get("netOffset", "0,0").split(",")[1])
    proj_str = loc.get("projParameter", "")
    try:
        to_utm = Transformer.from_crs("EPSG:4326", proj_str, always_xy=True)
    except Exception:
        to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)

    utm_x, utm_y = to_utm.transform(event_center[0], event_center[1])
    scx = utm_x + net_ox
    scy = utm_y + net_oy

    nearby = []
    for edge in net_tree.findall(".//edge"):
        eid = edge.get("id", "")
        if eid.startswith(":"):
            continue
        shape_str = edge.get("shape", "")
        if not shape_str:
            continue
        for pt in shape_str.split():
            parts = pt.split(",")
            if len(parts) == 2:
                ex, ey = float(parts[0]), float(parts[1])
                if ((ex - scx) ** 2 + (ey - scy) ** 2) ** 0.5 < radius_m:
                    nearby.append(eid)
                    break
    return nearby


def _fill_result(result: SumoSimResult, tripinfo_path: str) -> SumoSimResult:
    if os.path.exists(tripinfo_path):
        result.vehicle_logs = _parse_tripinfo(tripinfo_path)
        if result.vehicle_logs:
            durations = [v.get("duration", 0) for v in result.vehicle_logs]
            speeds = [v.get("routeLength", 0) / max(v.get("duration", 1), 1)
                      for v in result.vehicle_logs]
            result.vehicles_arrived = sum(1 for v in result.vehicle_logs if v.get("vaporized", "") == "")
            result.vehicles_inserted = len(result.vehicle_logs)
            if durations:
                result.avg_duration_s = sum(durations) / len(durations)
                result.avg_speed_ms = sum(speeds) / len(speeds)
    return result


def _parse_tripinfo(path: str) -> list[dict]:
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
    vehroute_path: str, network_path: str,
) -> list[dict]:
    import xml.etree.ElementTree as ET
    from pyproj import Transformer

    net_tree = ET.parse(network_path)
    location = net_tree.find(".//location")
    net_offset_x = float(location.get("netOffset", "0,0").split(",")[0]) if location is not None else 0
    net_offset_y = float(location.get("netOffset", "0,0").split(",")[1]) if location is not None else 0
    proj_str = location.get("projParameter", "") if location is not None else ""
    try:
        to_wgs84 = Transformer.from_crs(proj_str, "EPSG:4326", always_xy=True)
    except Exception:
        to_wgs84 = Transformer.from_crs("EPSG:32650", "EPSG:4326", always_xy=True)

    # Build junction coordinate lookup
    junction_coords = {}
    for junction in net_tree.findall(".//junction"):
        jid = junction.get("id", "")
        jx = float(junction.get("x", 0))
        jy = float(junction.get("y", 0))
        utm_x = jx - net_offset_x
        utm_y = jy - net_offset_y
        lon, lat = to_wgs84.transform(utm_x, utm_y)
        junction_coords[jid] = (lon, lat)

    # Build edge geometry lookup (shape or from-to junctions)
    edge_geoms = {}
    for edge in net_tree.findall(".//edge"):
        eid = edge.get("id", "")
        shape_str = edge.get("shape", "")
        if shape_str:
            coords = []
            for pt in shape_str.split():
                parts = pt.split(",")
                if len(parts) == 2:
                    sumo_x, sumo_y = float(parts[0]), float(parts[1])
                    utm_x = sumo_x - net_offset_x
                    utm_y = sumo_y - net_offset_y
                    lon, lat = to_wgs84.transform(utm_x, utm_y)
                    coords.append((lon, lat))
            if coords:
                edge_geoms[eid] = coords
        else:
            # Build from junction coordinates
            from_j = edge.get("from", "")
            to_j = edge.get("to", "")
            if from_j in junction_coords and to_j in junction_coords:
                edge_geoms[eid] = [junction_coords[from_j], junction_coords[to_j]]

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
                # Try common suffix patterns
                for suffix in ["#0", "#1", "#2", "_0", "_1"]:
                    if eid.endswith(suffix):
                        geom = edge_geoms.get(eid[: -len(suffix)])
                        if geom:
                            break
                if geom is None:
                    # Try prefix match (strip lane-addon suffixes)
                    base = eid.split("#")[0]
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
