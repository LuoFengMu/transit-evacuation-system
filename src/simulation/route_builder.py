"""Build SUMO trip files from dispatch plans. Uses SUMO's built-in router."""
import os
import subprocess


def _find_nearest_edge(lon: float, lat: float, network_path: str) -> str:
    """Find the nearest SUMO edge ID to a WGS84 coordinate."""
    import xml.etree.ElementTree as ET
    from pyproj import Transformer

    tree = ET.parse(network_path)
    loc = tree.find(".//location")
    if loc is None:
        return ""
    net_ox = float(loc.get("netOffset", "0,0").split(",")[0])
    net_oy = float(loc.get("netOffset", "0,0").split(",")[1])
    proj_str = loc.get("projParameter", "")

    try:
        to_utm = Transformer.from_crs("EPSG:4326", proj_str, always_xy=True)
    except Exception:
        to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)

    utm_x, utm_y = to_utm.transform(lon, lat)
    # Convert to SUMO internal: SUMO = UTM + netOffset
    sx = utm_x + net_ox
    sy = utm_y + net_oy

    best_eid = ""
    best_dist = float("inf")

    for edge in tree.findall(".//edge"):
        eid = edge.get("id", "")
        if eid.startswith(":"):
            continue
        shape_str = edge.get("shape", "")
        if shape_str:
            for pt in shape_str.split():
                parts = pt.split(",")
                if len(parts) == 2:
                    ex, ey = float(parts[0]), float(parts[1])
                    d = (ex - sx) ** 2 + (ey - sy) ** 2
                    if d < best_dist:
                        best_dist = d
                        best_eid = eid
        else:
            # Check junctions
            from_j = edge.get("from", "")
            to_j = edge.get("to", "")
            for jid in (from_j, to_j):
                jn = tree.find(f".//junction[@id='{jid}']")
                if jn is not None:
                    jx = float(jn.get("x", 0))
                    jy = float(jn.get("y", 0))
                    d = (jx - sx) ** 2 + (jy - sy) ** 2
                    if d < best_dist:
                        best_dist = d
                        best_eid = eid

    return best_eid


def _find_sumo_bin() -> str:
    candidates = [
        os.path.expanduser("~/Library/Python/3.9/bin/sumo"),
        os.path.expanduser("~/Library/Python/3.10/bin/sumo"),
        os.path.expanduser("~/Library/Python/3.11/bin/sumo"),
    ]
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        candidates.insert(0, os.path.join(sumo_home, "bin", "sumo"))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return "sumo"


_SUMO = _find_sumo_bin()


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
    """Convert dispatch plan to SUMO trip file, let SUMO route internally.

    SUMO reads trip files with fromLonLat/toLonLat and computes
    routes on its full network, avoiding duarouter's coordinate
    snapping issues.

    Returns: (trip_path, route_path) — route_path may equal trip_path
             when SUMO uses the trip file directly.
    """
    os.makedirs(output_dir, exist_ok=True)
    demand_pts = [g for g in demand_gdf.geometry]

    trip_path = os.path.join(output_dir, "bus_trips.trips.xml")
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<routes>']
    lines.append(
        f'  <vType id="{bus_type}" accel="1.5" decel="4.0" '
        f'length="12.0" maxSpeed="13.89" guiShape="bus"/>'
    )

    # Pre-compute nearest SUMO edges for all locations
    edge_cache: dict[tuple[float, float], str] = {}

    def _get_edge(lon: float, lat: float) -> str:
        key = (round(lon, 6), round(lat, 6))
        if key not in edge_cache:
            edge_cache[key] = _find_nearest_edge(lon, lat, network_path)
        return edge_cache[key]

    for vid, route in dispatch_result.vehicle_routes.items():
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

        vehicle_depart = float(hash(vid) % 60)
        for round_num in range(max_rounds):
            for k in range(len(seq_pts) - 1):
                from_lon, from_lat = seq_pts[k]
                to_lon, to_lat = seq_pts[k + 1]
                from_edge = _get_edge(from_lon, from_lat)
                to_edge = _get_edge(to_lon, to_lat)
                if not from_edge or not to_edge:
                    continue
                trip_id = f"{vid}_r{round_num}_leg{k}"
                lines.append(
                    f'  <trip id="{trip_id}" type="{bus_type}" '
                    f'depart="{vehicle_depart:.1f}" '
                    f'from="{from_edge}" to="{to_edge}"/>'
                )
                vehicle_depart += 5.0
            vehicle_depart += round_trip_s

    lines.append("</routes>")
    with open(trip_path, "w") as f:
        f.write("\n".join(lines))

    # Return trip_path as the route file — SUMO can use trip files
    # directly when run with the trip file as --route-files
    return trip_path, trip_path


def compute_rounds_needed(dispatch_result, vehicles, demand_gdf) -> int:
    sub_qty = dispatch_result.sub_demand_quantities
    total_served = sum(
        sub_qty[i] for i in range(len(sub_qty))
        if i not in dispatch_result.unserved_demand
    )
    total_demand = sum(sub_qty)
    if total_served <= 0:
        return 1
    return max(1, min(int(total_demand / max(total_served, 1)) + 1, 3))
