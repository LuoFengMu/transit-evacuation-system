"""Convert OSM road network to SUMO .net.xml format."""
import os
import subprocess


def _find_netconvert() -> str:
    candidates = [
        os.path.expanduser("~/Library/Python/3.9/bin/netconvert"),
        os.path.expanduser("~/Library/Python/3.10/bin/netconvert"),
        os.path.expanduser("~/Library/Python/3.11/bin/netconvert"),
    ]
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        candidates.insert(0, os.path.join(sumo_home, "bin", "netconvert"))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return "netconvert"


_NETCONVERT = _find_netconvert()


def build_sumo_network_from_osm(
    osm_path: str,
    output_dir: str,
    network_name: str = "xuzhou_network",
) -> str:
    """Convert an OSM XML file to a SUMO .net.xml file using netconvert.

    Args:
        osm_path: Path to .osm.xml file (from OSMnx save_graph_xml).
        output_dir: Directory to write the output.
        network_name: Base name for the output file.

    Returns:
        Path to the generated .net.xml file.
    """
    os.makedirs(output_dir, exist_ok=True)
    net_path = os.path.join(output_dir, f"{network_name}.net.xml")

    result = subprocess.run(
        [
            _NETCONVERT,
            "--osm-files", osm_path,
            "--output-file", net_path,
            "--geometry.remove",
            "--roundabouts.guess",
            "--ramps.guess",
            "--junctions.join",
            "--tls.guess-signals",
            "--tls.discard-simple",
            "--remove-edges.isolated",
        ],
        capture_output=True, text=True, timeout=600,
    )

    if not os.path.exists(net_path):
        raise RuntimeError(f"netconvert failed (exit {result.returncode}): {result.stderr}")

    return net_path


def crop_network(
    input_net: str,
    output_net: str,
    center_lon: float,
    center_lat: float,
    radius_m: float = 8000,
) -> str:
    """Crop a SUMO network to a bounding box around a center point.

    Uses netconvert --boundary to keep only edges within the box.
    Much faster simulation on the cropped sub-network.
    """
    import subprocess, os
    os.makedirs(os.path.dirname(output_net), exist_ok=True)

    # Convert meters to degrees (approximate)
    dlon = radius_m / 111320
    dlat = radius_m / 111320
    xmin = center_lon - dlon
    xmax = center_lon + dlon
    ymin = center_lat - dlat
    ymax = center_lat + dlat

    result = subprocess.run(
        [
            _NETCONVERT,
            "--sumo-net-file", input_net,
            "--output-file", output_net,
            "--boundary", f"{xmin},{ymin},{xmax},{ymax}",
            "--no-internal-links",
            "--ignore-errors", "--no-warnings",
        ],
        capture_output=True, text=True, timeout=120,
    )

    if not os.path.exists(output_net):
        raise RuntimeError(f"Network crop failed: {result.stderr}")

    return output_net


def download_xuzhou_osm(output_path: str) -> str:
    """Download Xuzhou OSM data via OSMnx and save as .osm.xml."""
    import osmnx as ox
    ox.settings.all_oneway = True
    G = ox.graph_from_place("Xuzhou, Jiangsu, China", network_type="drive")
    ox.save_graph_xml(G, filepath=output_path)
    return output_path
