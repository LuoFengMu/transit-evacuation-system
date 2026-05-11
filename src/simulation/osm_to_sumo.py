"""Convert OSM road network to SUMO .net.xml format.

Uses netconvert's native OSM parser for reliable conversion.
"""
import os
import subprocess


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
            "netconvert",
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


def download_xuzhou_osm(output_path: str) -> str:
    """Download Xuzhou OSM data via OSMnx and save as .osm.xml."""
    import osmnx as ox
    ox.settings.all_oneway = True
    G = ox.graph_from_place("Xuzhou, Jiangsu, China", network_type="drive")
    ox.save_graph_xml(G, filepath=output_path)
    return output_path
