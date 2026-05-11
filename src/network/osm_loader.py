"""Load and prepare OSM road network data."""
import osmnx as ox
import networkx as nx
import geopandas as gpd


def load_network_from_graphml(path: str) -> nx.MultiDiGraph:
    """Load a road network from a local GraphML file."""
    G = ox.load_graphml(path)
    return G


def load_network_from_place(place: str) -> nx.MultiDiGraph:
    """Download road network from OSM for a given place name."""
    G = ox.graph_from_place(place, network_type="drive", simplify=True)
    return G


def network_to_geodataframes(G: nx.MultiDiGraph) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Convert a NetworkX graph to node and edge GeoDataFrames.

    Both GeoDataFrames are returned with a flat integer index.
    """
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
    nodes_gdf = nodes_gdf.reset_index(drop=True)
    edges_gdf = edges_gdf.reset_index(drop=True)
    return nodes_gdf, edges_gdf


def get_network_bounds(G: nx.MultiDiGraph) -> tuple[float, float, float, float]:
    """Get the bounding box of the network. Returns (min_lon, min_lat, max_lon, max_lat)."""
    nodes_gdf, _ = ox.graph_to_gdfs(G)
    bounds = nodes_gdf.total_bounds  # [minx, miny, maxx, maxy]
    return bounds[0], bounds[1], bounds[2], bounds[3]
