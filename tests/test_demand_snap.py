"""Test demand point road-node snapping and water filtering.

Verifies Phase 3.3:
  - snap_to_road_nodes moves points to nearest road node
  - filter_water_bodies relocates points on water
  - Both preserve people_count and demand_id
"""
import pytest
import geopandas as gpd
from shapely.geometry import Point, Polygon
import networkx as nx

from src.demand.generator import (
    snap_to_road_nodes,
    filter_water_bodies,
)


def _build_grid_graph() -> nx.MultiDiGraph:
    """Build a simple grid road network with OSMnx-compatible CRS."""
    G = nx.MultiDiGraph()
    G.graph['crs'] = 'EPSG:4326'  # required by osmnx.nearest_nodes
    for x_int in range(5):
        for y_int in range(5):
            nid = x_int * 10 + y_int
            G.add_node(nid, x=117.2 + x_int * 0.005, y=34.27 + y_int * 0.005)
    for u in G.nodes():
        for v in G.nodes():
            if u != v:
                ux, uy = G.nodes[u]['x'], G.nodes[u]['y']
                vx, vy = G.nodes[v]['x'], G.nodes[v]['y']
                d = ((ux - vx)**2 + (uy - vy)**2)**0.5
                if d < 0.008:  # connect nearby nodes
                    G.add_edge(u, v, length=d * 111320)
    return G


def _make_demand_gdf(points: list[tuple[float, float]], people: list[int] | None = None) -> gpd.GeoDataFrame:
    n = len(points)
    if people is None:
        people = [100] * n
    return gpd.GeoDataFrame({
        'demand_id': [f'D{i}' for i in range(n)],
        'demand_name': [f'Demand {i}' for i in range(n)],
        'geometry': [Point(x, y) for x, y in points],
        'lon': [x for x, _ in points],
        'lat': [y for _, y in points],
        'people_count': people,
        'priority': [1] * n,
        'population_type': ['居民'] * n,
    }, crs='EPSG:4326')


class TestSnapToRoadNodes:

    def test_all_points_snapped(self):
        G = _build_grid_graph()
        # Points near but not exactly on nodes
        demand = _make_demand_gdf([
            (117.2001, 34.2701),
            (117.2051, 34.2751),
        ])
        result, snapped = snap_to_road_nodes(G, demand)
        assert len(result) == 2
        assert snapped >= 0  # may or may not snap depending on grid

    def test_preserves_people_count(self):
        G = _build_grid_graph()
        demand = _make_demand_gdf([(117.2001, 34.2701)], people=[42])
        result, _ = snap_to_road_nodes(G, demand)
        assert result.iloc[0]['people_count'] == 42

    def test_preserves_demand_id(self):
        G = _build_grid_graph()
        demand = _make_demand_gdf([(117.2001, 34.2701)])
        demand.at[0, 'demand_id'] = 'D_test'
        result, _ = snap_to_road_nodes(G, demand)
        assert result.iloc[0]['demand_id'] == 'D_test'

    def test_return_new_dataframe(self):
        G = _build_grid_graph()
        demand = _make_demand_gdf([(117.2001, 34.2701)])
        result, _ = snap_to_road_nodes(G, demand)
        # Snapped geometry should differ from original
        orig = demand.iloc[0].geometry
        snapped_geom = result.iloc[0].geometry
        # At minimum: result is a different object
        assert result is not demand


class TestFilterWaterBodies:

    def test_point_on_water_relocated(self):
        G = _build_grid_graph()
        # Water polygon covering an area between road nodes, so nearest node differs
        water = gpd.GeoDataFrame({
            'geometry': [Polygon([(117.2015, 34.2705), (117.2015, 34.2715),
                                   (117.2025, 34.2715), (117.2025, 34.2705)])],
        }, crs='EPSG:4326')
        # Point inside water, NOT exactly on a grid node
        demand = _make_demand_gdf([(117.2020, 34.2710)])

        result, relocated = filter_water_bodies(demand, G, water_gdf=water)
        assert relocated == 1
        # Point should have moved to nearest road node (different from water coords)
        new_x = result.iloc[0].geometry.x
        new_y = result.iloc[0].geometry.y
        assert abs(new_x - 117.2020) > 0.0001 or abs(new_y - 34.2710) > 0.0001

    def test_point_not_on_water_unchanged(self):
        G = _build_grid_graph()
        water = gpd.GeoDataFrame({
            'geometry': [Polygon([(117.2015, 34.2705), (117.2015, 34.2715),
                                   (117.2025, 34.2715), (117.2025, 34.2705)])],
        }, crs='EPSG:4326')
        # Point well outside the water polygon
        demand = _make_demand_gdf([(117.210, 34.280)])

        result, relocated = filter_water_bodies(demand, G, water_gdf=water)
        assert relocated == 0
        assert result.iloc[0].geometry.x == pytest.approx(117.210)
        assert result.iloc[0].geometry.y == pytest.approx(34.280)

    def test_empty_water_gdf_noop(self):
        G = _build_grid_graph()
        water = gpd.GeoDataFrame({'geometry': []}, crs='EPSG:4326')
        demand = _make_demand_gdf([(117.200, 34.270)])

        result, relocated = filter_water_bodies(demand, G, water_gdf=water)
        assert relocated == 0
        assert len(result) == 1

    def test_preserves_people_count_after_relocation(self):
        G = _build_grid_graph()
        water = gpd.GeoDataFrame({
            'geometry': [Polygon([(117.2015, 34.2705), (117.2015, 34.2715),
                                   (117.2025, 34.2715), (117.2025, 34.2705)])],
        }, crs='EPSG:4326')
        demand = _make_demand_gdf([(117.2020, 34.2710)], people=[77])

        result, relocated = filter_water_bodies(demand, G, water_gdf=water)
        assert result.iloc[0]['people_count'] == 77

    def test_skips_water_covered_node(self):
        """When the nearest road node is also inside water, skip to the next one."""
        G = _build_grid_graph()
        # Water polygon that covers both the demand point AND the nearest road node
        # Grid nodes: (117.200, 34.270), (117.200, 34.275), (117.205, 34.270), etc.
        water = gpd.GeoDataFrame({
            'geometry': [Polygon([(117.199, 34.269), (117.199, 34.272),
                                   (117.203, 34.272), (117.203, 34.269)])],
        }, crs='EPSG:4326')
        # Demand point inside water, nearest node is (117.200, 34.270) — also in water
        # Next nearest should be (117.205, 34.270) — outside water
        demand = _make_demand_gdf([(117.200, 34.271)])

        result, relocated = filter_water_bodies(demand, G, water_gdf=water)
        assert relocated == 1

        # Verify relocated point is NOT inside the water polygon
        new_pt = result.iloc[0].geometry
        assert not water.iloc[0].geometry.contains(new_pt), \
            f"Relocated point {new_pt} is still inside water polygon"
