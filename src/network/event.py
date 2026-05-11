"""Emergency event definition and affected-area analysis."""
from dataclasses import dataclass, field
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd


@dataclass
class EmergencyEvent:
    event_id: str
    event_type: str          # flood, earthquake, fire, accident, crowd
    center: Point
    radius_m: float
    affected_road_ids: list[str] = field(default_factory=list)
    start_time: str = ""
    capacity_reduction_ratio: float = 0.5


def create_event_from_yaml(config: dict) -> EmergencyEvent:
    """Parse an emergency event from a scenario YAML dict."""
    ev = config.get("event", {})
    center = ev.get("center", [117.20, 34.27])
    return EmergencyEvent(
        event_id=config.get("scenario_id", "event_001"),
        event_type=ev.get("type", "flood"),
        center=Point(center[0], center[1]),
        radius_m=ev.get("radius_m", 1000),
        affected_road_ids=ev.get("blocked_roads", []),
        start_time=ev.get("start_time", ""),
        capacity_reduction_ratio=ev.get("capacity_reduction_ratio", 0.5),
    )


def get_affected_roads(event: EmergencyEvent, edges_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return the subset of road edges that intersect the event radius.

    If specific road IDs are given in the event, filter by ID.
    Otherwise, filter by spatial distance from the event center.
    """
    if event.affected_road_ids:
        mask = pd.Series(False, index=edges_gdf.index)
        for col in ["osmid", "road_id", "name"]:
            if col in edges_gdf.columns:
                mask = mask | edges_gdf[col].astype(str).isin(event.affected_road_ids)
        if mask.any():
            return edges_gdf[mask].copy()
        return edges_gdf.head(0)

    # Spatial filter: roads whose midpoint is within the event radius
    event_buffer = event.center.buffer(event.radius_m / 111320)  # approximate degrees
    edges_gdf = edges_gdf.copy()
    edges_gdf["midpoint"] = edges_gdf.geometry.apply(
        lambda g: g.interpolate(0.5, normalized=True)
    )
    mask = edges_gdf["midpoint"].within(event_buffer)
    return edges_gdf[mask].drop(columns=["midpoint"]).copy()
