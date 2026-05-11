"""Streamlit interactive map rendering for evacuation visualization."""
from typing import Optional
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from shapely.geometry import Point

from src.network.event import EmergencyEvent
from src.network.pathfinder import PathResult


def _color_for_type(event_type: str) -> str:
    colors = {
        "flood": "#3498db", "earthquake": "#e74c3c", "fire": "#e67e22",
        "accident": "#f1c40f", "crowd": "#9b59b6",
    }
    return colors.get(event_type, "#e74c3c")


def _edges_to_single_trace(edges_gdf: gpd.GeoDataFrame, color: str, width: int,
                           name: str, sample_n: int = 200) -> go.Scattermapbox:
    """Combine multiple LineStrings into one trace using None separators.

    This avoids creating hundreds of individual traces, making the map
    dramatically faster to render.
    """
    all_lons: list[float] = []
    all_lats: list[float] = []
    edges_sample = (edges_gdf.sample(n=sample_n, random_state=42)
                    if len(edges_gdf) > sample_n else edges_gdf)

    for _, edge in edges_sample.iterrows():
        geom = edge.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            coords = list(geom.coords)
        elif geom.geom_type == "MultiLineString":
            coords = []
            for line in geom.geoms:
                coords.extend(list(line.coords))
        else:
            continue
        for c in coords:
            all_lons.append(c[0])
            all_lats.append(c[1])
        all_lons.append(None)  # break between lines
        all_lats.append(None)

    return go.Scattermapbox(
        lon=all_lons, lat=all_lats,
        mode="lines",
        line=dict(width=width, color=color),
        name=name,
        hoverinfo="skip",
        showlegend=True,
    )


def render_map(
    edges_gdf: gpd.GeoDataFrame,
    event: EmergencyEvent,
    demand_gdf: gpd.GeoDataFrame,
    shelters_gdf: gpd.GeoDataFrame,
    paths: Optional[list[PathResult]] = None,
    affected_roads: Optional[gpd.GeoDataFrame] = None,
    bus_routes: Optional[list[dict]] = None,
    depot_locations: Optional[list[Point]] = None,
) -> None:
    """Render the main evacuation map using Plotly + Streamlit."""
    fig = go.Figure()

    # ── 1. Road network — single trace, 200 edges max ──────────
    fig.add_trace(_edges_to_single_trace(
        edges_gdf, color="#bdc3c7", width=1, name="道路网络", sample_n=200,
    ))

    # ── 2. Affected roads — single trace ──────────────────────
    if affected_roads is not None and len(affected_roads) > 0:
        fig.add_trace(_edges_to_single_trace(
            affected_roads, color="#e74c3c", width=4, name="封闭道路", sample_n=500,
        ))

    # ── 3. Event center ───────────────────────────────────────
    ev_color = _color_for_type(event.event_type)
    fig.add_trace(go.Scattermapbox(
        lon=[event.center.x], lat=[event.center.y],
        mode="markers+text",
        marker=dict(size=20, color=ev_color, opacity=0.8),
        text=["⚠"],
        textposition="middle center",
        textfont=dict(size=16, color="white"),
        name=f"事件: {event.event_type}",
        hovertemplate=f"<b>{event.event_type}</b><br>半径: {event.radius_m}m<extra></extra>",
    ))

    # ── 4. Shelters — green squares, larger ──────────────────
    shelter_lons = [g.x for g in shelters_gdf.geometry]
    shelter_lats = [g.y for g in shelters_gdf.geometry]
    shelter_names = shelters_gdf.get("shelter_name", "")
    shelter_caps = shelters_gdf.get("capacity", 0)
    fig.add_trace(go.Scattermapbox(
        lon=shelter_lons, lat=shelter_lats,
        mode="markers",
        marker=dict(size=14, color="#27ae60", opacity=0.85),
        name=f"避难点 ({len(shelters_gdf)})",
        text=[f"<b>{n}</b><br>容量: {c:,}人" for n, c in zip(shelter_names, shelter_caps)],
        hovertemplate="%{text}<extra></extra>",
    ))

    # ── 5. Demand points — red circles, size ∝ people ────────
    demand_lons = [g.x for g in demand_gdf.geometry]
    demand_lats = [g.y for g in demand_gdf.geometry]
    demand_names = demand_gdf.get("demand_name", "")
    demand_people = demand_gdf["people_count"]
    # Scale marker sizes: 200 people → size 8, 800 people → size 18
    demand_sizes = (demand_people / 40).clip(lower=6, upper=22)
    fig.add_trace(go.Scattermapbox(
        lon=demand_lons, lat=demand_lats,
        mode="markers",
        marker=dict(
            size=demand_sizes.tolist(),
            color="#e74c3c",
            opacity=0.9,
        ),
        name=f"疏散需求点 ({len(demand_gdf)})",
        text=[f"<b>{n}</b><br>人数: {p:,}" for n, p in zip(demand_names, demand_people)],
        hovertemplate="%{text}<extra></extra>",
    ))

    # ── 6. Pedestrian evacuation paths — black, thick ─────────
    if paths:
        shown = 0
        for p in paths:
            if shown >= 30:
                break
            if p.path_geometry and p.path_geometry.geom_type == "LineString":
                coords = list(p.path_geometry.coords)
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                label = f"{p.origin_name} → {p.destination_name}"
                dist_km = p.distance_m / 1000
                time_min = p.travel_time_s / 60
                fig.add_trace(go.Scattermapbox(
                    lon=lons, lat=lats,
                    mode="lines",
                    line=dict(width=4, color="#000000"),
                    name="行人疏散路径",
                    text=label,
                    hovertemplate=(
                        f"<b>{label}</b><br>"
                        f"距离: {dist_km:.1f} km<br>"
                        f"耗时: {time_min:.1f} min<extra></extra>"
                    ),
                    showlegend=(shown == 0),
                ))
                shown += 1

    # ── 7. Bus depots (staging areas) ─────────────────────────
    if depot_locations:
        depot_lons = [p.x for p in depot_locations]
        depot_lats = [p.y for p in depot_locations]
        depot_names = ["集结区_东", "集结区_西", "集结区_南", "集结区_北"][:len(depot_locations)]
        fig.add_trace(go.Scattermapbox(
            lon=depot_lons, lat=depot_lats,
            mode="markers+text",
            marker=dict(size=22, color="#2980b9", opacity=0.9),
            text=depot_names,
            textposition="top center",
            textfont=dict(size=12, color="#2980b9"),
            name=f"公交集结区 ({len(depot_locations)})",
            hovertemplate="<b>%{text}</b><extra></extra>",
        ))

    # ── 8. Bus routes — blue dashed, per-vehicle ──────────────
    if bus_routes:
        shown = 0
        for br in bus_routes:
            if shown >= 30:
                break
            coords = br.get("coords", [])
            if len(coords) < 2:
                continue
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            vid = br.get("vehicle_id", "")
            n_stops = br.get("n_stops", 0)
            fig.add_trace(go.Scattermapbox(
                lon=lons, lat=lats,
                mode="lines+markers",
                line=dict(width=3, color="#2980b9"),
                marker=dict(size=5, color="#2980b9"),
                name="公交行驶路线",
                text=f"<b>{vid}</b><br>停靠: {n_stops} 个需求点",
                hovertemplate="%{text}<extra></extra>",
                showlegend=(shown == 0),
            ))
            shown += 1

    # ── Layout ────────────────────────────────────────────────
    if shelter_lats and demand_lats:
        center_lat = (sum(demand_lats) / len(demand_lats) + sum(shelter_lats) / len(shelter_lats)) / 2
        center_lon = (sum(demand_lons) / len(demand_lons) + sum(shelter_lons) / len(shelter_lons)) / 2
    else:
        center_lat, center_lon = 34.27, 117.20

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=11,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=650,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left", x=0,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_path_table(paths: list[PathResult]) -> None:
    """Display a table of evacuation path details."""
    if not paths:
        st.info("暂无路径数据")
        return

    rows = []
    for p in paths:
        rows.append({
            "需求点": p.origin_name or p.origin_id,
            "目的地": p.destination_name or p.destination_id,
            "距离 (km)": round(p.distance_m / 1000, 2),
            "耗时 (min)": round(p.travel_time_s / 60, 1),
            "状态": "有效" if p.node_path else "无路网路径",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_metrics(paths: list[PathResult], demand_gdf: gpd.GeoDataFrame) -> None:
    """Display summary metrics for the evacuation analysis."""
    col1, col2, col3, col4 = st.columns(4)

    total_people = int(demand_gdf["people_count"].sum())
    valid_paths = [p for p in paths if p.node_path]
    avg_dist = sum(p.distance_m for p in valid_paths) / len(valid_paths) if valid_paths else 0
    avg_time = sum(p.travel_time_s for p in valid_paths) / len(valid_paths) if valid_paths else 0
    max_dist = max((p.distance_m for p in valid_paths), default=0)

    col1.metric("需求点总数", len(demand_gdf))
    col2.metric("待疏散人数", f"{total_people:,}")
    col3.metric("平均疏散距离", f"{avg_dist / 1000:.2f} km")
    col4.metric("平均疏散时间", f"{avg_time / 60:.1f} min")
