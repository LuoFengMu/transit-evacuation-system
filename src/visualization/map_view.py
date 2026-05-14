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


def _make_circle(cx: float, cy: float, radius_m: float, n: int = 48):
    """Generate lat/lon points for a circle of given radius (meters)."""
    import numpy as np
    angles = np.linspace(0, 2 * np.pi, n)
    lons = cx + (radius_m / 111320) * np.cos(angles)
    lats = cy + (radius_m / 111320) * np.sin(angles)
    return lons.tolist(), lats.tolist()


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
    rail_stations: Optional[list] = None,
    rail_pressures: Optional[list] = None,
    show_shelters: bool = True,
    walking_paths: Optional[list[dict]] = None,
) -> None:
    """Render the main evacuation map using Plotly + Streamlit.

    Tip: click legend items to toggle layer visibility.
    """
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

    # ── 3. Event center — dark red square, white ⚠ ──────────
    fig.add_trace(go.Scattermapbox(
        lon=[event.center.x], lat=[event.center.y],
        mode="markers+text",
        marker=dict(size=30, color="#c0392b", opacity=1.0),
        text=["⚠"],
        textposition="middle center",
        textfont=dict(size=22, color="white"),
        name="事件中心",
        hovertemplate=f"<b>大客流</b><br>聚集半径: {event.radius_m}m<extra></extra>",
    ))

    # Event influence circle (semi-transparent)
    circle_lons, circle_lats = _make_circle(event.center.x, event.center.y, event.radius_m)
    fig.add_trace(go.Scattermapbox(
        lon=circle_lons, lat=circle_lats,
        mode="lines",
        line=dict(width=2, color="#c0392b"),
        fill="toself",
        fillcolor="rgba(231,76,60,0.08)",
        name="聚集范围",
        hoverinfo="skip",
        showlegend=False,
    ))

    # ── 4. Shelters — green squares (only if enabled) ────────
    if show_shelters:
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

    # ── 6. Pedestrian paths — orange, visible ──────────────────
    if paths:
        for i, p in enumerate(paths):
            if i >= 30:
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
                    line=dict(width=2.5, color="#e67e22"),
                    opacity=0.75,
                    name="步行接入估计线",
                    legendgroup="pedestrian",
                    text=label,
                    hovertemplate=(
                        f"<b>{label}</b><br>"
                        f"距离: {dist_km:.1f} km<br>"
                        f"耗时: {time_min:.1f} min<extra></extra>"
                    ),
                    showlegend=(i == 0),
                ))

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

    # ── 7b. Rail stations — metro icons ────────────────────────
    if rail_stations:
        r_lons = [s.lon for s in rail_stations]
        r_lats = [s.lat for s in rail_stations]
        r_names = [s.station_name for s in rail_stations]
        # Color by pressure if available
        if rail_pressures:
            press_map = {p.station_id: p for p in rail_pressures}
            r_colors = []
            r_labels = []
            for s in rail_stations:
                p = press_map.get(s.station_id)
                if p and p.level == "severe":
                    r_colors.append("#e74c3c")
                    r_labels.append(f"{s.station_name} (严重过载)")
                elif p and p.level == "overloaded":
                    r_colors.append("#e67e22")
                    r_labels.append(f"{s.station_name} (过载)")
                elif p and p.level == "saturated":
                    r_colors.append("#f39c12")
                    r_labels.append(f"{s.station_name} (饱和)")
                elif p and p.arrivals > 0:
                    r_colors.append("#27ae60")
                    r_labels.append(f"{s.station_name} (正常)")
                else:
                    r_colors.append("#3498db")
                    r_labels.append(s.station_name)
        else:
            r_colors = ["#3498db"] * len(rail_stations)
            r_labels = r_names

        fig.add_trace(go.Scattermapbox(
            lon=r_lons, lat=r_lats,
            mode="markers+text",
            marker=dict(size=16, color=r_colors, symbol="marker", opacity=0.9),
            text=r_names,
            textposition="top center",
            textfont=dict(size=9, color="#2c3e50"),
            name=f"轨道站 ({len(rail_stations)})",
            hovertemplate="<b>%{text}</b><extra></extra>",
            customdata=r_labels,
        ))

    # ── 7c. Walking paths — gray dashed ─────────────────────
    if walking_paths:
        for i, wp in enumerate(walking_paths):
            if i >= 30:
                break
            coords = wp.get("coords", [])
            if len(coords) < 2:
                continue
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            fig.add_trace(go.Scattermapbox(
                lon=lons, lat=lats,
                mode="lines",
                line=dict(width=1.5, color="#95a5a6"),
                opacity=0.45,
                name="OD连接 (需求→轨道站)",
                hoverinfo="skip",
                showlegend=(i == 0),
            ))

    # ── 8. Bus routes — per-vehicle ──
    if bus_routes:
        is_sumo = all("n_stops" not in br for br in bus_routes)
        route_label = "SUMO 公交轨迹" if is_sumo else "公交行驶路线"
        colors_10 = ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6",
                     "#1abc9c","#e67e22","#2980b9","#27ae60","#8e44ad"]

        for i, br in enumerate(bus_routes):
            if i >= 30:
                break
            coords = br.get("coords", [])
            if len(coords) < 2:
                continue
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            vid = br.get("vehicle_id", "")
            n_stops = br.get("n_stops", 0)
            vcolor = colors_10[i % len(colors_10)] if is_sumo else "#2980b9"
            hover_text = f"<b>{vid}</b>"
            if is_sumo:
                hover_text += f"<br>出发: {br.get('depart', 0):.0f}s<br>到达: {br.get('arrival', 0):.0f}s"
            else:
                hover_text += f"<br>停靠: {n_stops} 个需求点"
            fig.add_trace(go.Scattermapbox(
                lon=lons, lat=lats,
                mode="lines",
                line=dict(width=5, color=vcolor),
                name=route_label,
                legendgroup="bus",
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                showlegend=(i == 0),
            ))

    # ── Layout ────────────────────────────────────────────────
    if demand_lats:
        if show_shelters and shelter_lats:
            center_lat = (sum(demand_lats) / len(demand_lats) + sum(shelter_lats) / len(shelter_lats)) / 2
            center_lon = (sum(demand_lons) / len(demand_lons) + sum(shelter_lons) / len(shelter_lons)) / 2
        elif rail_stations:
            center_lat = (sum(demand_lats) / len(demand_lats) + sum(r_lats) / len(r_lats)) / 2
            center_lon = (sum(demand_lons) / len(demand_lons) + sum(r_lons) / len(r_lons)) / 2
        else:
            center_lat = sum(demand_lats) / len(demand_lats)
            center_lon = sum(demand_lons) / len(demand_lons)
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
