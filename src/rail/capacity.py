"""Rail station capacity model for v0.4.0.

Models:
  K_s = K_hall + K_platform        (static capacity, people)
  Q_s = min(Q_entry, Q_gate, Q_plat, Q_line)  (dynamic capacity, pax/h)
  Q_line = train_capacity × frequency         (line throughput, pax/h)
  P_s = A_s(Δt) / C_s(Δt)                    (station pressure, unitless)
"""
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import geopandas as gpd


@dataclass
class RailStation:
    station_id: str
    station_name: str
    line_id: str
    lon: float
    lat: float
    static_capacity: int         # K_s: max people in station
    dynamic_capacity_pax_h: int  # Q_s: bottleneck throughput
    line_rate_pax_h: int         # Q_line: line evacuation capacity
    is_transfer: bool = False
    transfer_lines: str = ""


@dataclass
class RailLine:
    line_id: str
    line_name: str
    train_capacity: int          # people per train
    emergency_freq_per_h: int    # trains per hour in emergency
    line_capacity_pax_h: int     # = train_capacity × emergency_freq


@dataclass
class StationPressure:
    station_id: str
    station_name: str
    pressure: float              # P_s
    arrivals: int                # A_s: allocated people
    capacity_used: int           # people that can be processed
    level: str                   # normal/saturated/overloaded/severe


def load_stations(path: str) -> list[RailStation]:
    gdf = gpd.read_file(path)
    stations = []
    for _, r in gdf.iterrows():
        stations.append(RailStation(
            station_id=r["station_id"], station_name=r["station_name"],
            line_id=r["line_id"], lon=r["lon"], lat=r["lat"],
            static_capacity=int(r["static_capacity"]),
            dynamic_capacity_pax_h=int(r["dynamic_capacity_pax_h"]),
            line_rate_pax_h=int(r["line_rate_pax_h"]),
            is_transfer=bool(r.get("is_transfer", False)),
            transfer_lines=str(r.get("transfer_lines", "")),
        ))
    return stations


def load_lines(path: str) -> list[RailLine]:
    df = pd.read_csv(path)
    lines = []
    for _, r in df.iterrows():
        lines.append(RailLine(
            line_id=r["line_id"], line_name=r["line_name"],
            train_capacity=int(r["train_capacity"]),
            emergency_freq_per_h=int(r["emergency_freq_per_h"]),
            line_capacity_pax_h=int(r["line_capacity_emergency_pax_h"]),
        ))
    return lines


def compute_pressure(
    stations: list[RailStation],
    allocations: dict[str, int],  # station_id → people allocated
    time_window_h: float = 1.0,
    background_flow: Optional[dict[str, int]] = None,
) -> list[StationPressure]:
    """Compute pressure for each station given allocation.

    P_s = (A_s + B_s) / (Q_s × Δt)

    Pressure levels:
      ≤ 0.8  → normal
      ≤ 1.0  → saturated
      ≤ 1.2  → overloaded
      > 1.2  → severe
    """
    results = []
    for s in stations:
        arrivals = allocations.get(s.station_id, 0)
        if background_flow:
            arrivals += background_flow.get(s.station_id, 0)
        capacity_used = int(s.dynamic_capacity_pax_h * time_window_h)
        if capacity_used > 0:
            pressure = arrivals / capacity_used
        else:
            pressure = float("inf")

        if pressure <= 0.8:
            level = "normal"
        elif pressure <= 1.0:
            level = "saturated"
        elif pressure <= 1.2:
            level = "overloaded"
        else:
            level = "severe"

        results.append(StationPressure(
            station_id=s.station_id, station_name=s.station_name,
            pressure=round(pressure, 3), arrivals=arrivals,
            capacity_used=capacity_used, level=level,
        ))
    return results
