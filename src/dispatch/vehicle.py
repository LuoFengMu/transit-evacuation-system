"""Bus vehicle and depot data management."""
from dataclasses import dataclass
from typing import Optional
import geopandas as gpd
import numpy as np
from shapely.geometry import Point


@dataclass
class BusVehicle:
    vehicle_id: str
    depot_id: str
    capacity: int           # max passengers
    available_time: float = 0.0  # seconds from scenario start
    max_speed_kmh: float = 50.0
    current_status: str = "idle"

    def __hash__(self):
        return hash(self.vehicle_id)


@dataclass
class BusDepot:
    depot_id: str
    depot_name: str
    lon: float
    lat: float
    available_vehicle_count: int
    geometry: Point

    @property
    def location(self) -> Point:
        return self.geometry


def generate_sample_vehicles(
    depot: BusDepot,
    n: int = 5,
    capacity: int = 60,
    seed: int = 42,
) -> list[BusVehicle]:
    """Generate sample bus vehicles for a depot."""
    rng = np.random.default_rng(seed)
    vehicles = []
    for i in range(n):
        vehicles.append(BusVehicle(
            vehicle_id=f"{depot.depot_id}_bus_{i:02d}",
            depot_id=depot.depot_id,
            capacity=capacity + int(rng.integers(-10, 20)),
            available_time=rng.uniform(0, 300),
            max_speed_kmh=rng.uniform(40, 60),
        ))
    return vehicles


def generate_sample_depots(
    bbox: tuple[float, float, float, float],
    n: int = 3,
    seed: int = 42,
) -> list[BusDepot]:
    """Generate sample depots within a bounding box.

    bbox: (min_lon, min_lat, max_lon, max_lat)
    """
    rng = np.random.default_rng(seed)
    depots = []
    names = ["徐州公交北站", "徐州公交东站", "徐州公交南站",
             "徐州公交西站", "徐州公交中心站"]
    for i in range(min(n, len(names))):
        lon = rng.uniform(bbox[0], bbox[2])
        lat = rng.uniform(bbox[1], bbox[3])
        depots.append(BusDepot(
            depot_id=f"depot_{i:02d}",
            depot_name=names[i],
            lon=lon,
            lat=lat,
            available_vehicle_count=rng.integers(10, 30),
            geometry=Point(lon, lat),
        ))
    return depots


def create_sample_data(
    bbox: tuple[float, float, float, float],
    n_depots: int = 3,
    buses_per_depot: int = 5,
    capacity_per_bus: int = 60,
    seed: int = 42,
) -> tuple[list[BusDepot], list[BusVehicle]]:
    """Create a complete sample dataset: depots + vehicles."""
    depots = generate_sample_depots(bbox, n_depots, seed)
    all_vehicles = []
    for i, depot in enumerate(depots):
        vehicles = generate_sample_vehicles(
            depot, buses_per_depot, capacity_per_bus, seed + i * 100,
        )
        all_vehicles.extend(vehicles)
    return depots, all_vehicles
