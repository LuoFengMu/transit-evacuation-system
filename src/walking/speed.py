"""Walking speed parameters."""
NORMAL_WALK_KMH = 5.0
CROWDED_WALK_KMH = 3.0
NORMAL_WALK_MS = NORMAL_WALK_KMH * 1000 / 3600   # 1.389 m/s
CROWDED_WALK_MS = CROWDED_WALK_KMH * 1000 / 3600  # 0.833 m/s

def walk_time_s(dist_m: float, crowded: bool = False) -> float:
    speed = CROWDED_WALK_MS if crowded else NORMAL_WALK_MS
    return dist_m / speed if speed > 0 else float("inf")
