from __future__ import annotations

class Grid:
    def __init__(self, min_lat: float, min_lon: float, max_lat: float, max_lon: float, step_distance: float):
        self.coordinates = [[Location(lat, lon) for lon in range(min_lon, max_lon, step_distance)] for lat in range(min_lat, max_lat, step_distance)]

class Location:
    def __init__(self, lat: float, lon: float):
        self.lat = lat
        self.lon = lon
    
    def distance_to(self, other: Location) -> float:
        return abs(self.lat - other.lat) + abs(self.lon - other.lon)
