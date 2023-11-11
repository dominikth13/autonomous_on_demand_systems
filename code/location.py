from __future__ import annotations

class Location:
    def __init__(self, lat: float, lon: float):
        self.lat = lat
        self.lon = lon
    
    # define distance between two points as 1 kilometer
    def distance_to(self, other: Location) -> float:
        return abs(self.lat - other.lat) + abs(self.lon - other.lon)
