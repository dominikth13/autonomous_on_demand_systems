from __future__ import annotations

# the inputs of lat & lon are in meter
class Location:
    def __init__(self, lat: float, lon: float):
        self.lat = lat
        self.lon = lon
    
    # define distance between two points in meter
    def distance_to(self, other: Location) -> float:
        return (111.3 * abs(self.lat - other.lat) + 71.5 * abs(self.lon - other.lon)) * 1000
