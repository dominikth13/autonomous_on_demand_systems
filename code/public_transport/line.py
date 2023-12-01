from location.location import Location
from public_transport.station import Station

class Line:
    def __init__(self, stations: list[Station], name: str) -> None:
        self.stations = stations
        self.name = name
    
    def get_closest_station(self, location: Location) -> Station:
        closest: Station = None
        for station in self.stations:
            if closest == None or closest.position.distance_to(location) > station.position.distance_to(location):
                closest = station
        
        return closest
