import random
from location import Location
from utils import IdProvider

ID_PROVIDER = IdProvider()

class Station:
    def __init__(self, position: Location) -> None:
        self.id = ID_PROVIDER.get_id()
        self.position = position

# Apply shortest path algorithm to station network
class FastestStationConnectionNetwork:
    def __init__(self, stations: list[Station]) -> None:
        self.stations = sorted(stations, key=lambda x: x.id)
        self.connection_network = {stations[i]: {stations[j]: ([], random.Random(i+j).random() * 20) for j in range(i+1, len(stations))} for i in range(len(stations))}

    # Returns: tuple[List of stations, transit time]
    def get_fastest_connection(self, start: Station, end: Station) -> tuple[list[Station], float]:
        if start.id <= end.id:
            return self.connection_network[start][end]
        else:
            return self.connection_network[end][start]