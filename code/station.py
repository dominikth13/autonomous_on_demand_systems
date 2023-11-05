from location import Location
from utils import IdProvider

ID_PROVIDER = IdProvider()

class Station:
    def __init__(self, position: Location) -> None:
        self.id = ID_PROVIDER.get_id()
        self.position = position

# Apply shortest path algorithm to station network
class FastestStationConnectionNetwork:
    def __init__(self, connections: list[tuple[Station, float, Station]]) -> None:
        from model_builder import solve_all_pair_shortest_path_problem

        self.connection_network = solve_all_pair_shortest_path_problem(connections)

    # Returns: tuple[List of stations, transit time]
    def get_fastest_connection(self, start: Station, end: Station) -> tuple[list[Station], float]:
        return self.connection_network[start.id][end.id]

# TODO add stations and connections
_stations = []
_connections = []
FASTEST_STATION_CONNECTION_NETWORK: FastestStationConnectionNetwork = FastestStationConnectionNetwork(_connections)