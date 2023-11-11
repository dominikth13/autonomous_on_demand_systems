from location import Location
from utils import IdProvider
import csv

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

        station_set = set()
        for connection in self.connection_network:
            station_set.add(connection[0])
            station_set.add(connection[2])

        self.stations = sorted(station_set, lambda x: x.id)

    # Returns: tuple[List of stations, transit time]
    def get_fastest_connection(self, start: Station, end: Station) -> tuple[list[Station], float]:
        return self.connection_network[start.id][end.id]

##############################################################################################################

# _stations = []
# Pfad zur CSV-Datei
stations_csv_file_path = 'stations.csv'

# Erstellung der _stations Liste durch Einlesen der CSV-Datei
_stations = []
with open(stations_csv_file_path, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        latitude = int(row['X'])
        longitude = int(row['Y'])
        _stations.append(Station(position=Location(lat=latitude, lon=longitude)))

# _connections = []

connections_csv_file_path = 'connections.csv'
_connections = []
with open(connections_csv_file_path, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        from_station_id = int(row['FromStationID'])
        to_station_id = int(row['ToStationID'])
        distance = float(row['Distance'])
        
        # Finden der Stationen in der _stations Liste anhand ihrer ID
        from_station = next((s for s in _stations if s.id == from_station_id), None)
        to_station = next((s for s in _stations if s.id == to_station_id), None)
        
        if from_station is not None and to_station is not None:
            # Hinzufügen der Verbindung zur Liste
            _connections.append((from_station, distance, to_station))


fastest_connection_network: FastestStationConnectionNetwork

def FASTEST_CONNECTION_NETWORK() -> FastestStationConnectionNetwork:
    if not fastest_connection_network:
        fastest_connection_network = FastestStationConnectionNetwork(_connections)
    return fastest_connection_network