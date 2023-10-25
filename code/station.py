from location import Location


class Station:
    def __init__(self, id: int, position: Location) -> None:
        self.id = id
        self.position = position

# Apply shortest path algorithm to station network
class FastestStationConnectionNetwork:
    def __init__(self, stations: list[Station]) -> None:
        pass

    # Returns: tuple[List of stations, transit time]
    def get_fastest_connection(start: Station, end: Station) -> tuple[list[Station], float]:
        pass