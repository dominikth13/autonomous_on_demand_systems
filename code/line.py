from station import Station

class Line:
    def __init__(self, stations: list[Station], name: str) -> None:
        self.stations = stations
        self.name = name