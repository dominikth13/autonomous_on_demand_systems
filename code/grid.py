from location import Location
from utils import IdProvider

ID_PROVIDER = IdProvider()

# We have the problem that zones not all the time match some straight lines
# We define our zones as sets of smaller squares where it is super easy to
# find the fitting square in a coordinate system (2-layer binary search)
# Initial setup with some api (google maps)
class Zone:
    def __init__(self, name: str) -> None:
        self.id = ID_PROVIDER.get_id()
        self.name = name


def to_zone(lat: float, lon: float):
    # Do some dark magic to find zone of location
    # Be aware that object references are still different
    return Zone("")


class Grid:
    def __init__(
        self,
        zones: list[Zone],
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
        step_distance: float,
    ):
        self.zones_dict = {zone.name: zone for zone in zones}
        self.cells = [
            [
                GridCell(Location(lat, lon), to_zone(lat, lon))
                for lon in range(min_lon, max_lon, step_distance)
            ]
            for lat in range(min_lat, max_lat, step_distance)
        ]

    def find_zone(self, location: Location) -> Zone:
        low = 0
        high = len(self.cells) - 1
        mid = 0

        first_selection = []
        # Use binary search for lat
        while low <= high:
            mid = (high + low) // 2

            if self.cells[mid][0].center.lat < location.lat:
                if self.cells[mid + 1][0].center.lat > location.lat:
                    first_selection = (
                        self.cells[mid]
                        if abs(self.cells[mid][0].center.lat - location.lat)
                        <= abs(self.cells[mid + 1][0].center.lat - location.lat)
                        else self.cells[mid + 1]
                    )
                    break
                else:
                    low = mid + 1
            elif self.cells[mid][0].center.lat > location.lat:
                if self.cells[mid - 1][0].center.lat < location.lat:
                    first_selection = (
                        self.cells[mid]
                        if abs(self.cells[mid][0].center.lat - location.lat)
                        <= abs(self.cells[mid - 1][0].center.lat - location.lat)
                        else self.cells[mid + 1]
                    )
                    break
                else:
                    high = mid - 1

        if len(first_selection) == 0:
            raise Exception(f"Latitute {location.lat} not in range")

        low = 0
        high = len(first_selection) - 1
        mid = 0

        final_cell = None
        # Use binary search for lon
        while low <= high:
            mid = (high + low) // 2

            if first_selection[mid].center.lon < location.lon:
                if first_selection[mid + 1].center.lon > location.lon:
                    final_cell = (
                        first_selection[mid]
                        if abs(first_selection[mid].center.lon - location.lon)
                        <= abs(first_selection[mid + 1].center.lon - location.lon)
                        else first_selection[mid + 1]
                    )
                    break
                else:
                    low = mid + 1
            elif first_selection[mid].center.lon > location.lon:
                if first_selection[mid - 1].center.lon < location.lon:
                    final_cell = (
                        first_selection[mid]
                        if abs(first_selection[mid].center.lon - location.lon)
                        <= abs(first_selection[mid - 1].center.lon - location.lon)
                        else first_selection[mid - 1]
                    )
                    break
                else:
                    high = mid - 1

        if final_cell == None:
            raise Exception(f"Longitude {location.lon} not in range")

        return final_cell.zone


class GridCell:
    def __init__(self, center: Location, zone: Zone) -> None:
        self.center = center
        self.zone = zone