from __future__ import annotations
from program_params import DISCOUNT_RATE, LEARNING_RATE
from location import Location


class Time:
    def __init__(self, hour: int, minute: int) -> None:
        self.hour = hour
        self.minute = minute

    def distance_to(self, other: Time) -> int:
        return abs(60 * (self.hour - other.hour)) + abs(self.minute - other.minute)

    def is_before(self, other: Time) -> bool:
        return self.hour <= other.hour or (
            self.hour == other.hour and self.minute <= other.minute
        )

    def is_after(self, other: Time) -> bool:
        return self.hour >= other.hour or (
            self.hour == other.hour and self.minute >= other.minute
        )


class GridInterval:
    def __init__(self, start: Time, end: Time) -> None:
        self.start = start
        self.end = end


class TimeSeries:
    def __init__(self, start: Time, end: Time, intervalLength: int) -> None:
        self.intervals = [
            GridInterval(start, start + intervalLength - 1)
            for start in range(start, end, intervalLength)
        ]

    def find_interval(self, time: Time) -> GridInterval:
        low = 0
        high = len(self.intervals) - 1
        mid = 0

        interval = None
        while low <= high:
            mid = (high + low) // 2

            if self.intervals[mid].start.is_before(time):
                if self.intervals[mid].end.is_after(time):
                    interval = self.intervals[mid]
                    break
                else:
                    low = mid + 1
            elif self.intervals[mid].end.is_before(time):
                low = mid + 1
            else:
                high = mid - 1

        if interval == None:
            raise Exception(f"Interval to time {time} not found")

        return interval


# We have the problem that zones not all the time match some straight lines
# We define our zones as sets of smaller squares where it is super easy to
# find the fitting square in a coordinate system (2-layer binary search)
# Initial setup with some api (google maps)
class Zone:
    def __init__(self, name: str) -> None:
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
        self.zones = zones
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


class StateValueTable:
    def __init__(self, grid: Grid, time_series: TimeSeries) -> None:
        self.value_grid = {
            interval: {zone: 0 for zone in grid.zones}
            for interval in time_series.intervals
        }
        self.grid = grid
        self.time_series = time_series

    def adjst_state_value(
        self,
        time: Time,
        location: Location,
        n_time: Time,
        n_location: Location,
        reward: float,
    ) -> None:
        # 1. find current state
        c_zone = self.grid.find_zone(location)
        c_interval = self.time_series.find_interval(time)

        # 2. find next state
        n_zone = self.grid.find_zone(n_location)
        n_interval = self.time_series.find_interval(n_time)

        self.value_grid[c_interval][c_zone] = self.value_grid[c_interval][
            c_zone
        ] + LEARNING_RATE * (
            reward
            + DISCOUNT_RATE * self.value_grid[n_interval][n_zone]
            - self.value_grid[c_interval][c_zone]
        )
