from __future__ import annotations
from utils import IdProvider
from program_params import *
from location import Location

ID_PROVIDER = IdProvider()

class Time:
    def __init__(self, hour: int, minute: int) -> None:
        self.hour = hour
        self.minute = minute

    def of_total_minutes(minutes: int) -> Time:
        return Time(minutes // 60, minutes % 60)

    def distance_to(self, other: Time) -> int:
        return abs(60 * (self.hour - other.hour)) + abs(self.minute - other.minute)

    def distance_to_in_seconds(self, other: Time) -> int:
        return self.distance_to(other) * 60

    def add_minutes(self, minutes: int) -> None:
        if self.minute + minutes > 59:
            minutes_to_next_hour = 60 - self.minute
            self.minute = 0
            minutes -= minutes_to_next_hour

            while minutes > 59:
                self.hour += 1
                if self.hour == 24:
                    self.hour = 0
                minutes -= 60

        self.minute += minutes

    def is_before(self, other: Time) -> bool:
        return self.hour <= other.hour or (
            self.hour == other.hour and self.minute <= other.minute
        )

    def is_after(self, other: Time) -> bool:
        return self.hour >= other.hour or (
            self.hour == other.hour and self.minute >= other.minute
        )

    def to_total_minutes(self):
        return self.hour * 60 + self.minute


# Intervals work inclusive -> 12:33:22 part of 12:33
class GridInterval:
    def __init__(self, start: Time, end: Time) -> None:
        self.start = start
        self.end = end
        self.next_interval = None

    def set_next_interval(self, next_interval: GridInterval) -> None:
        self.next_interval = next_interval


class TimeSeries:
    def __init__(self, start: Time, end: Time, intervalLength: int) -> None:
        self.intervals: list[GridInterval] = []
        last_interval = None

        # Build an single linked array list
        for start in range(
            start.to_total_minutes(), end.to_total_minutes(), intervalLength
        ):
            interval = GridInterval(
                Time.of_total_minutes(start),
                Time.of_total_minutes(start + intervalLength - 1),
            )
            self.intervals.append(interval)

            if last_interval != None:
                last_interval.set_next_interval(interval)
            last_interval = interval

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

    def adjust_state_value(
        self,
        current_time: Time,
        current_interval: GridInterval,
        current_location: Location,
        current_zone: Zone,
        next_time: Time,
        next_interval: GridInterval,
        next_location: Location,
        next_zone: Zone,
        reward: float,
    ) -> None:
        if current_zone and current_location:
            raise Exception("Only current zone or location is allowed")
        if not current_zone and not current_location:
            raise Exception("No current zone or location was specified")

        if current_time and current_interval:
            raise Exception("Only current time or interval is allowed")
        if not current_time and not current_interval:
            raise Exception("No current time or interval was specified")

        if next_location and next_zone:
            raise Exception("Only next zone or location is allowed")
        if not next_location and not next_zone:
            raise Exception("No next zone or location was specified")

        if next_interval and next_time:
            raise Exception("Only next time or interval is allowed")
        if not next_interval and not next_time:
            raise Exception("No next time or interval was specified")

        if current_zone == None:
            current_zone = self.grid.find_zone(current_location)
        if current_interval == None:
            current_interval = self.time_series.find_interval(current_time)
        if next_zone == None:
            next_zone = self.grid.find_zone(next_location)
        if next_interval == None:
            next_interval = self.time_series.find_interval(next_time)

        self.value_grid[current_interval][current_zone] = self.value_grid[
            current_interval
        ][current_interval] + LEARNING_RATE * (
            reward
            + DISCOUNT_FACTOR(current_interval.start, next_interval.start)
            * self.value_grid[next_interval][next_zone]
            - self.value_grid[next_interval][next_zone]
        )

    def get_state_value(self, zone: Zone, interval: GridInterval) -> float:
        return self.value_grid[interval][zone]


STATE_VALUE_TABLE: StateValueTable = StateValueTable(
    Grid([Zone("Zone_1"),Zone("Zone_2"),Zone("Zone_3"),Zone("Zone_4")], 0, 0, 10, 10, 1), TimeSeries(Time(3, 0), Time(12, 0), 1)
)
