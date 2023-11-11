from grid import Grid, Zone
from time_interval import GridInterval, TimeSeries, Time
from utils import IdProvider
from program_params import *
from location import Location

ID_PROVIDER = IdProvider()


class StateValueTable:
    def __init__(self, grid: Grid, time_series: TimeSeries) -> None:
        self.value_grid = {
            interval: {zone: 0 for zone in grid.zones_dict.values()}
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
