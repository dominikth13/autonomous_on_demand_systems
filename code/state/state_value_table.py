from __future__ import annotations
import itertools
from grid.grid import Grid
from grid.grid_cell import GridCell
from location.location import Location
from interval.time_series import GridInterval, TimeSeries, Time
from location.zone import Zone
from utils import IdProvider
from program_params import *

ID_PROVIDER = IdProvider()

# Singleton class
class StateValueTable:
    _state_value_table: StateValueTable = None

    def get_state_value_table() -> StateValueTable:
        if StateValueTable._state_value_table == None:
            StateValueTable._state_value_table = StateValueTable(
                Grid.get_instance(),
                TimeSeries.get_instance(),
            )
        return StateValueTable._state_value_table

    def __init__(self, grid: Grid, time_series: TimeSeries) -> None:
        # TODO fix
        self.value_grid = {
            interval: {zone: 0 for zone in grid.zones_dict.values()}
            for interval in time_series.intervals
        }

    def adjust_state_value(
        self,
        reward: float,
        current_time: Time = None,
        current_interval: GridInterval = None,
        current_location: Location = None,
        current_zone: GridCell = None,
        next_time: Time = None,
        next_interval: GridInterval = None,
        next_location: Location = None,
        next_zone: GridCell = None,
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
            current_zone = Grid.get_instance().find_zone(current_location)
        if current_interval == None:
            current_interval = TimeSeries.get_instance().find_interval(current_time)
        if next_zone == None:
            next_zone = Grid.get_instance().find_zone(next_location)
        if next_interval == None:
            next_interval = TimeSeries.get_instance().find_interval(next_time)

        self.value_grid[current_interval][current_zone] = self.value_grid[
            current_interval
        ][current_zone] + LEARNING_RATE * (
            reward
            + DISCOUNT_FACTOR(current_interval.start, next_interval.start)
            * self.value_grid[next_interval][next_zone]
            - self.value_grid[current_interval][current_zone]
        )

    def get_state_value(self, zone: Zone, interval: GridInterval) -> float:
        return self.value_grid[interval][zone]
