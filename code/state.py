from location import *
from time_interval import *

# Define here how the grid and intervals should look like
grid = Grid(0, 0, 10, 10, 1)
time_series = TimeSeries(0, 100, 1)

class States:
    def __init__(self):
        self.states = [[[State(point, time_interval) for point in grid.coordinates[i]] for i in range(0, len(grid.coordinates))] for time_interval in time_series.intervals]

class State:
    def __init__(self, point_in_grid, time_interval):
        self.point_in_grid = point_in_grid
        self.time_interval = time_interval