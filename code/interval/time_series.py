from __future__ import annotations
from interval.grid_interval import GridInterval
from interval.time import Time
from utils import IdProvider

ID_PROVIDER = IdProvider()

class TimeSeries: 
    def __init__(self, start: Time, end: Time, intervalLengthInSeconds: int) -> None:
        self.start_time = start
        self.end_time = end
        self.intervals: list[GridInterval] = []

    # start, end, interval length are all in second
        counter = 0
        start_seconds = start.to_total_seconds()
        end_seconds = end.to_total_seconds()
        
        for current_seconds in range(start_seconds, end_seconds, intervalLengthInSeconds):
            interval_start = Time(0, 0, current_seconds)
            interval_end = Time(0, 0, current_seconds + intervalLengthInSeconds - 1)
            interval = GridInterval(counter, interval_start, interval_end)
            self.intervals.append(interval)
            counter += 1

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
    
    def get_next_interval(self, current_interval: GridInterval) -> GridInterval:
        if len(self.intervals) == current_interval.index + 1:
            return None
        return self.intervals[current_interval.index + 1]