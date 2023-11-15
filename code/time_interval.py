from __future__ import annotations
from utils import IdProvider

ID_PROVIDER = IdProvider()

class Time:
    
    # delete other variables, only create total_seconds
    def __init__(self, hour: int, minute: int, second: int) -> None:
        self.total_seconds = hour * 3600 + minute * 60 + second

    def of_total_minutes(minutes: float) -> Time:
        return Time(minutes // 60, minutes % 60, (minutes % 1) * 60 )

    # Calculate time difference(distance) in seconds
    def distance_to(self, other: Time) -> int:
        return abs(self.total_seconds - other.total_seconds)
    
    def add_minutes(self, minutes: int):
        seconds = minutes * 60
        return self.add_seconds(seconds)
    
    def add_seconds(self, seconds: int) -> Time:
        new_total_second = self.total_seconds + seconds
        hour = new_total_second // 3600
        minute = (new_total_second % 3600) // 60
        second = new_total_second % 60 
        return Time(hour, minute, second)
    
    def is_before(self, other: Time) -> bool:
        return self.total_seconds <= other.total_seconds

    def is_after(self, other: Time) -> bool:
        return self.total_seconds >= other.total_seconds

    def to_total_minutes(self):
        return self.total_seconds // 60
    
    def to_total_seconds(self):
        return self.total_seconds

        # in case need to print time
    def __str__(self) -> str:
        hours, minutes, seconds = self.to_hours_minutes_seconds()
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def to_hours_minutes_seconds(self):
        hours = self.total_seconds // 3600
        minutes = (self.total_seconds % 3600) // 60
        seconds = self.total_seconds % 60
        return hours, minutes, seconds


# Intervals work inclusive -> 12:33:22 part of 12:33
class GridInterval:
    def __init__(self, index: int, start: Time, end: Time) -> None:
        self.id = ID_PROVIDER.get_id()
        self.index = index
        self.start = start
        self.end = end

class TimeSeries: 
    def __init__(self, start: Time, end: Time, intervalLengthInSeconds: int) -> None:
        self.start_time = start
        self.end_time = end
        self.intervals: list[GridInterval] = []

    # start, end, intervalength are all in second
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