from __future__ import annotations
from utils import IdProvider

ID_PROVIDER = IdProvider()

class Time:
    
    # add new attribute "seconds"
    def __init__(self, hour: int, minute: int, second: int) -> None:
        self.hour = hour
        self.minute = minute
        self.second = second  

    def of_total_minutes(minutes: float) -> Time:
        return Time(minutes // 60, minutes % 60, (minutes % 1) * 60 )

    # Calculate time difference(distance) in seconds
    def distance_to(self, other: Time) -> int:
        return abs(self.to_total_seconds() - other.to_total_seconds())

    def add_minutes(self, minutes: int) -> Time:
        minute = self.minute
        hour = self.hour
        second = self.second
        if minute + minutes > 59:
            minutes_to_next_hour = 60 - minute
            minute = 0
            minutes -= minutes_to_next_hour

            while minutes > 59:
                hour += 1
                if hour == 24:
                    hour = 0
                minutes -= 60
        minute += minutes
        return Time(hour, minute, second)
    
    def add_seconds(self, seconds: int):
        minute = self.minute
        hour = self.hour
        second = self.second
        second = (second + seconds) % 60 
        minutes = (second + seconds) // 60
        return self.add_minutes(minutes)

    def is_before(self, other: Time) -> bool:
        return self.to_total_seconds() <= other.to_total_seconds()

    def is_after(self, other: Time) -> bool:
        return self.to_total_seconds() >= other.to_total_seconds()

    def to_total_minutes(self):
        return self.hour * 60 + self.minute + self.second % 60
    
    def to_total_seconds(self):
        return self.to_total_minutes() * 60


# Intervals work inclusive -> 12:33:22 part of 12:33
class GridInterval:
    def __init__(self, index: int, start: Time, end: Time) -> None:
        self.id = ID_PROVIDER.get_id()
        self.index = index
        self.start = start
        self.end = end

class TimeSeries:
    def __init__(self, start: Time, end: Time, intervalLength: int) -> None:
        self.start_time = start
        self.end_time = end
        self.intervals: list[GridInterval] = []

        counter = 0
        for start in range(
            start.to_total_minutes(), end.to_total_minutes(), intervalLength
        ):
            interval = GridInterval(
                counter,
                Time.of_total_minutes(start),
                Time.of_total_minutes(start + intervalLength - 1),
            )
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