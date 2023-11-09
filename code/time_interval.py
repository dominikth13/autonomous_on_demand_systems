from __future__ import annotations

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
    
    def to_total_seconds(self):
        return self.to_total_minutes() * 60


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