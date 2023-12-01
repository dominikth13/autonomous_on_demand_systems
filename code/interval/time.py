from __future__ import annotations

class Time:
    
    # delete other variables, only create total_seconds
    def __init__(self, hour: int, minute: int, second: int) -> None:
        self.total_seconds = hour * 3600 + minute * 60 + second

    def of_total_minutes(minutes: float) -> Time:
        return Time(minutes // 60, minutes % 60, (minutes % 1) * 60 )
    
    def of_total_seconds(seconds: float) -> Time:
        return Time(seconds // 3600, (seconds % 3600) // 60, seconds % 60)

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