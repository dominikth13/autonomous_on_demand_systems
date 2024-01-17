from __future__ import annotations

class Time:
    
    # delete other variables, only create total_seconds
    def __init__(self, hour: int, minute: int, second = 0) -> None:
        assert hour >= 0 and hour <= 23
        assert minute >= 0 and minute <= 59
        assert second >= 0 and second <= 59
        self.total_seconds = int(hour * 3600 + minute * 60 + second)

    def of_total_minutes(minutes: int) -> Time:
        while minutes >= 1440:
            minutes -= 1440
        return Time(minutes // 60, minutes % 60, 0)
    
    def of_total_seconds(seconds: int) -> Time:
        if seconds >= 86400:
            seconds -= 86400
        return Time(seconds // 3600, (seconds % 3600) // 60, seconds % 60)

    # Calculate time difference(distance) in seconds
    def distance_to(self, other: Time) -> int:
        return abs(self.total_seconds - other.total_seconds)
    
    def add_minutes(self, minutes: int):
        seconds = minutes * 60
        return self.add_seconds(seconds)
    
    def add_seconds(self, seconds: int) -> Time:
        return Time.of_total_seconds(self.total_seconds + seconds)
    
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
        hours = int(self.total_seconds // 3600)
        minutes = int((self.total_seconds % 3600) // 60)
        seconds = int(self.total_seconds % 60)
        return hours, minutes, seconds
    
    # Override equals implementation
    def __eq__(self, other):
        if isinstance(other, Time):
            return self.total_seconds == other.total_seconds
        return False

    # Override hash implementation
    def __hash__(self):
      return self.total_seconds