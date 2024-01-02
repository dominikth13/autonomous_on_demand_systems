from location.location import Location

class DriverJob:
    # Total driving time in seconds
    def __init__(self, total_driving_time: int, new_position: Location, is_relocation: bool) -> None:
        self.total_driving_time: int = total_driving_time
        self.new_position: Location = new_position
        self.is_relocation = is_relocation
