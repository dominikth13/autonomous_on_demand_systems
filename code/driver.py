from state import Location

class Driver:
    def __init__(self, start_position: Location) -> None:
        current_position = start_position
        occupied = False