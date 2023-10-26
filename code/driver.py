from utils import IdProvider
from state import Location

ID_PROVIDER = IdProvider()

class Driver:
    def __init__(self, start_position: Location) -> None:
        self.id = ID_PROVIDER.get_id()
        self.current_position = start_position
        self.occupied = False
    
    def is_occupied(self):
        return self.occupied