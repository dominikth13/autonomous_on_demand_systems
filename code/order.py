from utils import IdProvider
from location import Location

ID_PROVIDER = IdProvider()

class Order:
    def __init__(self, start: Location, end: Location) -> None:
        self.id = ID_PROVIDER.get_id()
        self.start = start
        self.end = end