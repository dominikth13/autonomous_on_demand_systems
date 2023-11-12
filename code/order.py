from utils import IdProvider
from location import Location
from program_params import *

ID_PROVIDER = IdProvider()

class Order:
    def __init__(self, start: Location, end: Location) -> None:
        self.id = ID_PROVIDER.get_id()
        self.start = start
        self.end = end
        self.expires = ORDER_EXPIRY_DURATION