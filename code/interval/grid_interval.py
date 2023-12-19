from interval.time import Time
from utils import IdProvider

ID_PROVIDER = IdProvider()

# Intervals work inclusive -> 12:33:22 part of 12:33
class GridInterval:
    def __init__(self, index: int, start: Time, end: Time) -> None:
        self.id = ID_PROVIDER.get_id()
        self.index = index
        self.start = start
        self.end = end