from location.location import Location
from location.zone import Zone
from utils import IdProvider

ID_PROVIDER = IdProvider()

class GridCell:
    def __init__(self, center: Location, zone: Zone) -> None:
        self.id = ID_PROVIDER.get_id()
        self.center = center
        self.zone = zone