from utils import IdProvider
from driver import Driver
from route import Route

ID_PROVIDER = IdProvider()

class Action:

    def __init__(self, route: Route) -> None:
        self.id = ID_PROVIDER.get_id()
        self.route = route
        self.idling = route == None
    
    def is_idling(self) -> bool:
        return self.idling
    
    def is_route(self) -> bool:
        return not self.idling

class DriverActionPair:
    def __init__(self, driver: Driver, action: Action, weight: float) -> None:
        self.driver = driver
        self.action = action
        self.weight = weight