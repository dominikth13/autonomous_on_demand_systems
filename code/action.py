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
    
    def __str__(self):
        return f"{f'Route {self.route.id}' if self.is_route() else 'Idling'}"

class DriverActionPair:
    def __init__(self, driver: Driver, action: Action, weight: float) -> None:
        self.driver = driver
        self.action = action
        self.weight = weight

    def __str__(self):
        return f"[Driver {self.driver.id} - Action: {self.action} - State-Action-Value {self.weight}]"