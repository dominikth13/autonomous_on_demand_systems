from utils import IdProvider


ID_PROVIDER = IdProvider()


class Action:
    from route import Route

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
    from driver import Driver
    from location import Location

    def __init__(self, driver: Driver, action: Action, weight: float) -> None:
        self.driver = driver
        self.action = action
        self.weight = weight

    def __str__(self):
        return f"[Driver {self.driver.id} - Action: {self.action} - State-Action-Value {self.weight}]"

    def get_total_vehicle_travel_time_in_seconds(self) -> int:
        from program_params import VEHICLE_SPEED
        if self.action.is_idling():
            return 0
        return self.get_total_vehicle_distance() // VEHICLE_SPEED

    def get_total_vehicle_distance(self) -> float:
        if self.action.is_idling():
            return 0
        if self.action.route.stations == []:
            vehicle_distance_with_passenger = self.action.route.origin.distance_to(
                self.action.route.destination
            )
        else:
            vehicle_distance_with_passenger = self.action.route.origin.distance_to(
                self.action.route.stations[0]
            )
        return vehicle_distance_with_passenger + self.driver.current_position.distance_to(self.action.route.origin)

    def get_vehicle_destination(self) -> Location:
        if self.action.is_idling():
            return self.driver.current_position
        if self.action.route.stations == []:
            return self.action.route.destination
        return self.action.route.stations[0]
