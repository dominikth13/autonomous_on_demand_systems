from action.action import Action
from driver.driver import Driver
from location.location import Location

class DriverActionPair:

    def __init__(self, driver: Driver, action: Action, weight: float) -> None:
        self.driver = driver
        self.action = action
        self.weight = weight

    def __str__(self):
        return f"[Driver {self.driver.id} - Action: {self.action} - State-Action-Value {self.weight}]"

    def get_total_vehicle_travel_time_in_seconds(self) -> int:
        from program_params import ProgramParams
        if self.action.is_idling():
            return 0
        return self.get_total_vehicle_distance() // ProgramParams.VEHICLE_SPEED

    def get_total_vehicle_distance(self) -> float:
        if self.action.is_idling():
            return 0
        if self.action.route.stations == []:
            vehicle_distance_with_passenger = self.action.route.origin.distance_to(
                self.action.route.destination
            )
        else:
            vehicle_distance_with_passenger = self.action.route.origin.distance_to(
                self.action.route.stations[0].position
            )
        return vehicle_distance_with_passenger + self.driver.current_position.distance_to(self.action.route.origin)

    def get_vehicle_destination(self) -> Location:
        if self.action.is_idling():
            return self.driver.current_position
        if self.action.route.stations == []:
            return self.action.route.destination
        return self.action.route.stations[0].position