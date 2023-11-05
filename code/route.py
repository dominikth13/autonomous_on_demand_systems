from state_value_table import STATE_VALUE_TABLE
from location import Location
from order import Order
from station import *
from program_params import *

ID_PROVIDER = IdProvider()

# Class route contains data model of route object
class Route:
    def __init__(
        self,
        order: Order,
        origin: Location,
        destination: Location,
        stations: list[Station],
        vehicle_time: float,
        transit_time: float,
        walking_time: float,
        other_time: float,
        total_time: float,
        vehicle_price: float,
        price: float,
    ) -> None:
        self.id = ID_PROVIDER.get_id()
        self.order = order
        self.origin = origin
        self.destination = destination
        self.stations = stations
        self.transit_time = transit_time
        self.walking_time = walking_time
        self.other_time = other_time
        self.total_time = total_time
        self.price = price

        self.vehicle_price = vehicle_price
        self.vehicle_time = vehicle_time
        self.vehicle_destination_zone = STATE_VALUE_TABLE.grid.find_zone(destination if stations == [] else stations[0])
    
    def is_regular_route(self) -> bool:
        return self.stations == []


def regular_route(order: Order) -> Route:
    vehicle_time = order.start.distance_to(order.end) * VEHICLE_SPEED
    # TODO calculate time and price for regular routes
    #hier wird es ge'ndert
    #gugug
    return Route(order, order.start, order.end, [], vehicle_time, 0, 0, 0, vehicle_time, 5, 5)