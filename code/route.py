from state import Location
from order import Order
from station import *
from program_params import *

FASTEST_STATION_NETWORK = FastestStationConnectionNetwork(STATIONS)


class Route:
    def __init__(
        self,
        origin: Location,
        destination: Location,
        stations: list[Station],
        vehicle_time: float,
        transit_time: float,
        walking_time: float,
        other_time: float,
        total_time: float,
        price: float,
    ) -> None:
        self.origin = origin
        self.destination = destination
        self.stations = stations
        self.vehicle_time = vehicle_time
        self.transit_time = transit_time
        self.walking_time = walking_time
        self.other_time = other_time
        self.total_time = total_time
        self.price = price


def regular_route(origin: Location, destination: Location) -> Route:
    vehicle_time = origin.distance_to(destination) * VEHICLE_SPEED
    # TODO calculate time and price for regular routes
    return Route(origin, destination, [], vehicle_time, 0, 0, 0, vehicle_time, 5)