from __future__ import annotations
import csv
import random
from grid.grid import Grid
from interval.time import Time
from location.zone import Zone
from logger import LOGGER
from program.program_params import ProgramParams
from utils import IdProvider
from location.location import Location
import pandas as pd

ID_PROVIDER = IdProvider()


class Order:
    _orders_by_time = None

    # Resets the orders
    def reset() -> None:
        Order._orders_by_time = None

    def get_orders_by_time() -> dict[Time, list[Order]]:
        if Order._orders_by_time == None:
            Order._orders_by_time = {
                Time(hour, minute): [] for minute in range(60) for hour in range(24)
            }
            with open(ProgramParams.ORDERS_FILE_PATH(), mode="r") as file:
                reader = csv.DictReader(file)
                for i, row in enumerate(reader):
                    if i % 50000 == 0:
                        LOGGER.debug(f"Processed {(i // 50000)*50000} orders...")
                    # Extract hour and minute from the pickup datetime
                    hour = int(row["pickup_time"][0:2])
                    minute = int(row["pickup_time"][3:5])
                    # Create a tuple of Pickup and Dropoff Zone IDs
                    pu_zone_id = int(row["PULocationID"])
                    do_zone_id = int(row["DOLocationID"])
                    order = Order(
                        Time(hour, minute),
                        random.Random(i)
                        .choice(Grid.get_instance().cells_dict[pu_zone_id])
                        .center,
                        random.Random(i + 1)
                        .choice(Grid.get_instance().cells_dict[do_zone_id])
                        .center,
                        Grid.get_instance().zones_dict[pu_zone_id],
                    )
                    Order._orders_by_time[Time(hour, minute)].append(order)
        return Order._orders_by_time

    def __init__(
        self, dispatch_time: Time, start: Location, end: Location, zone: Zone
    ) -> None:
        self.id = ID_PROVIDER.get_id()
        self.dispatch_time = dispatch_time
        self.start = start
        self.end = end
        self.zone = zone
        # Initialized as not dispatched
        self.expires = None
        self.direct_connection = None

    def dispatch(self) -> None:
        self.expires = ProgramParams.ORDER_EXPIRY_DURATION

        # Init fastest connection with walking speed
        fastest_connection = (
            [],
            self.start.distance_to(self.end) / ProgramParams.WALKING_SPEED,
        )
        from public_transport.fastest_station_connection_network import (
            FastestStationConnectionNetwork,
        )

        fastest_connection_network = FastestStationConnectionNetwork.get_instance()

        # 1. Get the closest start and end station for each line
        from public_transport.station import Station

        origins: list[Station] = []
        destinations: list[Station] = []
        for line in fastest_connection_network.lines:
            origins.append(line.get_closest_station(self.start))
            destinations.append(line.get_closest_station(self.end))

        # 2. Find the most fastest connection without any autonomous on-demand services
        for origin in origins:
            for destination in destinations:
                if origin == destination:
                    continue
                connection = fastest_connection_network.get_fastest_connection(
                    origin, destination
                )
                walking_time = (
                    self.start.distance_to(origin.position)
                    + destination.position.distance_to(self.end)
                ) / ProgramParams.WALKING_SPEED
                # include entry, exit and waiting time
                other_time = (
                    2 * ProgramParams.PUBLIC_TRANSPORT_ENTRY_EXIT_TIME
                    + ProgramParams.PUBLIC_TRANSPORT_WAITING_TIME(self.dispatch_time)
                )
                total_additional_time = walking_time + other_time
                if fastest_connection[1] > total_additional_time + connection[1]:
                    fastest_connection = (
                        connection[0],
                        connection[1] + total_additional_time,
                    )

        self.direct_connection: tuple[list[Station], float] = fastest_connection
