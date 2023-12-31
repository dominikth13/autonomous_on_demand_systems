from __future__ import annotations
import random
from grid.grid import Grid
from logger import LOGGER
from utils import IdProvider
from location.location import Location
from program_params import *
import pandas as pd

ID_PROVIDER = IdProvider()


class Order:
    _orders_by_time = None

    def get_orders_by_time() -> dict[Time, list[Order]]:
        if Order._orders_by_time == None:
            Order._orders_by_time = {
                Time(hour, minute): [] for minute in range(60) for hour in range(24)
            }
            df = pd.read_csv("code/data/orders_2015-07-01.csv")
            counter = 0
            for _, row in df.iterrows():
                if counter % 50000 == 0:
                    LOGGER.debug(f"Processed {(counter // 50000)*50000} orders...")
                # Extract hour and minute from the pickup datetime
                hour = int(row["pickup_time"][0:2])
                minute = int(row["pickup_time"][3:5])
                # Create a tuple of Pickup and Dropoff Zone IDs
                pu_zone_id = int(row["PULocationID"])
                do_zone_id = int(row["DOLocationID"])
                Order._orders_by_time[Time(hour, minute)].append(
                    Order(
                        Time(hour, minute),
                        random.Random(counter)
                        .choice(Grid.get_instance().cells_dict[pu_zone_id])
                        .center,
                        random.Random(counter + 1)
                        .choice(Grid.get_instance().cells_dict[do_zone_id])
                        .center,
                    )
                )
                counter += 1
        return Order._orders_by_time

    def __init__(self, dispatch_time: Time, start: Location, end: Location) -> None:
        self.id = ID_PROVIDER.get_id()
        self.dispatch_time = dispatch_time
        self.start = start
        self.end = end
        # Initialized as not dispatched
        self.expires = None
        self.direct_connection = None

    def dispatch(self) -> None:
        self.expires = ORDER_EXPIRY_DURATION

        fastest_connection = None
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
                total_walking_time = (
                    self.start.distance_to(origin.position)
                    + destination.position.distance_to(self.end)
                ) / WALKING_SPEED

                if (
                    fastest_connection == None
                    or fastest_connection[1] > total_walking_time + connection[1]
                ):
                    fastest_connection = (
                        connection[0],
                        connection[1] + total_walking_time,
                    )

        self.direct_connection = fastest_connection
