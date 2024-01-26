import csv
from datetime import datetime, timedelta
from grid.grid import Grid
from interval.time import Time
from logger import LOGGER

from program.program_params import ProgramParams


def analyse_trip_data():
    grid = Grid.get_instance()
    start_date = datetime(2015, 7, 6)
    trips = []
    bool_dict = {"True": True, "False": False}
    for _ in range(7):
        csv_file_path = f"./store/baseline/baseline_0.005_1/tripdata{start_date.strftime('%Y-%m-%d')}.csv"
        with open(csv_file_path, mode="r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                trips.append(
                    {
                        "date": start_date,
                        "time": Time.of_total_seconds(int(row["total_seconds"])),
                        "combi_route": bool_dict[row["combi_route"]],
                        "pu_zone_id": int(row["passenger_pickup_zone_id"]),
                        "do_zone_id": int(row["passenger_dropoff_zone_id"]),
                        "destination_zone_id": int(row["destination_id"]),
                        "driver_zone_id": int(row["driver_start_zone_id"]),
                        "time_reduction": float(row["time_reduction"]),
                    }
                )
        start_date += timedelta(1)

    direct_routes = list(filter(lambda x: not x["combi_route"], trips))
    combi_routes = list(filter(lambda x: x["combi_route"], trips))
    LOGGER.info(f"Total amount of trips: {len(trips)}")
    LOGGER.info(f"Amount of direct routes: {len(direct_routes)}")
    LOGGER.info(f"Amount of combi routes: {len(combi_routes)}")
    LOGGER.info(f"Combi route quota: {round(len(combi_routes)/len(trips), 2)}%")

    average_trip_length = int(
        sum(
            list(
                map(
                    lambda x: grid.zones_dict[
                        x["pu_zone_id"]
                    ].central_location.distance_to(
                        grid.zones_dict[x["do_zone_id"]].central_location
                    ),
                    trips,
                )
            )
        )
        / len(trips)
    )
    average_direct_route_length = int(
        sum(
            list(
                map(
                    lambda x: grid.zones_dict[
                        x["pu_zone_id"]
                    ].central_location.distance_to(
                        grid.zones_dict[x["do_zone_id"]].central_location
                    ),
                    direct_routes,
                )
            )
        )
        / len(direct_routes)
    )
    average_combi_route_length = int(
        sum(
            list(
                map(
                    lambda x: grid.zones_dict[
                        x["pu_zone_id"]
                    ].central_location.distance_to(
                        grid.zones_dict[x["do_zone_id"]].central_location
                    ),
                    combi_routes,
                )
            )
        )
        / len(combi_routes)
    )
    average_total_combi_route_length = int(
        sum(
            list(
                map(
                    lambda x: grid.zones_dict[
                        x["pu_zone_id"]
                    ].central_location.distance_to(
                        grid.zones_dict[x["destination_zone_id"]].central_location
                    ),
                    combi_routes,
                )
            )
        )
        / len(combi_routes)
    )
    average_pickup_distance = int(
        sum(
            list(
                map(
                    lambda x: grid.zones_dict[
                        x["driver_zone_id"]
                    ].central_location.distance_to(
                        grid.zones_dict[x["pu_zone_id"]].central_location
                    ),
                    trips,
                )
            )
        )
        / len(trips)
    )
    LOGGER.info(f"Average trip length: {average_trip_length} meters")
    LOGGER.info(f"Average direct route length: {average_direct_route_length} meters")
    LOGGER.info(
        f"Average combi route length: {average_total_combi_route_length} meters"
    )
    LOGGER.info(
        f"Average combi route length with vehicle: {average_combi_route_length} meters"
    )
    LOGGER.info(f"Average pickup distance: {average_pickup_distance} meters")

    average_time_reduction = int(
        sum(list(map(lambda x: x["time_reduction"], trips))) / len(trips)
    )
    average_direct_time_reduction = int(
        sum(list(map(lambda x: x["time_reduction"], direct_routes))) / len(direct_routes)
    )
    average_combi_time_reduction = int(
        sum(list(map(lambda x: x["time_reduction"], combi_routes))) / len(combi_routes)
    )
    LOGGER.info(f"Average time reduction: {average_time_reduction} seconds")
    LOGGER.info(f"Average direct route time reduction: {average_direct_time_reduction} seconds")
    LOGGER.info(f"Average combi route time reduction: {average_combi_time_reduction} seconds")
