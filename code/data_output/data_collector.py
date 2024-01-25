import csv
from interval.time import Time
from location.location import Location
from location.zone import Zone
from program.program_params import ProgramParams


class DataCollector:
    # [(total_seconds, num_of_occupied_driver)]
    workload = []

    # [(total_seconds, num_of_relocated_drivers)]
    relocation = []

    # [(total_seconds, id, status, lat, lon)]
    # Will be saved each hour
    driver_data = []

    # [(total_seconds, quota_of_unserved_orders, num_of_served_orders)]
    orders_data = []

    # [(total_seconds, quota_of_saved_time_for_all_served_orders)]
    time_reduction_quota = []

    # [(total_seconds, driver_start_zone_id, passenger_pickup_zone_id, passenger_dropoff_zone_id, destination_id, vehicle_trip_time, time_reduction, combi_route)]
    trip_data = []

    def append_workload(current_time: Time, num_of_occupied_driver: int):
        DataCollector.workload.append(
            (current_time.to_total_seconds(), num_of_occupied_driver)
        )

    def append_relocation(current_time: Time, num_of_relocated_drivers: int):
        DataCollector.relocation.append(
            (current_time.to_total_seconds(), num_of_relocated_drivers)
        )

    def append_driver_data(
        current_time: Time, id: int, status: str, position: Location
    ):
        DataCollector.driver_data.append(
            (current_time.to_total_seconds(), id, status, position.lat, position.lon)
        )

    def append_orders_data(
        current_time: Time, quota_of_unserved_orders: float, num_of_served_orders: int
    ):
        DataCollector.orders_data.append(
            (
                current_time.to_total_seconds(),
                quota_of_unserved_orders,
                num_of_served_orders,
            )
        )

    def append_time_reduction_quota(
        current_time: Time, quota_of_saved_time_for_all_served_orders: float
    ):
        DataCollector.time_reduction_quota.append(
            (current_time.to_total_seconds(), quota_of_saved_time_for_all_served_orders)
        )

    def append_trip(
        current_time: Time,
        driver_zone: Zone,
        passenger_pu_zone: Zone,
        passenger_do_zone: Zone,
        destination_zone: Zone,
        total_vehicle_time: int,
        time_reduction: int,
        combi_route: bool,
    ):
        DataCollector.trip_data.append(
            (
                current_time.to_total_seconds(),
                driver_zone.id,
                passenger_pu_zone.id,
                passenger_do_zone.id,
                destination_zone.id,
                total_vehicle_time,
                time_reduction,
                combi_route,
            )
        )

    def export_all_data():
        csv_file_path = f"code/data_output/workload{ProgramParams.SIMULATION_DATE.strftime('%Y-%m-%d')}.csv"
        with open(csv_file_path, mode="w") as file:
            writer = csv.writer(file)
            writer.writerow(["total_seconds", "num_of_occupied_driver"])
            for w in DataCollector.workload:
                writer.writerow([w[0], w[1]])

        csv_file_path = f"code/data_output/relocation{ProgramParams.SIMULATION_DATE.strftime('%Y-%m-%d')}.csv"
        with open(csv_file_path, mode="w") as file:
            writer = csv.writer(file)
            writer.writerow(["total_seconds", "num_of_relocated_drivers"])
            for w in DataCollector.relocation:
                writer.writerow([w[0], w[1]])

        csv_file_path = f"code/data_output/driverdata{ProgramParams.SIMULATION_DATE.strftime('%Y-%m-%d')}.csv"
        with open(csv_file_path, mode="w") as file:
            writer = csv.writer(file)
            writer.writerow(["total_seconds", "id", "status", "lat", "lon"])
            for w in DataCollector.driver_data:
                writer.writerow([w[0], w[1], w[2], w[3], w[4]])

        csv_file_path = f"code/data_output/ordersdata{ProgramParams.SIMULATION_DATE.strftime('%Y-%m-%d')}.csv"
        with open(csv_file_path, mode="w") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["total_seconds", "quota_of_unserved_orders", "num_of_served_orders"]
            )
            for w in DataCollector.orders_data:
                writer.writerow([w[0], w[1], w[2]])

        csv_file_path = f"code/data_output/average_time_reduction{ProgramParams.SIMULATION_DATE.strftime('%Y-%m-%d')}.csv"
        with open(csv_file_path, mode="w") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["total_seconds", "quota_of_saved_time_for_all_served_orders"]
            )
            for w in DataCollector.time_reduction_quota:
                writer.writerow([w[0], w[1]])

        csv_file_path = f"code/data_output/tripdata{ProgramParams.SIMULATION_DATE.strftime('%Y-%m-%d')}.csv"
        with open(csv_file_path, mode="w") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "total_seconds",
                    "driver_start_zone_id",
                    "passenger_pickup_zone_id",
                    "passenger_dropoff_zone_id",
                    "destination_id",
                    "vehicle_trip_time",
                    "time_reduction",
                    "combi_route",
                ]
            )
            for w in DataCollector.driver_data:
                writer.writerow([w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]])
    
    def clear():
        DataCollector.driver_data.clear()
        DataCollector.orders_data.clear()
        DataCollector.relocation.clear()
        DataCollector.workload.clear()
        DataCollector.trip_data.clear()
        DataCollector.time_reduction_quota.clear()
