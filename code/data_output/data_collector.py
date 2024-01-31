import csv
from interval.time import Time
from order import Order
from location.location import Location
from location.zone import Zone
from program.program_params import ProgramParams


class DataCollector:
    # [(total_seconds, num_of_occupied_driver)]
    workload = []

    # [(total_seconds, start_zone_id, end_zone_id, distance)]
    relocation_trip_data = []

    # [(total_seconds, id, status, lat, lon)]
    # Will be saved each hour
    driver_data = []

    # [(total_seconds, quota_of_unserved_orders, num_of_served_orders)]
    orders_data = []
    orders_dataa = []


    # [(total_seconds, quota_of_saved_time_for_all_served_orders)]
    time_reduction_quota = []

    zone_id_list = []
    # [(total_seconds, driver_start_zone_id, passenger_pickup_zone_id, passenger_dropoff_zone_id, destination_id, vehicle_trip_time, time_reduction, combi_route, total_vehicle_distance)]
    trip_data = []

    def append_workload(current_time: Time, num_of_occupied_driver: int):
        DataCollector.workload.append(
            (current_time.to_total_seconds(), num_of_occupied_driver)
        )

    def append_relocation_trip_data(current_time: Time, start_zone: Zone, end_zone: Zone, distance: int):
        DataCollector.relocation_trip_data.append(
            (current_time.to_total_seconds(), start_zone.id, end_zone.id, distance)
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


    def append_orders_dataa(
        current_time: Time, order: Order, destination_vehicle: Location, destination_time:float):
        DataCollector.orders_dataa.append(
            (
                current_time.to_total_seconds(),
                order.start.lat,
                order.start.lon,
                order.end.lat ,
                order.end.lon,
                destination_vehicle.lat,
                destination_vehicle.lon , 
                destination_time
                
            )
        )

    def append_time_reduction_quota(
        current_time: Time, quota_of_saved_time_for_all_served_orders: float
    ):
        DataCollector.time_reduction_quota.append(
            (current_time.to_total_seconds(), quota_of_saved_time_for_all_served_orders)
        )

    def append_zone_id(current_time: Time, zone_id: int):
        DataCollector.zone_id_list.append(
            (current_time.to_total_seconds(), zone_id)
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
        total_vehicle_distance: int
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
                total_vehicle_distance
            )
        )

    def export_all_data():
        csv_file_path = f"code/data_output/workload{ProgramParams.SIMULATION_DATE.strftime('%Y-%m-%d')}.csv"
        with open(csv_file_path, mode="w") as file:
            writer = csv.writer(file)
            writer.writerow(["total_seconds", "num_of_occupied_driver"])
            for w in DataCollector.workload:
                writer.writerow([w[0], w[1]])

        csv_file_path = f"code/data_output/relocation_trip_data{ProgramParams.SIMULATION_DATE.strftime('%Y-%m-%d')}.csv"
        with open(csv_file_path, mode="w") as file:
            writer = csv.writer(file)
            writer.writerow(["total_seconds", "start_zone_id", "end_zone_id", "distance"])
            for w in DataCollector.relocation_trip_data:
                writer.writerow([w[0], w[1], w[2], w[3]])

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

        csv_file_path = "code/data_output/orders_dataa.csv"
        with open(csv_file_path, mode="w") as file:
            writer = csv.writer(file)
            writer.writerow(["total_seconds", "start_lat", "start_lon","end_lat","end_lon" , "veh_des_lat", "veh_des_lon","veh_time"])
            for w in DataCollector.orders_dataa:
                writer.writerow([w[0], w[1], w[2],w[3],w[4],w[5],w[6],w[7]])
        

        csv_file_path = (f"code/data_output/time_reduction_quota_{ProgramParams.SIMULATION_DATE.strftime('%Y-%m-%d')}.csv")
        csv_file_path = f"code/data_output/average_time_reduction{ProgramParams.SIMULATION_DATE.strftime('%Y-%m-%d')}.csv"
        with open(csv_file_path, mode="w") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["total_seconds", "quota_of_saved_time_for_all_served_orders"]
            )
            for w in DataCollector.time_reduction_quota:
                writer.writerow([w[0], w[1]])

        csv_file_path = "code/data_output/cell_id.csv"
        with open(csv_file_path, mode="w") as file:
            writer = csv.writer(file)
            writer.writerow(["total_seconds", "cell_id"])
            for w in DataCollector.zone_id_list:
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
                    "total_vehicle_distance"
                ]
            )
            for w in DataCollector.trip_data:
                writer.writerow([w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8]])
    
    def clear():
        DataCollector.driver_data.clear()
        DataCollector.orders_data.clear()
        DataCollector.relocation_trip_data.clear()
        DataCollector.workload.clear()
        DataCollector.trip_data.clear()
        DataCollector.time_reduction_quota.clear()
        DataCollector.orders_dataa.clear()
        DataCollector.zone_id_list.clear()