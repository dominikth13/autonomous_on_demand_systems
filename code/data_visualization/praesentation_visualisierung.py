import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from shapely.wkt import loads
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import re
from datetime import datetime



def average_number_of_drivers_per_day():
    tripdata_path = "code/data_output"

    tripdata_files = [
        f
        for f in os.listdir(tripdata_path)
        if f.endswith(".csv") and f.startswith("tripdata")
    ]

    average_occupied_drivers = []
    # Extrahieren Sie die Daten aus den Dateinamen der tripdata-Dateien und sortieren Sie sie
    dates = [re.search(r"(\d{4}-\d{2}-\d{2})", f).group(1) for f in tripdata_files]
    dates.sort(key=lambda date: datetime.strptime(date, "%Y-%m-%d"))

    for date in dates:
        workload_file_name = f"workload{date}.csv"
        workload_file_path = os.path.join(tripdata_path, workload_file_name)
        if os.path.exists(workload_file_path):
            workload_data = pd.read_csv(workload_file_path)
            average_occupied_drivers.append(
                workload_data["num_of_occupied_driver"].mean()
            )
        else:
            average_occupied_drivers.append(float("nan"))



    # Erstellen Sie eine Figur mit vier Subplots
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 12))
    #ax1 = axs.flatten()

    # Erster Plot: Anzahl der Routen pro Tag
    ax1.bar(dates, average_occupied_drivers, color="blue")
    ax1.set_xlabel("Datum")
    ax1.set_ylabel("Durchschnittlich besetzte Fahrer")
    ax1.set_title("Durchschnittlich besetzte Fahrer pro Tag")
    ax1.set_xticklabels(dates, rotation=45)
    ax1.set_ylim(0, 100)
    plt.show()

average_number_of_drivers_per_day()


def number_of_routes_per_day():
    
    tripdata_path = "code/data_output"
    tripdata_files = [
        f
        for f in os.listdir(tripdata_path)
        if f.endswith(".csv") and f.startswith("tripdata")
    ]
    routes_per_day = []

    dates = [re.search(r"(\d{4}-\d{2}-\d{2})", f).group(1) for f in tripdata_files]
    dates.sort(key=lambda date: datetime.strptime(date, "%Y-%m-%d"))

    for date in dates:
    # Tripdata-Daten
        tripdata_file_name = f"tripdata{date}.csv"
        tripdata_file_path = os.path.join(tripdata_path, tripdata_file_name)
        if os.path.exists(tripdata_file_path):
            tripdata = pd.read_csv(tripdata_file_path)
            routes_per_day.append(len(tripdata))
        else:
            routes_per_day.append(float("nan"))

    fig, ax2 = plt.subplots(1, 1, figsize=(15, 12))
    # Zweiter Plot: Summierte Zeitersparnis pro Tag
    ax2.bar(dates, routes_per_day, color="blue")
    ax2.set_xlabel("Datum")
    ax2.set_ylabel("Anzahl der Routen")
    ax2.set_title("Anzahl der Routen pro Tag")
    ax2.set_xticklabels(dates, rotation=45)

    plt.show()
    
number_of_routes_per_day()


    # # Liste alle Dateien im Verzeichnis auf

    # # Extrahieren Sie die Daten aus den Dateinamen der tripdata-Dateien und sortieren Sie sie
    # dates = [re.search(r"(\d{4}-\d{2}-\d{2})", f).group(1) for f in tripdata_files]
    # dates.sort(key=lambda date: datetime.strptime(date, "%Y-%m-%d"))

    # # Initialisieren Sie Listen für die Diagrammdaten
    # routes_per_day = []
    # total_time_reduction = []
    # average_time_reduction_per_day = []
    # average_occupied_drivers = []
    # total_time_reduction_per_car_in_minutes = []
    # average_route_length = {
    #     "driver_to_pickup_distance_km": [],
    #     "pickup_to_dropoff_distance_km": [],
    # }
    # average_combi_route_length = {
    #     "driver_to_pickup_distance_km": [],
    #     "pickup_to_dropoff_distance_km": [],
    #     "dropoff_to_destination_distance_km": [],
    #     "driver_to_dropoff_distance_km": [],
    # }

    # # Berechnen Sie die Daten für alle Diagramme

    # for date in dates:
    #     workload_file_name = f"workload{date}.csv"
    #     workload_file_path = os.path.join(tripdata_path, workload_file_name)
    #     if os.path.exists(workload_file_path):
    #         workload_data = pd.read_csv(workload_file_path)
    #         average_occupied_drivers.append(
    #             workload_data["num_of_occupied_driver"].mean()
    #         )
    #     else:
    #         average_occupied_drivers.append(float("nan"))

    # taxi_zones_file_path = "code/data/taxi_zones.csv"
    # taxi_zones = pd.read_csv(taxi_zones_file_path)
    # # Convert polygon strings to shapely polygon objects
    # taxi_zones["polygon"] = taxi_zones["the_geom"].apply(loads)
    # # Calculate the latitude and longitude of the center for each zone
    # taxi_zones["center"] = taxi_zones["polygon"].apply(
    #     lambda p: (p.centroid.y, p.centroid.x)
    # )
    # # Create a dictionary to map zone IDs to their center latitude and longitude
    # zone_centers = dict(zip(taxi_zones["LocationID"], taxi_zones["center"]))

    # # Define a function to calculate the distance between two zones in kilometers
    # def calculate_distance_km(zone1_id, zone2_id):
    #     if zone1_id in zone_centers and zone2_id in zone_centers:
    #         # Use geodesic from geopy to calculate the distance
    #         return geodesic(zone_centers[zone1_id], zone_centers[zone2_id]).kilometers
    #     else:
    #         raise Exception(f"{zone1_id} or {zone2_id} not found")

    # for date in dates:
    #     # Tripdata-Daten
    #     tripdata_file_name = f"tripdata{date}.csv"
    #     tripdata_file_path = os.path.join(tripdata_path, tripdata_file_name)
    #     if os.path.exists(tripdata_file_path):
    #         tripdata = pd.read_csv(tripdata_file_path)
    #         routes_per_day.append(len(tripdata))
    #         total_time_reduction.append(tripdata["time_reduction"].sum())
    #         # Calculate distances for each trip in the dataset
    #         tripdata["driver_to_pickup_distance_km"] = tripdata.apply(
    #             lambda x: calculate_distance_km(
    #                 x["driver_start_zone_id"], x["passenger_pickup_zone_id"]
    #             ),
    #             axis=1,
    #         )
    #         tripdata["pickup_to_dropoff_distance_km"] = tripdata.apply(
    #             lambda x: calculate_distance_km(
    #                 x["passenger_pickup_zone_id"], x["passenger_dropoff_zone_id"]
    #             ),
    #             axis=1,
    #         )
    #         tripdata["dropoff_to_destination_distance_km"] = tripdata.apply(
    #             lambda x: calculate_distance_km(
    #                 x["passenger_dropoff_zone_id"], x["destination_id"]
    #             ),
    #             axis=1,
    #         )
    #         average_route_length["driver_to_pickup_distance_km"].append(
    #             tripdata[tripdata["combi_route"]]["driver_to_pickup_distance_km"].mean()
    #         )
    #         average_route_length["pickup_to_dropoff_distance_km"].append(
    #             tripdata[tripdata["combi_route"]][
    #                 "pickup_to_dropoff_distance_km"
    #             ].mean()
    #         )

    #         average_combi_route_length["driver_to_pickup_distance_km"].append(
    #             tripdata[tripdata["combi_route"]]["driver_to_pickup_distance_km"].mean()
    #         )

    #         average_combi_route_length["pickup_to_dropoff_distance_km"].append(
    #             tripdata[tripdata["combi_route"]][
    #                 "pickup_to_dropoff_distance_km"
    #             ].mean()
    #         )

    #         average_combi_route_length["driver_to_dropoff_distance_km"].append(
    #             average_combi_route_length["driver_to_pickup_distance_km"][-1]
    #             + average_combi_route_length["pickup_to_dropoff_distance_km"][-1]
    #         )

    #         average_combi_route_length["dropoff_to_destination_distance_km"].append(
    #             tripdata[tripdata["combi_route"]][
    #                 "dropoff_to_destination_distance_km"
    #             ].mean()
    #         )

    #     else:
    #         routes_per_day.append(float("nan"))
    #         total_time_reduction.append(float("nan"))

    #     # Orders-Daten

    # i = 0
    # for date in dates:
    #     # Orders-Daten
    #     orders_file_name = f"orders_{date}.csv"
    #     orders_file_path = os.path.join(orders_path, orders_file_name)
    #     if os.path.exists(orders_file_path):
    #         orders = pd.read_csv(orders_file_path)
    #         num_orders = len(orders)
    #         average_time_reduction_per_day.append(
    #             total_time_reduction[i] / num_orders if num_orders > 0 else float("nan")
    #         )
    #     else:
    #         average_time_reduction_per_day.append(float("nan"))
    #     i += 1

    # for total_time in total_time_reduction:
    #     # Teile die Gesamtzeitersparnis durch 100 (für die Anzahl der Autos) und dann durch 60 (für Minuten)
    #     if not pd.isna(total_time):
    #         time_per_car_in_minutes = (total_time / 100) / 60 / 24
    #     else:
    #         time_per_car_in_minutes = float("nan")
    #     total_time_reduction_per_car_in_minutes.append(time_per_car_in_minutes)

    # # Erstellen Sie eine Figur mit vier Subplots
    # fig, axs = plt.subplots(2, 3, figsize=(15, 12))
    # ax1, ax2, ax3, ax4, ax5, ax6 = axs.flatten()

    # # Erster Plot: Anzahl der Routen pro Tag



    # ax3.bar(dates, total_time_reduction_per_car_in_minutes, color="red")
    # ax3.set_xlabel("Datum")
    # ax3.set_ylabel("Zeitersparnis pro Auto (Minuten)")
    # ax3.set_title("Summierte Zeitersparnis pro Stunde pro Auto (Minuten)")
    # ax3.set_xticklabels(dates, rotation=45)

    # # Vierter Plot: Hier fügen Sie Ihren Code für den vierten Plot ein
    # ax4.bar(dates, average_time_reduction_per_day, color="orange")
    # ax4.set_xlabel("Datum")
    # ax4.set_ylabel("Durchschnittliche Zeitersparnis pro Order pro Tag")
    # ax4.set_title("Durchschnittliche Zeitersparnis pro Order pro Tag")
    # ax4.set_xticklabels(dates, rotation=45)

    # # Combined plot
    # ax5.bar(
    #     dates,
    #     average_route_length["driver_to_pickup_distance_km"],
    #     color="#90EE90",
    #     label="Fahrerposition bis Abholpunkt",
    # )
    # ax5.bar(
    #     dates,
    #     average_route_length["pickup_to_dropoff_distance_km"],
    #     color="green",
    #     alpha=0.6,
    #     label="Abholpunkt bis Ziel",
    #     bottom=average_route_length["driver_to_pickup_distance_km"],
    # )
    # ax5.set_xlabel("Datum")
    # ax5.set_ylabel("Durchschnittliche Tripdistanzen pro Tag in km")
    # ax5.set_title("Durchschnittliche Tripdistanzen pro Tag für Direktrouten")
    # ax5.set_xticklabels(dates, rotation=45)

    # # Combined plot
    # ax6.bar(
    #     dates,
    #     average_combi_route_length["driver_to_pickup_distance_km"],
    #     color="#90EE90",
    #     label="Fahrerposition bis Abholpunkt",
    # )
    # ax6.bar(
    #     dates,
    #     average_combi_route_length["pickup_to_dropoff_distance_km"],
    #     color="green",
    #     label="Abholpunkt bis Station",
    #     bottom=average_combi_route_length["driver_to_pickup_distance_km"],
    # )
    # ax6.bar(
    #     dates,
    #     average_combi_route_length["dropoff_to_destination_distance_km"],
    #     color="#006400",
    #     label="Station bis Ziel",
    #     bottom=average_combi_route_length["driver_to_dropoff_distance_km"],
    # )
    # ax6.set_xlabel("Datum")
    # ax6.set_ylabel("Durchschnittliche Tripdistanzen pro Tag in km")
    # ax6.set_title("Durchschnittliche Tripdistanzen pro Tag für Kombinationsrouten")
    # ax6.set_xticklabels(dates, rotation=45)

    # # Zeige die Figur an
    # plt.tight_layout()