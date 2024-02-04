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
    tripdata_path = "store/for_hire/rl_relocation/drivers/1000"
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

    # Erster Plot: Anzahl der Routen pro Tag
    ax1.bar(dates, average_occupied_drivers, color="blue")
    ax1.set_xlabel("Datum")
    ax1.set_ylabel("Durchschnittlich besetzte Fahrer")
    ax1.set_title("Durchschnittlich besetzte Fahrer pro Tag")
    ax1.set_xticklabels(dates, rotation=45)
    ax1.set_ylim(0, 1000)
    plt.show()

average_number_of_drivers_per_day()


def number_of_routes_per_day():
    
    tripdata_path = "store/for_hire/rl_relocation/drivers/1000"
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

def total_time_reduction_per_car_in_minutes():
    tripdata_path = "store/for_hire/rl_relocation/drivers/1000"
    total_time_reduction = []
    total_time_reduction_per_car_in_minutes = []

    tripdata_files = [
        f
        for f in os.listdir(tripdata_path)
        if f.endswith(".csv") and f.startswith("tripdata")
    ]
    dates = [re.search(r"(\d{4}-\d{2}-\d{2})", f).group(1) for f in tripdata_files]
    dates.sort(key=lambda date: datetime.strptime(date, "%Y-%m-%d"))

    for date in dates:
        # Tripdata-Daten
        tripdata_file_name = f"tripdata{date}.csv"
        tripdata_file_path = os.path.join(tripdata_path, tripdata_file_name)
        if os.path.exists(tripdata_file_path):
            tripdata = pd.read_csv(tripdata_file_path)
            total_time_reduction.append(tripdata["time_reduction"].sum())
        else:
            total_time_reduction.append(float("nan"))

    for total_time in total_time_reduction:
    # Teile die Gesamtzeitersparnis durch 100 (für die Anzahl der Autos) und dann durch 60 (für Minuten)
        if not pd.isna(total_time):
            time_per_car_in_minutes = (total_time / 1000) / 60 / 24
        else:
            time_per_car_in_minutes = float("nan")
        total_time_reduction_per_car_in_minutes.append(time_per_car_in_minutes)

    fig, ax3 = plt.subplots(1, 1, figsize=(15, 12))
    ax3.bar(dates, total_time_reduction_per_car_in_minutes, color="red")
    ax3.set_xlabel("Datum")
    ax3.set_ylabel("Zeitersparnis pro Auto (Minuten)")
    ax3.set_title("Summierte Zeitersparnis pro Stunde pro Auto (Minuten)")
    ax3.set_xticklabels(dates, rotation=45)
    plt.show()

total_time_reduction_per_car_in_minutes()

def average_time_reduction_per_day(): 
    print("v4")
    orders_path = "code/data/for_hire"
    tripdata_path = "store/for_hire/rl_relocation/drivers/1000"
    total_time_reduction = []
    average_time_reduction_per_day = []
    total_time_reduction_per_car_in_minutes = []

    tripdata_files = [
        f
        for f in os.listdir(tripdata_path)
        if f.endswith(".csv") and f.startswith("tripdata")
    ]
    dates = [re.search(r"(\d{4}-\d{2}-\d{2})", f).group(1) for f in tripdata_files]
    dates.sort(key=lambda date: datetime.strptime(date, "%Y-%m-%d"))

    for date in dates:
        # Tripdata-Daten
        tripdata_file_name = f"tripdata{date}.csv"
        tripdata_file_path = os.path.join(tripdata_path, tripdata_file_name)
        if os.path.exists(tripdata_file_path):
            tripdata = pd.read_csv(tripdata_file_path)
            total_time_reduction.append(tripdata["time_reduction"].sum())
        else:
            total_time_reduction.append(float("nan"))

    i = 0
    for date in dates:
        # Orders-Daten
        orders_file_name = f"orders_{date}.csv"
        orders_file_path = os.path.join(orders_path, orders_file_name)
        if os.path.exists(orders_file_path):
            orders = pd.read_csv(orders_file_path)
            num_orders = len(orders)
            average_time_reduction_per_day.append(
                total_time_reduction[i] / num_orders if num_orders > 0 else float("nan")
            )
        else:
            average_time_reduction_per_day.append(float("nan"))
        i += 1

    for total_time in total_time_reduction:
        # Teile die Gesamtzeitersparnis durch 100 (für die Anzahl der Autos) und dann durch 60 (für Minuten)
        if not pd.isna(total_time):
            time_per_car_in_minutes = (total_time / 1000) / 60 / 24
        else:
            time_per_car_in_minutes = float("nan")
        total_time_reduction_per_car_in_minutes.append(time_per_car_in_minutes)


    fig, ax4 = plt.subplots(1, 1, figsize=(15, 12))
    # Vierter Plot: Hier fügen Sie Ihren Code für den vierten Plot ein
    ax4.bar(dates, average_time_reduction_per_day, color="orange")
    ax4.set_xlabel("Datum")
    ax4.set_ylabel("Durchschnittliche Zeitersparnis pro Order pro Tag")
    ax4.set_title("Durchschnittliche Zeitersparnis pro Order pro Tag")
    ax4.set_xticklabels(dates, rotation=45)
    plt.show()

average_time_reduction_per_day()

def average_trip_distances_per_day_for_direct_routes():
    print("v5")
    orders_path = "code/data/for_hire"
    tripdata_path = "store/for_hire/rl_relocation/drivers/1000"
    total_time_reduction = []
    routes_per_day = []
    routes_per_day = []
    total_time_reduction = []
    average_route_length = {
        "driver_to_pickup_distance_km": [],
        "pickup_to_dropoff_distance_km": [],
    }
    average_combi_route_length = {
        "driver_to_pickup_distance_km": [],
        "pickup_to_dropoff_distance_km": [],
        "dropoff_to_destination_distance_km": [],
        "driver_to_dropoff_distance_km": [],
    }

    taxi_zones_file_path = "code/data/taxi_zones.csv"
    taxi_zones = pd.read_csv(taxi_zones_file_path)
    # Convert polygon strings to shapely polygon objects
    taxi_zones["polygon"] = taxi_zones["the_geom"].apply(loads)
    # Calculate the latitude and longitude of the center for each zone
    taxi_zones["center"] = taxi_zones["polygon"].apply(
        lambda p: (p.centroid.y, p.centroid.x)
    )
    # Create a dictionary to map zone IDs to their center latitude and longitude
    zone_centers = dict(zip(taxi_zones["LocationID"], taxi_zones["center"]))

    # Define a function to calculate the distance between two zones in kilometers
    def calculate_distance_km(zone1_id, zone2_id):
        if zone1_id in zone_centers and zone2_id in zone_centers:
            # Use geodesic from geopy to calculate the distance
            return geodesic(zone_centers[zone1_id], zone_centers[zone2_id]).kilometers
        else:
            raise Exception(f"{zone1_id} or {zone2_id} not found")
    tripdata_files = [
        f
        for f in os.listdir(tripdata_path)
        if f.endswith(".csv") and f.startswith("tripdata")
    ]
    dates = [re.search(r"(\d{4}-\d{2}-\d{2})", f).group(1) for f in tripdata_files]
    dates.sort(key=lambda date: datetime.strptime(date, "%Y-%m-%d"))
    for date in dates:
        # Tripdata-Daten
        tripdata_file_name = f"tripdata{date}.csv"
        tripdata_file_path = os.path.join(tripdata_path, tripdata_file_name)
        if os.path.exists(tripdata_file_path):
            tripdata = pd.read_csv(tripdata_file_path)
            routes_per_day.append(len(tripdata))
            total_time_reduction.append(tripdata["time_reduction"].sum())
            # Calculate distances for each trip in the dataset
            tripdata["driver_to_pickup_distance_km"] = tripdata.apply(
                lambda x: calculate_distance_km(
                    x["driver_start_zone_id"], x["passenger_pickup_zone_id"]
                ),
                axis=1,
            )
            tripdata["pickup_to_dropoff_distance_km"] = tripdata.apply(
                lambda x: calculate_distance_km(
                    x["passenger_pickup_zone_id"], x["passenger_dropoff_zone_id"]
                ),
                axis=1,
            )
            tripdata["dropoff_to_destination_distance_km"] = tripdata.apply(
                lambda x: calculate_distance_km(
                    x["passenger_dropoff_zone_id"], x["destination_id"]
                ),
                axis=1,
            )
            average_route_length["driver_to_pickup_distance_km"].append(
                tripdata[tripdata["combi_route"]]["driver_to_pickup_distance_km"].mean()
            )
            average_route_length["pickup_to_dropoff_distance_km"].append(
                tripdata[tripdata["combi_route"]][
                    "pickup_to_dropoff_distance_km"
                ].mean()
            )


        else:
            routes_per_day.append(float("nan"))
            total_time_reduction.append(float("nan"))

        # Orders-Daten

    fig, ax5 = plt.subplots(1, 1, figsize=(15, 12))
    ax5.bar(
        dates,
        average_route_length["driver_to_pickup_distance_km"],
        color="#90EE90",
        label="Fahrerposition bis Abholpunkt",
    )
    ax5.bar(
        dates,
        average_route_length["pickup_to_dropoff_distance_km"],
        color="green",
        alpha=0.6,
        label="Abholpunkt bis Ziel",
        bottom=average_route_length["driver_to_pickup_distance_km"],
    )
    ax5.set_xlabel("Datum")
    ax5.set_ylabel("Durchschnittliche Tripdistanzen pro Tag in km")
    ax5.set_title("Durchschnittliche Tripdistanzen pro Tag für Direktrouten")
    ax5.set_xticklabels(dates, rotation=45)
    plt.show()
#average_trip_distances_per_day_for_direct_routes()


def average_trip_distances_per_day_for_combination_routes():
    print("v6")

    orders_path = "code/data/for_hire"
    tripdata_path = "store/for_hire/rl_relocation/drivers/1000"
    total_time_reduction = []
    routes_per_day = []
    routes_per_day = []
    total_time_reduction = []
    average_route_length = {
        "driver_to_pickup_distance_km": [],
        "pickup_to_dropoff_distance_km": [],
    }
    average_combi_route_length = {
        "driver_to_pickup_distance_km": [],
        "pickup_to_dropoff_distance_km": [],
        "dropoff_to_destination_distance_km": [],
        "driver_to_dropoff_distance_km": [],
    }

    taxi_zones_file_path = "code/data/taxi_zones.csv"
    taxi_zones = pd.read_csv(taxi_zones_file_path)
    # Convert polygon strings to shapely polygon objects
    taxi_zones["polygon"] = taxi_zones["the_geom"].apply(loads)
    # Calculate the latitude and longitude of the center for each zone
    taxi_zones["center"] = taxi_zones["polygon"].apply(
        lambda p: (p.centroid.y, p.centroid.x)
    )
    # Create a dictionary to map zone IDs to their center latitude and longitude
    zone_centers = dict(zip(taxi_zones["LocationID"], taxi_zones["center"]))

    # Define a function to calculate the distance between two zones in kilometers
    def calculate_distance_km(zone1_id, zone2_id):
        if zone1_id in zone_centers and zone2_id in zone_centers:
            # Use geodesic from geopy to calculate the distance
            return geodesic(zone_centers[zone1_id], zone_centers[zone2_id]).kilometers
        else:
            raise Exception(f"{zone1_id} or {zone2_id} not found")
    tripdata_files = [
        f
        for f in os.listdir(tripdata_path)
        if f.endswith(".csv") and f.startswith("tripdata")
    ]
    dates = [re.search(r"(\d{4}-\d{2}-\d{2})", f).group(1) for f in tripdata_files]
    dates.sort(key=lambda date: datetime.strptime(date, "%Y-%m-%d"))
    for date in dates:
        # Tripdata-Daten
        tripdata_file_name = f"tripdata{date}.csv"
        tripdata_file_path = os.path.join(tripdata_path, tripdata_file_name)
        if os.path.exists(tripdata_file_path):
            tripdata = pd.read_csv(tripdata_file_path)
            routes_per_day.append(len(tripdata))
            total_time_reduction.append(tripdata["time_reduction"].sum())
            # Calculate distances for each trip in the dataset
            tripdata["driver_to_pickup_distance_km"] = tripdata.apply(
                lambda x: calculate_distance_km(
                    x["driver_start_zone_id"], x["passenger_pickup_zone_id"]
                ),
                axis=1,
            )
            tripdata["pickup_to_dropoff_distance_km"] = tripdata.apply(
                lambda x: calculate_distance_km(
                    x["passenger_pickup_zone_id"], x["passenger_dropoff_zone_id"]
                ),
                axis=1,
            )
            tripdata["dropoff_to_destination_distance_km"] = tripdata.apply(
                lambda x: calculate_distance_km(
                    x["passenger_dropoff_zone_id"], x["destination_id"]
                ),
                axis=1,
            )

            average_combi_route_length["driver_to_pickup_distance_km"].append(
                tripdata[tripdata["combi_route"]]["driver_to_pickup_distance_km"].mean()
            )

            average_combi_route_length["pickup_to_dropoff_distance_km"].append(
                tripdata[tripdata["combi_route"]][
                    "pickup_to_dropoff_distance_km"
                ].mean()
            )

            average_combi_route_length["driver_to_dropoff_distance_km"].append(
                average_combi_route_length["driver_to_pickup_distance_km"][-1]
                + average_combi_route_length["pickup_to_dropoff_distance_km"][-1]
            )

            average_combi_route_length["dropoff_to_destination_distance_km"].append(
                tripdata[tripdata["combi_route"]][
                    "dropoff_to_destination_distance_km"
                ].mean()
            )

        else:
            routes_per_day.append(float("nan"))
            total_time_reduction.append(float("nan"))

    fig, ax6 = plt.subplots(1, 1, figsize=(15, 12))
        # Combined plot
    ax6.bar(
        dates,
        average_combi_route_length["driver_to_pickup_distance_km"],
        color="#90EE90",
        label="Fahrerposition bis Abholpunkt",
    )
    ax6.bar(
        dates,
        average_combi_route_length["pickup_to_dropoff_distance_km"],
        color="green",
        label="Abholpunkt bis Station",
        bottom=average_combi_route_length["driver_to_pickup_distance_km"],
    )
    ax6.bar(
        dates,
        average_combi_route_length["dropoff_to_destination_distance_km"],
        color="#006400",
        label="Station bis Ziel",
        bottom=average_combi_route_length["driver_to_dropoff_distance_km"],
    )
    ax6.set_xlabel("Datum")
    ax6.set_ylabel("Durchschnittliche Tripdistanzen pro Tag in km")
    ax6.set_title("Durchschnittliche Tripdistanzen pro Tag für Kombinationsrouten")
    ax6.set_xticklabels(dates, rotation=45)
    plt.show()
#average_trip_distances_per_day_for_combination_routes()


## fig 1.
def Zeitersparnis_Anzahl_der_Autos():
    base_path = "store/for_hire/rl_relocation/drivers"   # manuell eingegeben werden 
    driver_counts = [10, 100, 1000]
    time_savings_avg = {}

    for count in driver_counts:
        total_time_reduction = 0
        file_count = 0
        path = os.path.join(base_path, str(count))
        for file_name in os.listdir(path):
            if file_name.startswith("tripdata") and file_name.endswith(".csv"):
                df = pd.read_csv(os.path.join(path, file_name))
                total_time_reduction += df['time_reduction'].sum()
                file_count += 1
        
        if file_count > 0:
            avg_time_savings_per_hour_per_car = (total_time_reduction / 60) / (24 * file_count) 
            time_savings_avg[count] = avg_time_savings_per_hour_per_car / int(count)

    # Plotting
    plt.figure(figsize=(10, 6))
    bar_width = 0.4  # You can adjust the width as needed
    positions = range(len(time_savings_avg))
    
    plt.bar(positions, time_savings_avg.values(), color='skyblue', width=bar_width)
    plt.xlabel('Anzahl der Autos')
    plt.ylabel('Zeitersparnis pro Stunde pro Auto (Minuten)')
    plt.title('Zeitersparnis pro Stunde pro Autos（for_hire/rl_relocation）')
    
    plt.xticks(positions, labels=[str(k) for k in time_savings_avg.keys()])
    
    plt.show()

Zeitersparnis_Anzahl_der_Autos()


## fig 2.
def Ablehnungsqoute_in_unterschiedlicher_Anzahl_der_Autos():
    base_path = "store/for_hire/rl_relocation/drivers"   # manuell eingegeben werden 
    driver_counts = [10, 100, 1000]  
    quota_sums = {}
    file_counts = {}

    for count in driver_counts:
        path = os.path.join(base_path, str(count))
        for file_name in os.listdir(path):
            if file_name.startswith("ordersdata") and file_name.endswith(".csv"):
                file_path = os.path.join(path, file_name)
                df = pd.read_csv(file_path)
                quota_sum = df['quota_of_unserved_orders'].sum()
                
                
                if count not in quota_sums:
                    quota_sums[count] = 0
                    file_counts[count] = 0
                
                quota_sums[count] += quota_sum
                file_counts[count] += len(df)


    avg_quota_per_driver = {count: quota_sums[count] / file_counts[count] for count in driver_counts}
    plt.figure(figsize=(10, 6))
    positions = range(len(avg_quota_per_driver))
    plt.bar(positions, avg_quota_per_driver.values(), color='skyblue')
    plt.xlabel('Anzahl der Autos')
    plt.ylabel('Ablehnungsqoute')
    plt.title('Ablehnungsqoute in unterschiedlicher Anzahl der Autos（for_hire/rl_relocation）')
    plt.xticks(positions, labels=[str(k) for k in avg_quota_per_driver.keys()])
    plt.show()

Ablehnungsqoute_in_unterschiedlicher_Anzahl_der_Autos()



## fig 3.
## Achtung: die CSV-Datei werden direkt im Ordner „store/for_hire/rl“ oder „store/for_hire/drl“ gespeichert 
#            und nicht im Ordner „store/for_hire/rl/drivers/10“ gespeichert.
## Das bedeutet, dass der Ordner „store/for_hire/rl“ viele CSV-Dateien (für fig. 3) und Treiberordner "driver" (für fig. 1 oder fig. 2) enthält.
def calculate_percentage(combi_route_counts, total_counts):
    combi_route_percentages = {}
    for method, count in combi_route_counts.items():
        total = total_counts[method]
        percentage = (count / total) * 100 if total > 0 else 0
        combi_route_percentages[method] = percentage
    return combi_route_percentages

def process_directory(data_path, combi_route_counts, total_counts):
    for entry in os.listdir(data_path):
        entry_path = os.path.join(data_path, entry)
        if entry.endswith(".csv"):
            df = pd.read_csv(entry_path)
            combi_true_count = df[df['combi_route'] == True].shape[0]
            modeling_method = os.path.basename(data_path)  
            combi_route_counts[modeling_method] += combi_true_count
            total_counts[modeling_method] += df.shape[0]

def Anzahl_der_Combirouten_in_Prozent():
    data_path = "store/for_hire"   # manuell eingegeben werden 
    combi_route_counts = {}
    total_counts = {}

    for method_dir in os.listdir(data_path):
        method_path = os.path.join(data_path, method_dir)
        if os.path.isdir(method_path):
            combi_route_counts[method_dir] = 0
            total_counts[method_dir] = 0
            process_directory(method_path, combi_route_counts, total_counts)


    combi_route_percentages = calculate_percentage(combi_route_counts, total_counts)

    method_names = {
        'baseline': 'Baseline',
        'rl': 'RL',
        'rl_relocation': 'RL mit relocation',
        'drl': 'DRL'
    }

    labels = [method_names.get(method, method) for method in combi_route_percentages.keys()]
    percentages = list(combi_route_percentages.values())

    plt.figure(figsize=(10, 6))
    plt.bar(labels, percentages, color='skyblue')
    plt.xlabel('Modellierungsmethode')
    plt.ylabel('Prozentualer Anteil der Combirouten (%)')
    plt.title('Anzahl der Combirouten in Prozent')
    plt.xticks(rotation=45)
    plt.show()

Anzahl_der_Combirouten_in_Prozent()



## fig 4.
def process_directory_for_route_distribution(data_path, combi_route_counts, direct_route_counts, file_days):
    for method_dir in os.listdir(data_path):
        method_path = os.path.join(data_path, method_dir)
        if os.path.isdir(method_path):
            combi_route_counts[method_dir] = 0
            direct_route_counts[method_dir] = 0
            file_days[method_dir] = 0

            for file_name in os.listdir(method_path):
                if file_name.endswith(".csv"):
                    df = pd.read_csv(os.path.join(method_path, file_name))
                    combi_true_count = df[df['combi_route'] == True].shape[0]
                    combi_false_count = df[df['combi_route'] == False].shape[0]

                    combi_route_counts[method_dir] += combi_true_count
                    direct_route_counts[method_dir] += combi_false_count
                    file_days[method_dir] += 1

def Routen_Aufteilung_pro_Stunde():
    data_path = "store/for_hire"   # Manuell eingegebener Pfad

    combi_route_counts = {}
    direct_route_counts = {}
    file_days = {}

    process_directory_for_route_distribution(data_path, combi_route_counts, direct_route_counts, file_days)
    hours_per_day = 24
    combi_avg_per_hour = {method: (combi_route_counts[method] / (file_days[method] * hours_per_day)) 
                          if file_days[method] > 0 else 0
                          for method in combi_route_counts}
    direct_avg_per_hour = {method: (direct_route_counts[method] / (file_days[method] * hours_per_day)) 
                           if file_days[method] > 0 else 0
                           for method in direct_route_counts}

    labels = ['Baseline', 'RL', 'RL mit relocation', 'DRL']
    mapping = {'baseline': 'Baseline', 'rl': 'RL', 'rl_relocation': 'RL mit relocation', 'drl': 'DRL'}
    combi_data = [combi_avg_per_hour.get(method, 0) for method in mapping]
    direct_data = [direct_avg_per_hour.get(method, 0) for method in mapping]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, combi_data, width, label='Combirouten')
    rects2 = ax.bar(x + width/2, direct_data, width, label='Direktrouten')

    ax.set_xlabel('Modellierungsmethode')
    ax.set_ylabel('Anzahl der Routen pro Stunde')
    ax.set_title('Aufteilung der Routen in Combi und Direktroute pro Stunde')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.show()

Routen_Aufteilung_pro_Stunde()


## fig 5.
def Combirouten_pro_Stunde_in_unterschiedlicher_Autoanzahl():
    base_path = "store/for_hire/drl/drivers" # Manuell eingegebener Pfad
    driver_counts = [10, 100, 1000]  
    combi_counts = {}
    file_days = {}

    for count in driver_counts:
        path = os.path.join(base_path, str(count))
        if os.path.isdir(path):
            combi_counts[count] = 0
            file_days[count] = 0
            
            
            for file_name in os.listdir(path):
                if file_name.startswith("tripdata") and file_name.endswith(".csv"):
                    file_path = os.path.join(path, file_name)
                    
                    
                    df = pd.read_csv(file_path)
                    combi_true_count = (df['combi_route'] == True).sum()
                    
                    
                    combi_counts[count] += combi_true_count
                    file_days[count] += 1

    
    hours_per_day = 24
    combi_avg_per_hour = {count: (combi_counts[count] / (file_days[count] * hours_per_day * int(count))) 
                          for count in combi_counts}

    plt.figure(figsize=(10, 6))
    plt.bar([str(count) for count in driver_counts], [combi_avg_per_hour.get(count, 0) for count in driver_counts], color='skyblue')
    plt.xlabel('Anzahl der Autos')
    plt.ylabel('Anzahl der Combirouten pro Stunde')
    plt.title('Anzahl der Combirouten pro Stunde in unterschiedlicher Anzahl von Autos')
    plt.xticks([str(count) for count in driver_counts])
    plt.show()

Combirouten_pro_Stunde_in_unterschiedlicher_Autoanzahl()


## fig 6.
def durchschnittlich_gefahrene_Distanz_Anzahl_der_Autos():
    base_path = "store/for_hire/drl/drivers" # Manuell eingegebener Pfad
    driver_counts = [10, 100, 1000]
    status_speed_kmh = {'occupied': 6.33 * 3.6, 'idling': 0, 'relocation': 6.33 * 3.6}  

    status_time_proportion = {driver: {'occupied': [], 'idling': [], 'relocation': []} for driver in driver_counts}

    for count in driver_counts:
        drivers_path = os.path.join(base_path, str(count))
        for file_name in os.listdir(drivers_path):
            if file_name.startswith("driverdata"):
                file_path = os.path.join(drivers_path, file_name)
                df = pd.read_csv(file_path)
                total_records = len(df) 
                for status in status_time_proportion[count]:
                
                    status_records = df[df['status'] == status].shape[0]
                    status_proportion = status_records / total_records if total_records > 0 else 0
                    status_time_proportion[count][status].append(status_proportion)

    avg_status_proportion = {driver: {status: np.mean(proportions) for status, proportions in statuses.items()} for driver, statuses in status_time_proportion.items()}
  
    avg_distance_data = {driver: {status: avg_status_proportion[driver][status] * status_speed_kmh[status] for status in status_speed_kmh} for driver in driver_counts}

    
    fig, ax = plt.subplots()
    width = 0.25
    ind = np.arange(len(driver_counts))

    for i, (status, color) in enumerate(zip(['occupied', 'idling', 'relocation'], ['skyblue', 'orange', 'green'])):
        avg_distances = [avg_distance_data[driver][status] for driver in driver_counts]
        ax.bar(ind + i * width, avg_distances, width, label=status.capitalize(), color=color)

    ax.set_xlabel('Anzahl der Autos')
    ax.set_ylabel('Durchschnittlich gefahrene Distanz pro Stunde (km)')
    ax.set_title('Durchschnittlich gefahrene Distanz nach Status')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(driver_counts)
    ax.legend()

    plt.tight_layout()
    plt.show()

durchschnittlich_gefahrene_Distanz_Anzahl_der_Autos()


## fig 7.
def durchschnittlich_gefahrene_Distan_modeling_method():
    base_path = "store/for_hire"   # Manuell eingegebener Pfad
    methods = ['baseline', 'rl', 'rl_relocation', 'drl']
    status_speed = {'occupied': 6.33, 'idling': 0, 'relocation': 6.33}  


    status_proportions = {method: {'occupied': [], 'idling': [], 'relocation': []} for method in methods}
    total_counts = {method: [] for method in methods}  


    for method in methods:
        method_path = os.path.join(base_path, method)
        for file_name in os.listdir(method_path):
            if file_name.startswith("driverdata"):  
                file_path = os.path.join(method_path, file_name)
                df = pd.read_csv(file_path)
                total_count = len(df)
                for status in status_speed:
                    status_count = df[df['status'] == status].shape[0]
                    status_proportions[method][status].append(status_count / total_count if total_count else 0)

    avg_proportions = {method: {status: np.mean(proportions) for status, proportions in status_data.items()} 
                       for method, status_data in status_proportions.items()}

    avg_distance_data = {method: {status: avg_proportions[method][status] * status_speed[status] * 3.6 for status in status_speed}
                         for method in methods}

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.2
    ind = np.arange(len(methods))
    for i, (status, color) in enumerate(zip(['occupied', 'idling', 'relocation'], ['skyblue', 'orange', 'green'])):
        avg_distances = [avg_distance_data[method][status] for method in methods]
        ax.bar(ind + i * width, avg_distances, width, label=status.capitalize(), color=color)

    ax.set_xlabel('Modellierungsmethode')
    ax.set_ylabel('Durchschnittlich gefahrene Distanz pro Stunde (km)')
    ax.set_title('Durchschnittlich gefahrene Distanz nach Methode und Status')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(methods)
    ax.legend()

    plt.tight_layout()
    plt.show()

durchschnittlich_gefahrene_Distan_modeling_method()
