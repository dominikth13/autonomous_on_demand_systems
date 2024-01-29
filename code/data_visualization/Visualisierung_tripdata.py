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


def visualize_trip_data():
    
    print("HI")
    tripdata_path = "code/data_output"
    orders_path = "code/data/yellow_cab"  # Pfad zu den orders-Dateien

    # Liste alle Dateien im Verzeichnis auf
    tripdata_files = [f for f in os.listdir(tripdata_path) if f.endswith('.csv') and f.startswith('tripdata')]
    orders_files = [f for f in os.listdir(orders_path) if f.startswith('orders_2015_07')]

    # Extrahieren Sie die Daten aus den Dateinamen der tripdata-Dateien und sortieren Sie sie
    dates = [re.search(r'(\d{4}-\d{2}-\d{2})', f).group(1) for f in tripdata_files]
    dates.sort(key=lambda date: datetime.strptime(date, '%Y-%m-%d'))

    # Initialisieren Sie Listen für die Diagrammdaten
    routes_per_day = []
    total_time_reduction = []
    average_time_reduction_per_day = []
    average_occupied_drivers = []
    total_time_reduction_per_car_in_minutes = []

    # Berechnen Sie die Daten für alle Diagramme

    workload_files = [f for f in os.listdir(tripdata_path) if f.endswith('.csv') and f.startswith('workload')]
    for date in dates:
        workload_file_name = f"workload{date}.csv"
        workload_file_path = os.path.join(tripdata_path, workload_file_name)
        if os.path.exists(workload_file_path):
            workload_data = pd.read_csv(workload_file_path)
            average_occupied_drivers.append(workload_data["num_of_occupied_driver"].mean())
        else:
            average_occupied_drivers.append(float('nan'))


    for date in dates:
        # Tripdata-Daten
        tripdata_file_name = f"tripdata{date}.csv"
        tripdata_file_path = os.path.join(tripdata_path, tripdata_file_name)
        if os.path.exists(tripdata_file_path):
            tripdata = pd.read_csv(tripdata_file_path)
            routes_per_day.append(len(tripdata))
            total_time_reduction.append(tripdata["time_reduction"].sum())
        else:
            routes_per_day.append(float('nan'))
            total_time_reduction.append(float('nan'))
    

        # Orders-Daten

    i = 0 
    for date in dates:
        # Orders-Daten
        orders_file_name = f"orders_{date}.csv"
        orders_file_path = os.path.join(orders_path, orders_file_name)
        if os.path.exists(orders_file_path):
            orders = pd.read_csv(orders_file_path)
            num_orders = len(orders)
            average_time_reduction_per_day.append(total_time_reduction[i] / num_orders if num_orders > 0 else float('nan'))
        else:
            average_time_reduction_per_day.append(float('nan'))
        i += 1

    for total_time in total_time_reduction:
        # Teile die Gesamtzeitersparnis durch 100 (für die Anzahl der Autos) und dann durch 60 (für Minuten)
        if not pd.isna(total_time):
            time_per_car_in_minutes = (total_time / 100) / 60 / 24
        else:
            time_per_car_in_minutes = float('nan')
        total_time_reduction_per_car_in_minutes.append(time_per_car_in_minutes)


    # Erstellen Sie eine Figur mit vier Subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    ax1, ax2, ax3, ax4 = axs.flatten()

    # Erster Plot: Anzahl der Routen pro Tag
    ax1.bar(dates, average_occupied_drivers, color='blue')
    ax1.set_xlabel('Datum')
    ax1.set_ylabel('Durchschnittlich besetzte Fahrer')
    ax1.set_title('Durchschnittlich besetzte Fahrer pro Tag')
    ax1.set_xticklabels(dates, rotation=45)
    ax1.set_ylim(0, 100)

    # Zweiter Plot: Summierte Zeitersparnis pro Tag
    ax2.bar(dates, routes_per_day, color='blue')
    ax2.set_xlabel('Datum')
    ax2.set_ylabel('Anzahl der Routen')
    ax2.set_title('Anzahl der Routen pro Tag')
    ax2.set_xticklabels(dates, rotation=45)


    ax3.bar(dates, total_time_reduction_per_car_in_minutes, color='red')
    ax3.set_xlabel('Datum')
    ax3.set_ylabel('Zeitersparnis pro Auto (Minuten)')
    ax3.set_title('Summierte Zeitersparnis pro Stunde pro Auto (Minuten)')
    ax3.set_xticklabels(dates, rotation=45)

    # Vierter Plot: Hier fügen Sie Ihren Code für den vierten Plot ein
    ax4.bar(dates, average_time_reduction_per_day, color='orange')
    ax4.set_xlabel('Datum')
    ax4.set_ylabel('Durchschnittliche Zeitersparnis pro Order pro Tag')
    ax4.set_title('Durchschnittliche Zeitersparnis pro Order pro Tag')
    ax4.set_xticklabels(dates, rotation=45)
    # Zeige die Figur an
    plt.tight_layout()
    plt.savefig(f'code/data_visualization/waaa.png')
    


    prozentual_combined_orders = []
    for date in dates:
        # Tripdata-Daten
        tripdata_file_name = f"tripdata{date}.csv"
        tripdata_file_path = os.path.join(tripdata_path, tripdata_file_name)
        if os.path.exists(tripdata_file_path):
            tripdata = pd.read_csv(tripdata_file_path)
            prozentual_combined_orders.append(tripdata["combi_route"].sum()/len(tripdata))
            total_time_reduction.append(tripdata["time_reduction"].sum())
        #else:
         #   routes_per_day.append(float('nan'))
          #  total_time_reduction.append(float('nan'))

    print(prozentual_combined_orders)
    print(sum(prozentual_combined_orders)/len(prozentual_combined_orders))

    plt.show()
    


    
    
    
    