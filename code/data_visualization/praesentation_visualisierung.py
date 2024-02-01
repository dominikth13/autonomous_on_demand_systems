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



def plot1():
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

plot1()





