import os
import re
import pandas as pd
import shutil
import matplotlib.pyplot as plt

def load_and_merge_data(base_path, start_date, end_date):
    date_range = pd.date_range(start_date, end_date)
    dfs = []  # Eine Liste zum Speichern der einzelnen DataFrames
    for single_date in date_range:
        formatted_date = single_date.strftime("%Y-%m-%d")
        file_path = f"{base_path}{formatted_date}.csv"
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
        except FileNotFoundError:
            print(f"Datei nicht gefunden: {file_path}")
    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
    else:
        merged_df = pd.DataFrame()
    return merged_df

# Start- und Enddatum für den gewünschten Zeitraum festlegen
start_date = "2023-07-10"
end_date = "2023-07-30"

# Basispfade ohne Datum für jede Dateiart
base_paths = {
    "tripdata": "store/for_hire/rl_relocation/tripdata",
    "relocation_trip_data": "store/for_hire/rl_relocation/relocation_trip_data",
    "driverdata": "store/for_hire/rl_relocation/driverdata",
    "orders": "code/data/for_hire/orders_"
}

data = load_and_merge_data(base_paths["tripdata"], start_date, end_date)
datarl = load_and_merge_data(base_paths["relocation_trip_data"], start_date, end_date)
datadriver = load_and_merge_data(base_paths["driverdata"], start_date, end_date)
dataorders = load_and_merge_data(base_paths["orders"], start_date, end_date)

print(f'Anzahl Orders pro Tag: {len(dataorders)/21}')