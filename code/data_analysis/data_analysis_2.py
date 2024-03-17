import os
import re
import pandas as pd
import shutil
import matplotlib.pyplot as plt

#a second data analysis. The data must be in the /store/ folder

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
start_date = "2023-07-17"
end_date = "2023-07-23"

# Basispfade ohne Datum für jede Dateiart
base_paths = {
    "tripdata": "store/for_hire/rl_relocation/tripdata",
    "relocation_trip_data": "store/for_hire/rl_relocation/relocation_trip_data",
    "driverdata": "store/for_hire/rl_relocation/driverdata",
    "orders": "code/data/for_hire/orders_"
}

# Laden und Zusammenführen der Daten für jeden Dateityp
data = load_and_merge_data(base_paths["tripdata"], start_date, end_date)
datarl = load_and_merge_data(base_paths["relocation_trip_data"], start_date, end_date)
datadriver = load_and_merge_data(base_paths["driverdata"], start_date, end_date)
dataorders = load_and_merge_data(base_paths["orders"], start_date, end_date)


print(f'Durchschnittliche Zeitersparnis: {round(sum(data["time_reduction"]/60/24/7/2000), 2)}')
print(f'Anzahl Orders pro Tag: {len(data)/7}')
print(f'Prozentualer Anteil Combirouten: {round(sum(data["combi_route"]/len(data)), 2)}')


print(f'Durschnittliche Routenlänge: {round(sum(data["total_vehicle_distance"]/len(data)),2)}')

anzahl_occupied = datadriver.loc[datadriver["status"] == "occupied", "status"].count()
anzahl_relocation = datadriver.loc[datadriver["status"] == "relocation", "status"].count()
anzahl_idling = datadriver.loc[datadriver["status"] == "idling", "status"].count()
print(f'Anzahl der "occupied": {anzahl_occupied/len(datadriver), anzahl_relocation/len(datadriver), anzahl_idling/len(datadriver)}')


print(f'Anzahl der Routen: {len(data)}')

gesamte_zeitersparnis =  round(sum(data["time_reduction"]/60), 2)
print(f'Gesame Zeitersparnis in Minute: {gesamte_zeitersparnis}')
print(f'Durchschnittliche Zeitersparnis pro Minute pro Order: {gesamte_zeitersparnis/len(dataorders)}')
print(f'Prozentualer Anteil Orders: {len(data)/len(dataorders)}')

print(f'Anzahl an relocation: {len(datarl)}')
print(f'Durchschnittliche Entfernung relocation: {round(sum(datarl["distance"]/len(datarl)), 2)}')

pathstate = "code/training_data/state_value_table.csv"
datastate = pd.read_csv(pathstate)

print("##########################################################################################")
anzahl_max_statevalue = datastate.loc[datastate["state_value"] == 1000, "state_value"].count()
print(f'Anzahl state_value_Werte = maximum: {anzahl_max_statevalue}')
print(f'Prozentualer Anteil: {anzahl_max_statevalue/len(datastate)}')

anzahl_min_statevalue = datastate.loc[datastate["state_value"] == -1000, "state_value"].count()
print(f'Anzahl state_value_Werte = min: {anzahl_min_statevalue}')
print(f'Prozentualer Anteil: {anzahl_min_statevalue/len(datastate)}')

anzahl_kleiner0_statevalue = datastate.loc[datastate["state_value"] <= 0, "state_value"].count()
print(f'Anzahl state_value_Werte = keliner 0: {anzahl_kleiner0_statevalue}')
print(f'Prozentualer Anteil: {anzahl_kleiner0_statevalue/len(datastate)}')
