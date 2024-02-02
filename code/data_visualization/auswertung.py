import os
import re
import pandas as pd
import shutil
import matplotlib.pyplot as plt


path = "code/data_output/tripdata2023-07-11.csv"
pathrl = "code/data_output/relocation_trip_data2023-07-11.csv"
pathdriver = "code/data_output/driverdata2023-07-11.csv"
data = pd.read_csv(path)
datarl = pd.read_csv(pathrl)
datadriver = pd.read_csv(pathdriver)
print(f'Anzahl der Routen: {len(data)}')
print(f'Prozentualer Anteil Combirouten: {round(sum(data["combi_route"]/len(data)), 2)}')
print(f'Durschnittliche Routenlänge: {round(sum(data["total_vehicle_distance"]/len(data)),2)}')
print(f'Durchschnittliche Zeitersparnis: {round(sum(data["time_reduction"]/60/24/100), 2)}')
print(f'Anzahl an relocation: {len(datarl)}')
print(f'Durchschnittliche Entfernung relocation: {round(sum(datarl["distance"]/len(datarl)), 2)}')

anzahl_occupied = datadriver.loc[datadriver["status"] == "occupied", "status"].count()
anzahl_relocation = datadriver.loc[datadriver["status"] == "relocation", "status"].count()
anzahl_idling = datadriver.loc[datadriver["status"] == "idling", "status"].count()
print(f'Anzahl der "occupied": {anzahl_occupied/len(datadriver), anzahl_relocation/len(datadriver), anzahl_idling/len(datadriver)}')
