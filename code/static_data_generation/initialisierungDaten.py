import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt

import pandas as pd

# Lese die existierende CSV-Datei ein
df = pd.read_csv('training_data\stationen_NY_full.csv')

# Filtere die Daten, um nur Zeilen mit "Borough" = "M" zu behalten. Ich erhalte damitz nicht soo gute Ergebnisse. Nur verwenden, wenn Laufzeitproblem
#filtered_df = df[df['borough'] == 'M']
# Speichere die gefilterten Daten in einer neuen CSV-Datei
#filtered_df.to_csv('training_data\stationen_NY_M_only.csv', index=False)


def haversine(lon1, lat1, lon2, lat2):
    """
    Berechnet den Kreisabstand zwischen zwei Punkten auf der Erdoberfläche 
    anhand ihrer Längen- und Breitengrade.
    """
    # Umwandeln von Dezimalgraden in Radianten
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine-Formel
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius der Erde in Kilometern
    return c * r

def manhattan_distance(lon1, lat1, lon2, lat2):
    """
    Berechnet die Manhattan-Distanz zwischen zwei Punkten auf der Erdoberfläche 
    anhand ihrer Längen- und Breitengrade.
    """
    # Berechne die Ost-West-Distanz (Längendifferenz)
    ew_distance = haversine(lon1, lat1, lon2, lat1)

    # Berechne die Nord-Süd-Distanz (Breitendifferenz)
    ns_distance = haversine(lon1, lat1, lon1, lat2)

    # Addiere beide Distanzen für die Manhattan-Distanz
    return ew_distance + ns_distance

def calculate_travel_times(zones_df, speed):
    """
    Berechnet die Umsteigezeiten zwischen allen Zonen.
    """
    # Initialisierung einer leeren Liste für die Ergebnisse
    travel_times = []

    # Berechnung der Umsteigezeiten für jedes Zonenpaar
    for i in range(len(zones_df)):
        for j in range(len(zones_df)):
            if i != j:
                distance = manhattan_distance(zones_df.iloc[i]['Long'], zones_df.iloc[i]['Lat'],
                                              zones_df.iloc[j]['Long'], zones_df.iloc[j]['Lat'])
                time = distance*1000 / speed
                travel_times.append([zones_df.iloc[i]['Zone'], zones_df.iloc[j]['Zone'], distance, time])

    return pd.DataFrame(travel_times, columns=['Startzone', 'Endzone', 'Distanz (km)', 'Zeit (h)'])

# CSV-Datei 'zones.csv' mit den Spalten 'Zone', 'Lat', 'Long'
zones_df = pd.read_csv('training_data\zones.csv')

# Geschwindigkeit (angenommen in km/h)# unbedingt ädnern!
speed = 1.3  # Diese Variable können Sie je nach Bedarf anpassen

# Berechnung der Umsteigezeiten (dieser Schritt wird ausgeführt, sobald Sie die Daten haben)
travel_times_df = calculate_travel_times(zones_df, speed)

# Speichern der Ergebnisse als CSV-Datei
travel_times_df.to_csv('training_data\zone_travel_times.csv', index=False)



# Hinweis: Der Code ist so konzipiert, dass Sie Ihre eigene CSV-Datei mit den Zonendaten einlesen können.
### Stationen: https://data.ny.gov/Transportation/MTA-Subway-Stations/39hk-dx4f
#https://new.mta.info/map/5341
#https://new.mta.info/document/9426
#https://data.ny.gov/widgets/i9wp-a4ja