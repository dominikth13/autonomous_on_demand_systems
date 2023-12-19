import pandas as pd
from datetime import datetime, timedelta

# Laden der .parquet-Datei
df = pd.read_parquet('code\data\orders.parquet')
# Liste der Werte, die in PULocationID und DOLocationID nicht mehr vorkommen dürfen
df = df[['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'DOLocationID', 'PULocationID']]
verbotene_werte = [23, 44, 84, 204, 5, 6, 109, 110, 176, 172, 214, 221, 206, 156, 187, 118, 99, 264, 1, 265, 57, 251, 245, 115, 105, 104]  # Ersetzen Sie dies mit Ihren eigenen Werten
#                   1, 5, 6, 23, 44, 57, 84, 99, 104, 105, 109, 110, 115, 118, 156, 172, 176, 187, 204, 206, 214, 221, 245, 251
# Filtern des DataFrames
df = df[~df['PULocationID'].isin(verbotene_werte) & ~df['DOLocationID'].isin(verbotene_werte)]

# Umwandeln von 'tpep_pickup_datetime' in ein Datumsformat
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
print(df.head())

# Extrahieren des Datums und der Uhrzeit in separate Spalten
df['pickup_date'] = df['tpep_pickup_datetime'].dt.date
df['pickup_time'] = df['tpep_pickup_datetime'].dt.time

# Start- und Enddatum des Zeitraums festlegen
start_date = datetime(2015, 7, 1)
end_date = datetime(2015, 7, 4)

# Für jeden Tag im Zeitraum
current_date = start_date
while current_date <= end_date:
    # Filtern der Daten für den aktuellen Tag
    day_data = df[df['pickup_date'] == current_date.date()].copy()  # Verwenden von 'pickup_date'

    # Sortieren der Daten nach 'tpep_pickup_datetime'
    day_data.sort_values(by='tpep_pickup_datetime', inplace=True)

    # Speichern der gefilterten Daten in einer neuen Datei im 'code\data\' Verzeichnis
    day_data.to_csv(f'code\data\orders_{current_date.strftime("%Y-%m-%d")}.csv', index=False)
    
    # Zum nächsten Tag übergehen
    current_date += timedelta(days=1)

print(df['pickup_time'].dtype)