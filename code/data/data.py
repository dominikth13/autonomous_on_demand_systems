import pandas as pd
from datetime import datetime, timedelta

# Pfad zur Originaldatei im Parquet-Format
input_file = 'yellow_tripdata_201-07(1).parquet'

# Datumseinstellungen
start_date = datetime(2015, 7, 1)
end_date = datetime(2015, 7, 30)

# Spalten, die behalten werden sollen
columns_to_keep = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'DOLocationID', 'PULocationID', 'pickup_date', 'pickup_time']

# DataFrame aus der Parquet-Datei erstellen und Spalten filtern
df = pd.read_parquet(input_file)[columns_to_keep]

# Datumsspalte in datetime umwandeln, falls nötig
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])

# Daten für jeden Tag filtern und in eine separate Datei speichern
current_date = start_date
while current_date <= end_date:
    # Daten für den aktuellen Tag filtern
    filtered_df = df[df['tpep_pickup_datetime'].dt.date == current_date.date()]

    # Dateiname für den gefilterten Datensatz
    output_file = f'orders_{current_date.strftime("%Y-%m-%d")}.csv'

    # Gefilterten Datensatz in CSV speichern
    filtered_df.to_csv(output_file, index=False)
    
    # Zum nächsten Tag wechseln
    current_date += timedelta(days=1)
