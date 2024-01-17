import pandas as pd

# Pfad zur CSV-Datei
input_file = 'code/data/orders_2015-07-01.csv'

# Laden der CSV-Datei in einen DataFrame
df = pd.read_csv(input_file)

# Extrahieren von 5% der Daten
sampled_df = df.sample(frac=0.05, random_state=1)  # random_state für reproduzierbare Ergebnisse

# Optional: Speichern des extrahierten Datensatzes in einer neuen CSV-Datei
sampled_df.to_csv('code/data/orders_2015-07-01__5_percent.csv', index=False)
