import pandas as pd

# Pfad zu Ihrer Parquet-Datei
file_path = 'yellow_tripdata_2015-01.parquet'

# Laden der Parquet-Datei in einen DataFrame
df = pd.read_parquet(file_path)


#csv_file_path = 'yellow_tripdata_2015-01.csv'

# Speichern des DataFrames als CSV
#df.to_csv(csv_file_path, index=False)

pd.set_option('display.max_columns', None)
# Anzeigen der ersten Zeilen des DataFrame
print(df.head())
print(df.columns)

print(df[(df["DOLocationID"] == 166) & (df["PULocationID"] == 141)])
