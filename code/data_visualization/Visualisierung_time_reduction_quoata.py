import pandas as pd
import matplotlib.pyplot as plt
import os

# Pfad zu den CSV-Dateien
path = "code/data_output/"

# Liste für die durchschnittliche nicht kumulierte Zeitersparnis pro Zeile und die Daten
average_non_cumulative_savings = []
dates = []
previous_day_savings = 0

for day in range(1, 31):
    date_str = f"2015-07-{day:02d}"
    file_path = os.path.join(path, f"time_reduction_quota_{date_str}.csv")
    if os.path.exists(file_path):
        # Lese die CSV-Datei ein
        df = pd.read_csv(file_path)
        # Berechne die kumulierte Zeitersparnis für diesen Tag
        cumulative_savings = df['quota_of_saved_time_for_all_served_orders'].sum()
        # Berechne die nicht kumulierte Zeitersparnis
        daily_savings = cumulative_savings - previous_day_savings
        # Berechne die durchschnittliche Zeitersparnis pro Zeile
        average_daily_savings = daily_savings / len(df) if len(df) > 0 else 0
        average_non_cumulative_savings.append(average_daily_savings)
        previous_day_savings = cumulative_savings
        dates.append(date_str)
    else:
        print(f"Datei nicht gefunden: {file_path}")

# Erstellen des Balkendiagramms
plt.bar(dates, average_non_cumulative_savings)
plt.gca().invert_yaxis()  # Y-Achse umkehren
plt.xlabel("Datum")
plt.ylabel("Durchschnittliche nicht kumulierte Zeitersparnis pro Zeile")
plt.xticks(rotation=45)
plt.title("Durchschnittliche nicht kumulierte Zeitersparnis pro Tag")
plt.show()