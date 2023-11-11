#import random
from location import Location
from utils import IdProvider
import csv
ID_PROVIDER = IdProvider()


class Driver:
    def __init__(self, start_position: Location) -> None:
        self.id = ID_PROVIDER.get_id()
        self.current_position = start_position
        self.job: tuple[int, Location] = None

    def is_occupied(self) -> bool:
        return self.job != None

    # Total driving time in seconds
    def set_new_job(self, total_driving_time: int, new_position: Location) -> None:
        self.job = (total_driving_time, new_position)

    # Duration in seconds
    def update_job_status(self, duration: int) -> None:
        if self.job[0] - duration < 0:
            # Job is finished by next interval
            self.current_position = self.job[1]
            self.job = None
        else:
            self.job[0] -= duration

##################ab hier änderungen#####################



# Pfad zur CSV-Datei
csv_file_path = 'drivers.csv'

# Liste für geladene Fahrer
DRIVERS = []
with open(csv_file_path, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # Wir verwenden die korrekten Spaltennamen
        lat = int(row['X']) ##float für genauere Koordinaten 
        lon = int(row['Y']) ##float für genauere Koordinaten
        DRIVERS.append(
            Driver(
                start_position=Location(lat, lon)
            )
        )


        # DRIVERS: list[Driver] = [
#     Driver(
#         Location(
#             random.Random(i).randint(0, 10000),
#             random.Random(i * i).randint(0, 10000),
#         )
#     )
#     for i in range(100)
# ]

