import random
from state_value_table import Time
from location import Location
from station import Station
import csv
# Minimal trip time for routes to be eligible for combined routes in seconds
L1 = 600

# Maximum difference between direct route time and combined route time in seconds
L2 = 1200

# Set of stations
#STATIONS = [Station(Location(random.Random(i**10).random() * 10000, random.Random(i**9).random() * 10000)) for i in range (0, 10)]
# _stations = []
# Pfad zur CSV-Datei
stations_csv_file_path = 'stations.csv'

# Erstellung der _stations Liste durch Einlesen der CSV-Datei
STATIONS = []
with open(stations_csv_file_path, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        latitude = int(row['X'])
        longitude = int(row['Y'])
        STATIONS.append(Station(position=Location(lat=latitude, lon=longitude)))
        
# Static vehicle speed in m/s
VEHICLE_SPEED = 15

# Static walking speed in m/s
WALKING_SPEED = 1

# Pick-up distance threshold (how far away driver consider new orders) in meter
PICK_UP_DISTANCE_THRESHOLD = 10000

LEARNING_RATE = 0.00001

def DISCOUNT_FACTOR(current_time: Time, time_after_action: Time) -> float:
    DISCOUNT_RATE = 0.98
    LS = 0.99
    return DISCOUNT_RATE ** (time_after_action.distance_to_in_seconds(current_time) / LS)

# Duration how long orders can be matched with drivers in seconds
ORDER_EXPIRY_DURATION = 120