import random
from location import Location
from station import Station

# Minimal trip time for routes to be eligible for combined routes in seconds
L1 = 600

# Maximum difference between direct route time and combined route time in seconds
L2 = 1200

# Set of stations
STATIONS = [Station(Location(random.Random(i**10).random() * 10000, random.Random(i**9).random() * 10000)) for i in range (0, 10)]

# Static vehicle speed in m/s
VEHICLE_SPEED = 15

# Static walking speed in m/s
WALKING_SPEED = 1

# Pick-up distance threshold (how far away driver consider new orders)
PICK_UP_DISTANCE_THRESHOLD = 10000