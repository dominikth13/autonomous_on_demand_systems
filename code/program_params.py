from state_value_table import Time

# Minimal trip time for routes to be eligible for combined routes in seconds
L1 = 600

# Maximum difference between direct route time and combined route time in seconds
L2 = 1200

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

# Time it takes for customers to enter or leave the public transport system in seconds
PUBLIC_TRANSPORT_ENTRY_EXIT_TIME = 120

# Waiting time till next train depending on current time in seconds
def PUBLIC_TRANSPORT_WAITING_TIME(time: Time):
    five = Time(5,0)
    six = Time(6,0)
    seven = Time(7,0)
    if time.is_before(five):
        return 600
    if time.is_before(six):
        return 420
    if time.is_before(seven):
        return 240
    return 120

PUBLIC_TRANSPORT_TICKET_PRICE = 2

# Taxi price per kilometer
TAXI_PRICE = 1.5

# Time it takes until the simulation updates in seconds
SIMULATION_UPDATE_RATE = 60