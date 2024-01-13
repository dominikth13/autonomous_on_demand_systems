from enum import Enum
from interval.time import Time

class ProgramParams:

    # Minimal trip time for routes to be eligible for combined routes in seconds
    L1 = 600

    # Maximum difference between direct route time and combined route time in seconds
    L2 = 1200

    # Static vehicle speed in m/s -> assume these small busses driving in Berlin
    VEHICLE_SPEED = 10

    # Static walking speed in m/s
    WALKING_SPEED = 1.5

    # Pick-up distance threshold (how far away driver consider new orders) in meter
    PICK_UP_DISTANCE_THRESHOLD = 1000

    LEARNING_RATE = 0.001

    def DISCOUNT_FACTOR(duration_in_seconds: int) -> float:
        DISCOUNT_RATE = 0.95
        LS = 0.9
        return DISCOUNT_RATE ** (duration_in_seconds / LS)

    # Duration how long orders can be matched with drivers in seconds
    ORDER_EXPIRY_DURATION = 120

    # Time it takes for customers to enter or leave the public transport system in seconds
    PUBLIC_TRANSPORT_ENTRY_EXIT_TIME = 120

    # Medium waiting time
    def PUBLIC_TRANSPORT_WAITING_TIME(time: Time):
        five = Time(5,0,0)
        six = Time(6,0,0)
        seven = Time(7,0,0)
        if time.is_before(five):
            return 600
        if time.is_before(six):
            return 420
        if time.is_before(seven):
            return 240
        return 120

    # Time it takes until the simulation updates in seconds
    SIMULATION_UPDATE_RATE = 60

    # Time the driver need to idle until he can relocate
    MAX_IDLING_TIME = 150

    EXECUTION_MODE = None

    # Number of iterations until the weights of main net are copied to target net
    MAIN_AND_TARGET_NET_SYNC_ITERATIONS = 100

class Mode(Enum):
    TABULAR = "Tabular"
    DEEP_NEURAL_NETWORKS = "Deep Neural Networks"