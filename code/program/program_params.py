from datetime import datetime
from enum import Enum
from interval.time import Time


class ProgramParams:
    # Minimal trip time for routes to be eligible for combined routes in seconds
    L1 = 0

    # Maximum difference between direct route time and combined route time in seconds
    L2 = 1800

    # Static vehicle speed in m/s -> assume these small busses driving in Berlin
    VEHICLE_SPEED = 6.33  # FIX, wie im Paper Feng et al. 2022

    # Static walking speed in m/s
    WALKING_SPEED = 1  # FIX, wie im Paper Feng et al. 2022

    # Pick-up distance threshold (how far away driver consider new orders) in meter
    # Equal to 5 minutes
    PICK_UP_DISTANCE_THRESHOLD = 1900  # im Paper 950 Meter

    LEARNING_RATE = 0.005  # im Paper Feng et al. 2022 ist es 0.005

    # Hyperparameter that says how the online policy learning should be 
    # influenced by offline policy learning
    OMEGA = 0.2

    def DISCOUNT_FACTOR(duration_in_seconds: int) -> float:
        DISCOUNT_RATE = 0.95  # im Paper Feng et al. 2022 ist es 0.95
        LS = 0.9
        return DISCOUNT_RATE ** (duration_in_seconds / LS)

    # Duration how long orders can be matched with drivers in seconds
    ORDER_EXPIRY_DURATION = 120

    # Time it takes for customers to enter or leave the public transport system in seconds
    PUBLIC_TRANSPORT_ENTRY_EXIT_TIME = 120

    # Medium waiting time
    def PUBLIC_TRANSPORT_WAITING_TIME(time: Time):
        rush_hours_morning = Time(6, 30, 0)
        middays = Time(9, 30, 0)
        rush_hours_afternoon = Time(15, 30, 0)
        evenings = Time(20, 0, 0)

        if time.is_before(rush_hours_morning):
            return 600  # late nights waiting duration
        if time.is_before(middays):
            return 150  # rush hours morning waiting duration
        if time.is_before(rush_hours_afternoon):
            return 300  # middays waiting duration
        if time.is_before(evenings):
            return 150  # rush hours afternoon waiting duration
        return 450  # evenings waiting duration
        # Quelle: https://www.introducingnewyork.com/subway
        # https://www.humiliationstudies.org/documents/NYsubwaymap.pdf
    # Time it takes until the simulation updates in seconds
    SIMULATION_UPDATE_RATE = 60

    # Time the driver need to idle until he can relocate
    MAX_IDLING_TIME = 150

    EXECUTION_MODE = None

    # Number of iterations until the weights of main net are copied to target net
    MAIN_AND_TARGET_NET_SYNC_ITERATIONS = 60

    AMOUNT_OF_DRIVERS = 100

    # Radius for relocation in 100 meters
    RELOCATION_RADIUS = 20

    # Inilization of the static_data
    STATION_DURATION = 80  # Fahrzeit für eine Station
    TRANSFER_SAME_STATION = 300  # Setzen Sie hier den Wert für Umsteige_selbe_Station
    MAX_WALKING_DURATION = 600

    SIMULATION_DATE = datetime(2015, 7, 18)

    def TIME_SERIES_BREAKPOINTS() -> list[int]:
        wd = ProgramParams.SIMULATION_DATE.weekday()
        wd_to_bkps = {
            0: [150, 300, 450, 1050, 1350],
            1: [150, 300, 450, 1050, 1350],
            2: [150, 300, 450, 1050, 1350],
            3: [150, 300, 450, 1050, 1350],
            4: [150, 300, 450, 1050, 1350],
            5: [150, 300, 450, 750, 1050],
            6: [150, 300, 450, 600, 1350],
        }
        return wd_to_bkps[wd]

    # File paths to orders
    ORDERS_FILE_PATH = f"code/data/orders_{SIMULATION_DATE.strftime('%Y-%m-%d')}.csv"

    # If algorithm should do relocation
    FEATURE_RELOCATION_ENABLED = True


class Mode(Enum):
    TABULAR = "Tabular"
    DEEP_NEURAL_NETWORKS = "Deep Neural Networks"
    # Just solve the optimization problem without knowing state values
    BASELINE_PERFORMANCE = "Baseline Performance"
