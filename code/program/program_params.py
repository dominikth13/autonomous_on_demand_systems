from datetime import datetime
from enum import Enum
from interval.time import Time

class Mode(Enum):
    TABULAR = "Tabular"
    DEEP_NEURAL_NETWORKS = "Deep Neural Networks"
    # Just solve the optimization problem without knowing state values
    BASELINE_PERFORMANCE = "Baseline Performance"

class DataSet(Enum):
    YELLOW_CAB = "yellow_cab"
    FOR_HIRE = "for_hire"

class ProgramParams:

    SIMULATION_DATE = datetime(2015, 7, 6)

    DATA_SET = DataSet.YELLOW_CAB

    # If algorithm should do relocation
    FEATURE_RELOCATION_ENABLED = False

    FEATURE_ADD_IDLING_COST_TO_TARGET = True

    IDLING_COST = 1

    AMOUNT_OF_DRIVERS = 100

    LEARNING_RATE = 0.005  # im Paper Feng et al. 2022 ist es 0.005

    MAXIMUM_STATE_VALUE = 10000

    MINIMUM_STATE_VALUE = -10000

    # Time it takes for customers to enter or leave the public transport system in seconds
    PUBLIC_TRANSPORT_ENTRY_EXIT_TIME = 120   

    # Radius for relocation in 100 meters
    RELOCATION_RADIUS = 20

    # Number of iterations until the weights of main net are copied to target net
    MAIN_AND_TARGET_NET_SYNC_ITERATIONS = 60

    # Pick-up distance threshold (how far away driver consider new orders) in meter
    # Equal to 5 minutes
    PICK_UP_DISTANCE_THRESHOLD = 1900  # im Paper 950 Meter

    # Duration how long orders can be matched with drivers in seconds
    ORDER_EXPIRY_DURATION = 120

    ######################################################################################################
    ############### Deep Reinforcement Learning ###############

    # Hyperparameter that says how the online policy learning should be 
    # influenced by offline policy learning
    OMEGA = 0.2

    ######################################################################################################
    ############### Variable, aber wollen wir kaum verändern ###############

    FEATURE_ORDERS_AS_WIN = False

    # Time the driver need to idle until he can relocate
    MAX_IDLING_TIME = 150

    EXECUTION_MODE = None

    # Minimal trip time for routes to be eligible for combined routes in seconds
    L1 = 0

    # Maximum difference between route without vehicles time and route with vehicles time in seconds
    L2 = 1800

    def DISCOUNT_FACTOR(duration_in_seconds: int) -> float:
        DISCOUNT_RATE = 0.95  # im Paper Feng et al. 2022 ist es 0.95
        LS = 0.9
        return DISCOUNT_RATE ** (duration_in_seconds / LS)
    

    STATE_VALUE_TABLE_UPDATE_RATE = 1800
    
    ##########################################################################################################
    ############### Inilization of the static_data ###############
    # Hierfür muss die Datei graph_für_OEPNV_Netz.py geändert werden
    # zusätzlich zu den Parametern, wird WALKING_SPEED für graph_für_OEPNV_Netz.py verwendet
    STATION_DURATION = 80  # Fahrzeit für eine Station
    TRANSFER_SAME_STATION = 300  # Setzen Sie hier den Wert für Umsteige_selbe_Station
    MAX_WALKING_DURATION = 600

    ######################################################################################################
    ############### FIX ###############
    # File paths to orders
    def ORDERS_FILE_PATH() :
        return f"code/data/{ProgramParams.DATA_SET.value}/orders_{ProgramParams.SIMULATION_DATE.strftime('%Y-%m-%d')}.csv"

    def TIME_SERIES_BREAKPOINTS() -> list[int]:
        wd = ProgramParams.SIMULATION_DATE.weekday()
        wd_to_bkps = {
            0: [0, 150, 300, 450, 1050, 1350],
            1: [0, 150, 300, 450, 1050, 1350],
            2: [0, 150, 300, 450, 1050, 1350],
            3: [0, 150, 300, 450, 1050, 1350],
            4: [0, 150, 300, 450, 1050, 1350],
            5: [0, 150, 300, 450, 750, 1050],
            6: [0, 150, 300, 450, 600, 1350],
        }
        return wd_to_bkps[wd]
    
    # Time it takes until the simulation updates in seconds
    SIMULATION_UPDATE_RATE = 60 #FIX

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
    
    
    # Static vehicle speed in m/s -> assume these small busses driving in Berlin
    VEHICLE_SPEED = 6.33  # FIX, wie im Paper Feng et al. 2022

    # Static walking speed in m/s
    WALKING_SPEED = 1  # FIX, wie im Paper Feng et al. 2022