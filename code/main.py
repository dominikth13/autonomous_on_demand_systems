import random
import time
from algorithm import (
    generate_driver_action_pairs,
    generate_routes,
    solve_optimization_problem,
)
from time_interval import Time
from state import STATE
from location import Location
from order import Order
import pandas as pd
from state_value_table import STATE_VALUE_TABLE
from logger import LOGGER

def q_learning():
    start_time = time.time()
    # import_state_values_from_csv()

    # 1. Find all shortest paths in public transport network
    # Is done automatically in station.py

    # 2. Run Q-Learning algorithm to train state value table
    counter = 1
    while STATE.current_interval.next_interval != None:
        LOGGER.info(f"Starting iteration {counter}")
        LOGGER.debug("Collecting orders")
        # Collect new orders
        orders = [
            Order(
                Location(
                    random.Random(i + 15).randint(0, 10000),
                    random.Random((i + 15) ** 3).randint(0, 10000),
                ),
                Location(
                    random.Random((i + 15) ** 2).randint(0, 10000),
                    random.Random((i + 15) ** 4).randint(0, 10000),
                ),
            )
            for i in range(random.randint(0, 20))
        ]
        # Add orders to state
        STATE.add_orders(orders)
        # Generate routes
        LOGGER.debug("Generate routes")
        order_routes_dict = generate_routes(orders)
        # Generate Action-Driver pairs with all available routes and drivers
        LOGGER.debug("Generate driver-action-pairs")
        driver_action_pairs = generate_driver_action_pairs(order_routes_dict)
        # Find Action-Driver matches based on a min-cost-flow problem
        LOGGER.debug("Generate driver-action matches")
        matches = solve_optimization_problem(driver_action_pairs)
        # Apply state changes based on Action-Driver matches and existing driver jobs
        LOGGER.debug("Apply state-value changes")
        STATE.apply_state_change(matches)
        # Update the expiry durations of still open orders
        STATE.update_order_expiry_duration()
        # Increment to next interval
        STATE.increment_time_interval()
        counter += 1
    LOGGER.info("Exporting results...")
    export_epoch_to_csv()
    LOGGER.info(f"Algorithm took {time.time() - start_time} seconds to run.")

def export_epoch_to_csv():
    export_table = pd.DataFrame(columns=["start_time", "end_time","zone_name","state_value"])
    for time_interval in STATE_VALUE_TABLE.value_grid:
        for zone in STATE_VALUE_TABLE.value_grid[time_interval]:
            export_table.loc[len(export_table)] = [time_interval.start.to_total_minutes(), time_interval.end.to_total_minutes(), zone.name, STATE_VALUE_TABLE.value_grid[time_interval][zone]]

    export_table.to_csv("training_data/state_value_table.csv")

def import_state_values_from_csv():
    import_table = pd.read_csv("training_data/state_value_table.csv")
    
    for i in range(len(import_table)):
        start_time = Time.of_total_minutes(int(import_table[i]["start_time"]))
        interval = STATE_VALUE_TABLE.time_series.find_interval(start_time)
        zone = STATE_VALUE_TABLE.grid.zones_dict[import_table[i]["zone_name"]]
        state_value = float(import_table[i]["state_value"])
        STATE_VALUE_TABLE.value_grid[interval][zone] = state_value

q_learning()