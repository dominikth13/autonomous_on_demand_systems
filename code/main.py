import random
import time
from algorithm.algorithm import generate_driver_action_pairs, generate_routes, solve_optimization_problem
from interval.time_series import Time
from location.location import Location
from order import Order
import pandas as pd
from logger import LOGGER
from state.state import State
from state.state_value_table import StateValueTable


def q_learning():
    start_time = time.time()
    #import_state_values_from_csv()

    # 1. Find all shortest paths in public transport network
    # Is done automatically in station.py

    # 2. Run Q-Learning algorithm to train state value table
    counter = 1
    for start_minutes in range(
        StateValueTable.get_state_value_table().time_series.start_time.to_total_minutes(),
        # TimeSeries is two hours longer than the simulation
        StateValueTable.get_state_value_table().time_series.end_time.to_total_minutes() - 360,
    ):
        current_time = Time.of_total_minutes(start_minutes)
        LOGGER.info(f"Simulate time {current_time}")
        LOGGER.debug("Collecting orders")
        # Collect new orders
        orders = [
            Order(
                Location(
                    random.randint(40534522, 40925205) / 1000000,
                    random.randint(-74050826, -73685841) / 1000000,
                ),
                Location(
                    random.randint(40534522, 40925205) / 1000000,
                    random.randint(-74050826, -73685841) / 1000000,
                ),
            )
            for i in range(random.randint(0, 50))
        ]

        # Add orders to state
        State.get_state().add_orders(orders)
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
        State.get_state().apply_state_change(matches)
        # Update the expiry durations of still open orders
        State.get_state().update_order_expiry_duration()
        # Increment to next interval
        State.get_state().increment_time_interval(current_time)
        counter += 1
    LOGGER.info("Exporting results...")
    export_epoch_to_csv()
    LOGGER.info(f"Algorithm took {time.time() - start_time} seconds to run.")


def export_epoch_to_csv():
    export_table = pd.DataFrame(
        columns=["start_time", "end_time", "zone_name", "state_value"]
    )
    for time_interval in StateValueTable.get_state_value_table().value_grid:
        for zone in StateValueTable.get_state_value_table().value_grid[time_interval]:
            export_table.loc[len(export_table)] = [
                time_interval.start.to_total_seconds(),
                time_interval.end.to_total_seconds(),
                zone.id,
                StateValueTable.get_state_value_table().value_grid[time_interval][zone],
            ]

    export_table.to_csv("code/training_data/state_value_table.csv")


def import_state_values_from_csv():
    import_table = pd.read_csv("code/training_data/state_value_table.csv")

    for _, row in import_table.iterrows():
        start_time = Time.of_total_seconds(int(row["start_time"]))
        interval = StateValueTable.get_state_value_table().time_series.find_interval(start_time)
        zone = StateValueTable.get_state_value_table().grid.zones_dict[row["zone_name"]]
        state_value = float(row["state_value"])
        StateValueTable.get_state_value_table().value_grid[interval][zone] = state_value

q_learning()
