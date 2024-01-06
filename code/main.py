import csv
import random
import time
from algorithm.algorithm import (
    generate_driver_action_pairs,
    generate_routes,
    solve_optimization_problem,
)
from grid.grid import Grid
from interval.time_series import Time, TimeSeries
from order import Order
from logger import LOGGER
from public_transport.fastest_station_connection_network import (
    FastestStationConnectionNetwork,
)
from state.state import State
from state.state_value_table import StateValueTable
import pandas as pd
from program_params import *
from static_data_generation.grid_builder import create_cell_grid
from static_data_generation.trajectory_data_builder import generate_trajectories


def q_learning():
    start_time = time.time()
    LOGGER.info("Initialize Grid")
    Grid.get_instance()
    LOGGER.info("Initialize time series")
    TimeSeries.get_instance()
    LOGGER.info("Initialize state value table")
    StateValueTable.get_state_value_table()
    LOGGER.info("Initialize state")
    State.get_state()
    LOGGER.info("Initialize fastest connection network")
    FastestStationConnectionNetwork.get_instance()
    LOGGER.info("Initialize orders")
    Order.get_orders_by_time()
    # import_state_values_from_csv()

    # 1. Initialize environment data

    # Is done automatically in station.py

    # 2. Run Q-Learning algorithm to train state value table
    for current_total_minutes in range(
        TimeSeries.get_instance().start_time.to_total_minutes(),
        TimeSeries.get_instance().end_time.to_total_minutes() - 360,
    ):
        current_time = Time.of_total_minutes(current_total_minutes)
        LOGGER.info(f"Simulate time {current_time}")

        LOGGER.debug(f"Dispatch orders")
        orders = Order.get_orders_by_time()[current_time]
        for order in orders:
            order.dispatch()
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

        if (
            current_total_minutes
            - TimeSeries.get_instance().start_time.to_total_minutes() % MAX_IDLING_TIME
            == 0
        ):
            LOGGER.debug("Relocate long time idle drivers")
            State.get_state().relocate()

        # Update the expiry durations of still open orders
        State.get_state().update_order_expiry_duration()

        # Increment to next interval
        State.get_state().increment_time_interval(current_time)

    LOGGER.info("Exporting results...")
    export_epoch_to_csv()
    LOGGER.info(f"Algorithm took {time.time() - start_time} seconds to run.")


def export_epoch_to_csv():
    with open("code/training_data/state_value_table.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(["start_time", "end_time", "zone_id", "state_value"])
        for time_interval in StateValueTable.get_state_value_table().value_grid:
            for zone in StateValueTable.get_state_value_table().value_grid[
                time_interval
            ]:
                writer.writerow(
                    [
                        int(time_interval.start.to_total_seconds()),
                        int(time_interval.end.to_total_seconds()),
                        int(zone.id),
                        StateValueTable.get_state_value_table().value_grid[
                            time_interval
                        ][zone],
                    ]
                )

def import_state_values_from_csv():
    import_table = pd.read_csv("code/training_data/state_value_table.csv")

    for _, row in import_table.iterrows():
        start_time = Time.of_total_seconds(int(row["start_time"]))
        interval = TimeSeries.get_instance().find_interval(start_time)
        zone = Grid.get_instance().zones_dict[row["zone_id"]]
        state_value = float(row["state_value"])
        StateValueTable.get_state_value_table().value_grid[interval][zone] = state_value


while True:
    user_input = input(
        "Which script do you want to start? (Reinforcement Learning -> 1, Grid Builder -> 2, Trajectory Builder -> 3) "
    )
    if user_input == "1":
        q_learning()
        break
    elif user_input == "2":
        create_cell_grid()
        break
    elif user_input == "3":
        generate_trajectories()
        break
    else:
        print("This option is not allowed. Please try again.")
# INFO:algorithm:Simulate time 03:00:00
# Laufzeit filtered_orders: 0.0 Sekunden
# Laufzeit orders: 620.99849152565 Sekunden
# Laufzeit add_orders: 0.029900312423706055 Sekunden
# Laufzeit generate Routes: 345.75252294540405 Sekunden
# Laufzeit generate_driver_action_pairs: 794.6348361968994 Sekunden
# Laufzeit solve optimization_problem: 31.162750244140625 Sekunden
# Laufzeit rest: 0.7596573829650879 Sekunden
# INFO:algorithm:Simulate time 03:01:00
# Laufzeit filtered_orders: 0.0 Sekunden
