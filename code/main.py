import csv
import random
import time
from algorithm.algorithm import (
    generate_driver_action_pairs,
    generate_routes,
    solve_optimization_problem,
)
from data_output.data_collector import DataCollector
from deep_reinforcement_learning.deep_rl_training import train
from deep_reinforcement_learning.offline_policy_evaluation import train_ope
from driver.drivers import Drivers
from grid.grid import Grid
from interval.time_series import Time, TimeSeries
from order import Order
from logger import LOGGER
from public_transport.fastest_station_connection_network import (
    FastestStationConnectionNetwork,
)
from state.state import State
from state.state_value_networks import StateValueNetworks
from state.state_value_table import StateValueTable
import pandas as pd
from program.program_params import Mode, ProgramParams
from static_data_generation.grid_builder import create_cell_grid
from static_data_generation.initial_driver_positions import initialize_driver_positions
from static_data_generation.trajectory_data_builder import generate_trajectories, remove_idle_trajectories
from data_visualization.Datavisualisierung import visualize_drivers


def start():
    # 1. Initialize environment data
    start_time = time.time()
    LOGGER.info("Initialize Grid")
    Grid.get_instance()
    LOGGER.info("Initialize time series")
    TimeSeries.get_instance()
    if ProgramParams.EXECUTION_MODE == Mode.TABULAR:
        LOGGER.info("Initialize state value table")
        StateValueTable.get_state_value_table()
    else:
        LOGGER.info("Initialize state value networks")
        StateValueNetworks.get_instance()
    LOGGER.info("Initialize state")
    State.get_state()
    LOGGER.info("Initialize fastest connection network")
    FastestStationConnectionNetwork.get_instance()
    LOGGER.info("Initialize orders")
    Order.get_orders_by_time()
    LOGGER.info("Initialize vehicles")
    Drivers.get_drivers()

    if ProgramParams.EXECUTION_MODE == Mode.TABULAR:
        StateValueTable.get_state_value_table().import_state_value_table_from_csv()
    else:
        StateValueNetworks.get_instance().import_weights()


    # 2. Run Q-Learning/DQ-Learning algorithm to train state value table/network
    for current_total_minutes in range(
        TimeSeries.get_instance().start_time.to_total_minutes() ,
        TimeSeries.get_instance().end_time.to_total_minutes() + 1,
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
            - TimeSeries.get_instance().start_time.to_total_minutes() % ProgramParams.MAX_IDLING_TIME
            == 0
        ):
            LOGGER.debug("Relocate long time idle drivers")
            State.get_state().relocate()
        
        if (current_total_minutes % 60 == 0):
            LOGGER.debug("Save current driver positions")
            for driver in Drivers.get_drivers():
                status = "idling" if not driver.is_occupied() else ("relocation" if driver.job.is_relocation else "occupied")
                DataCollector.append_driver_data(current_time, driver.id, status, driver.current_position)

        # Update the expiry durations of still open orders
        State.get_state().update_order_expiry_duration()

        # Increment to next interval
        State.get_state().increment_time_interval(current_time)

    LOGGER.info("Exporting final driver positions")
    Drivers.export_drivers()
    LOGGER.info("Exporting data")
    DataCollector.export_all_data()
    LOGGER.info("Exporting training results")
    if ProgramParams.EXECUTION_MODE == Mode.TABULAR:
        StateValueTable.get_state_value_table().export_state_value_table_to_csv()
    else:
        StateValueNetworks.get_instance().export_weights()
    LOGGER.info(f"Algorithm took {time.time() - start_time} seconds to run.")    

while True:
    user_input = input(
        "Which menu you want to enter? (Tabular Reinforcement Learning -> 1, Deep Reinforcement Learning -> 2, Static Data Generation -> 3, Visualization -> 4) "
    )
    if user_input == "1":
        while True:
            user_input = input(
                "Which script do you want to start? (Start Q-Learning -> 1) "
            )
            if user_input == "1":
                ProgramParams.EXECUTION_MODE = Mode.TABULAR
                start()
                break
            else:
                print("This option is not allowed. Please try again.")
        break


    elif user_input == "2":
        while True:
            user_input = input(
                "Which script do you want to start? (Offline Policy Evaluation (Dominik) -> 1, Offline Policy Evaluation (Malik) -> 2, Start DRL -> 3) "
            )
            if user_input == "1":
                train_ope()
                break
            elif user_input == "2":
                train()
                break
            elif user_input == "3":
                ProgramParams.EXECUTION_MODE = Mode.DEEP_NEURAL_NETWORKS
                start()
                break
            else:
                print("This option is not allowed. Please try again.")
        break

    elif user_input == "3":
        while True:
            user_input = input(
                "Which script do you want to start? (Grid Cell Creation -> 1, Generate Trajectories -> 2, Remove Idle Trajectories -> 3, Initialize Drivers -> 4) "
            )
            if user_input == "1":
                create_cell_grid()
                break
            elif user_input == "2":
                generate_trajectories()
                break
            elif user_input == "3":
                remove_idle_trajectories()
                break
            elif user_input == "4":
                initialize_driver_positions()
                break
            else:
                print("This option is not allowed. Please try again.")
        break
    elif user_input == "4":
        while True:
            user_input = input(
                "Which script do you want to start? (Visualize driver positions -> 1) "
            )
            if user_input == "1":
                visualize_drivers()
                break
            else:
                print("This option is not allowed. Please try again.")
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
