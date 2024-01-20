import csv
from datetime import timedelta
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
from static_data_generation.initial_driver_positions import initialize_driver_positions, initialize_driver_positions_for_trajectories
from static_data_generation.time_series_discretization import TimeSeriesDiscretization
from static_data_generation.trajectory_data_builder import (
    generate_trajectories,
    remove_idle_trajectories,
)
from data_visualization.Datavisualisierung import visualize_drivers


def start_q_learning():
    # 1. Initialize environment data
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
    LOGGER.info("Initialize vehicles")
    Drivers.get_drivers()

    StateValueTable.get_state_value_table().import_state_value_table_from_csv()

    # 2. Run Q-Learning algorithm to train state value table
    for current_total_minutes in range(
        TimeSeries.get_instance().start_time.to_total_minutes(),
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

        if ProgramParams.FEATURE_RELOCATION_ENABLED and current_time.to_total_seconds() % ProgramParams.MAX_IDLING_TIME == 0:
            LOGGER.debug("Relocate long time idle drivers")
            State.get_state().relocate()

        if current_total_minutes % 60 == 0:
            LOGGER.debug("Save current driver positions")
            for driver in Drivers.get_drivers():
                status = (
                    "idling"
                    if not driver.is_occupied()
                    else ("relocation" if driver.job.is_relocation else "occupied")
                )
                DataCollector.append_driver_data(
                    current_time, driver.id, status, driver.current_position
                )
            #visualize_drivers(f"drivers_{ProgramParams.SIMULATION_DATE.strftime('%Y-%m-%d')}_{current_total_minutes}_eod.png")
        # Update the expiry durations of still open orders
        State.get_state().update_order_expiry_duration()

        # Increment to next interval
        State.get_state().increment_time_interval(current_time)

    LOGGER.info("Exporting final driver positions")
    Drivers.export_drivers()
    LOGGER.info("Exporting data")
    DataCollector.export_all_data()
    LOGGER.info("Exporting training results")
    StateValueTable.get_state_value_table().export_state_value_table_to_csv()
    LOGGER.info(f"Algorithm took {time.time() - start_time} seconds to run.")

def start_drl():
    # 1. Initialize environment data
    start_time = time.time()
    LOGGER.info("Initialize Grid")
    Grid.get_instance()
    LOGGER.info("Initialize time series")
    TimeSeries.get_instance()
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

    StateValueNetworks.get_instance().import_weights()

    # 2. Run DQ-Learning algorithm to train state value network
    for current_total_minutes in range(
        TimeSeries.get_instance().start_time.to_total_minutes(),
        TimeSeries.get_instance().end_time.to_total_minutes() + 1,
    ):
        current_time = Time.of_total_minutes(current_total_minutes)
        LOGGER.info(f"Simulate time {current_time}")

        if current_total_minutes in ProgramParams.TIME_SERIES_BREAKPOINTS():
            LOGGER.debug("Reinitialize network weights with offline trained weights")
            StateValueNetworks.get_instance().load_offline_policy_weights(current_total_minutes)

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

        if ProgramParams.FEATURE_RELOCATION_ENABLED and current_time.to_total_seconds() % ProgramParams.MAX_IDLING_TIME == 0:
            LOGGER.debug("Relocate long time idle drivers")
            State.get_state().relocate()

        if current_total_minutes % 60 == 0:
            LOGGER.debug("Save current driver positions")
            for driver in Drivers.get_drivers():
                status = (
                    "idling"
                    if not driver.is_occupied()
                    else ("relocation" if driver.job.is_relocation else "occupied")
                )
                DataCollector.append_driver_data(
                    current_time, driver.id, status, driver.current_position
                )

        # Update the expiry durations of still open orders
        State.get_state().update_order_expiry_duration()

        # Increment to next interval
        State.get_state().increment_time_interval(current_time)

    LOGGER.info("Exporting final driver positions")
    Drivers.export_drivers()
    LOGGER.info("Exporting data")
    DataCollector.export_all_data()
    LOGGER.info("Exporting training results")
    StateValueNetworks.get_instance().export_weights()
    LOGGER.info(f"Algorithm took {time.time() - start_time} seconds to run.")


while True:
    user_input = input(
        "Which menu you want to enter? (Tabular Reinforcement Learning -> 1, Deep Reinforcement Learning -> 2, Static Data Generation -> 3, Visualization -> 4) "
    )
    if user_input == "1":
        ProgramParams.EXECUTION_MODE = Mode.TABULAR
        while True:
            user_input = input(
                "Which script do you want to start? (Online Training -> 1, Start Q-Learning -> 2) "
            )
            if user_input == "1":
                # Train the algorithm On-Policy
                for i in range(30):
                    
                    start_q_learning()
                    Order.reset()
                    State.reset()
                    ProgramParams.SIMULATION_DATE += timedelta(1)
                break
            if user_input == "2":
                start_q_learning()
                break
            else:
                print("This option is not allowed. Please try again.")
        break

    elif user_input == "2":
        while True:
            ProgramParams.EXECUTION_MODE = Mode.DEEP_NEURAL_NETWORKS
            user_input = input(
                "Which script do you want to start? (Offline Policy Evaluation -> 1, Start DRL -> 2) "
            )
            if user_input == "1":
                train_ope()
                break
            elif user_input == "2":
                start_drl()
                break
            else:
                print("This option is not allowed. Please try again.")
        break

    elif user_input == "3":
        while True:
            user_input = input(
                "Which script do you want to start? (Grid Cell Creation -> 1, Generate Trajectories -> 2, Remove Idle Trajectories -> 3, Initialize Drivers -> 4, Discretize days -> 5) "
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
            elif user_input == "5":
                TimeSeriesDiscretization.discretize_day()
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
                visualize_drivers(f"drivers_{ProgramParams.SIMULATION_DATE.strftime('%Y-%m-%d')}_eod.png")
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
