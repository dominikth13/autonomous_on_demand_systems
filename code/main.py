from datetime import timedelta
import time
from algorithm.algorithm import (
    generate_driver_action_pairs,
    generate_routes,
    solve_optimization_problem,
)

from data_analysis.data_analysis import analyse_trip_data
from data_output.data_collector import DataCollector
from data_visualization.Visualisierung_tripdata import visualize_trip_data
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
from data_visualization.Datavisualisierung import visualize_drivers, visualize_orders


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
        
        for match in matches:
            if match.action.is_route():
                matched_order = match.action.route.order
                destination_vehicle = match.action.route.vehicle_destination
                destination_time = match.action.route.vehicle_time
                DataCollector.append_orders_dataa(current_time,matched_order,destination_vehicle,destination_time)
            

        # Apply state changes based on Action-Driver matches and existing driver jobs
        LOGGER.debug("Apply state-value changes")
        State.get_state().apply_state_change(matches)

        if ProgramParams.FEATURE_RELOCATION_ENABLED and current_time.to_total_seconds() % ProgramParams.MAX_IDLING_TIME == 0:
            LOGGER.debug("Relocate long time idle drivers")
            State.get_state().relocate()

        
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
            DataCollector.append_zone_id(
                current_time, Grid.get_instance().find_cell(driver.current_position).id
            )
        # if current_total_minutes % 60 == 0:
        #     LOGGER.debug("Save current driver positions")
        #     for driver in Drivers.get_drivers():
        #         status = (
        #             "idling"
        #             if not driver.is_occupied()
        #             else ("relocation" if driver.job.is_relocation else "occupied")
        #         )
        #         DataCollector.append_driver_data(
        #             current_time, driver.id, status, driver.current_position
        #         )
            #visualize_drivers(f"drivers_{ProgramParams.SIMULATION_DATE.strftime('%Y-%m-%d')}_{current_total_minutes}.png")
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

    DataCollector.clear()

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
    # entries = []

    # for minutes in ProgramParams.TIME_SERIES_BREAKPOINTS():
    #     StateValueNetworks.get_instance().load_offline_policy_weights(minutes)
    #     timet = Time.of_total_minutes(minutes)
    #     for zone in Grid.get_instance().zones_dict.values():
    #         mains = StateValueNetworks.get_instance().get_main_state_value(zone, timet)
    #         targets = StateValueNetworks.get_instance().get_target_state_value(zone, timet)
    #         entries.append((zone.id, zone.central_location.lat, zone.central_location.lon, timet, mains, targets))

    # csv_file_path = "code/data_output/state_values_over_time.csv"
    # with open(csv_file_path, mode="w") as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["zone_id", "lat", "lon", "total_minutes", "main_sv", "target_sv"])
    #     for w in entries:
    #         writer.writerow([w[0], w[1], w[2], w[3], w[4], w[5]])

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

        for match in matches:
            if match.action.is_route():
                matched_order = match.action.route.order
                destination_vehicle = match.action.route.vehicle_destination
                destination_time = match.action.route.vehicle_time
                DataCollector.append_orders_dataa(current_time,matched_order,destination_vehicle,destination_time)

        # Apply state changes based on Action-Driver matches and existing driver jobs
        LOGGER.debug("Apply state-value changes")
        State.get_state().apply_state_change(matches)

        if ProgramParams.FEATURE_RELOCATION_ENABLED and current_time.to_total_seconds() % ProgramParams.MAX_IDLING_TIME == 0:
            LOGGER.debug("Relocate long time idle drivers")
            State.get_state().relocate()

        for driver in Drivers.get_drivers():
            status = (
                "idling"
                if not driver.is_occupied()
                else ("relocation" if driver.job.is_relocation else "occupied")
            )
            DataCollector.append_driver_data(
                current_time, driver.id, status, driver.current_position
            )
            DataCollector.append_zone_id(
                current_time, Grid.get_instance().find_cell(driver.current_position).id
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

    DataCollector.clear()

def start_baseline_performance():
    # 1. Initialize environment data
    start_time = time.time()
    LOGGER.info("Initialize Grid")
    Grid.get_instance()
    LOGGER.info("Initialize time series")
    TimeSeries.get_instance()
    LOGGER.info("Initialize state")
    State.get_state()
    LOGGER.info("Initialize fastest connection network")
    FastestStationConnectionNetwork.get_instance()
    LOGGER.info("Initialize orders")
    Order.get_orders_by_time()
    LOGGER.info("Initialize vehicles")
    Drivers.get_drivers()

    # 2. Run DQ-Learning algorithm to train state value network
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
        for match in matches:
            if match.action.is_route():
                matched_order = match.action.route.order
                destination_vehicle = match.action.route.vehicle_destination
                destination_time = match.action.route.vehicle_time
                DataCollector.append_orders_dataa(current_time,matched_order,destination_vehicle,destination_time)

        # Apply state changes based on Action-Driver matches and existing driver jobs
        LOGGER.debug("Apply state-value changes")
        State.get_state().apply_state_change(matches)

        if ProgramParams.FEATURE_RELOCATION_ENABLED and current_time.to_total_seconds() % ProgramParams.MAX_IDLING_TIME == 0:
            LOGGER.debug("Relocate long time idle drivers")
            State.get_state().relocate()

        # if current_total_minutes % 60 == 0:
        #     #visualize_drivers(f"drivers_{current_total_minutes}.png")
        #     LOGGER.debug("Save current driver positions")
        #     for driver in Drivers.get_drivers():
        #         status = (
        #             "idling"
        #             if not driver.is_occupied()
        #             else ("relocation" if driver.job.is_relocation else "occupied")
        #         )
        #         DataCollector.append_driver_data(
        #             current_time, driver.id, status, driver.current_position
        #         )

        for driver in Drivers.get_drivers():
            status = (
                "idling"
                if not driver.is_occupied()
                else ("relocation" if driver.job.is_relocation else "occupied")
            )
            DataCollector.append_driver_data(
                current_time, driver.id, status, driver.current_position
            )
            DataCollector.append_zone_id(
                current_time, Grid.get_instance().find_cell(driver.current_position).id
            )
    
      
        # Update the expiry durations of still open orders
        State.get_state().update_order_expiry_duration()

        # Increment to next interval
        State.get_state().increment_time_interval(current_time)

    LOGGER.info("Exporting final driver positions")
    Drivers.export_drivers()
    LOGGER.info("Exporting data")
    DataCollector.export_all_data()
    LOGGER.info(f"Algorithm took {time.time() - start_time} seconds to run.")

    DataCollector.clear()

while True:
    user_input = input(
        "Which menu you want to enter? (Tabular Reinforcement Learning -> 1, Deep Reinforcement Learning -> 2, Baseline Performance -> 3, Static Data Generation -> 4, Visualization -> 5, Data Analysis -> 6) "
    )
    if user_input == "1":
        ProgramParams.EXECUTION_MODE = Mode.TABULAR
        while True:
            user_input = input(
                "Which script do you want to start? (Online Training and Testing -> 1, Start Q-Learning (one day)-> 2) "
            )
            if user_input == "1":
                StateValueTable.get_state_value_table().raze_state_value_table()
                initialize_driver_positions()
                # Train the algorithm On-Policy
                for i in range(14):
                    start_q_learning()
                    Order.reset()
                    State.reset()
                    ProgramParams.SIMULATION_DATE += timedelta(1)
                # Testing
                Drivers.raze_drivers()
                initialize_driver_positions()
                # Train the algorithm On-Policy
                for i in range(7):
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
                "Which script do you want to start? (Offline Policy Evaluation -> 1, Online Training -> 2, Start DRL (one day) -> 3) "
            )
            if user_input == "1":
                initialize_driver_positions()
                # Train the ope networks with online data
                for i in range(14):
                    train_ope()
                    Order.reset()
                    State.reset()
                    ProgramParams.SIMULATION_DATE += timedelta(1)
                break
            elif user_input == "2":
                initialize_driver_positions()
                # Train the algorithm On-Policy
                # You have to set the date in program_params on the Date you want (old date + train duration)
                for i in range(7):
                    start_drl()
                    Order.reset()
                    State.reset()
                    ProgramParams.SIMULATION_DATE += timedelta(1)
                break
            elif user_input == "3":
                start_drl()
                break
            else:
                print("This option is not allowed. Please try again.")
        break

    elif user_input == "3":
        while True:
            ProgramParams.EXECUTION_MODE = Mode.BASELINE_PERFORMANCE
            user_input = input(
                "Which script do you want to start? (Run Baseline Performance -> 1, Run Baseline Performance (one day) -> 2) "
            )
            if user_input == "1":
                initialize_driver_positions()
                # Train the algorithm On-Policy
                for i in range(14):
                    start_baseline_performance()
                    Order.reset()
                    State.reset()
                    ProgramParams.SIMULATION_DATE += timedelta(1)
                #Testing
                Drivers.raze_drivers()
                initialize_driver_positions()
                # Train the algorithm On-Policy
                for i in range(7):
                    start_baseline_performance()
                    Order.reset()
                    State.reset()
                    ProgramParams.SIMULATION_DATE += timedelta(1)           
                break
            if user_input == "2":
                start_baseline_performance()
                break
            else:
                print("This option is not allowed. Please try again.")
        break

    elif user_input == "4":
        while True:
            user_input = input(
                "Which script do you want to start? (Grid Cell Creation -> 1, Generate Trajectories -> 2, Remove Idle Trajectories -> 3, Initialize Drivers -> 4, Discretize days -> 5) "
            )
            if user_input == "1":
                create_cell_grid()
                break
            elif user_input == "2":
                initialize_driver_positions_for_trajectories()
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
    elif user_input == "5":
        while True:
            user_input = input(
                "Which script do you want to start? (Visualize driver positions -> 1, Visualize order positions -> 2, Visualize trip data -> 3) "
            )
            if user_input == "1":
                visualize_drivers(f"drivers_{ProgramParams.SIMULATION_DATE.strftime('%Y-%m-%d')}.png")
                break
            elif user_input == "2":
                visualize_orders(f"orders_{ProgramParams.SIMULATION_DATE.strftime('%Y-%m-%d')}.png")
                break
            elif user_input == "3":
                visualize_trip_data()
                break
            else:
                print("This option is not allowed. Please try again.")
        break
    elif user_input == "6":
        while True:
            user_input = input(
                "Which script do you want to start? (Analyse trip data (data_analysis.py need to be adapted[date and folder structure]) -> 1) "
            )
            if user_input == "1":
                analyse_trip_data()
                break
            else:
                print("This option is not allowed. Please try again.")
        break

    else:
        print("This option is not allowed. Please try again.")

