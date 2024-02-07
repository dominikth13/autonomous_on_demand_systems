# This file contains the code for the offline policy evaluation (training with random samples of driver trajectories)

import csv
import math
import random
import time
from numpy import mean
import torch
import torch.optim as optim
from algorithm.algorithm import generate_driver_action_pairs, generate_routes, solve_optimization_problem
from data_output.data_collector import DataCollector
from deep_reinforcement_learning.neuro_net import NeuroNet
from deep_reinforcement_learning.deep_rl_training import import_trajectories
from deep_reinforcement_learning.temporal_difference_loss import TemporalDifferenceLoss
from driver.drivers import Drivers
from grid.grid import Grid
from interval.time import Time
from interval.time_series import TimeSeries
from logger import LOGGER
from order import Order
from program.program_params import ProgramParams
from public_transport.fastest_station_connection_network import FastestStationConnectionNetwork
from state.state import State
from state.state_value_networks import StateValueNetworks

def train_ope():
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
            LOGGER.debug("Load other ope training weights")
            StateValueNetworks.get_instance().import_offline_policy_weights(current_total_minutes)


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
    StateValueNetworks.get_instance().export_offline_policy_weights(ProgramParams.TIME_SERIES_BREAKPOINTS()[-1])

    LOGGER.info("Exporting final driver positions")
    Drivers.export_drivers()
    LOGGER.info("Exporting data")
    DataCollector.export_all_data()
    LOGGER.info("Exporting training results")
    StateValueNetworks.get_instance().export_weights()
    LOGGER.info(f"Algorithm took {time.time() - start_time} seconds to run.")

    DataCollector.clear()

# def train_ope() -> None:
#     # wd, sat or sun
#     TRAINING_MODE = "wd"
#     time_series_breakpoints = {
#         "wd": [0, 150, 300, 450, 1050, 1350],
#         "sat": [0, 150, 300, 450, 750, 1050],
#         "sun": [0, 150, 300, 450, 600, 1350],
#     }
#     LOGGER.info("Initialize environment")
#     trajectories_by_bp = import_trajectories(
#         TRAINING_MODE, time_series_breakpoints[TRAINING_MODE]
#     )
#     all_losses = []
#     for bp in trajectories_by_bp:
#         all_losses.append(train_for(TRAINING_MODE, bp, trajectories_by_bp[bp]))

#     LOGGER.info("Finished Training")

#     with open(f"code/training_data/training_{TRAINING_MODE}.csv", "w") as file:
#         writer = csv.writer(file)
#         writer.writerow(["break_point", "losses"])
#         for i, bp in enumerate(time_series_breakpoints[TRAINING_MODE]):
#             writer.writerow([bp, all_losses[i]])


# def train_for(
#     train_mode: str,
#     time_series_breakpoint_group: int,
#     trajectories: dict[int, dict[str, float]],
# ) -> list[float]:
#     state_value_net = NeuroNet()
#     target_net = NeuroNet()
#     keys = list(trajectories.keys())
#     random.Random(time_series_breakpoint_group).shuffle(keys)

#     # Berechnen der Größe jedes Teilsets
#     size, extra = divmod(len(keys), 11)

#     # Aufteilung in Teilsets
#     all_groups = [
#         keys[i * size + min(i, extra) : (i + 1) * size + min(i + 1, extra)]
#         for i in range(11)
#     ]
#     epoch_groups = all_groups[:-1]
#     test_group = all_groups[-1]

#     # Used in Tang et al. (2021)
#     # Loss function
#     loss_fn = TemporalDifferenceLoss()
#     # Optimizer
#     optimizer = optim.Adam(
#         state_value_net.parameters(), lr=3 * math.exp(-4)
#     )  # Stochastic Gradient Descent
#     all_losses = []
#     for epoch, group in enumerate(epoch_groups):
#         LOGGER.info(f"Epoch {epoch}")
#         LOGGER.info("Start training")
#         state_value_net.train()
#         target_net.train()
#         # Training loop
#         M = 100
#         N = len(group) // M
#         for i in range(N):
#             if i % 100 == 0:
#                 LOGGER.debug("Transfer weights from main to target network")
#                 target_net.load_state_dict(state_value_net.state_dict())
#             LOGGER.debug(f"Training loop {i}")
#             # Sample random batch of trajectories
#             batch = random.Random(epoch + i**epoch).sample(group, k=M)

#             LOGGER.debug("Forward propagation")
#             state_values = []
#             for key in batch:
#                 trajectory = trajectories[key]
#                 output_current = state_value_net(
#                     torch.Tensor(
#                         [
#                             trajectory["current_zone"],
#                         ]
#                     )
#                 )
#                 output_target = target_net(
#                     torch.Tensor(
#                         [
#                             trajectory["target_zone"],
#                         ]
#                     )
#                 )
#                 state_values.append((trajectory, output_current, output_target))

#             LOGGER.debug("Backward propagation and optimization")
#             # Backward and optimize
#             optimizer.zero_grad()
#             # Compute loss
#             loss = loss_fn(state_value_net, state_values)
#             loss.backward()
#             optimizer.step()

#         LOGGER.info("Start evaluation")
#         state_value_net.eval()
#         target_net.eval()
#         losses = []
#         N = len(test_group) // M // 10
#         for i in range(N):
#             # Sample random batch of trajectories
#             batch = random.Random(epoch + i**epoch).sample(test_group, k=M)
#             for key in batch:
#                 trajectory = trajectories[key]
#                 output_current = state_value_net(
#                     torch.Tensor(
#                         [
#                             trajectory["current_zone"],
#                         ]
#                     )
#                 )
#                 output_target = target_net(
#                     torch.Tensor(
#                         [
#                             trajectory["target_zone"],
#                         ]
#                     )
#                 )
#                 state_values.append((trajectory, output_current, output_target))

#             # Compute loss
#             losses.append(loss_fn(state_value_net, state_values).item())

#         LOGGER.info(f"Medium difference error: {float(mean(losses))}")
#         all_losses.append(float(mean(losses)))
#     torch.save(
#         state_value_net.state_dict(),
#         f"code/training_data/ope_{train_mode}_{time_series_breakpoint_group}.pth",
#     )
#     torch.save(
#         target_net.state_dict(),
#         f"code/training_data/ope_target_{train_mode}_{time_series_breakpoint_group}.pth",
#     )
#     LOGGER.info(
#         f"Finished Training for breakpoint {time_series_breakpoint_group} on mode {train_mode}"
#     )

#     return all_losses
