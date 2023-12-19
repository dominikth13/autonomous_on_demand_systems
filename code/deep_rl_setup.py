import torch
import torch.nn as nn
import torch.optim as optim
from state import STATE
from state_value_table import STATE_VALUE_TABLE
from model_builder import (
    or_tools_min_cost_flow,
    solve_as_bipartite_matching_problem,
    solve_as_min_cost_flow_problem,
)
from action import Action, DriverActionPair
from driver import DRIVERS, Driver
from route import *
from order import Order
from program_params import *
import pandas as pd


# Define a simple neural network
class NeuroNet(nn.Module):
    def __init__(self):
        super(NeuroNet, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        #  ReLU activation function after first layer
        x = nn.ReLU()(self.fc1(x))

        #  ReLU activation function after second layer
        x = nn.ReLU()(self.fc2(x))

        x = self.fc3(x)

# Define a loss function and optimizer
def td_error(output_net, output_target_net,reward):
    loss = (reward + output_target_net - output_net)**2
    return loss
 # Commonly used for classification problems


def generate_driver_action_pairs_without_weights(
    order_routes_dict: dict[Order, list[Route]]
) -> list[DriverActionPair]:
    driver_to_orders_to_routes_dict: dict[
        Driver, dict[Order, list[DriverActionPair]]
    ] = {driver: {} for driver in DRIVERS}
    driver_to_idling_dict = {driver: None for driver in DRIVERS}

    # Central idling action
    idling = Action(None)

    # 1. Generate DriverActionPair for each pair that is valid (distance), add DriverIdlingPair for each driver
    for driver in DRIVERS:
        driver_to_idling_dict[driver] = DriverActionPair(driver, idling, 0)
        if driver.is_occupied():
            # If driver is occupied he cannot take any new order
            continue

        for order in order_routes_dict:
            if (
                order.start.distance_to(driver.current_position)
                > PICK_UP_DISTANCE_THRESHOLD
            ):
                # If driver is currently to far away for this order he ignores it
                continue
            driver_to_orders_to_routes_dict[driver][order] = []
            for route in order_routes_dict[order]:
                driver_to_orders_to_routes_dict[driver][order].append(
                    DriverActionPair(driver, Action(route), 0)
                )
    data = pd.DataFrame(columns=['Reward', 'Target Destination', 'Target Arrival', 'Current Time', 'Current Position'])
    # 2. For each DriverActionPair: calculate edge weight based on reward and state value function
    for driver in driver_to_orders_to_routes_dict:
        for order in driver_to_orders_to_routes_dict[driver]:
            for pair in driver_to_orders_to_routes_dict[driver][order]:
                arrival_interval = STATE_VALUE_TABLE.time_series.find_interval(
                    STATE.current_interval.start.add_minutes(
                        pair.get_total_vehicle_travel_time_in_seconds() // 60
                    )
                )
                # TODO potentially bring in a Malus for long trips since they bind the car more longer
                # weight = time reduction for passenger + state value after this option

                reward = pair.action.route.time_reduction
                target_destination = pair.action.route.vehicle_destination_zone
                target_arrival = arrival_interval.end
                current_time = STATE.current_interval.start
                current_position = pair.driver.current_position

                row = {
                'Driver': driver,
                'Order': order,
                'Reward': reward,
                'Target Destination': target_destination,
                'Target Arrival': target_arrival,
                'Current Time': current_time,
                'Current Position': current_position    
                }

                 # Append to DataFrame
                data = data.append(row, ignore_index=True)
    return data