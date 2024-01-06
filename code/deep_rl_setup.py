import random
import torch
import torch.nn as nn
import torch.optim as optim
from action.action import Action
from action.driver_action_pair import DriverActionPair
from algorithm.algorithm import generate_routes
from driver.driver import Driver
from driver.drivers import Drivers
from order import Order
from program_params import *
import pandas as pd
import torch.nn.functional as F

from route import Route



import torch.nn.functional as F

class NeuroNet(nn.Module):
    def __init__(self):
        super(NeuroNet, self).__init__()
        # Define separate layers for each input
        self.fc_x = nn.Linear(1, 16)  # Assuming x is a scalar
        self.fc_y = nn.Linear(1, 16)  # Assuming y is a scalar
        self.fc_time = nn.Linear(1, 16)  # Assuming time is a scalar

        # Layers after merging
        self.fc_combined = nn.Linear(16 * 3, 32)
        self.fc_final = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.relu(self.fc3(x))

        return output

# Example usage:
# model = NeuroNet()
# output = model(input_x, input_y, input_time)



# Define a loss function and optimizer
def td_error(output_net, output_target_net, reward):
    loss = (reward + output_target_net - output_net) ** 2
    return loss


# Commonly used for classification problems


def generate_driver_action_pairs_without_weights() -> list[tuple]:
    orders = []
    for i in range(1440):
        sub_orders = Order.get_orders_by_time()[Time.of_total_minutes(i)]
        if len(sub_orders) == 0:
            continue
        orders.extend(random.sample(sub_orders,1))

    driver_to_orders_to_routes_dict: dict[
        Driver, dict[Order, list[DriverActionPair]]
    ] = {driver: {} for driver in Drivers.get_drivers()}

    # 1. Generate DriverActionPair for each pair that is valid (distance), add DriverIdlingPair for each driver
    for driver in Drivers.get_drivers():

        for order in orders:
            if (
                order.start.distance_to(driver.current_position)
                > PICK_UP_DISTANCE_THRESHOLD
            ):
                # If driver is currently to far away for this order he ignores it
                continue
            driver_to_orders_to_routes_dict[driver][order] = []
            order.dispatch()
            for route  in generate_routes([order])[order]:
                driver_to_orders_to_routes_dict[driver][order].append(
                    DriverActionPair(driver, Action(route), 0)
                )

    # data = pd.DataFrame(
    #     columns=[
    #         "Reward",
    #         "Target Time",
    #         "Target Position",
    #         "Current Time",
    #         "Current Position",
    #     ]
    # )
    data = []
    # 2. For each DriverActionPair: calculate edge weight based on reward and state value function
    for driver in driver_to_orders_to_routes_dict:
        for order in driver_to_orders_to_routes_dict[driver]:
            for pair in driver_to_orders_to_routes_dict[driver][order]:
                # TODO potentially bring in a Malus for long trips since they bind the car more longer
                # weight = time reduction for passenger + state value after this option

                reward = pair.action.route.time_reduction
                target_position = pair.action.route.vehicle_destination
                target_time = pair.action.route.order.dispatch_time.add_seconds(
                    pair.action.route.vehicle_time
                ).to_total_seconds()
                current_time = pair.action.route.order.dispatch_time.to_total_seconds()
                current_position = pair.driver.current_position
                
                data.append((reward, target_time, target_position, current_time, current_position))
    return data
