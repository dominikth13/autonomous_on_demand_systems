import csv
import random

import pandas as pd
from action.action import Action
from action.driver_action_pair import DriverActionPair
from algorithm.algorithm import generate_routes
from deep_reinforcement_learning.deep_rl_training import import_trajectories
from driver.driver import Driver
from driver.drivers import Drivers
from interval.time import Time
from logger import LOGGER
from order import Order
from program_params import ProgramParams


def generate_trajectories() -> None:
    orders = []
    LOGGER.debug("Generate orders")
    for i in range(1440):
        sub_orders = Order.get_orders_by_time()[Time.of_total_minutes(i)]
        if len(sub_orders) == 0:
            continue
        orders.extend(random.sample(sub_orders, len(sub_orders) // 10))

    driver_to_orders_to_routes_dict: dict[
        Driver, dict[Order, list[DriverActionPair]]
    ] = {driver: {} for driver in Drivers.get_drivers()}
    LOGGER.debug("Dispatch orders")
    # 1. Generate DriverActionPair for each pair that is valid (distance), add DriverIdlingPair for each driver
    for driver in Drivers.get_drivers():
        for order in orders:
            if (
                order.start.distance_to(driver.current_position)
                > ProgramParams.PICK_UP_DISTANCE_THRESHOLD
            ):
                # If driver is currently to far away for this order he ignores it
                continue
            driver_to_orders_to_routes_dict[driver][order] = []
            order.dispatch()
            for route in generate_routes([order])[order]:
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
    LOGGER.debug("Create trajectories")
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

                if target_time - current_time == 0:
                    continue

                data.append(
                    (
                        reward,
                        target_time,
                        target_position,
                        current_time,
                        current_position,
                    )
                )

    LOGGER.debug("Export trajectories")
    with open("code/data/trajectories.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "reward",
                "target_time",
                "target_lat",
                "target_lon",
                "current_time",
                "current_lat",
                "current_lon",
            ]
        )
        for trajectory in data:
            writer.writerow(
                [
                    trajectory[0],
                    trajectory[1],
                    trajectory[2].lat,
                    trajectory[2].lon,
                    trajectory[3],
                    trajectory[4].lat,
                    trajectory[4].lon,
                ]
            )

def remove_idle_trajectories():
    trajectories = import_trajectories()
    valid_trajectories = []
    for trajectory in trajectories:
        if trajectory["target_time"] - trajectory["current_time"] == 0:
            continue
        valid_trajectories.append(trajectory)
    with open("code/data/trajectories.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "reward",
                "target_time",
                "target_lat",
                "target_lon",
                "current_time",
                "current_lat",
                "current_lon",
            ]
        )
        for trajectory in valid_trajectories:
            writer.writerow(
                [
                    trajectory["reward"],
                    trajectory["target_time"],
                    trajectory["target_lat"],
                    trajectory["target_lon"],
                    trajectory["current_time"],
                    trajectory["current_lat"],
                    trajectory["current_lon"],
                ]
            )