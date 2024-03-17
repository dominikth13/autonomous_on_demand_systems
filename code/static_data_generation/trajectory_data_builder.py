import csv
import random

import pandas as pd
from action.action import Action
from action.driver_action_pair import DriverActionPair
from algorithm.algorithm import generate_routes
from deep_reinforcement_learning.deep_rl_training import import_trajectories
from driver.driver import Driver
from driver.drivers import Drivers
from grid.grid import Grid
from interval.time import Time
from logger import LOGGER
from order import Order
from program.program_params import ProgramParams
from route import Route


def generate_trajectories() -> None:
    orders = []
    LOGGER.debug("Generate orders")
    # Build trajectories with 10% of the data
    for i in range(0, 1440):
        sub_orders = Order.get_orders_by_time()[Time.of_total_minutes(i)]
        if len(sub_orders) == 0:
            continue
        sample_size = 30 if len(sub_orders) >= 30 else len(sub_orders)
        orders.extend(random.sample(sub_orders, sample_size))

    LOGGER.debug("Dispatch orders")
    for order in orders:
        order.dispatch()

    LOGGER.debug("Generate routes")
    order_to_route_dict: dict[Order, list[Route]] = {
        order: generate_routes([order])[order] for order in orders
    }
    LOGGER.debug("Filter routes")
    for order in order_to_route_dict.keys():
        sorted_routes = list(
            sorted(
                order_to_route_dict[order],
                key=lambda x: x.time_reduction,
            )
        )
        # For each driver order pair only return best pair
        best = sorted_routes[-1]
        order_to_route_dict[order] = best

    driver_to_orders_to_routes_dict: dict[
        Driver, dict[Order, list[DriverActionPair]]
    ] = {driver: {} for driver in Drivers.get_drivers()}

    LOGGER.debug("Generate vehicle order pairs & trajectories")
    data = []
    # 1. Generate DriverActionPair for each pair that is valid (distance), add DriverIdlingPair for each driver
    total_count = 0
    for driver in Drivers.get_drivers():
        for order in orders:
            total_count += 1
            if total_count % 1000000 == 0:
                LOGGER.debug(
                    f"Generated {total_count}/{len(Drivers.get_drivers())*len(orders)} driver-order matches ({(total_count/(len(Drivers.get_drivers())*len(orders)))*100} %)"
                )
            if (
                order.start.distance_to(driver.current_position)
                > ProgramParams.PICK_UP_DISTANCE_THRESHOLD
            ):
                # If driver is currently to far away for this order he ignores it
                continue
            driver_to_orders_to_routes_dict[driver][order] = []
            route = order_to_route_dict[order]
            driver_to_orders_to_routes_dict[driver][order].append(
                DriverActionPair(driver, Action(route), 0)
            )

            for pair in driver_to_orders_to_routes_dict[driver][order]:
                reward = pair.action.route.time_reduction
                target_zone = pair.action.route.vehicle_destination_cell.zone
                target_time = pair.action.route.order.dispatch_time.add_seconds(
                    pair.get_total_vehicle_travel_time_in_seconds()
                ).to_total_seconds()
                current_time = pair.action.route.order.dispatch_time.to_total_seconds()
                current_zone = Grid.get_instance().find_zone(pair.driver.current_position)

                if target_time - current_time == 0:
                    continue

                data.append(
                    (
                        reward,
                        target_time,
                        target_zone,
                        current_time,
                        current_zone
                    )
                )


    wd_str = {0: "mon", 1: "tue", 2: "wed", 3: "thu", 4: "fri", 5: "sat", 6: "sun"}
    calendar_week = ProgramParams.SIMULATION_DATE.isocalendar()[1]
    weekday = wd_str[ProgramParams.SIMULATION_DATE.weekday()]
    LOGGER.debug("Export trajectories")
    with open(
        f"code/trajectories/trajectories_{weekday}_week_{calendar_week}.csv", "w"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "reward",
                "target_time",
                "target_zone",
                "current_time",
                "current_zone"
            ]
        )
        for trajectory in data:
            writer.writerow(
                [
                    trajectory[0],
                    trajectory[1],
                    trajectory[2].id,
                    trajectory[3],
                    trajectory[4].id
                ]
            )


def remove_idle_trajectories():
    trajectories = import_trajectories()
    valid_trajectories = []
    for trajectory in trajectories:
        if trajectory["target_time"] - trajectory["current_time"] == 0:
            continue
        valid_trajectories.append(trajectory)
    with open("code/trajectories/trajectories.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "reward",
                "target_time",
                "target_zone",
                "current_time",
                "current_zone"
            ]
        )
        for trajectory in valid_trajectories:
            writer.writerow(
                [
                    trajectory["reward"],
                    trajectory["target_time"],
                    trajectory["target_zone"],
                    trajectory["current_time"],
                    trajectory["current_zone"]
                ]
            )
