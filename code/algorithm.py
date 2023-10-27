import copy
import random
from model_builder import or_tools_min_cost_flow, solve_as_bipartite_matching_problem, solve_as_min_cost_flow_problem
from action import Action, DriverActionPair
from driver import Driver
from route import *
from order import Order
from station import FastestStationConnectionNetwork
from program_params import *

from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable


# The so called 'Algorithm 1'
def generate_routes(orders: list[Order]) -> dict[Order, list[Route]]:
    routes_per_order = {order: [] for order in orders}
    for order in orders:
        default_route = regular_route(order)
        routes_per_order[order].append(default_route)
        start = order.start
        end = order.end

        if default_route.total_time > L1:
            for origin in STATIONS:
                for destination in STATIONS:
                    if origin == destination:
                        continue
                    connection = FASTEST_STATION_NETWORK.get_fastest_connection(
                        origin, destination
                    )
                    # Distance
                    vehicle_time = start.distance_to(origin.position) * VEHICLE_SPEED
                    walking_time = destination.position.distance_to(end) * WALKING_SPEED
                    transit_time = connection[1]
                    stations = connection[0]
                    # TODO include entry, exit and waiting time
                    other_time = 0
                    total_time = vehicle_time + walking_time + transit_time + other_time

                    if total_time < default_route.total_time + L2:
                        # TODO include price calculation
                        price = 4
                        if price < default_route.price:
                            routes_per_order[order].append(
                                Route(
                                    order,
                                    start,
                                    end,
                                    stations,
                                    vehicle_time,
                                    transit_time,
                                    walking_time,
                                    other_time,
                                    total_time,
                                    price,
                                )
                            )
    return routes_per_order


# TODO build class for return value
def generate_driver_action_pairs(
    order_routes_dict: dict[Order, list[Route]], drivers: list[Driver]
) -> list[DriverActionPair]:
    driver_to_orders_to_routes_dict = {driver: {} for driver in drivers}
    driver_to_idling_dict = {driver: None for driver in drivers}

    # Central idling action
    idling = Action(None)

    # 1. Generate DriverActionPair for each pair that is valid (distance), add DriverIdlingPair for each driver
    for driver in drivers:
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

    # 2. For each DriverActionPair: calculate edge weight based on value state function
    # TODO implement real value computation, for now consistent shuffle
    # shuffled_dict = {
    #     driver: {
    #         order: random.Random(driver.id).sample(
    #             order_routes_dict[order], len(order_routes_dict[order])
    #         )
    #         for order in driver_to_orders_to_routes_dict[driver]
    #     }
    #     for driver in drivers
    # }
    # for driver in driver_to_orders_to_routes_dict:
    #     driver_to_idling_dict[driver].weight = random.Random(driver.id).random() * 10

    #     for order in driver_to_orders_to_routes_dict[driver]:
    #         routes = shuffled_dict[driver][order]
    #         for pair in driver_to_orders_to_routes_dict[driver][order]:
    #             pair.weight = routes.index(pair.action.route)
    for driver in driver_to_orders_to_routes_dict:
        # Implement calculation of weights  (price + value state after this option)
        pass

    # 3. Filter out all DriverRoutePairs which are not the one with highest edge value for each order and each driver
    driver_action_pairs = []
    for driver in driver_to_orders_to_routes_dict:
        driver_action_pairs.append(driver_to_idling_dict[driver])
        for order in driver_to_orders_to_routes_dict[driver]:
            best_action_route = sorted(
                driver_to_orders_to_routes_dict[driver][order],
                key=lambda x: x.weight,
                reverse=True,
            )[0]
            driver_action_pairs.append(best_action_route)

    return driver_action_pairs


def solve_optimization_problem(driver_action_pairs: list[DriverActionPair]):
    #return solve_as_bipartite_matching_problem(driver_action_pairs)
    #solve_as_min_cost_flow_problem(driver_action_pairs)
    return or_tools_min_cost_flow(driver_action_pairs)
