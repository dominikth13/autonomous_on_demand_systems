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
                    connection = FASTEST_STATION_CONNECTION_NETWORK.get_fastest_connection(
                        origin, destination
                    )
                    # Distance
                    vehicle_time = start.distance_to(origin.position) / VEHICLE_SPEED
                    walking_time = destination.position.distance_to(end) / WALKING_SPEED
                    transit_time = connection[1]
                    stations = connection[0]
                    # TODO include entry, exit and waiting time
                    other_time = 0
                    total_time = vehicle_time + walking_time + transit_time + other_time

                    if total_time < default_route.total_time + L2:
                        # if the route contains transit ticket for public transport needs to be added to overall price
                        if transit_time > 0:
                            opnv_ticket = 2
                        else:
                            opnv_ticket = 0

                        #1.5 euro for each km with the vehicle 
                        vehicle_price = start.distance_to(origin.position)*1.5 
                        price = vehicle_price + opnv_ticket
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
                                    vehicle_price,
                                    price,
                                )
                            )
    return routes_per_order


# TODO build class for return value
def generate_driver_action_pairs(
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

    # 2. For each DriverActionPair: calculate edge weight based on reward and state value function
    for driver in driver_to_orders_to_routes_dict:
        for order in driver_to_orders_to_routes_dict[driver]:
            for pair in driver_to_orders_to_routes_dict[driver][order]:
                arrival_interval = STATE_VALUE_TABLE.time_series.find_interval(
                    STATE.current_interval.start.add_minutes(
                        pair.get_total_vehicle_travel_time_in_seconds() // 60
                    )
                )
                # weight = revenue for driver + state value after this option
                weight = (
                    pair.action.route.vehicle_price
                    + DISCOUNT_FACTOR(
                        STATE.current_interval.start, arrival_interval.start
                    )
                    * STATE_VALUE_TABLE.get_state_value(pair.action.route.vehicle_destination_zone, arrival_interval)
                )
                pair.weight = weight

    # 3. Filter out all DriverRoutePairs which are not the one with highest edge value for each order and each driver
    driver_action_pairs = []
    for driver in driver_to_orders_to_routes_dict:
        driver_action_pairs.append(driver_to_idling_dict[driver])
        for order in driver_to_orders_to_routes_dict[driver]:
            best_action_route = driver_to_orders_to_routes_dict[driver][order][0]
            for pair in driver_to_orders_to_routes_dict[driver][order]:
                if pair.weight > best_action_route.weight:
                    best_action_route = pair
            driver_action_pairs.append(best_action_route)

    return driver_action_pairs


def solve_optimization_problem(
    driver_action_pairs: list[DriverActionPair],
) -> list[DriverActionPair]:
    # return solve_as_bipartite_matching_problem(driver_action_pairs)
    # solve_as_min_cost_flow_problem(driver_action_pairs)
    return or_tools_min_cost_flow(driver_action_pairs)
