from action.driver_action_pair import DriverActionPair
from algorithm.model_builder import or_tools_min_cost_flow
from driver.drivers import Drivers
from interval.time_series import TimeSeries
from public_transport.fastest_station_connection_network import FastestStationConnectionNetwork

from action.action import Action
from driver.driver import Driver
from order import Order
from program_params import *
from route import Route, regular_route
from state.state import State
from state.state_value_table import StateValueTable

# The so called 'Algorithm 1'
def generate_routes(orders: list[Order]) -> dict[Order, list[Route]]:
    routes_per_order = {order: [] for order in orders}
    fastest_connection_network = FastestStationConnectionNetwork.get_instance()
    for order in orders:
        default_route = regular_route(order)
        routes_per_order[order].append(default_route)
        start = order.start
        end = order.end

        if default_route.total_time > L1:
            # 1. Get the closest start and end station for each line
            from public_transport.station import Station
            origins: list[Station] = []
            destinations: list[Station] = []
            for line in fastest_connection_network.lines:
                origins.append(line.get_closest_station(start))
                destinations.append(line.get_closest_station(end))

            # 2. Generate combination routes
            for origin in origins:
                for destination in destinations:
                    if origin == destination:
                        continue
                    connection = fastest_connection_network.get_fastest_connection(
                        origin, destination
                    )

                    # Distance (time in second)
                    vehicle_time = start.distance_to(origin.position) / VEHICLE_SPEED
                    walking_time = destination.position.distance_to(end) / WALKING_SPEED
                    transit_time = connection[1]
                    stations = connection[0]
                    # include entry, exit and waiting time
                    other_time = 2 * PUBLIC_TRANSPORT_ENTRY_EXIT_TIME + PUBLIC_TRANSPORT_WAITING_TIME(State.get_state().current_interval.start)
                    total_time = vehicle_time + walking_time + transit_time + other_time

                    if total_time < default_route.total_time + L2:
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
                                order.direct_connection[1] - total_time
                            )
                        )
    return routes_per_order


def generate_driver_action_pairs(
    order_routes_dict: dict[Order, list[Route]]
) -> list[DriverActionPair]:
    driver_to_orders_to_routes_dict: dict[
        Driver, dict[Order, list[DriverActionPair]]
    ] = {driver: {} for driver in Drivers.get_drivers()}
    driver_to_idling_dict = {driver: None for driver in Drivers.get_drivers()}

    # Central idling action
    idling = Action(None)

    # 1. Generate DriverActionPair for each pair that is valid (distance), add DriverIdlingPair for each driver
    for driver in Drivers.get_drivers():
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
                arrival_interval = TimeSeries.get_instance().find_interval(
                    State.get_state().current_interval.start.add_minutes(
                        pair.get_total_vehicle_travel_time_in_seconds() // 60
                    )
                )
                # TODO potentially bring in a Malus for long trips since they bind the car more longer
                # weight = time reduction for passenger + state value after this option
                weight = (
                    pair.action.route.time_reduction
                    + DISCOUNT_FACTOR(
                        State.get_state().current_interval.start.distance_to(arrival_interval.start)
                    )
                    * StateValueTable.get_state_value_table().get_state_value(pair.action.route.vehicle_destination_cell.zone, arrival_interval)
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
    # solve_as_min_cost_flow_problem(driver_action_pairs)
    return or_tools_min_cost_flow(driver_action_pairs)
