from action.driver_action_pair import DriverActionPair
from algorithm.model_builder import or_tools_min_cost_flow
from data_output.data_collector import DataCollector
from driver.drivers import Drivers
from grid.grid import Grid
from interval.time_series import TimeSeries
from logger import LOGGER
from program.program_params import Mode, ProgramParams
from public_transport.fastest_station_connection_network import (
    FastestStationConnectionNetwork,
)

from action.action import Action
from driver.driver import Driver
from order import Order
from route import Route, regular_route
from state.state import State
from state.state_value_networks import StateValueNetworks
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

        if order.direct_connection[1] > ProgramParams.L1:
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
                    vehicle_time = (
                        start.distance_to(origin.position) / ProgramParams.VEHICLE_SPEED
                    )
                    walking_time = (
                        destination.position.distance_to(end)
                        / ProgramParams.WALKING_SPEED
                    )
                    transit_time = connection[1]
                    stations = connection[0]
                    # include entry, exit and waiting time
                    other_time = (
                        2 * ProgramParams.PUBLIC_TRANSPORT_ENTRY_EXIT_TIME
                        + ProgramParams.PUBLIC_TRANSPORT_WAITING_TIME(
                            order.dispatch_time
                        )
                    )
                    total_time = vehicle_time + walking_time + transit_time + other_time

                    # Since we want people to use public transport, here we check against the direct_connection without any bus
                    if total_time < order.direct_connection[1] + ProgramParams.L2:
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
        if ProgramParams.FEATURE_ADD_IDLING_COST_TO_TARGET:
            # We add a cost term of -60 to every idling action
            reward = (-1) * ProgramParams.IDLING_COST
        else:
            reward = 0
        if ProgramParams.EXECUTION_MODE == Mode.TABULAR:
            state_value = StateValueTable.get_state_value_table().get_state_value(
                Grid.get_instance().find_zone(driver.current_position),
                TimeSeries.get_instance().find_interval(
                    State.get_state().current_time.add_seconds(
                        ProgramParams.SIMULATION_UPDATE_RATE
                    )
                ),
            )
        elif ProgramParams.EXECUTION_MODE == Mode.DEEP_NEURAL_NETWORKS:
            state_value = StateValueNetworks.get_instance().get_target_state_value(
                Grid.get_instance().find_zone(driver.current_position),
                State.get_state().current_time.add_seconds(
                    ProgramParams.SIMULATION_UPDATE_RATE
                ),
            )
        else:
            state_value = 0

        weight = reward + state_value
        driver_to_idling_dict[driver] = DriverActionPair(driver, idling, weight)
        if driver.is_occupied():
            # If driver is occupied he cannot take any new order
            continue

        for order in order_routes_dict:
            if (
                order.start.distance_to(driver.current_position)
                > ProgramParams.PICK_UP_DISTANCE_THRESHOLD
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
                arrival_time = State.get_state().current_interval.start.add_seconds(
                    pair.get_total_vehicle_travel_time_in_seconds()
                )
                arrival_interval = TimeSeries.get_instance().find_interval(arrival_time)
                # weight = time reduction for passenger + state value after this option
                if ProgramParams.EXECUTION_MODE == Mode.TABULAR:
                    state_value = (
                        StateValueTable.get_state_value_table().get_state_value(
                            pair.action.route.vehicle_destination_cell.zone,
                            arrival_interval,
                        )
                    )
                elif ProgramParams.EXECUTION_MODE == Mode.DEEP_NEURAL_NETWORKS:
                    state_value = (
                        StateValueNetworks.get_instance().get_target_state_value(
                            pair.action.route.vehicle_destination_cell.zone,
                            arrival_time,
                        )
                    )
                else:
                    # Baseline Performance
                    state_value = 0
                weight = (
                    pair.action.route.time_reduction
                    + ProgramParams.DISCOUNT_FACTOR(
                        State.get_state().current_interval.start.distance_to(
                            arrival_interval.start
                        )
                    )
                    * state_value
                )
                if pair.action.is_route() and pair.action.route.is_regular_route():
                    weight = weight * ProgramParams.DIRECT_TRIP_DISCOUNT_FACTOR
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
    driver_action_pairs = or_tools_min_cost_flow(driver_action_pairs)
    occupied_drivers = len(
        list(filter(lambda x: x.driver.is_occupied(), driver_action_pairs))
    )
    relocated_drivers = len(
        list(
            filter(
                lambda x: x.driver.is_occupied() and x.driver.job.is_relocation,
                driver_action_pairs,
            )
        )
    )
    idling_drivers = (
        len(list(filter(lambda x: x.action.is_idling(), driver_action_pairs)))
        - occupied_drivers
    )
    matched_drivers = len(driver_action_pairs) - idling_drivers - occupied_drivers
    occupied_drivers = occupied_drivers - relocated_drivers
    LOGGER.debug(
        f"Matched drivers: {matched_drivers}, Occupied drivers: {occupied_drivers}, Relocated drivers: {relocated_drivers}, Idling drivers: {idling_drivers}"
    )
    DataCollector.append_workload(State.get_state().current_time, occupied_drivers)
    for pair in driver_action_pairs:
        if pair.action.is_idling():
            continue
        current_time = State.get_state().current_time
        driver_zone = Grid.get_instance().find_zone(pair.driver.current_position)
        passenger_pu_zone = pair.action.route.order.zone
        passenger_do_zone = Grid.get_instance().find_zone(
            pair.get_vehicle_destination()
        )
        destination_zone = Grid.get_instance().find_zone(pair.action.route.destination)
        vehicle_trip_time = pair.action.route.vehicle_time
        time_reduction = pair.action.route.time_reduction
        combi_route = not pair.action.route.is_regular_route()
        total_vehicle_distance = pair.get_total_vehicle_distance()
        DataCollector.append_trip(
            current_time,
            driver_zone,
            passenger_pu_zone,
            passenger_do_zone,
            destination_zone,
            vehicle_trip_time,
            time_reduction,
            combi_route,
            total_vehicle_distance,
        )
    return driver_action_pairs
