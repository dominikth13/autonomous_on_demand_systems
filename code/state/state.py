from __future__ import annotations
import math
import random
from action.driver_action_pair import DriverActionPair
from data_output.data_collector import DataCollector
from grid.grid import Grid
from driver.drivers import Drivers
from interval.time import Time
from interval.time_series import TimeSeries
from logger import LOGGER
from program.program_params import Mode, ProgramParams
from state.state_value_networks import StateValueNetworks
from state.state_value_table import StateValueTable
from order import Order
from grid.grid_cell import GridCell


# Define here how the grid and intervals should look like
class State:
    _state: State = None

    # Resets the state
    def reset() -> None:
        State._state = None

    def get_state() -> State:
        if State._state == None:
            State._state = State()
        return State._state

    def __init__(self) -> None:
        # Dict containing orders mapped by id
        self.orders_dict: dict[int, Order] = {}

        self.current_interval = TimeSeries.get_instance().intervals[0]
        self.current_time = self.current_interval.start

        # if EXECUTION_MODE == Mode.TABULAR -> [(reward, current_interval, current_zone, next_interval, next_zone)]
        # else -> [(reward, current_location, current_time, next_location, next_time)]
        self.action_tuples = []

    def apply_state_change(self, driver_action_pairs: list[DriverActionPair]) -> None:
        order_time_reduction_quota = []
        # Apply changes in simulation
        for pair in driver_action_pairs:
            driver = pair.driver
            action = pair.action

            if action.is_idling():
                if ProgramParams.EXECUTION_MODE == Mode.TABULAR:
                    self.action_tuples.append(
                        (
                            pair.weight,
                            Grid.get_instance().find_zone(driver.current_position),
                            self.current_interval,
                            Grid.get_instance().find_zone(driver.current_position),
                            TimeSeries.get_instance().find_interval(
                                self.current_time.add_minutes(1)
                            ),
                        )
                    )
                elif ProgramParams.EXECUTION_MODE == Mode.DEEP_NEURAL_NETWORKS:
                    self.action_tuples.append(
                        (
                            pair.weight,
                            Grid.get_instance().find_zone(driver.current_position),
                            self.current_time,
                            Grid.get_instance().find_zone(driver.current_position),
                            self.current_time.add_minutes(1),
                        )
                    )
            else:
                # Here we have to plan a route for the driver to take. The route consists of two parts:
                # pick up the person and drive the person to the desired location. Afterwards we calculate the total
                # travel time. This one and the driver finals position are saved together with him. In each interval,
                # the remaining travel time is discounted. When the travel time reaches zero, the driver reaches his
                # final position

                route = action.route
                driver_final_destination = pair.get_vehicle_destination()

                # Schedule new driver job and update it for the next state
                driver.set_new_job(
                    pair.get_total_vehicle_travel_time_in_seconds(),
                    pair.get_pickup_travel_time_in_seconds(),
                    pair.action.route.origin,
                    driver_final_destination,
                )

                if Grid.get_instance().find_zone(driver_final_destination).id == 9999:
                    LOGGER.warn(
                        f"Driver {driver.id} goes to forbidden zone by order {pair.action.route.order.id}"
                    )

                if ProgramParams.FEATURE_ORDERS_AS_WIN:
                    # We calculate the reward as the product of time reduction and amount of orders in this zone in the last 30 minutes
                    start_minutes = (
                        self.current_time.to_total_minutes() - 30
                        if self.current_time.to_total_minutes() - 30 > 0
                        else 0
                    )
                    amount_of_orders = len(
                        list(
                            filter(
                                lambda x: x.zone.id == route.order.zone.id,
                                [
                                    Order.get_orders_by_time(
                                        Time.of_total_minutes(total_minute)
                                    )
                                    for total_minute in range(
                                        start_minutes,
                                        self.current_time.to_total_minutes(),
                                    )
                                ],
                            )
                        )
                    )
                    reward = action.route.time_reduction * amount_of_orders
                else:
                    reward = action.route.time_reduction

                if ProgramParams.EXECUTION_MODE == Mode.TABULAR:
                    self.action_tuples.append(
                        (
                            reward,
                            Grid.get_instance().find_zone(driver.current_position),
                            self.current_interval,
                            action.route.vehicle_destination_cell.zone,
                            TimeSeries.get_instance().find_interval(
                                self.current_time.add_seconds(
                                    pair.get_total_vehicle_travel_time_in_seconds()
                                )
                            ),
                        )
                    )
                elif ProgramParams.EXECUTION_MODE == Mode.DEEP_NEURAL_NETWORKS:
                    self.action_tuples.append(
                        (
                            reward,
                            Grid.get_instance().find_zone(driver.current_position),
                            self.current_time,
                            action.route.vehicle_destination_cell.zone,
                            self.current_time.add_seconds(
                                pair.get_total_vehicle_travel_time_in_seconds()
                            ),
                        )
                    )

                # Save reduction quota
                if route.order.direct_connection[1] > 0:
                    order_time_reduction_quota.append(
                        (
                            (
                                action.route.time_reduction
                                - route.order.direct_connection[1]
                            )
                            / route.order.direct_connection[1]
                        )
                    )
                # Remove order from open orders set
                del self.orders_dict[route.order.id]

        self.apply_state_changes_to_value_function()

        amount_of_unserved_orders = len(self.orders_dict)
        DataCollector.append_orders_data(
            self.current_time,
            amount_of_unserved_orders
            / (amount_of_unserved_orders + len(order_time_reduction_quota))
            if (amount_of_unserved_orders + len(order_time_reduction_quota)) > 0
            else 0,
            len(order_time_reduction_quota),
        )
        average_time_reduction_quota = (
            sum(order_time_reduction_quota) / len(order_time_reduction_quota)
            if len(order_time_reduction_quota) > 0
            else 0
        )
        DataCollector.append_time_reduction_quota(
            self.current_time, average_time_reduction_quota
        )

        # Compute job state changes for all drivers
        for driver in Drivers.get_drivers():
            driver.update_job_status(ProgramParams.SIMULATION_UPDATE_RATE)

    def apply_state_changes_to_value_function(self) -> None:
        if ProgramParams.EXECUTION_MODE == Mode.TABULAR:
            for tup in self.action_tuples:
                StateValueTable.get_state_value_table().adjust_state_value(
                    reward=tup[0],
                    current_zone=tup[1],
                    current_interval=tup[2],
                    next_zone=tup[3],
                    next_interval=tup[4],
                )
        elif ProgramParams.EXECUTION_MODE == Mode.DEEP_NEURAL_NETWORKS:
            StateValueNetworks.get_instance().adjust_state_values(self.action_tuples)

        self.action_tuples = []

    def update_order_expiry_duration(self) -> None:
        duration = ProgramParams.SIMULATION_UPDATE_RATE
        orders_to_delete = []
        for id in self.orders_dict:
            self.orders_dict[id].expires -= duration
            if self.orders_dict[id].expires <= 0:
                orders_to_delete.append(id)
        # Delete expired orders
        for id in orders_to_delete:
            del self.orders_dict[id]

    def add_orders(self, orders: list[Order]) -> None:
        for order in orders:
            self.orders_dict[order.id] = order

    def increment_time_interval(self, current_time) -> None:
        if self.current_interval.end.is_before(current_time):
            self.current_interval = TimeSeries.get_instance().intervals[
                self.current_interval.index + 1
            ]
        self.current_time = current_time

    def relocate(self) -> None:
        from grid.grid import Grid

        # Relocate drivers which idle for long time
        for driver in Drivers.get_drivers():
            if driver.idle_time >= ProgramParams.MAX_IDLING_TIME:
                # Calculate probability distribution
                current_cell = Grid.get_instance().find_cell(driver.current_position)
                zones = current_cell.zone.find_adjacent_zone_ids(
                    ProgramParams.RELOCATION_RADIUS
                )
                cells = [current_cell]
                for zone in zones:
                    cells = list(
                        filter(
                            lambda x: not x.is_empty(),
                            Grid.get_instance().cells_dict[zone],
                        )
                    )

                    if len(cells) == 0:
                        continue
                    cells.append(random.choice(cells))

                cells_to_weight = {}
                min_state_value = float("inf")
                for cell in cells:
                    driving_time = int(
                        current_cell.center.distance_to(cell.center)
                        / ProgramParams.VEHICLE_SPEED
                    )
                    if ProgramParams.EXECUTION_MODE == Mode.TABULAR:
                        state_value = (
                            StateValueTable.get_state_value_table().get_state_value(
                                cell.zone,
                                TimeSeries.get_instance().find_interval(
                                    self.current_time.add_seconds(driving_time)
                                ),
                            )
                        )
                    elif ProgramParams.EXECUTION_MODE == Mode.DEEP_NEURAL_NETWORKS:
                        state_value = (
                            StateValueNetworks.get_instance().get_target_state_value(
                                cell.zone, self.current_time.add_seconds(driving_time)
                            )
                        )
                    else:
                        state_value = random.randint(1, 100)
                    if min_state_value > state_value:
                        min_state_value = state_value
                    cells_to_weight[cell] = state_value

                for cell in cells_to_weight:
                    # We don't want negative or 0 values
                    state_value = cells_to_weight[cell] + abs(min_state_value) + 1
                    cells_to_weight[cell] = (
                        ProgramParams.DISCOUNT_FACTOR(driving_time) * state_value
                    )

                # Get the relocation target based on weighted stochastic choices
                if len(cells_to_weight) > 0:
                    total_weight = sum(cells_to_weight.values())
                    cell_list = []
                    probability_list = []
                    for cell in cells_to_weight:
                        cell_list.append(cell)
                        probability_list.append(cells_to_weight[cell] / total_weight)
                        relocation_cell = random.choices(
                            cell_list, weights=probability_list, k=1
                        )[0]
                else:
                    # When we can't switch just stay
                    continue

                # Create relocation job
                driving_time = int(
                    driver.current_position.distance_to(relocation_cell.center)
                    / ProgramParams.VEHICLE_SPEED
                )
                driver.set_new_relocation_job(driving_time, relocation_cell.center)
                driver.idle_time = 0
                DataCollector.append_relocation_trip_data(
                    self.current_time,
                    current_cell.zone,
                    relocation_cell.zone,
                    int(driver.current_position.distance_to(relocation_cell.center)),
                )
