from __future__ import annotations
import math
import random
from action.driver_action_pair import DriverActionPair
from grid.grid import Grid
from driver.drivers import Drivers
from interval.time_series import TimeSeries
from program_params import Mode, ProgramParams
from state.state_value_networks import StateValueNetworks
from state.state_value_table import StateValueTable
from order import Order


# Define here how the grid and intervals should look like
class State:
    _state: State = None

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
        # Apply changes in simulation
        for pair in driver_action_pairs:
            driver = pair.driver
            action = pair.action

            if action.is_idling():
                if ProgramParams.EXECUTION_MODE == Mode.TABULAR:
                    self.action_tuples.append(
                        (
                            0,
                            Grid.get_instance().find_zone(driver.current_position),
                            self.current_interval,
                            Grid.get_instance().find_zone(driver.current_position),
                            TimeSeries.get_instance().find_interval(
                                self.current_time.add_minutes(1)
                            ),
                        )
                    )
                else:
                    self.action_tuples.append(
                        (
                            0,
                            driver.current_position,
                            self.current_time,
                            driver.current_position,
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
                    int(pair.get_total_vehicle_travel_time_in_seconds()),
                    driver_final_destination,
                )
                if ProgramParams.EXECUTION_MODE == Mode.TABULAR:
                    self.action_tuples.append(
                        (
                            action.route.time_reduction,
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
                else:
                    self.action_tuples.append(
                        (
                            action.route.time_reduction,
                            driver.current_position,
                            self.current_time,
                            action.route.vehicle_destination,
                            self.current_time.add_seconds(
                                pair.get_total_vehicle_travel_time_in_seconds()
                            ),
                        )
                    )

                # Remove order from open orders set
                del self.orders_dict[route.order.id]

        self.apply_state_changes_to_value_function()

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
        else:
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
                cells = Grid.get_instance().find_n_adjacent_cells(current_cell, 2)
                cells_to_weight = {}

                for cell in cells:
                    driving_time = int(
                        current_cell.center.distance_to(cell.center) / ProgramParams.VEHICLE_SPEED
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
                    else:
                        state_value = (
                            StateValueNetworks.get_instance().get_target_state_value(
                                cell.center, self.current_time.add_seconds(driving_time)
                            )
                        )

                    cells_to_weight[cell] = math.exp(
                        ProgramParams.DISCOUNT_FACTOR(driving_time) * state_value
                    )

                total_weight = sum(cells_to_weight.values())
                cell_list = []
                probability_list = []
                for cell in cells:
                    cell_list.append(cell)
                    probability_list.append(cells_to_weight[cell] / total_weight)

                # Get the relocation target based on weighted stochastic choices
                relocation_cell = random.choice(cell_list, weights=probability_list)

                # Create relocation job
                driving_time = int(
                    driver.current_position.distance_to(relocation_cell.center)
                    / ProgramParams.VEHICLE_SPEED
                )
                driver.set_new_relocation_job(driving_time, relocation_cell.center)
                driver.idle_time = 0

                # Maybe this is needed but I'm not sure now
                # if EXECUTION_MODE == Mode.TABULAR:
                #     self.action_tuples.append(
                #         (
                #             0,
                #             Grid.get_instance().find_zone(driver.current_position),
                #             self.current_interval,
                #             relocation_cell.zone,
                #             TimeSeries.get_instance().find_interval(
                #                 self.current_time.add_seconds(driving_time)
                #             ),
                #         )
                #     )
                # else:
                #     self.action_tuples.append(
                #         (
                #             0,
                #             driver.current_position,
                #             self.current_time,
                #             relocation_cell.center,
                #             self.current_time.add_seconds(driving_time),
                #         )
                #     )
