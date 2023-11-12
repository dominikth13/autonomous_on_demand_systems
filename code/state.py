from state_value_table import STATE_VALUE_TABLE
from order import Order
from program_params import *
from driver import DRIVERS, Driver
from location import *
from time_interval import *

# Define here how the grid and intervals should look like
class State:
    from action import Action, DriverActionPair

    def __init__(self) -> None:
        # Dict containing orders mapped by id
        self.orders_dict: dict[int, Order] = {}

        self.current_interval = STATE_VALUE_TABLE.time_series.intervals[0]
        self.current_time = self.current_interval.start

    def apply_state_change(self, driver_action_pairs: list[DriverActionPair]) -> None:
        # Apply driver action pairs
        for pair in driver_action_pairs:
            driver = pair.driver
            action = pair.action

            if action.is_idling():
                STATE_VALUE_TABLE.adjust_state_value(
                    0,
                    current_interval=self.current_interval,
                    current_location=driver.current_position,
                    next_interval=STATE_VALUE_TABLE.time_series.get_next_interval(self.current_interval),
                    next_location=driver.current_position
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
                driver.set_new_job(int(pair.get_total_vehicle_travel_time_in_seconds()), driver_final_destination)

                # Adjust state value
                STATE_VALUE_TABLE.adjust_state_value(
                    action.route.vehicle_price,
                    current_interval=self.current_interval,
                    current_location=driver.current_position,
                    next_time=self.current_time.add_seconds(pair.get_total_vehicle_travel_time_in_seconds()),
                    next_zone=action.route.vehicle_destination_zone,
                )

                # Remove order from open orders set
                del self.orders_dict[route.order.id]

        # Compute job state changes for all drivers
        for driver in DRIVERS:
            driver.update_job_status(SIMULATION_UPDATE_RATE)
    
    def update_order_expiry_duration(self) -> None:
        duration = SIMULATION_UPDATE_RATE
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
            self.current_interval = STATE_VALUE_TABLE.time_series.intervals[self.current_interval.index + 1]
        self.current_time = current_time

STATE: State = State()
        
                

