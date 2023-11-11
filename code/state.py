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
        # Dict containing orders mapped by id and the amount of remaining seconds to serve the order
        self.expiring_orders_dict: dict[int, tuple[Order, int]] = {}

        self.current_interval = STATE_VALUE_TABLE.time_series.intervals[0]

    def apply_state_change(self, driver_action_pairs: list[DriverActionPair]) -> None:
        # Build a set to check if all drivers applied a state change
        remaining_drivers = set(DRIVERS)

        # Apply driver action pairs
        for pair in driver_action_pairs:
            driver = pair.driver
            action = pair.action

            if action.is_idling():
                STATE_VALUE_TABLE.adjust_state_value(
                    current_interval=self.current_interval,
                    current_location=driver.current_position,
                    next_interval=self.current_interval.next_interval,
                    next_location=driver.current_position,
                    reward=0
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
                driver.set_new_job(int(pair.get_total_vehicle_travel_time_in_seconds), driver_final_destination)
                driver.update_job_status(self.current_interval.start.distance_to_in_seconds(self.current_interval.next_interval.start))

                # Adjust state value
                STATE_VALUE_TABLE.adjust_state_value(
                    current_interval=self.current_interval,
                    current_location=driver.current_position,
                    next_interval=self.current_interval.next_interval,
                    next_zone=action.route.vehicle_destination_zone,
                    reward=action.route.vehicle_price
                )

                # Remove order from open orders set
                del self.expiring_orders_dict[route.order.id]
            remaining_drivers.remove(driver)

        # Compute job state changes for occupied drivers
        for driver in remaining_drivers:
            driver.update_job_status(self.current_interval.start.distance_to_in_seconds(self.current_interval.next_interval.start))
    
    def update_order_expiry_duration(self) -> None:
        duration = self.current_interval.start.distance_to_in_seconds(self.current_interval.next_interval.start)
        for id in self.expiring_orders_dict:
            self.expiring_orders_dict[id][1] -= duration
            if self.expiring_orders_dict[id][1] <= 0:
                # Delete expired orders
                del self.expiring_orders_dict[id]

    def add_orders(self, orders: list[Order]) -> None:
        for order in orders:
            self.expiring_orders_dict[order.id] = (order, ORDER_EXPIRY_DURATION)
    
    def increment_time_interval(self) -> None:
        self.current_interval = self.current_interval.next_interval

STATE: State = State()
        
                

