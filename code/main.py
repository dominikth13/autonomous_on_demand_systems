import random
from algorithm import (
    generate_driver_action_pairs,
    generate_routes,
    solve_optimization_problem,
)
from program_fixtures import *
from state_value_table import Grid, Time, TimeSeries
from state import State
from order import Order
from location import Location
from driver import Driver


def test():
    drivers = [
        Driver(
            Location(
                random.Random(i).randint(0, 10000),
                random.Random(i * i).randint(0, 10000),
            )
        )
        for i in range(400)
    ]
    orders = [
        Order(
            Location(
                random.Random(i + 15).randint(0, 10000),
                random.Random((i + 15) ** 3).randint(0, 10000),
            ),
            Location(
                random.Random((i + 15) ** 2).randint(0, 10000),
                random.Random((i + 15) ** 4).randint(0, 10000),
            ),
        )
        for i in range(300)
    ]

    order_routes_dict = generate_routes(orders)
    driver_action_pairs = generate_driver_action_pairs(order_routes_dict, drivers)
    result_pairs = solve_optimization_problem(driver_action_pairs)
    driver_r = list(map(lambda x: x[0], result_pairs))
    if len(driver_r) != len(set(driver_r)):
        print("Double drivers")
        exit(1)
    action_r = list(map(lambda x: x[1], result_pairs))
    order_r = list(
        map(lambda x: x.route.order, list(filter(lambda x: x.is_route(), action_r)))
    )
    if len(order_r) != len(set(order_r)):
        print("Double orders")
        exit(1)
    print("Optimization result optimal and valid for original problem")


def q_learning():

    while STATE.current_interval.next_interval != None:
        # Collect new orders
        orders = [
            Order(
                Location(
                    random.Random(i + 15).randint(0, 10000),
                    random.Random((i + 15) ** 3).randint(0, 10000),
                ),
                Location(
                    random.Random((i + 15) ** 2).randint(0, 10000),
                    random.Random((i + 15) ** 4).randint(0, 10000),
                ),
            )
            for i in range(random.randint(0, 30))
        ]
        # Add orders to state
        STATE.add_orders(orders)
        # Generate routes
        order_routes_dict = generate_routes(orders)
        # Generate Action-Driver pairs with all available routes and drivers
        driver_action_pairs = generate_driver_action_pairs(order_routes_dict)
        # Find Action-Driver matches based on a min-cost-flow problem
        matches = solve_optimization_problem(driver_action_pairs)
        # Apply state changes based on Action-Driver matches and existing driver jobs
        STATE.apply_state_change(matches)
        # Update the expiry durations of still open orders
        STATE.update_order_expiry_duration()
        # Increment to next interval
        STATE.increment_time_interval()

test()
