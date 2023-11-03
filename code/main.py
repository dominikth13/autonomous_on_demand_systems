import random
from algorithm import (
    generate_driver_action_pairs,
    generate_routes,
    solve_optimization_problem,
)
from location import Location
from station import Station
from program_fixtures import *
from state_value_table import Grid, Time, TimeSeries
from order import Order
from driver import Driver
from model_builder import solve_all_pair_shortest_path_problem


def test_q_learning_step():
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

def test_shortest_path_solver():
    connections = []
    stations = [Station(Location(1,1)) for i in range(20)]
    for station1 in stations:
        for station2 in stations:
            if station1 == station2:
                continue
            connections.append((station1, random.Random(station1.id*station2.id).randint(0, 10000), station2))
    
    result_dict = solve_all_pair_shortest_path_problem(connections)
    print(result_dict)


def q_learning():
    initialize()

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

test_shortest_path_solver()
