import random
from algorithm import (
    generate_driver_action_pairs,
    generate_routes,
    solve_optimization_problem,
)
from state import STATE
from location import Location
from order import Order

def q_learning():
    # 1. Find all shortest paths in public transport network
    # Is done automatically in station.py

    # 2. Run Q-Learning algorithm to train state value table
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