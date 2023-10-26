import random
from algorithm import (
    generate_driver_action_pairs,
    generate_routes,
    solve_optimization_problem,
)
from customer import Customer
from order import Order
from location import Location
from driver import Driver


def test():
    drivers = [
        Driver(
            Location(
                random.Random(i).randint(0, 10000), random.Random(i * i).randint(0, 10000)
            )
        )
        for i in range(400)
    ]
    customers = [Customer() for i in range(1000)]
    orders = [
        Order(
            customers[i],
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
    print(result_pairs)

test()