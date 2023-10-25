from bipartite_graph import BipartiteGraph
from driver import Driver
from route import *
from order import Order
from station import FastestStationConnectionNetwork
from program_params import *

from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable


# The so called 'Algorithm 1'
def generate_routes(
    orders: list[Order], station_network: FastestStationConnectionNetwork
) -> dict[Order, list[Route]]:
    routes_per_order = {order: [] for order in orders}
    for order in orders:
        default_route = regular_route(order.start, order.end)
        routes_per_order[order].append(regular_route)
        start = order.start
        end = order.end

        if default_route.total_time > L1:
            for origin in STATIONS:
                for destination in STATIONS:
                    if origin == destination:
                        continue
                    connection = FASTEST_STATION_NETWORK.get_fastest_connection(
                        origin, destination
                    )
                    # Distance
                    vehicle_time = start.distance_to(origin.position) * VEHICLE_SPEED
                    walking_time = destination.position.distance_to(end) * WALKING_SPEED
                    transit_time = connection[1]
                    stations = connection[0]
                    # TODO include entry, exit and waiting time
                    other_time = 0
                    total_time = vehicle_time + walking_time + transit_time + other_time

                    if total_time < default_route.total_time + L2:
                        # TODO include price calculation
                        price = 4
                        if price < default_route.price:
                            routes_per_order[order].append(
                                Route(
                                    start,
                                    end,
                                    stations,
                                    vehicle_time,
                                    transit_time,
                                    walking_time,
                                    other_time,
                                    total_time,
                                    price,
                                )
                            )
    return routes_per_order

# TODO build class for return value
def generate_route_driver_pairs(routes: dict[Order, list[Route]], drivers: list[Driver]) -> tuple:
    pass

def solve_bipartite_matching_problem(graph: BipartiteGraph):
    # Create the model
    model = LpProblem(name="bipartite-matching-problem", sense=LpMaximize)
    # Define the decision variables
    # TODO add names
    x = {driver: {route: LpVariable(name=f"x[{driver},{route}]", lowBound=0) for route in graph.adjacency_matrix[driver]} for driver in graph.adjacency_matrix.keys}
    # TODO see https://realpython.com/linear-programming-python/