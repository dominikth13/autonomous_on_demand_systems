import time
from station import Station
from driver import Driver
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable

import networkx as nx

# Not used because of computation complexity
from action import Action, DriverActionPair
def build_bipartite_matching_problem(
    driver_action_pairs: list[DriverActionPair],
) -> tuple[LpProblem, dict[LpVariable, tuple[Driver, Action]]]:
    # Build adjacency matrix with driver -> {action -> weight}
    adjacency_matrix: dict[Driver, dict[Action, float]] = {}
    for pair in driver_action_pairs:
        if pair.driver not in adjacency_matrix:
            adjacency_matrix[pair.driver] = {}
        adjacency_matrix[pair.driver][pair.action] = pair.weight

    # Create the model, see https://realpython.com/linear-programming-python/#using-pulp
    model = LpProblem(name="bipartite-matching-problem", sense=LpMaximize)
    # Define the decision variables
    # x -> (driver,action)
    var_pair_dict = {}
    # x -> weight
    var_weight_dict = {}
    # order -> [x]
    order_var_dict = {}
    # driver -> [x]
    driver_var_dict = {}
    for driver in adjacency_matrix:
        for action in adjacency_matrix[driver]:
            x = LpVariable(name=f"x[{action},{driver}]", cat="Binary")
            var_pair_dict[x] = (driver, action)
            var_weight_dict[x] = adjacency_matrix[driver][action]

            if driver not in driver_var_dict:
                driver_var_dict[driver] = []
            driver_var_dict[driver].append(x)

            if action.is_idling():
                # In this case we don't have an order
                continue

            if action.route.order not in order_var_dict:
                order_var_dict[action.route.order] = []
            order_var_dict[action.route.order].append(x)

    # Build target function
    model += lpSum([var_weight_dict[x] * x] for x in var_weight_dict)

    # Add constraints
    # Only one action per driver
    for driver in driver_var_dict:
        model += (lpSum(driver_var_dict[driver]) <= 1, f"driver {driver.id}")
    # Only one driver per order
    for order in order_var_dict:
        model += (lpSum(order_var_dict[order]) <= 1, f"order {order.id}")

    return (model, var_pair_dict)

# Not used because of computation complexity
# This method solves the optimization problem of DriverActionPairs as a bipartite matching problem
# This counts as a combinatorial problem which is NP hard
def solve_as_bipartite_matching_problem(
    driver_action_pairs: list[DriverActionPair],
) -> list[DriverActionPair]:
    start_time = time.time()
    (model, var_pair_dict) = build_bipartite_matching_problem(driver_action_pairs)
    medium_time = time.time()
    # Solve the model
    status = model.solve()

    end_time = time.time()
    print(
        f"The calculation took {end_time - medium_time} seconds, while preparation took {medium_time - start_time} seconds"
    )
    # Print results
    print(f"Model name: {model.name}, status: {LpStatus[model.status]}")

    result_pairs = []
    for x in var_pair_dict:
        if x.value() == 1:
            result_pairs.append(var_pair_dict[x])

    return result_pairs

# Not used anymore because of runtime inefficiencies
def solve_as_min_cost_flow_problem(
    driver_action_pairs: list[DriverActionPair],
) -> list[DriverActionPair]:
    start_time = time.time()
    graph = nx.DiGraph(name="min-cost-flow-problem")

    driver_nodes = set()
    route_action_nodes = set()
    idling_action_node = None
    order_nodes = set()
    t_node = "T-Node"

    for pair in driver_action_pairs:
        is_idling = pair.action.is_idling()
        driver = pair.driver
        action = pair.action
        order = pair.action.route.order if not is_idling else None

        # Check whether nodes are already in the graph
        if driver not in driver_nodes:
            driver_nodes.add(driver)
            graph.add_node(driver, demand=-1)
        # if is_idling:
        #     if action not in idling_action_nodes:
        #         idling_action_nodes.add(action)
        #         graph.add_node(action)
        # else:
        #     if action not in route_action_nodes:
        #         route_action_nodes.add(action)
        #         graph.add_node(action)
        #     if order not in order_nodes:
        #         order_nodes.add(order)

        # Add edge with negative weight and capacity 1
        graph.add_edge(pair.driver, pair.action, weight=pair.weight * (-1), capacity=1)
        driver_nodes.add(pair.driver)
        if is_idling and idling_action_node == None:
            idling_action_node = pair.action

        if not is_idling and not pair.action in route_action_nodes:
            # Add edge from route to order to limit on one route per order
            graph.add_edge(pair.action, pair.action.route.order, weight=0, capacity=1)
            route_action_nodes.add(pair.action)
            order_nodes.add(pair.action.route.order)

    graph.add_node(t_node, demand=len(driver_nodes))

    # Add edges between order/idling nodes and T-Node
    graph.add_edge(idling_action_node, t_node, weight=0, capacity=len(driver_nodes))
    for order in order_nodes:
        graph.add_edge(order, t_node, weight=0, capacity=1)

    medium_time = time.time()
    path = nx.min_cost_flow(graph)
    end_time = time.time()

    print(
        f"The calculation took {end_time - medium_time} seconds, while preparation took {medium_time - start_time} seconds"
    )


import numpy as np
from ortools.graph.python import min_cost_flow

# Solve the bipartite matching problem as a min-cost-flow problem to use the efficient C++ solver
def or_tools_min_cost_flow(driver_action_pairs: list[DriverActionPair]) -> list[tuple[Driver, Action]]:
    start_time = time.time()
    smcf = min_cost_flow.SimpleMinCostFlow()
    edge_sum = 0
    # 1. Calculate how many edges and nodes we totally have, create mappers to indices
    pair_to_index = {driver_action_pairs[i]: i for i in range(len(driver_action_pairs))}
    edge_sum += len(pair_to_index)
    # Route -> Order
    route_order_set = set()
    for pair in driver_action_pairs:
        if pair.action.is_route():
            if (pair.action, pair.action.route.order) not in route_order_set:
                route_order_set.add((pair.action, pair.action.route.order))
    route_order_list = list(route_order_set)
    del route_order_set
    route_order_to_index = {
        route_order_list[i - edge_sum]: i
        for i in range(edge_sum, edge_sum + len(route_order_list))
    }
    edge_sum += len(route_order_list)

    # Route -> T-Node
    order_t_set = set()
    for tup in route_order_list:
        order_t_set.add(tup[1])
    order_t_list = list(order_t_set)
    del order_t_set
    order_t_to_index = {
        order_t_list[i - edge_sum]: i for i in range(edge_sum, edge_sum + len(order_t_list))
    }
    edge_sum += len(order_t_list)

    # Idling -> T-Node
    idling_t_index = edge_sum
    edge_sum += 1

    node_sum = 0
    # Nodes
    # Drivers
    driver_set = set()
    for pair in driver_action_pairs:
        driver_set.add(pair.driver)
    driver_list = list(driver_set)
    del driver_set
    driver_to_index = {driver_list[i]: i for i in range(0, len(driver_list))}
    node_sum += len(driver_list)

    # Actions
    action_set = set()
    idling_node = None
    for pair in driver_action_pairs:
        if pair.action.is_idling():
            idling_node = pair.action
        action_set.add(pair.action)
    action_list = list(action_set)
    del action_set
    action_to_index = {
        action_list[i - node_sum]: i for i in range(node_sum, node_sum + len(action_list))
    }
    idling_to_index = action_to_index[idling_node]
    node_sum += len(action_list)

    # Orders
    order_to_index = {order_t_list[i - node_sum]: i for i in range(node_sum, node_sum + len(order_t_list))}
    node_sum += len(order_to_index)
    t_to_index = node_sum
    node_sum += 1

    # 2. Create arrays for edge and node definitions
    # Edges
    start_nodes = [0 for i in range(edge_sum)]
    end_nodes = [0 for i in range(edge_sum)]
    capacities = [1 for i in range(edge_sum)]
    weights = [0 for i in range(edge_sum)]
    for pair in pair_to_index:
        start_nodes[pair_to_index[pair]] = driver_to_index[pair.driver]
        end_nodes[pair_to_index[pair]] = action_to_index[pair.action]
        capacities[pair_to_index[pair]] = 1
        weights[pair_to_index[pair]] = pair.weight * (-1)
    
    for tup in route_order_to_index:
        start_nodes[route_order_to_index[tup]] = action_to_index[tup[0]]
        end_nodes[route_order_to_index[tup]] = order_to_index[tup[1]]
    
    for order in order_t_to_index:
        start_nodes[order_t_to_index[order]] = order_to_index[order]
        end_nodes[order_t_to_index[order]] = t_to_index
    
    start_nodes[idling_t_index] = idling_to_index
    end_nodes[idling_t_index] = t_to_index
    capacities[idling_t_index] = len(driver_list)

    start_arr = np.array(start_nodes)
    end_arr = np.array(end_nodes)
    capacities_arr = np.array(capacities)
    weights_arr = np.array(weights)

    # Nodes
    supplies = [0 for i in range(node_sum)]
    for driver in driver_to_index:
        supplies[driver_to_index[driver]] = 1
    
    supplies[t_to_index] = len(driver_to_index) * (-1)
    
    # 3. Fill the model
    # Add arcs, capacities and costs in bulk using numpy.
    all_arcs = smcf.add_arcs_with_capacity_and_unit_cost(
        start_arr, end_arr, capacities_arr, weights_arr
    )

    print(all_arcs)

    # Add supply for each nodes.
    smcf.set_nodes_supplies(np.arange(0, len(supplies)), supplies)

    medium_time = time.time()
    # 4. Solve
    # Find the min cost flow.
    status = smcf.solve()
    end_time = time.time()
    if status != smcf.OPTIMAL:
        print("There was an issue with the min cost flow input.")
        print(f"Status: {status}")
        exit(1)
    print("Optimal solution found!")
    print(
        f"The calculation took {end_time - medium_time} seconds, while preparation took {medium_time - start_time} seconds"
    )
    solution_flows = smcf.flows(all_arcs)
    
    return list(filter(lambda pair: solution_flows[pair_to_index[pair]] == 1, driver_action_pairs))

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall

# Use the Floyd-Warshall algorithm to solve the all-pair shortest path problem
# Input undirected edges as tuples [station1, weight, station2]
def solve_all_pair_shortest_path_problem(connections: list[tuple[Station, float, Station]]) -> dict[int, dict[int, tuple[list[Station], float]]]:
    # Build a dict containing all stations mapped by their id
    station_id_dict = {}
    for connection in connections:
        station1 = connection[0]
        station2 = connection[2]

        if station1.id not in station_id_dict:
            station_id_dict[station1.id] = station1
        if station2.id not in station_id_dict:
            station_id_dict[station2.id] = station2
    
    # Create Mapper from station id -> index and other direction in graph matrix 
    sorted_station_ids = sorted(station_id_dict.keys())
    index_to_id_dict = {i: sorted_station_ids[i] for i in range(len(sorted_station_ids))}
    id_to_index_dict = {sorted_station_ids[i]: i for i in range(len(sorted_station_ids))}

    graph = [[0 for i in range(len(sorted_station_ids))] for j in range(len(sorted_station_ids))]

    # Fill connections in graph
    for connection in connections:
        graph[id_to_index_dict[connection[0].id]][id_to_index_dict[connection[2].id]] = connection[1]
    
    # Convert graph to network
    graph = csr_matrix(graph)

    # Solve the problem
    dist_matrix, predecessors = floyd_warshall(csgraph=graph, directed=False, return_predecessors=True)

    result_dict = {id: {id2: None for id2 in sorted_station_ids} for id in sorted_station_ids}
    print(type(dist_matrix))
    for idx1 in range(len(dist_matrix)):
        for idx2 in range(len(dist_matrix)):
            idx = predecessors[idx1][idx2]
            station1 = station_id_dict[index_to_id_dict[idx1]]
            station2 = station_id_dict[index_to_id_dict[idx2]]

            if idx == -9999:
                del result_dict[station1.id][station2.id]
                continue
            
            stations = [station1]
            while idx != idx1:
                stations.append(station_id_dict[index_to_id_dict[idx]])
                idx = predecessors[idx1][idx]
            stations.append(station2)
            

            result_dict[station1.id][station2.id] = (stations, dist_matrix[idx1][idx2])
    
    return result_dict
