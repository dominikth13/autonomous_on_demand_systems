from action import Action, DriverActionPair
from driver import Driver
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable

# Adjacency matrix is a Driver x Routes dict, which contains the q values or edge weights as values
class BipartiteGraph:
    def __init__(self, adjacency_matrix: dict[int, dict[int, float]]) -> None:
        self.adjacency_matrix = adjacency_matrix

def build_bipartite_matching_problem(driver_action_pairs: list[DriverActionPair]) -> tuple[LpProblem, dict[LpVariable, tuple[Driver, Action]]]:
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

# This method solves the optimization problem of DriverActionPairs as a bipartite matching problem
# This counts as a combinatorial problem which is NP hard
def solve_as_bipartite_matching_problem(driver_action_pairs: list[DriverActionPair]) -> list[DriverActionPair]:
    (model, var_pair_dict) = build_bipartite_matching_problem(driver_action_pairs)
    
    # Solve the model
    status = model.solve()

    # Print results
    print(f"Model name: {model.name}, status: {LpStatus[model.status]}")

    result_pairs = []
    for x in var_pair_dict:
        if x.value() == 1:
            result_pairs.append(var_pair_dict[x])
    
    return result_pairs
