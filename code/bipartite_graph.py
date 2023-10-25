from driver import Driver

# Adjacency matrix is a Driver x Routes dict, which contains the q values or edge weights as values
class BipartiteGraph:
    def __init__(self, adjacency_matrix: dict[int, dict[int, float]]) -> None:
        self.adjacency_matrix = adjacency_matrix