import networkx as nx
import matplotlib.pyplot as plt
import time
import random

random.seed(42)
start_time = time.time()
G = nx.DiGraph()
# Knoten A-Z zu G hinzufügen
nodes = [chr(65 + i) for i in range(62)]
G.add_nodes_from(nodes)

        # Wenn keine zufälligkeit gewüscht ist 
        # G = nx.DiGraph()
        # G.add_node("a", demand=-5)
        # G.add_node("d", demand=5)
        # G.add_edge("a", "b", weight=3, capacity=4)
        # G.add_edge("a", "c", weight=6, capacity=10)
        # G.add_edge("b", "d", weight=1, capacity=9)
        # G.add_edge("c", "d", weight=2, capacity=5)
        # flowDict = nx.min_cost_flow(G)

# 100 zufällige Kanten zu G hinzufügen
edges_added = set()  # Zum Nachverfolgen der bereits hinzugefügten Kanten
while len(edges_added) < 10000:
    start = random.choice(nodes)
    end = random.choice(nodes)
    while start == end or (start, end) in edges_added:
        end = random.choice(nodes)
    weight = random.randint(1, 15)
    capacity = random.randint(5, 10)
    G.add_edge(start, end, weight=weight, capacity=capacity)
    edges_added.add((start, end))

# Find the shortest path from node A to node E
path = nx.min_cost_flow(G)
print(path)

# Ermitteln des Min-Cost-Flow
flow = nx.min_cost_flow(G)

#time!
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Die Ausführung dauerte {elapsed_time} Sekunden.")


# Erstellen einer Liste aller Kanten und Festlegen der Farben basierend auf ihrem Flusswert
edge_colors = [
    "red" if flow[u][v] > 0 else "black" for u, v in G.edges()
]

# Visualisierung des Graphen
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos)
nx.draw_networkx_edges(G, pos, edge_color=edge_colors)
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edge_labels(
G, pos, edge_labels={(u, v): d["weight"] for u, v, d in G.edges(data=True)}
)

plt.show()

