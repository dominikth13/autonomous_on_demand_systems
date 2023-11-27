from collections import deque, namedtuple
import csv
import math

Edge = namedtuple('Edge', 'start, end, cost')

def create_edge(start, end, cost):
    return Edge(start, end, cost)

class Graph:
    def __init__(self, edges):
        self.edges = [create_edge(*e) for e in edges]

    def vertices(self): 
        return set(e.start for e in self.edges).union(e.end for e in self.edges)
    
    def get_neighbours(self, v):
        neighbours = []
        for e in self.edges:
            if e.start == v:
                neighbours.append((e.end, e.cost))
        return neighbours

    def dijkstra(self, source, destination): 
        distances = {v: float('inf') for v in self.vertices()} # jeder Knoten bekommt eine Distanz von unendlich
        prev_v = {v: None for v in self.vertices()} # jeder Knoten bekommt einen Vorgänger von None

        distances[source] = 0 # die Distanz des Startknotens wird auf 0 gesetzt
        vertices = list(self.vertices())[:]
        while len(vertices) > 0:
            v = min(vertices, key=lambda u: distances[u]) # der Knoten mit der kleinsten Distanz wird ausgewählt
            vertices.remove(v) # der Knoten wird aus der Liste entfernt 
            if distances[v] == float('inf'):
                break # wenn die Distanz des Knotens unendlich ist, dann gibt es keinen Pfad zum Zielknoten
            for neighbour, cost in self.get_neighbours(v):
                path_cost = distances[v] + cost # die Distanz des Nachbarn wird berechnet
                if path_cost < distances[neighbour]:
                    distances[neighbour] = path_cost
                    prev_v[neighbour] = v
        path = []
        curr_v = destination
        while curr_v and prev_v[curr_v] is not None:
            path.insert(0, curr_v)
            curr_v = prev_v[curr_v] # der Vorgänger des Knotens wird zum aktuellen Knoten
        if curr_v:
            path.insert(0, curr_v)
        return path, distances[destination]


###############################################################################

def lat_lon_to_meters(lat1, lon1, lat2, lon2):
    # Umrechnungsfaktoren
    meters_per_degree_lat = 111000  # ungefähr 111 Kilometer pro Grad
    meters_per_degree_lon = meters_per_degree_lat * math.cos(math.radians((lat1 + lat2) / 2))

    # Umrechnung in Meter
    delta_lat_meters = (lat1 - lat2) * meters_per_degree_lat
    delta_lon_meters = (lon1 - lon2) * meters_per_degree_lon

    return delta_lat_meters, delta_lon_meters

def manhattan_distance(lat1, lon1, lat2, lon2):
    delta_lat_meters, delta_lon_meters = lat_lon_to_meters(lat1, lon1, lat2, lon2)
    return abs(delta_lat_meters) + abs(delta_lon_meters)

###############################################################################

def read_csv_data(filename):
    data = []
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append((int(row['station_id']), row['stop_name'], row['line'], float(row['gtfs_latitude']), float(row['gtfs_longitude'])))
    return data

###############################################################################

def create_edges(data):
    edges = []
    station_dauer = 80  # Setzen Sie hier den Wert für Station_Dauer
    umsteige_selbe_station = 300  # Setzen Sie hier den Wert für Umsteige_selbe_Station
    max_lauf_dauer = 600
    lauf_geschwindigkeit = 1.3 #in m/s (ca. 4,7km/h)


    
    # Zählvariablen für die verschiedenen Arten von Kanten
    kanten_in_linie_count = 0
    umstiege_kanten_count = 0
    kanten_rest_count = 0

    # Hilfsfunktionen und -strukturen
    id_to_name = {station_id: stop_name for station_id, stop_name, _ , _, _ in data}
    name_to_ids = {}
    for station_id, stop_name, _, _, _ in data:
        if stop_name not in name_to_ids:
            name_to_ids[stop_name] = []
        name_to_ids[stop_name].append(station_id)

    # Kanten für dieselbe Linie
    for i in range(len(data) - 1):
        id1, name1, line1, lat1, lon1 = data[i]
        id2, name2, line2, lat2, lon2 = data[i + 1]
        if line1 == line2:
            edges.append((id1, id2, station_dauer))
            edges.append((id2, id1, station_dauer))
            kanten_in_linie_count += 1

    # Kanten für Umstiege an derselben Station
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            id1, name1, line1, lat1, lon1 = data[i]
            id2, name2, line2, lat2, lon2 = data[j]
            if name1 == name2 and line1 != line2:
                edges.append((id1, id2, umsteige_selbe_station))
                edges.append((id2, id1, umsteige_selbe_station))
                umstiege_kanten_count += 1
    
    for i in range(len(data)):
            for j in range(i + 1, len(data)):
                id1, name1, line1, lat1, lon1 = data[i]
                id2, name2, line2, lat2, lon2 = data[j]
                distance = manhattan_distance(lat1, lon1, lat2, lon2)
                kosten = distance * lauf_geschwindigkeit  # Berechnung der Kosten

                if kosten < max_lauf_dauer and not (line1 == line2 or name1 == name2):
                    edges.append((id1, id2, kosten))
                    edges.append((id2, id1, kosten))
                    kanten_rest_count += 1

    print(f"Anzahl der Kanten innerhalb derselben Linie: {kanten_in_linie_count}")
    print(f"Anzahl der Kanten für Umstiege an derselben Station: {umstiege_kanten_count}")
    print(f"Anzahl der restlichen Kanten: {kanten_rest_count}")
    return edges

#berechnung kürzeste Wege
def calculate_shortest_paths(graph, nodes):
    paths = []
    for start_node in nodes:
        for end_node in nodes:
            if start_node != end_node:
                shortest_path = graph.dijkstra(start_node, end_node)
                paths.append((start_node, end_node, shortest_path))
    return paths

def save_paths_to_csv(paths, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Startstation', 'Endstation', 'Weg'])
        for start, end, path in paths:
            writer.writerow([start, end, ' -> '.join(map(str, path))])



if __name__ == "__main__": 
    station_data = read_csv_data('training_data\stationen_NY_full.csv')
    station_data.sort(key=lambda x: x[0])
    edges = create_edges(station_data)
    graph = Graph(edges)

    nodes = list(graph.vertices())  # Liste aller Knoten (Stationen)
    shortest_paths = calculate_shortest_paths(graph, nodes)
    print("Speichern beginnt")
    save_paths_to_csv(shortest_paths, 'training_data\shortest_paths.csv')
    #print(graph.dijkstra(1, 50)) # Start und Zielknoten müssen angepasst werden
    
            