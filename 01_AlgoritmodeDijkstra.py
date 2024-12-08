import heapq  # Para usar colas de prioridad
import networkx as nx  # Para la creación y visualización de grafos
import matplotlib.pyplot as plt  # Para graficar el grafo

def dijkstra(graph, start):
    """
    Algoritmo de Dijkstra para encontrar las rutas más cortas desde un nodo inicial.
    Muestra el proceso paso a paso en la consola.
    """
    # Inicializar las distancias como infinito para todos los nodos, excepto el inicial
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    # Inicializar la cola de prioridad con el nodo inicial
    priority_queue = [(0, start)]  # (distancia, nodo)
    
    # Mantener un registro de los nodos visitados
    visited = set()
    
    # Mantener un diccionario para rastrear los caminos más cortos
    shortest_paths = {node: [] for node in graph}
    shortest_paths[start] = [start]
    
    print("Inicio del algoritmo de Dijkstra:\n")
    
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        
        # Si ya visitamos este nodo, continuamos
        if current_node in visited:
            continue
        
        print(f"Visitando nodo: {current_node}, Distancia acumulada: {current_distance}")
        visited.add(current_node)
        
        # Relajar los vecinos del nodo actual
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            
            if distance < distances[neighbor]:  # Si encontramos una distancia más corta
                distances[neighbor] = distance
                shortest_paths[neighbor] = shortest_paths[current_node] + [neighbor]
                heapq.heappush(priority_queue, (distance, neighbor))
                print(f" - Actualizando distancia de {neighbor} a {distance}. Camino: {shortest_paths[neighbor]}")
    
    print("\nResultado final:")
    for node, distance in distances.items():
        print(f"Distancia más corta a {node}: {distance}, Camino: {shortest_paths[node]}")
    
    return distances, shortest_paths


def plot_graph(graph, shortest_paths=None, start=None):
    """
    Genera y muestra una gráfica del grafo.
    Resalta los caminos más cortos si se proporcionan.
    """
    G = nx.DiGraph()  # Crear un grafo dirigido
    
    # Añadir nodos y bordes al grafo
    for node, neighbors in graph.items():
        for neighbor, weight in neighbors.items():
            G.add_edge(node, neighbor, weight=weight)
    
    pos = nx.spring_layout(G)  # Posiciones de los nodos
    labels = nx.get_edge_attributes(G, 'weight')  # Pesos de los bordes
    
    # Dibujar el grafo
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="lightblue", font_size=10, font_weight="bold")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    
    # Resaltar los caminos más cortos
    if shortest_paths and start:
        for target, path in shortest_paths.items():
            if len(path) > 1:  # Solo dibujar si hay un camino válido
                path_edges = list(zip(path[:-1], path[1:]))
                nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="red", width=2)
    
    plt.title("Grafo y Caminos Más Cortos")
    plt.show()


# Ejemplo de uso:
if __name__ == "__main__":
    # Grafo de ejemplo representado como un diccionario
    graph = {
        'A': {'B': 1, 'C': 4},
        'B': {'C': 2, 'D': 6},
        'C': {'D': 3},
        'D': {}
    }
    
    start_node = 'A'  # Nodo inicial
    distances, shortest_paths = dijkstra(graph, start_node)
    plot_graph(graph, shortest_paths, start=start_node)
