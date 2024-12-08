import heapq
import networkx as nx
import matplotlib.pyplot as plt

def prim_algorithm(graph, start_node):
    """
    Implementación del algoritmo de Prim para encontrar el Árbol de Expansión Mínimo.
    Muestra paso a paso en la consola cómo se seleccionan los nodos y las aristas.
    
    :param graph: Grafo representado como un diccionario {nodo: [(vecino, peso)]}
    :param start_node: Nodo inicial
    :return: Lista de aristas del Árbol de Expansión Mínimo
    """
    visited = set()  # Conjunto de nodos visitados
    mst_edges = []  # Aristas del Árbol de Expansión Mínimo
    priority_queue = []  # Cola de prioridad para seleccionar la menor arista

    # Inicia con el nodo inicial y sus aristas
    visited.add(start_node)
    for neighbor, weight in graph[start_node]:
        heapq.heappush(priority_queue, (weight, start_node, neighbor))

    print(f"Nodo inicial: {start_node}")
    
    while priority_queue:
        # Selecciona la arista con el menor peso
        weight, from_node, to_node = heapq.heappop(priority_queue)
        
        # Si el nodo destino ya fue visitado, omitir
        if to_node in visited:
            continue
        
        # Añadir la arista al Árbol de Expansión Mínimo
        mst_edges.append((from_node, to_node, weight))
        visited.add(to_node)

        print(f"Seleccionada arista: ({from_node}, {to_node}) con peso {weight}")
        
        # Añadir las nuevas aristas del nodo visitado a la cola de prioridad
        for neighbor, weight in graph[to_node]:
            if neighbor not in visited:
                heapq.heappush(priority_queue, (weight, to_node, neighbor))

    return mst_edges

def draw_graph(graph, mst_edges):
    """
    Dibuja el grafo original y resalta el Árbol de Expansión Mínimo.
    
    :param graph: Grafo representado como un diccionario {nodo: [(vecino, peso)]}
    :param mst_edges: Lista de aristas del Árbol de Expansión Mínimo
    """
    G = nx.Graph()
    
    # Añadir todas las aristas al grafo
    for node, neighbors in graph.items():
        for neighbor, weight in neighbors:
            G.add_edge(node, neighbor, weight=weight)

    # Añadir las aristas del MST como un subgrafo
    mst_graph = nx.Graph()
    for from_node, to_node, weight in mst_edges:
        mst_graph.add_edge(from_node, to_node, weight=weight)

    pos = nx.spring_layout(G)  # Layout del grafo

    # Dibujar el grafo completo
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=700, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d['weight'] for u, v, d in G.edges(data=True)})

    # Resaltar el Árbol de Expansión Mínimo
    nx.draw(mst_graph, pos, with_labels=True, node_color='lightgreen', edge_color='red', node_size=700, font_size=10)
    
    plt.title("Árbol de Expansión Mínimo (Algoritmo de Prim)")
    plt.show()

# Grafo de ejemplo representado como un diccionario
graph = {
    'A': [('B', 4), ('C', 3)],
    'B': [('A', 4), ('C', 2), ('D', 5)],
    'C': [('A', 3), ('B', 2), ('D', 7)],
    'D': [('B', 5), ('C', 7)]
}

# Ejecutar el algoritmo de Prim
start_node = 'A'
mst_edges = prim_algorithm(graph, start_node)

# Mostrar los resultados
print("\nAristas del Árbol de Expansión Mínimo:")
for edge in mst_edges:
    print(edge)

# Dibujar el grafo y el MST
draw_graph(graph, mst_edges)

"""
prim_algorithm(graph, start_node):

Utiliza una cola de prioridad para seleccionar siempre la arista con el menor peso conectada a un nodo visitado.
Muestra paso a paso en la consola las aristas seleccionadas.
draw_graph(graph, mst_edges):

Dibuja el grafo original completo.
Resalta el Árbol de Expansión Mínimo con nodos en color verde y aristas en rojo.
Entrada del Grafo:

El grafo está representado como un diccionario, donde cada nodo tiene una lista de tuplas que representan sus vecinos y el peso de las aristas.

"""