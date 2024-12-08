import networkx as nx  # Librería para trabajar con grafos y visualizarlos
import matplotlib.pyplot as plt  # Librería para crear gráficos
from operator import itemgetter  # Para ordenar listas por un índice específico

def kruskal(graph, minimize=True):
    """
    Implementación del algoritmo de Kruskal para generar un Árbol de Expansión.
    Parámetros:
        - graph: Lista de aristas con pesos [(nodo1, nodo2, peso)].
        - minimize: Booleano. Si es True, genera el Árbol de Expansión Mínimo (MST). Si es False, genera el Árbol de Expansión Máximo.
    Retorna:
        - mst: Lista de aristas que forman el Árbol de Expansión.
        - total_weight: Peso total del Árbol de Expansión.
    """
    # Ordenar las aristas por peso:
    # Si minimize=True, orden ascendente; si minimize=False, orden descendente.
    graph = sorted(graph, key=itemgetter(2), reverse=not minimize)

    # Inicializamos las estructuras para el sistema de conjuntos disjuntos (Union-Find)
    parent = {}  # Diccionario para representar el padre de cada nodo
    rank = {}  # Diccionario para la profundidad del árbol de cada nodo

    # Función para encontrar el conjunto al que pertenece un nodo (con compresión de caminos)
    def find(node):
        if parent[node] != node:  # Si el nodo no es su propio padre (no es raíz)
            parent[node] = find(parent[node])  # Se comprime el camino
        return parent[node]

    # Función para unir dos conjuntos
    def union(node1, node2):
        root1 = find(node1)  # Encontrar la raíz del primer nodo
        root2 = find(node2)  # Encontrar la raíz del segundo nodo

        # Unir los conjuntos según el rango (para mantener el árbol más plano)
        if root1 != root2:  # Solo unimos si las raíces son diferentes
            if rank[root1] > rank[root2]:  # Si el rango de root1 es mayor
                parent[root2] = root1  # root1 se convierte en el padre de root2
            elif rank[root1] < rank[root2]:  # Si el rango de root2 es mayor
                parent[root1] = root2  # root2 se convierte en el padre de root1
            else:  # Si los rangos son iguales
                parent[root2] = root1  # root1 se convierte en el padre de root2
                rank[root1] += 1  # Incrementar el rango de root1

    # Inicializar cada nodo como un conjunto independiente
    nodes = set([edge[0] for edge in graph] + [edge[1] for edge in graph])  # Todos los nodos del grafo
    for node in nodes:
        parent[node] = node  # Cada nodo es su propio padre al inicio
        rank[node] = 0  # Todos los nodos tienen rango 0 al inicio

    # Lista para almacenar las aristas seleccionadas en el Árbol de Expansión
    mst = []
    total_weight = 0  # Peso total del Árbol de Expansión

    # Iterar sobre las aristas ordenadas
    for u, v, weight in graph:
        # Si los nodos no están en el mismo conjunto (evitar ciclos)
        if find(u) != find(v):
            union(u, v)  # Unir los conjuntos
            mst.append((u, v, weight))  # Añadir la arista al Árbol de Expansión
            total_weight += weight  # Sumar el peso de la arista
            # Mostrar el paso actual en consola
            print(f"Seleccionada arista ({u}, {v}) con peso {weight}")

    return mst, total_weight  # Retornar el Árbol de Expansión y su peso total

def draw_graph(edges, title):
    """
    Dibuja un grafo basado en las aristas seleccionadas.
    Parámetros:
        - edges: Lista de aristas [(nodo1, nodo2, peso)].
        - title: Título para la gráfica.
    """
    # Crear un grafo vacío
    G = nx.Graph()

    # Añadir las aristas al grafo
    for u, v, weight in edges:
        G.add_edge(u, v, weight=weight)

    # Generar posiciones para los nodos
    pos = nx.spring_layout(G)  # Layout para posicionar nodos de forma estética

    # Obtener etiquetas de los pesos de las aristas
    edge_labels = nx.get_edge_attributes(G, "weight")

    # Dibujar nodos y aristas
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=700, font_weight="bold")
    # Dibujar etiquetas de las aristas
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")

    # Añadir el título y mostrar la gráfica
    plt.title(title)
    plt.show()

# Grafo de ejemplo: Lista de aristas (nodo1, nodo2, peso)
graph = [
    ("A", "B", 4),
    ("A", "C", 3),
    ("B", "C", 1),
    ("B", "D", 2),
    ("C", "D", 4),
    ("C", "E", 2),
    ("D", "E", 3)
]

# Generar Árbol de Expansión Mínimo (MST)
print("Árbol de Expansión Mínimo (MST):")
mst, min_weight = kruskal(graph, minimize=True)
print(f"Peso total del MST: {min_weight}\n")
draw_graph(mst, "Árbol de Expansión Mínimo (MST)")

# Generar Árbol de Expansión Máximo (MAXT)
print("\nÁrbol de Expansión Máximo (MAXT):")
maxt, max_weight = kruskal(graph, minimize=False)
print(f"Peso total del MAXT: {max_weight}\n")
draw_graph(maxt, "Árbol de Expansión Máximo (MAXT)")
"""
Ordenar las aristas:

Si se busca el Árbol de Expansión Mínimo, las aristas se ordenan por peso en orden ascendente.
Si se busca el Árbol de Expansión Máximo, se ordenan en orden descendente.
Estructuras Union-Find:

El diccionario parent guarda la raíz del conjunto al que pertenece cada nodo.
El diccionario rank ayuda a mantener el árbol de cada conjunto lo más plano posible.
Evitando ciclos:

Al procesar cada arista, el algoritmo verifica si los dos nodos ya están en el mismo conjunto.
Si no están, se selecciona la arista y se unen los conjuntos.
Visualización:

Se utiliza networkx para construir el grafo y matplotlib para graficarlo.
Las aristas seleccionadas se dibujan con etiquetas que muestran sus pesos.
Resultados:

Se imprime en consola el proceso paso a paso.
Se generan dos gráficas: una para el MST y otra para el MAXT.
"""