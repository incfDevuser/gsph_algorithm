import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.spatial import cKDTree
import time
import os
import argparse
import glob
import csv
import math
from typing import List, Tuple, Dict, Optional
from collections import namedtuple
import datetime

# =============================================================================
# CONFIGURACIÓN GLOBAL
# =============================================================================
EPS_FRONTIER = 5
MAX_ITER_LOCAL = 800
RESULTS_DIR = "gsph_fc_results_professional"

# Tipos de datos para mayor claridad
Point = Tuple[float, float]
Route = List[Point]
Quadrants = Dict[str, Route]
Connections = List[Tuple[Point, Point]]

# Estructura para nodos del quadtree
Node = namedtuple("Node", ["points", "bbox", "children", "level"])

def load_bks_values() -> Dict[str, int]:
    """
    Carga los valores BKS (Best Known Solutions) desde el archivo.
    
    Returns:
        Diccionario con instancia -> valor BKS
    """
    bks_dict = {}
    bks_file = os.path.join("src", "BKS", "BKS_TSP.txt")
    
    if not os.path.exists(bks_file):
        print(f"⚠️  Archivo BKS no encontrado: {bks_file}")
        return {}
    
    try:
        with open(bks_file, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line and not line.startswith('#'):
                    # Formato: "instancia : valor" o "instancia : valor (tipo)"
                    parts = line.split(':')
                    if len(parts) >= 2:
                        instance = parts[0].strip()
                        value_part = parts[1].strip()
                        # Remover comentarios en paréntesis
                        if '(' in value_part:
                            value_part = value_part.split('(')[0].strip()
                        try:
                            bks_value = int(value_part)
                            bks_dict[instance] = bks_value
                        except ValueError:
                            continue
        print(f"✓ Cargados {len(bks_dict)} valores BKS")
        return bks_dict
    except Exception as e:
        print(f"⚠️  Error cargando BKS: {e}")
        return {}
# =============================================================================
# MÓDULO 1: LECTURA Y PROCESAMIENTO DE ARCHIVOS TSP
# =============================================================================

def read_tsplib(filename: str) -> List[Point]:
    """
    Lee un archivo TSPLIB y extrae las coordenadas de los nodos.
    Solo soporta formato EUC_2D.
    
    Args:
        filename: Ruta al archivo TSP
        
    Returns:
        Lista de puntos (x, y)
        
    Raises:
        ValueError: Si el archivo no es compatible o no tiene coordenadas
    """
    nodes = []
    with open(filename, 'r') as f:
        content = f.read()
        if "EUC_2D" not in content:
            raise ValueError(f"El archivo {filename} no usa formato EUC_2D. Solo se soporta EDGE_WEIGHT_TYPE: EUC_2D")
        f.seek(0)
        reading_nodes = False
        for line in f:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                reading_nodes = True
                continue
            if line == "EOF":
                break
            if reading_nodes and line:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        x, y = float(parts[1]), float(parts[2])
                        nodes.append((x, y))
                    except (ValueError, IndexError):
                        continue
    
    if not nodes:
        raise ValueError(f"No se encontraron coordenadas válidas en {filename}")
    
    return nodes
def find_tsp_file(tsp_filename: str) -> str:
    """
    Busca el archivo TSP en diferentes ubicaciones.
    
    Args:
        tsp_filename: Nombre del archivo TSP
        
    Returns:
        Ruta completa al archivo encontrado
        
    Raises:
        SystemExit: Si no se encuentra el archivo
    """
    if os.path.exists(tsp_filename):
        return tsp_filename
    src_path = os.path.join("src", "INSTANCES", tsp_filename)
    if os.path.exists(src_path):
        return src_path
    
    print(f"ERROR: No se pudo encontrar el archivo TSP: {tsp_filename}")
    exit(1)
def find_all_tsp_files() -> List[str]:
    """
    Encuentra todos los archivos TSP compatibles (EUC_2D) en el directorio de instancias.
    
    Returns:
        Lista de rutas a todos los archivos TSP EUC_2D encontrados
    """
    tsp_files = []
    instances_dir = os.path.join("src", "INSTANCES")
    if os.path.exists(instances_dir):
        pattern = os.path.join(instances_dir, "*.tsp")
        all_files = glob.glob(pattern)
        for file_path in all_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    if "EUC_2D" in content and "NODE_COORD_SECTION" in content:
                        tsp_files.append(file_path)
            except Exception:
                continue 
    
    current_pattern = "*.tsp"
    all_current_files = glob.glob(current_pattern)
    for file_path in all_current_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                if "EUC_2D" in content and "NODE_COORD_SECTION" in content:
                    tsp_files.append(file_path)
        except Exception:
            continue
    
    tsp_files = sorted(list(set(tsp_files)))
    
    return tsp_files


def get_instance_name(file_path: str) -> str:
    """
    Extrae el nombre de la instancia del path del archivo.
    
    Args:
        file_path: Ruta completa al archivo
        
    Returns:
        Nombre de la instancia sin extensión
    """
    return os.path.splitext(os.path.basename(file_path))[0]
# =============================================================================
# MÓDULO 2: CÁLCULOS DE DISTANCIA Y LONGITUD DE RUTAS
# =============================================================================

def euclidean_distance(p1: Point, p2: Point) -> int:
    """
    Calcula la distancia euclidiana entre dos puntos (redondeada a entero).
    
    Args:
        p1, p2: Puntos (x, y)
        
    Returns:
        Distancia redondeada a entero
    """
    dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return int(dist + 0.5)


def calculate_path_length(path: Route) -> int:
    """
    Calcula la longitud total de un camino.
    
    Args:
        path: Lista de puntos que forman el camino
        
    Returns:
        Longitud total del camino
    """
    if len(path) < 2:
        return 0
    return sum(euclidean_distance(path[i], path[i + 1]) for i in range(len(path) - 1))


def calculate_tour_length(tour: Route) -> int:
    """
    Calcula la longitud total de un tour (incluyendo regreso al inicio).
    
    Args:
        tour: Lista de puntos que forman el tour
        
    Returns:
        Longitud total del tour
    """
    if len(tour) < 2:
        return 0
    
    path_length = calculate_path_length(tour)
    # Agregar distancia de regreso al inicio
    return_distance = euclidean_distance(tour[-1], tour[0])
    return path_length + return_distance
# =============================================================================
# MÓDULO 3: OPTIMIZACIÓN LOCAL - ALGORITMO 2-OPT
# =============================================================================

def two_opt_swap(route: Route, i: int, k: int) -> Route:
    """
    Realiza un intercambio 2-opt en una ruta.
    
    Args:
        route: Ruta original
        i, k: Índices para el intercambio
        
    Returns:
        Nueva ruta con el segmento i:k+1 invertido
    """
    return route[:i] + route[i:k + 1][::-1] + route[k + 1:]


def optimize_route_2opt_delta(points: Route, max_iter: int = None) -> Route:
    """
    2-opt optimizado con evaluación delta O(1). Versión adaptativa.
    
    Args:
        points: Lista de puntos (x,y)
        max_iter: Iteraciones máximas (auto si None)
        
    Returns:
        Ruta optimizada
    """
    n = len(points)
    if n < 3:
        return points[:]
    
    # Iteraciones adaptativas basadas en el tamaño
    if max_iter is None:
        max_iter = min(50000, max(5000, n * 100))
    
    # Trabajamos sobre un orden de índices para no copiar puntos
    order = list(range(n))

    def d(i, j) -> float:
        """Distancia euclidiana rápida sin raíz cuadrada para comparaciones"""
        pi, pj = points[i], points[j]
        return (pi[0]-pj[0])**2 + (pi[1]-pj[1])**2

    improved = True
    it = 0
    best_improvement = 0
    
    while improved and it < max_iter:
        improved = False
        it += 1
        current_improvement = 0
        
        for i in range(n - 1):
            a, b = order[i], order[(i + 1) % n]
            dab = d(a, b)
            
            # k empieza en i+2 y evita cerrar con i
            k_max = n if i > 0 else n - 1
            for k in range(i + 2, k_max):
                c, dd = order[k], order[(k + 1) % n]
                dcd = d(c, dd)
                
                # Delta del swap 2-opt: (ab + cd) - (ac + bd)
                old_cost = dab + dcd
                new_cost = d(a, c) + d(b, dd)
                
                if old_cost > new_cost:
                    # Invertir segmento (i+1..k)
                    order[i + 1 : k + 1] = reversed(order[i + 1 : k + 1])
                    improved = True
                    current_improvement += old_cost - new_cost
                    break  # Reiniciar después de una mejora
        
        # Criterio de parada inteligente
        if current_improvement > best_improvement:
            best_improvement = current_improvement
        elif current_improvement < best_improvement * 0.001:  # Mejora muy pequeña
            break

    # Reconstruir la ruta
    return [points[idx] for idx in order]


def optimize_route_2opt(points: Route, max_iter: int = 800) -> Route:
    """Wrapper para compatibilidad hacia atrás"""
    return optimize_route_2opt_delta(points, max_iter)


def or_opt_optimization(route: Route, max_passes: int = 2) -> Route:
    """
    Optimización Or-opt: mueve cadenas de 1, 2 o 3 nodos a mejor posición.
    
    Args:
        route: Ruta a optimizar
        max_passes: Número máximo de pasadas
        
    Returns:
        Ruta optimizada
    """
    if len(route) < 6:
        return route[:]
    
    r = route[:]
    n = len(r)
    
    for _ in range(max_passes):
        improved = False
        
        for chain_len in (1, 2, 3):
            if chain_len >= n:
                continue
                
            for i in range(n - chain_len):
                chain = r[i:i+chain_len]
                remain = r[:i] + r[i+chain_len:]
                
                current_cost = calculate_path_length(r)
                best_gain = 0
                best_route = None
                
                # Probar todas las posiciones para insertar la cadena
                step = max(1, len(remain) // 15)  # Muestreo para instancias grandes
                for j in range(0, len(remain)+1, step):
                    candidate = remain[:j] + chain + remain[j:]
                    candidate_cost = calculate_path_length(candidate)
                    gain = current_cost - candidate_cost
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_route = candidate
                
                if best_gain > 0:
                    r = best_route
                    n = len(r)
                    improved = True
        
        if not improved:
            break
    
    return r


def three_opt_light(route: Route, max_iter: int = 1000) -> Route:
    """
    3-opt ligero para post-optimización.
    
    Args:
        route: Ruta a optimizar
        max_iter: Iteraciones máximas
        
    Returns:
        Ruta optimizada
    """
    if len(route) < 6:
        return route[:]
    
    best_route = route[:]
    best_length = calculate_path_length(best_route)
    n = len(route)
    
    for iteration in range(max_iter):
        improved = False
        
        # Muestreo aleatorio para 3-opt (más eficiente)
        import random
        indices = list(range(n))
        random.shuffle(indices)
        
        for idx in range(min(n//3, 100)):  # Limitar el número de intentos
            i = indices[idx]
            j = (i + random.randint(2, min(n//3, 20))) % n
            k = (j + random.randint(2, min(n//3, 20))) % n
            
            if i == j or j == k or i == k:
                continue
            
            # Asegurar orden correcto
            if i > j:
                i, j = j, i
            if j > k:
                j, k = k, j
            if i > j:
                i, j = j, i
            
            # Probar una reconexión 3-opt simple
            new_route = route[:i] + route[j:k] + route[i:j] + route[k:]
            new_length = calculate_path_length(new_route)
            
            if new_length < best_length:
                best_route = new_route
                best_length = new_length
                improved = True
                break
        
        if not improved:
            break
    
    return best_route
# =============================================================================
# MÓDULO 4: SUBDIVISIÓN JERÁRQUICA ADAPTATIVA
# =============================================================================

def calculate_midpoints(points: List[Point]) -> Tuple[float, float]:
    """
    Calcula los puntos medios para dividir el espacio en cuadrantes.
    
    Args:
        points: Lista de puntos
        
    Returns:
        Tupla (x_medio, y_medio)
    """
    xs, ys = zip(*points)
    x_mid = (min(xs) + max(xs)) / 2
    y_mid = (min(ys) + max(ys)) / 2
    return x_mid, y_mid


def calculate_bbox(points: List[Point]) -> Tuple[float, float, float, float]:
    """Calcula bounding box: (min_x, min_y, max_x, max_y)"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def split_points_adaptive(points: List[Point]) -> Tuple[List[Point], List[Point], List[Point], List[Point], float, float]:
    """
    Divide puntos en 4 cuadrantes usando punto medio adaptativo.
    
    Args:
        points: Lista de puntos a dividir
        
    Returns:
        Tupla (Q1, Q2, Q3, Q4, x_mid, y_mid)
    """
    x_mid, y_mid = calculate_midpoints(points)
    Q1, Q2, Q3, Q4 = [], [], [], []
    
    for p in points:
        quad_name = classify_point_quadrant(p, x_mid, y_mid)
        if quad_name == "Q1":
            Q1.append(p)
        elif quad_name == "Q2":
            Q2.append(p)
        elif quad_name == "Q3":
            Q3.append(p)
        else:  # Q4
            Q4.append(p)
    
    return Q1, Q2, Q3, Q4, x_mid, y_mid


def build_hierarchical_quadtree(points: List[Point], max_leaf_size: int = None, level: int = 0) -> Node:
    """
    Construye un quadtree jerárquico adaptativo.
    
    Args:
        points: Puntos a subdividir
        max_leaf_size: Tamaño máximo de hoja (auto si None)
        level: Nivel actual de recursión
        
    Returns:
        Nodo del quadtree
    """
    n = len(points)
    
    # Tamaño adaptativo de hoja basado en √n
    if max_leaf_size is None:
        max_leaf_size = max(25, int(1.5 * math.sqrt(n)))
    
    bbox = calculate_bbox(points)
    
    # Condiciones de parada
    if n <= max_leaf_size or level > 8:  # Límite de profundidad
        return Node(points=points, bbox=bbox, children=None, level=level)
    
    # Dividir en cuadrantes
    Q1, Q2, Q3, Q4, x_mid, y_mid = split_points_adaptive(points)
    
    # Verificar que la división sea productiva
    non_empty = [q for q in [Q1, Q2, Q3, Q4] if q]
    if len(non_empty) <= 1:
        return Node(points=points, bbox=bbox, children=None, level=level)
    
    # Recursión en cuadrantes no vacíos
    children = []
    for quadrant in [Q1, Q2, Q3, Q4]:
        if quadrant:
            child = build_hierarchical_quadtree(quadrant, max_leaf_size, level + 1)
            children.append(child)
    
    return Node(points=[], bbox=bbox, children=children, level=level)


def traverse_leaves(node: Node) -> List[List[Point]]:
    """
    Recorre el quadtree y extrae todas las hojas.
    
    Args:
        node: Nodo raíz del quadtree
        
    Returns:
        Lista de listas de puntos (una por hoja)
    """
    if node.children is None:  # Es hoja
        return [node.points]
    
    leaves = []
    for child in node.children:
        leaves.extend(traverse_leaves(child))
    
    return leaves


def adaptive_parameters(n: int) -> Tuple[float, int]:
    """
    Calcula parámetros adaptativos basados en el tamaño de la instancia.
    
    Args:
        n: Número de puntos
        
    Returns:
        Tupla (eps_frontier_adaptativo, max_iter_adaptativo)
    """
    # eps_frontier: más pequeño para instancias grandes
    if n <= 50:
        eps_frontier = 8.0
    elif n <= 200:
        eps_frontier = 6.0
    elif n <= 1000:
        eps_frontier = 4.0
    else:
        eps_frontier = 2.0
    
    # max_iter: escalado inteligente
    if n <= 100:
        max_iter = min(10000, n * 150)
    elif n <= 500:
        max_iter = min(20000, n * 100)
    elif n <= 2000:
        max_iter = min(50000, n * 50)
    else:
        max_iter = min(100000, n * 25)
    
    return eps_frontier, max_iter


def classify_point_quadrant(point: Point, x_mid: float, y_mid: float) -> str:
    """
    Clasifica un punto en uno de los cuatro cuadrantes.
    
    Args:
        point: Punto a clasificar
        x_mid, y_mid: Coordenadas del punto medio
        
    Returns:
        Nombre del cuadrante ('Q1', 'Q2', 'Q3', 'Q4')
    """
    x, y = point
    if x <= x_mid and y > y_mid:
        return 'Q1'  # Noroeste
    elif x > x_mid and y > y_mid:
        return 'Q2'  # Noreste
    elif x <= x_mid and y <= y_mid:
        return 'Q3'  # Suroeste
    else:
        return 'Q4'  # Sureste


# =============================================================================
# MÓDULO 4.5: GATES GLOBALES Y NEAREST NEIGHBOR
# =============================================================================

def nearest_neighbor_tour(points: Route) -> Route:
    """
    Construye un tour inicial usando el algoritmo del vecino más cercano.
    
    Args:
        points: Lista de puntos
        
    Returns:
        Tour construido con NN
    """
    if len(points) < 3:
        return points[:]
    
    unvisited = points[:]
    tour = [unvisited.pop(0)]  # Empezar con el primer punto
    
    while unvisited:
        last = tour[-1]
        # Encontrar el punto más cercano
        closest_idx = min(range(len(unvisited)), 
                         key=lambda i: euclidean_distance(last, unvisited[i]))
        tour.append(unvisited.pop(closest_idx))
    
    return tour


def find_best_gate_pair(cluster_a: Route, cluster_b: Route) -> Tuple[Point, Point]:
    """
    Encuentra el mejor par de gates entre dos clusters usando KDTree.
    
    Args:
        cluster_a, cluster_b: Clusters a conectar
        
    Returns:
        Tupla (punto_a, punto_b) con la mejor conexión
    """
    if not cluster_a or not cluster_b:
        return None, None
    
    # Usar KDTree para búsqueda eficiente
    tree_b = cKDTree(np.array(cluster_b))
    distances, indices = tree_b.query(np.array(cluster_a), k=1)
    
    # Encontrar la distancia mínima
    min_idx = np.argmin(distances)
    best_point_a = cluster_a[min_idx]
    best_point_b = cluster_b[indices[min_idx]]
    
    return best_point_a, best_point_b


def optimize_cluster_orientation(cluster: Route, entry_point: Point, exit_point: Point) -> Route:
    """
    Optimiza la orientación de un cluster para conectar eficientemente.
    
    Args:
        cluster: Cluster a orientar
        entry_point: Punto de entrada deseado
        exit_point: Punto de salida deseado
        
    Returns:
        Cluster reorientado
    """
    if len(cluster) < 2:
        return cluster[:]
    
    # Encontrar índices de puntos más cercanos a entry y exit
    entry_idx = min(range(len(cluster)), 
                   key=lambda i: euclidean_distance(cluster[i], entry_point))
    exit_idx = min(range(len(cluster)), 
                  key=lambda i: euclidean_distance(cluster[i], exit_point))
    
    # Rotar cluster para empezar cerca del entry_point
    rotated = cluster[entry_idx:] + cluster[:entry_idx]
    
    # Decidir si invertir basado en proximidad al exit_point
    if len(rotated) > 1:
        forward_exit_dist = euclidean_distance(rotated[-1], exit_point)
        reversed_cluster = list(reversed(rotated))
        backward_exit_dist = euclidean_distance(reversed_cluster[-1], exit_point)
        
        if backward_exit_dist < forward_exit_dist:
            return reversed_cluster
    
    return rotated


def stitch_clusters_with_global_gates(clusters: List[Route]) -> Route:
    """
    Conecta clusters usando gates globales y orientación optimizada.
    
    Args:
        clusters: Lista de clusters (rutas parciales)
        
    Returns:
        Tour global conectado
    """
    if not clusters:
        return []
    
    if len(clusters) == 1:
        return clusters[0]
    
    # Optimizar cada cluster individualmente
    optimized_clusters = []
    for cluster in clusters:
        if len(cluster) > 2:
            nn_tour = nearest_neighbor_tour(cluster)
            optimized_tour = optimize_route_2opt_delta(nn_tour)
            optimized_clusters.append(optimized_tour)
        else:
            optimized_clusters.append(cluster)
    
    # Conectar clusters secuencialmente
    global_tour = optimized_clusters[0]
    
    for i in range(1, len(optimized_clusters)):
        current_cluster = optimized_clusters[i]
        
        # Encontrar mejor gate pair
        gate_a, gate_b = find_best_gate_pair(global_tour, current_cluster)
        
        if gate_a and gate_b:
            # Optimizar orientación del cluster actual
            oriented_cluster = optimize_cluster_orientation(
                current_cluster, gate_b, current_cluster[-1]
            )
            
            # Conectar
            global_tour.extend(oriented_cluster)
    
    return global_tour


def subdivide_into_quadrants(points: List[Point]) -> Tuple[Quadrants, float, float]:
    """
    Subdivide los puntos en cuatro cuadrantes.
    
    Args:
        points: Lista de puntos a subdividir
        
    Returns:
        Tupla (diccionario_cuadrantes, x_medio, y_medio)
    """
    x_mid, y_mid = calculate_midpoints(points)
    
    quadrants = {'Q1': [], 'Q2': [], 'Q3': [], 'Q4': []}
    
    for point in points:
        quad_name = classify_point_quadrant(point, x_mid, y_mid)
        quadrants[quad_name].append(point)
    
    return quadrants, x_mid, y_mid

# =============================================================================
# MÓDULO 5: BÚSQUEDA DE CONEXIONES FRONTERA
# =============================================================================

def filter_frontier_points(points: List[Point], direction: str, mid_value: float, epsilon: float) -> List[Point]:
    """
    Filtra puntos que están cerca de la frontera entre cuadrantes.
    
    Args:
        points: Lista de puntos a filtrar
        direction: 'vertical' o 'horizontal'
        mid_value: Valor de la línea divisoria
        epsilon: Tolerancia para considerar un punto como frontera
        
    Returns:
        Lista de puntos cercanos a la frontera
    """
    if direction == 'vertical':
        return [p for p in points if abs(p[0] - mid_value) < epsilon]
    else:
        return [p for p in points if abs(p[1] - mid_value) < epsilon]


def find_best_frontier_connection(quadrant_a: List[Point], quadrant_b: List[Point], 
                                direction: str, mid_value: float, epsilon: float) -> Tuple[Optional[Tuple[Point, Point]], float]:
    """
    Encuentra la mejor conexión entre dos cuadrantes basada en puntos frontera.
    
    Args:
        quadrant_a, quadrant_b: Puntos de los cuadrantes A y B
        direction: 'vertical' o 'horizontal'
        mid_value: Valor de la línea divisoria
        epsilon: Tolerancia para la frontera
        
    Returns:
        Tupla ((punto_a, punto_b), distancia) o ((None, None), inf) si no hay conexión
    """
    candidates_a = filter_frontier_points(quadrant_a, direction, mid_value, epsilon)
    candidates_b = filter_frontier_points(quadrant_b, direction, mid_value, epsilon)
    
    if not candidates_a or not candidates_b:
        return (None, None), float('inf')
    
    best_pair = (None, None)
    best_distance = float('inf')
    
    for point_a in candidates_a:
        for point_b in candidates_b:
            distance = euclidean_distance(point_a, point_b)
            if distance < best_distance:
                best_pair = (point_a, point_b)
                best_distance = distance
    
    return best_pair, best_distance
# =============================================================================
# MÓDULO 6: OPTIMIZACIÓN DE CUADRANTES
# =============================================================================

def optimize_quadrant_routes(quadrants: Quadrants, max_iter: int) -> Quadrants:
    """
    Optimiza las rutas dentro de cada cuadrante usando 2-opt.
    
    Args:
        quadrants: Diccionario con los puntos de cada cuadrante
        max_iter: Número máximo de iteraciones para 2-opt
        
    Returns:
        Diccionario con las rutas optimizadas por cuadrante
    """
    optimized_routes = {}
    
    for quad_name, points in quadrants.items():
        if len(points) > 1:
            optimized_route = optimize_route_2opt(points, max_iter)
            optimized_routes[quad_name] = optimized_route
        else:
            optimized_routes[quad_name] = points[:]
    
    return optimized_routes


def calculate_quadrant_lengths(routes: Quadrants) -> Dict[str, int]:
    """
    Calcula la longitud de cada ruta de cuadrante.
    
    Args:
        routes: Diccionario con las rutas de cada cuadrante
        
    Returns:
        Diccionario con las longitudes por cuadrante
    """
    lengths = {}
    for quad_name, route in routes.items():
        lengths[quad_name] = calculate_path_length(route)
    return lengths
# =============================================================================
# MÓDULO 7: CONEXIONES ENTRE CUADRANTES
# =============================================================================

def define_neighbor_pairs(x_mid: float, y_mid: float) -> List[Tuple[str, str, str, float]]:
    """
    Define los pares de cuadrantes vecinos y sus características de conexión.
    
    Args:
        x_mid, y_mid: Coordenadas del punto medio
        
    Returns:
        Lista de tuplas (cuadrante1, cuadrante2, dirección, valor_medio)
    """
    return [
        ('Q1', 'Q2', 'vertical', x_mid),
        ('Q1', 'Q3', 'horizontal', y_mid),
        ('Q2', 'Q4', 'horizontal', y_mid),
        ('Q3', 'Q4', 'vertical', x_mid)
    ]


def find_inter_quadrant_connections(quadrants: Quadrants, x_mid: float, y_mid: float, 
                                   eps_frontier: float) -> Tuple[Connections, int]:
    """
    Encuentra las mejores conexiones entre cuadrantes vecinos.
    
    Args:
        quadrants: Diccionario con los puntos de cada cuadrante
        x_mid, y_mid: Coordenadas del punto medio
        eps_frontier: Tolerancia para la búsqueda de frontera
        
    Returns:
        Tupla (lista_conexiones, longitud_total_inter)
    """
    neighbor_pairs = define_neighbor_pairs(x_mid, y_mid)
    connections = []
    total_inter_length = 0
    
    for q1, q2, direction, mid_value in neighbor_pairs:
        (point_a, point_b), distance = find_best_frontier_connection(
            quadrants[q1], quadrants[q2], direction, mid_value, eps_frontier
        )
        
        if point_a and point_b:
            connections.append((point_a, point_b))
            total_inter_length += distance
    
    return connections, total_inter_length


# =============================================================================
# MÓDULO 8: ALGORITMO PRINCIPAL GSPH-FC
# =============================================================================

def gsph_fc_enhanced_algorithm(nodes: List[Point], eps_frontier: Optional[float] = None, 
                              max_iter_local: Optional[int] = None) -> Tuple[Quadrants, Connections, float, float, float, Dict]:
    """
    Algoritmo GSPH-FC mejorado con subdivisión jerárquica, gates globales y post-optimización.
    
    Args:
        nodes: Lista de puntos a procesar
        eps_frontier: Tolerancia adaptativa (opcional)
        max_iter_local: Iteraciones adaptativas (opcional)
        
    Returns:
        Tupla (rutas_cuadrantes, conexiones, longitud_total, x_medio, y_medio, estadísticas_detalladas)
    """
    n = len(nodes)
    start_time = time.time()
    
    # Parámetros adaptativos
    if eps_frontier is None or max_iter_local is None:
        adaptive_eps, adaptive_iter = adaptive_parameters(n)
        eps_frontier = eps_frontier or adaptive_eps
        max_iter_local = max_iter_local or adaptive_iter
    
    # Construcción del quadtree jerárquico
    quadtree_start = time.time()
    quadtree = build_hierarchical_quadtree(nodes)
    clusters = traverse_leaves(quadtree)
    quadtree_time = time.time() - quadtree_start
    
    # Construcción del tour global con gates
    tour_start = time.time()
    global_tour = stitch_clusters_with_global_gates(clusters)
    tour_time = time.time() - tour_start
    
    # Post-optimización
    post_opt_start = time.time()
    if len(global_tour) > 3:
        # Or-opt
        global_tour = or_opt_optimization(global_tour, max_passes=2)
        # 2-opt final
        global_tour = optimize_route_2opt_delta(global_tour, max_iter_local)
        # 3-opt ligero para instancias pequeñas
        if n <= 200:
            global_tour = three_opt_light(global_tour, max_iter=500)
    post_opt_time = time.time() - post_opt_start
    
    # Calcular midpoints para visualización
    x_mid, y_mid = calculate_midpoints(nodes)
    
    # Redistribuir tour en cuadrantes para visualización
    quads = {'Q1': [], 'Q2': [], 'Q3': [], 'Q4': []}
    for point in global_tour:
        quad_name = classify_point_quadrant(point, x_mid, y_mid)
        quads[quad_name].append(point)
    
    # Generar conexiones inter-cuadrante desde el tour
    connections = []
    for i in range(len(global_tour)):
        current_point = global_tour[i]
        next_point = global_tour[(i + 1) % len(global_tour)]
        
        current_quad = classify_point_quadrant(current_point, x_mid, y_mid)
        next_quad = classify_point_quadrant(next_point, x_mid, y_mid)
        
        if current_quad != next_quad:
            connections.append((current_point, next_point))
    
    # Remover duplicados
    connections = list(set(connections))
    
    # Calcular longitud total precisa
    total_length = calculate_tour_length(global_tour)
    total_time = time.time() - start_time
    
    # Estadísticas detalladas
    stats = {
        'num_clusters': len(clusters),
        'cluster_sizes': [len(c) for c in clusters],
        'quadtree_time': quadtree_time,
        'tour_construction_time': tour_time,
        'post_optimization_time': post_opt_time,
        'total_algorithm_time': total_time,
        'adaptive_eps_frontier': eps_frontier,
        'adaptive_max_iter': max_iter_local,
        'global_tour_length': len(global_tour),
        'num_connections': len(connections)
    }
    
    return quads, connections, total_length, x_mid, y_mid, stats


def gsph_fc_algorithm(nodes: List[Point], eps_frontier: Optional[float] = None, 
                     max_iter_local: Optional[int] = None) -> Tuple[Quadrants, Connections, int, float, float]:
    """
    Wrapper para compatibilidad hacia atrás.
    """
    quads, connections, total_length, x_mid, y_mid, _ = gsph_fc_enhanced_algorithm(
        nodes, eps_frontier, max_iter_local
    )
    return quads, connections, int(total_length), x_mid, y_mid
# =============================================================================
# MÓDULO 9: VISUALIZACIÓN Y GRÁFICOS
# =============================================================================

def setup_plot_colors_and_labels() -> Tuple[List[str], List[str]]:
    """
    Define colores y etiquetas para los cuadrantes.
    
    Returns:
        Tupla (lista_colores, lista_etiquetas)
    """
    colors = ['blue', 'green', 'red', 'orange']
    labels = ['Q1 (NW)', 'Q2 (NE)', 'Q3 (SW)', 'Q4 (SE)']
    return colors, labels


def plot_quadrant_routes(routes: Quadrants, colors: List[str], labels: List[str]) -> List[Point]:
    """
    Plotea las rutas de cada cuadrante.
    
    Args:
        routes: Diccionario con las rutas de cada cuadrante
        colors: Lista de colores para cada cuadrante
        labels: Lista de etiquetas para cada cuadrante
        
    Returns:
        Lista de puntos de conexión
    """
    connection_points = []
    
    for idx, (quad_name, route) in enumerate(routes.items()):
        if len(route) > 1:
            x, y = zip(*route)
            plt.plot(x, y, marker='o', color=colors[idx], linewidth=2, 
                    markersize=4, label=f'{labels[idx]} ({len(route)} puntos)')
            
            plt.scatter(x[0], y[0], color=colors[idx], s=80, marker='s', 
                       edgecolors='black', linewidth=1)
            plt.scatter(x[-1], y[-1], color=colors[idx], s=80, marker='^', 
                       edgecolors='black', linewidth=1)
        elif len(route) == 1:
            plt.scatter(*route[0], color=colors[idx], marker='x', s=100, 
                       label=f'{labels[idx]} (1 punto)')
    
    return connection_points


def plot_inter_quadrant_connections(routes: Quadrants) -> List[Point]:
    """
    Plotea las conexiones entre cuadrantes.
    
    Args:
        routes: Diccionario con las rutas de cada cuadrante
        
    Returns:
        Lista de puntos de conexión
    """
    subroutes = list(routes.values())
    connection_points = []
    
    for i in range(len(subroutes)):
        if len(subroutes[i]) > 0 and len(subroutes[(i+1) % len(subroutes)]) > 0:
            last_point = subroutes[i][-1]
            next_point = subroutes[(i+1) % len(subroutes)][0]
            
            plt.plot([last_point[0], next_point[0]], [last_point[1], next_point[1]],
                     linestyle='--', color='purple', linewidth=3, alpha=0.8,
                     label='Conexiones inter-cuadrante' if i == 0 else "")
            connection_points.extend([last_point, next_point])
    
    return connection_points


def plot_quadrant_divisions(x_mid: float, y_mid: float):
    """
    Plotea las líneas divisorias entre cuadrantes.
    
    Args:
        x_mid, y_mid: Coordenadas del punto medio
    """
    plt.axvline(x_mid, linestyle='-', color='gray', alpha=0.5, linewidth=2, 
                label='División vertical')
    plt.axhline(y_mid, linestyle='-', color='gray', alpha=0.5, linewidth=2, 
                label='División horizontal')


def add_quadrant_labels(routes: Quadrants, x_mid: float, y_mid: float):
    """
    Añade etiquetas de texto a cada cuadrante.
    
    Args:
        routes: Diccionario con las rutas de cada cuadrante
        x_mid, y_mid: Coordenadas del punto medio
    """
    all_x = [p[0] for route in routes.values() for p in route]
    all_y = [p[1] for route in routes.values() for p in route]
    
    if not all_x or not all_y:
        return
    
    x_range = max(all_x) - min(all_x)
    y_range = max(all_y) - min(all_y)
    offset_x = x_range * 0.02
    offset_y = y_range * 0.02
    
    plt.text(x_mid - offset_x, max(all_y) - offset_y, 'Q1 (NW)', 
             fontsize=12, fontweight='bold', ha='right')
    plt.text(x_mid + offset_x, max(all_y) - offset_y, 'Q2 (NE)', 
             fontsize=12, fontweight='bold', ha='left')
    plt.text(x_mid - offset_x, min(all_y) + offset_y, 'Q3 (SW)', 
             fontsize=12, fontweight='bold', ha='right')
    plt.text(x_mid + offset_x, min(all_y) + offset_y, 'Q4 (SE)', 
             fontsize=12, fontweight='bold', ha='left')


def create_complete_plot(routes: Quadrants, connections: Connections, x_mid: float, y_mid: float, 
                        total_length: int, save_path: Optional[str] = None, show_plot: bool = True):
    """
    Crea el gráfico completo con todas las visualizaciones.
    
    Args:
        routes: Diccionario con las rutas de cada cuadrante
        connections: Lista de conexiones entre cuadrantes
        x_mid, y_mid: Coordenadas del punto medio
        total_length: Longitud total de la solución
        save_path: Ruta donde guardar el gráfico (opcional)
        show_plot: Si mostrar el gráfico en pantalla
    """
    plt.figure(figsize=(12, 10))
    colors, labels = setup_plot_colors_and_labels()
    
    plot_quadrant_routes(routes, colors, labels)
    
    connection_points = plot_inter_quadrant_connections(routes)
    
    if connection_points:
        conn_x, conn_y = zip(*connection_points)
        plt.scatter(conn_x, conn_y, color='purple', s=60, marker='D', 
                   edgecolors='white', linewidth=1, alpha=0.8)
    
    plot_quadrant_divisions(x_mid, y_mid)
    add_quadrant_labels(routes, x_mid, y_mid)
    
    plt.title(f"GSPH-FC (Cuadrantes) - Longitud total: {total_length:.0f}", fontsize=14)
    plt.xlabel("Coordenada X", fontsize=12)
    plt.ylabel("Coordenada Y", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"✓ Gráfico guardado en: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
# =============================================================================
# MÓDULO 10: PROCESAMIENTO DE RESULTADOS Y REPORTES
# =============================================================================

def generate_professional_report(routes: Quadrants, total_length: float, execution_time: float, 
                               instance_name: str, bks_value: Optional[int] = None, 
                               stats: Optional[Dict] = None) -> str:
    """
    Genera un reporte profesional y detallado de los resultados.
    
    Args:
        routes: Diccionario con las rutas de cada cuadrante
        total_length: Longitud total de la solución
        execution_time: Tiempo de ejecución
        instance_name: Nombre de la instancia
        bks_value: Valor BKS para comparación
        stats: Estadísticas adicionales del algoritmo
        
    Returns:
        String con el reporte formateado profesionalmente
    """
    report = []
    report.append("="*80)
    report.append(f"REPORTE DETALLADO - ALGORITMO GSPH-FC MEJORADO")
    report.append("="*80)
    report.append(f"Instancia: {instance_name}")
    report.append(f"Fecha y hora: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Información de la instancia
    total_points = sum(len(route) for route in routes.values())
    report.append("INFORMACIÓN DE LA INSTANCIA:")
    report.append(f"  • Total de puntos: {total_points}")
    
    # Comparación con BKS
    if bks_value:
        gap = ((total_length - bks_value) / bks_value) * 100
        report.append(f"  • Valor BKS (Best Known Solution): {bks_value}")
        report.append(f"  • Valor obtenido: {total_length:.6f}")
        report.append(f"  • Gap vs BKS: {gap:.4f}%")
        if gap <= 0:
            report.append("  • ✅ MEJOR QUE BKS!")
        elif gap <= 5:
            report.append("  • ✅ Excelente resultado (≤5% gap)")
        elif gap <= 10:
            report.append("  • ✅ Buen resultado (≤10% gap)")
        elif gap <= 20:
            report.append("  • ⚠️  Resultado aceptable (≤20% gap)")
        else:
            report.append("  • ❌ Resultado por mejorar (>20% gap)")
    else:
        report.append(f"  • Valor obtenido: {total_length:.6f}")
        report.append("  • Valor BKS: No disponible")
    
    report.append("")
    
    # Detalles por cuadrante
    report.append("DISTRIBUCIÓN POR CUADRANTES:")
    total_intra_length = 0
    for quad_name, route in routes.items():
        if route:
            route_length = calculate_path_length(route)
            total_intra_length += route_length
            percentage = (len(route) / total_points) * 100
            density = route_length / len(route) if len(route) > 0 else 0
            
            report.append(f"  • {quad_name} (Cuadrante):")
            report.append(f"    - Número de nodos: {len(route)} ({percentage:.2f}%)")
            report.append(f"    - Longitud de ruta: {route_length:.6f}")
            report.append(f"    - Densidad promedio: {density:.4f}")
    
    report.append("")
    
    # Información de conexiones
    inter_length = total_length - total_intra_length
    if total_length > 0:
        intra_percentage = (total_intra_length / total_length) * 100
        inter_percentage = (inter_length / total_length) * 100
    else:
        intra_percentage = inter_percentage = 0
    
    report.append("ANÁLISIS DE CONEXIONES:")
    report.append(f"  • Longitud intra-cuadrante: {total_intra_length:.6f} ({intra_percentage:.2f}%)")
    report.append(f"  • Longitud inter-cuadrante: {inter_length:.6f} ({inter_percentage:.2f}%)")
    report.append(f"  • Longitud total: {total_length:.6f}")
    report.append("")
    
    # Información temporal
    report.append("ANÁLISIS TEMPORAL:")
    report.append(f"  • Tiempo total de ejecución: {execution_time:.6f} segundos")
    if total_points > 0:
        time_per_point = execution_time / total_points
        report.append(f"  • Tiempo por punto: {time_per_point:.6f} segundos")
    
    # Estadísticas del algoritmo si están disponibles
    if stats:
        report.append("")
        report.append("ESTADÍSTICAS DEL ALGORITMO:")
        report.append(f"  • Número de clusters generados: {stats.get('num_clusters', 'N/A')}")
        if 'cluster_sizes' in stats:
            avg_cluster_size = sum(stats['cluster_sizes']) / len(stats['cluster_sizes'])
            report.append(f"  • Tamaño promedio de cluster: {avg_cluster_size:.2f}")
            report.append(f"  • Rango de tamaños: {min(stats['cluster_sizes'])} - {max(stats['cluster_sizes'])}")
        
        report.append(f"  • Tiempo construcción quadtree: {stats.get('quadtree_time', 0):.6f}s")
        report.append(f"  • Tiempo construcción tour: {stats.get('tour_construction_time', 0):.6f}s")
        report.append(f"  • Tiempo post-optimización: {stats.get('post_optimization_time', 0):.6f}s")
        report.append(f"  • Parámetros adaptativos:")
        report.append(f"    - eps_frontier: {stats.get('adaptive_eps_frontier', 'N/A')}")
        report.append(f"    - max_iter: {stats.get('adaptive_max_iter', 'N/A')}")
        report.append(f"  • Conexiones inter-cuadrante: {stats.get('num_connections', 'N/A')}")
    
    report.append("")
    report.append("="*80)
    
    return "\n".join(report)


def save_professional_results(routes: Quadrants, total_length: float, execution_time: float, 
                            instance_name: str, bks_value: Optional[int] = None, 
                            stats: Optional[Dict] = None):
    """
    Guarda los resultados en una estructura de carpetas profesional.
    
    Args:
        routes: Diccionario con las rutas de cada cuadrante
        total_length: Longitud total de la solución
        execution_time: Tiempo de ejecución
        instance_name: Nombre de la instancia
        bks_value: Valor BKS para comparación
        stats: Estadísticas adicionales del algoritmo
    """
    # Crear estructura de directorios profesional
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = RESULTS_DIR
    instance_dir = os.path.join(base_dir, instance_name)
    run_dir = os.path.join(instance_dir, f"run_{timestamp}")
    
    os.makedirs(run_dir, exist_ok=True)
    
    # Generar reporte profesional
    report = generate_professional_report(routes, total_length, execution_time, 
                                        instance_name, bks_value, stats)
    
    # Guardar reporte principal
    report_path = os.path.join(run_dir, f"{instance_name}_detailed_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    # Guardar datos en CSV para análisis
    csv_path = os.path.join(run_dir, f"{instance_name}_data.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Cuadrante', 'Num_Nodos', 'Longitud_Ruta', 'Densidad'])
        
        for quad_name, route in routes.items():
            if route:
                route_length = calculate_path_length(route)
                density = route_length / len(route) if len(route) > 0 else 0
                writer.writerow([quad_name, len(route), f"{route_length:.6f}", f"{density:.6f}"])
    
    # Guardar resumen ejecutivo
    summary_path = os.path.join(run_dir, f"{instance_name}_executive_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"RESUMEN EJECUTIVO - {instance_name}\n")
        f.write("="*50 + "\n")
        f.write(f"Longitud obtenida: {total_length:.6f}\n")
        if bks_value:
            gap = ((total_length - bks_value) / bks_value) * 100
            f.write(f"Valor BKS: {bks_value}\n")
            f.write(f"Gap: {gap:.4f}%\n")
        f.write(f"Tiempo ejecución: {execution_time:.6f}s\n")
        f.write(f"Número de puntos: {sum(len(route) for route in routes.values())}\n")
    
    print(f"✅ Resultados profesionales guardados en: {run_dir}")
    return run_dir


def save_results_to_file(routes: Quadrants, total_length: int, execution_time: float, 
                        results_dir: str, instance_name: str = ""):
    """
    Función de compatibilidad hacia atrás. Usa la nueva función profesional.
    
    Args:
        routes: Diccionario con las rutas de cada cuadrante
        total_length: Longitud total de la solución
        execution_time: Tiempo de ejecución
        results_dir: Directorio donde guardar los resultados
        instance_name: Nombre de la instancia (opcional)
    """
    # Usar la nueva función profesional pero en formato simple
    save_professional_results(routes, float(total_length), execution_time, instance_name)


def print_enhanced_summary(routes: Quadrants, total_length: float, execution_time: float, 
                         instance_name: str, bks_value: Optional[int] = None, 
                         stats: Optional[Dict] = None):
    """
    Imprime un resumen mejorado de los resultados en consola.
    
    Args:
        routes: Diccionario con las rutas de cada cuadrante
        total_length: Longitud total de la solución
        execution_time: Tiempo de ejecución
        instance_name: Nombre de la instancia
        bks_value: Valor BKS para comparación
        stats: Estadísticas adicionales del algoritmo
    """
    print(f"\n{'='*80}")
    print(f"RESULTADOS DETALLADOS PARA {instance_name.upper()}")
    print(f"{'='*80}")
    
    # Información básica
    total_points = sum(len(route) for route in routes.values())
    print(f"Total de puntos: {total_points}")
    print(f"Longitud obtenida: {total_length:.6f}")
    
    # Comparación con BKS
    if bks_value:
        gap = ((total_length - bks_value) / bks_value) * 100
        print(f"Valor BKS: {bks_value}")
        print(f"Gap vs BKS: {gap:.4f}%", end="")
        if gap <= 0:
            print(" ✅ MEJOR QUE BKS!")
        elif gap <= 5:
            print(" ✅ Excelente")
        elif gap <= 10:
            print(" ✅ Muy bueno")
        elif gap <= 20:
            print(" ⚠️  Aceptable")
        else:
            print(" ❌ Por mejorar")
    
    print(f"Tiempo de ejecución: {execution_time:.6f} segundos")
    
    # Detalles por cuadrante
    print(f"\nDistribución por cuadrantes:")
    for quad_name, route in routes.items():
        if route:
            route_length = calculate_path_length(route)
            percentage = (len(route) / total_points) * 100
            print(f"  {quad_name}: {len(route)} nodos ({percentage:.1f}%), longitud: {route_length:.6f}")
    
    # Estadísticas del algoritmo
    if stats:
        print(f"\nEstadísticas del algoritmo:")
        print(f"  Clusters generados: {stats.get('num_clusters', 'N/A')}")
        print(f"  Parámetros adaptativos: eps={stats.get('adaptive_eps_frontier', 'N/A'):.2f}, "
              f"iter={stats.get('adaptive_max_iter', 'N/A')}")
        print(f"  Tiempo quadtree: {stats.get('quadtree_time', 0):.4f}s")
        print(f"  Tiempo post-opt: {stats.get('post_optimization_time', 0):.4f}s")


def print_results_summary(routes: Quadrants, total_length: int, execution_time: float, instance_name: str = ""):
    """
    Función de compatibilidad hacia atrás.
    """
    print_enhanced_summary(routes, float(total_length), execution_time, instance_name)


# =============================================================================
# MÓDULO 10B: PROCESAMIENTO MASIVO DE INSTANCIAS
# =============================================================================

def run_single_instance(tsp_file: str, eps_frontier: float, max_iter_local: int, 
                       save_plots: bool = True, show_plots: bool = False) -> Dict:
    """
    Ejecuta el algoritmo en una sola instancia.
    
    Args:
        tsp_file: Ruta al archivo TSP
        eps_frontier: Parámetro epsilon para fronteras
        max_iter_local: Iteraciones máximas para 2-opt
        save_plots: Si guardar gráficos
        show_plots: Si mostrar gráficos
        
    Returns:
        Diccionario con los resultados
    """
    instance_name = get_instance_name(tsp_file)
    
    try:
        # Cargar datos
        nodes = read_tsplib(tsp_file)
        num_points = len(nodes)
        
        # Cargar BKS para comparación
        bks_dict = load_bks_values()
        bks_value = bks_dict.get(instance_name)
        
        # Ejecutar algoritmo mejorado
        start_time = time.time()
        routes, connections, total_length, x_mid, y_mid, stats = gsph_fc_enhanced_algorithm(
            nodes, eps_frontier, max_iter_local
        )
        execution_time = time.time() - start_time
        
        # Guardar resultados profesionales
        run_dir = save_professional_results(routes, total_length, execution_time, 
                                          instance_name, bks_value, stats)
        
        # Crear visualización si se solicita
        if save_plots:
            plot_path = os.path.join(run_dir, f"grafico_{instance_name}.png")
            create_complete_plot(
                routes, connections, x_mid, y_mid, total_length, 
                save_path=plot_path, show_plot=show_plots
            )
        
        return {
            'instance': instance_name,
            'points': num_points,
            'total_length': total_length,
            'execution_time': execution_time,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        print(f"ERROR procesando {instance_name}: {str(e)}")
        return {
            'instance': instance_name,
            'points': 0,
            'total_length': float('inf'),
            'execution_time': 0,
            'success': False,
            'error': str(e)
        }


def run_all_instances(eps_frontier: float = 5, max_iter_local: int = 800, 
                     save_plots: bool = True, show_plots: bool = False) -> List[Dict]:
    """
    Ejecuta el algoritmo en todas las instancias TSP encontradas.
    
    Args:
        eps_frontier: Parámetro epsilon para fronteras
        max_iter_local: Iteraciones máximas para 2-opt
        save_plots: Si guardar gráficos
        show_plots: Si mostrar gráficos
        
    Returns:
        Lista de resultados para cada instancia
    """
    tsp_files = find_all_tsp_files()
    
    if not tsp_files:
        print("No se encontraron archivos TSP")
        return []
    
    print(f"Encontradas {len(tsp_files)} instancias TSP")
    print(f"Configuración: eps_frontier={eps_frontier}, max_iter_local={max_iter_local}")
    print(f"{'='*80}")
    
    results = []
    total_start_time = time.time()
    
    for i, tsp_file in enumerate(tsp_files, 1):
        instance_name = get_instance_name(tsp_file)
        print(f"\n[{i:3d}/{len(tsp_files)}] Procesando {instance_name}...")
        
        result = run_single_instance(tsp_file, eps_frontier, max_iter_local, save_plots, show_plots)
        results.append(result)
        
        if result['success']:
            print(f"    ✓ Completado: {result['points']} puntos, longitud: {result['total_length']:.0f}, tiempo: {result['execution_time']:.3f}s")
        else:
            print(f"    ✗ Error: {result['error']}")
    
    total_time = time.time() - total_start_time
    successful = sum(1 for r in results if r['success'])
    
    print(f"\n{'='*80}")
    print(f"RESUMEN FINAL:")
    print(f"  Total instancias: {len(results)}")
    print(f"  Exitosas: {successful}")
    print(f"  Con errores: {len(results) - successful}")
    print(f"  Tiempo total: {total_time:.2f} segundos")
    
    return results


def save_consolidated_report(results: List[Dict], eps_frontier: float, max_iter_local: int):
    """
    Guarda un reporte consolidado con todos los resultados.
    
    Args:
        results: Lista de resultados de todas las instancias
        eps_frontier: Parámetro epsilon usado
        max_iter_local: Iteraciones máximas usadas
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    csv_path = os.path.join(RESULTS_DIR, "resumen_consolidado.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['instancia', 'puntos', 'longitud_total', 'tiempo_ejecucion', 'exitoso', 'error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow({
                'instancia': result['instance'],
                'puntos': result['points'],
                'longitud_total': result['total_length'] if result['success'] else 'ERROR',
                'tiempo_ejecucion': result['execution_time'],
                'exitoso': 'SI' if result['success'] else 'NO',
                'error': result['error'] or ''
            })
    
    report_path = os.path.join(RESULTS_DIR, "reporte_consolidado.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("REPORTE CONSOLIDADO - ALGORITMO GSPH-FC\n")
        f.write("="*60 + "\n\n")
        f.write(f"Configuración utilizada:\n")
        f.write(f"  eps_frontier: {eps_frontier}\n")
        f.write(f"  max_iter_local: {max_iter_local}\n\n")
        
        successful_results = [r for r in results if r['success']]
        
        if successful_results:
            f.write(f"ESTADÍSTICAS GENERALES:\n")
            f.write(f"  Total de instancias procesadas: {len(results)}\n")
            f.write(f"  Instancias exitosas: {len(successful_results)}\n")
            f.write(f"  Instancias con error: {len(results) - len(successful_results)}\n\n")
            
            lengths = [r['total_length'] for r in successful_results]
            times = [r['execution_time'] for r in successful_results]
            points = [r['points'] for r in successful_results]
            
            f.write(f"ESTADÍSTICAS DE INSTANCIAS EXITOSAS:\n")
            f.write(f"  Longitud promedio: {sum(lengths)/len(lengths):.2f}\n")
            f.write(f"  Longitud mínima: {min(lengths):.2f}\n")
            f.write(f"  Longitud máxima: {max(lengths):.2f}\n")
            f.write(f"  Tiempo promedio: {sum(times)/len(times):.3f}s\n")
            f.write(f"  Tiempo mínimo: {min(times):.3f}s\n")
            f.write(f"  Tiempo máximo: {max(times):.3f}s\n")
            f.write(f"  Puntos promedio: {sum(points)/len(points):.1f}\n")
            f.write(f"  Puntos mínimo: {min(points)}\n")
            f.write(f"  Puntos máximo: {max(points)}\n\n")
        
        f.write("RESULTADOS DETALLADOS:\n")
        f.write("-" * 60 + "\n")
        for result in sorted(results, key=lambda x: x['instance']):
            f.write(f"Instancia: {result['instance']}\n")
            if result['success']:
                f.write(f"  Puntos: {result['points']}\n")
                f.write(f"  Longitud: {result['total_length']:.2f}\n")
                f.write(f"  Tiempo: {result['execution_time']:.3f}s\n")
            else:
                f.write(f"  ERROR: {result['error']}\n")
            f.write("\n")
    
    print(f"✓ Reporte CSV guardado en: {csv_path}")
    print(f"✓ Reporte detallado guardado en: {report_path}")


# =============================================================================
# MÓDULO 11: CONFIGURACIÓN Y ARGUMENTOS DE LÍNEA DE COMANDOS
# =============================================================================

def setup_command_line_parser() -> argparse.ArgumentParser:
    """
    Configura el parser de argumentos de línea de comandos.
    
    Returns:
        Parser configurado
    """
    parser = argparse.ArgumentParser(description="GSPH-FC (Quadrant-based) algorithm for TSP")
    
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--tsp", type=str, help="TSP file name for single instance")
    mode_group.add_argument("--all", action="store_true", help="Run on all TSP instances found")
    
    parser.add_argument("--eps_frontier", type=int, default=5, 
                       help="Frontier epsilon for boundary connections (default: 5)")
    parser.add_argument("--max_iter_local", type=int, default=800, 
                       help="Maximum iterations for local 2-opt (default: 800)")
    
    parser.add_argument("--no_plots", action="store_true", 
                       help="Don't generate plot files (faster for batch processing)")
    parser.add_argument("--show_plots", action="store_true", 
                       help="Show plots on screen (only for single instance)")
    
    parser.set_defaults(tsp="a280.tsp")
    
    return parser


def validate_and_prepare_parameters(args) -> Tuple[Optional[str], bool, float, int, bool, bool]:
    """
    Valida y prepara los parámetros del algoritmo.
    
    Args:
        args: Argumentos parseados de la línea de comandos
        
    Returns:
        Tupla (tsp_file_or_None, run_all, eps_frontier, max_iter_local, save_plots, show_plots)
    """
    if args.all:
        run_all = True
        tsp_file = None
    else:
        run_all = False
        tsp_file = find_tsp_file(args.tsp)
    
    eps_frontier = args.eps_frontier
    max_iter_local = args.max_iter_local
    save_plots = not args.no_plots
    show_plots = args.show_plots and not args.all
    
    return tsp_file, run_all, eps_frontier, max_iter_local, save_plots, show_plots


# =============================================================================
# MÓDULO 12: FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """
    Función principal que ejecuta todo el pipeline del algoritmo GSPH-FC.
    Soporta tanto ejecución en una sola instancia como en todas las instancias.
    """
    parser = setup_command_line_parser()
    args = parser.parse_args()
    
    tsp_file, run_all, eps_frontier, max_iter_local, save_plots, show_plots = validate_and_prepare_parameters(args)
    
    if run_all:
        print("🚀 MODO MASIVO: Ejecutando en todas las instancias TSP")
        print(f"Configuración: eps_frontier={eps_frontier}, max_iter_local={max_iter_local}")
        print(f"Gráficos: {'Guardando' if save_plots else 'No guardando'}")
        
        results = run_all_instances(eps_frontier, max_iter_local, save_plots, show_plots)
        
        if results:
            save_consolidated_report(results, eps_frontier, max_iter_local)
            print(f"\n✅ Procesamiento masivo completado. Resultados en: {RESULTS_DIR}")
        
    else:
        # MODO: Ejecutar en una sola instancia
        instance_name = get_instance_name(tsp_file)
        print(f"🎯 MODO INDIVIDUAL: Ejecutando en {instance_name}")
        print(f"Archivo: {tsp_file}")
        print(f"Configuración: eps_frontier={eps_frontier}, max_iter_local={max_iter_local}")
        
        # Cargar datos
        nodes = read_tsplib(tsp_file)
        print(f"Instancia cargada: {len(nodes)} puntos")
        
        # Cargar BKS para comparación
        bks_dict = load_bks_values()
        bks_value = bks_dict.get(instance_name)
        
        # Ejecutar algoritmo mejorado
        start_time = time.time()
        routes, connections, total_length, x_mid, y_mid, stats = gsph_fc_enhanced_algorithm(
            nodes, eps_frontier, max_iter_local
        )
        execution_time = time.time() - start_time
        
        # Mostrar resumen en consola
        print_enhanced_summary(routes, total_length, execution_time, instance_name, bks_value, stats)
        
        # Guardar resultados profesionales
        run_dir = save_professional_results(routes, total_length, execution_time, 
                                          instance_name, bks_value, stats)
        
        # Crear visualización
        if save_plots:
            plot_path = os.path.join(run_dir, f"grafico_{instance_name}.png")
            create_complete_plot(
                routes, connections, x_mid, y_mid, total_length, 
                save_path=plot_path, show_plot=show_plots
            )


if __name__ == "__main__":
    main()