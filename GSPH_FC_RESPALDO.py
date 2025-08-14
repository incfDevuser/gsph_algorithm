import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import time
import os
import argparse
import glob
import csv
import datetime
from typing import List, Tuple, Dict, Optional

# =============================================================================
# CONFIGURACIÓN GLOBAL
# =============================================================================
EPS_FRONTIER = 5
MAX_ITER_LOCAL = 800
RESULTS_DIR = "gsph_fc_results"

# Tipos de datos para mayor claridad
Point = Tuple[float, float]
Route = List[Point]
Quadrants = Dict[str, Route]
Connections = List[Tuple[Point, Point]]
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


def optimize_route_2opt(points: Route, max_iter: int = 800) -> Route:
    """
    Optimiza una ruta usando el algoritmo 2-opt.
    
    Args:
        points: Puntos a optimizar
        max_iter: Número máximo de iteraciones
        
    Returns:
        Ruta optimizada
    """
    if len(points) < 3:
        return points[:]
    
    best = points[:]
    best_dist = calculate_path_length(best)
    changed = True
    iteration = 0
    
    while changed and iteration < max_iter:
        changed = False
        iteration += 1
        
        for i in range(1, len(points) - 2):
            for k in range(i + 1, len(points)):
                new_route = two_opt_swap(best, i, k)
                new_dist = calculate_path_length(new_route)
                
                if new_dist < best_dist:
                    best = new_route
                    best_dist = new_dist
                    changed = True
    
    return best
# =============================================================================
# MÓDULO 4: SUBDIVISIÓN EN CUADRANTES
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

def gsph_fc_algorithm(nodes: List[Point], eps_frontier: Optional[float] = None, 
                     max_iter_local: Optional[int] = None) -> Tuple[Quadrants, Connections, int, float, float]:
    """
    Ejecuta el algoritmo completo GSPH-FC.
    
    Args:
        nodes: Lista de puntos a procesar
        eps_frontier: Tolerancia para búsqueda de frontera (opcional)
        max_iter_local: Iteraciones máximas para 2-opt (opcional)
        
    Returns:
        Tupla (rutas_cuadrantes, conexiones, longitud_total, x_medio, y_medio)
    """
    if eps_frontier is None:
        eps_frontier = EPS_FRONTIER
    if max_iter_local is None:
        max_iter_local = MAX_ITER_LOCAL
    quadrants, x_mid, y_mid = subdivide_into_quadrants(nodes)
    optimized_routes = optimize_quadrant_routes(quadrants, max_iter_local)
    connections, inter_length = find_inter_quadrant_connections(
        quadrants, x_mid, y_mid, eps_frontier
    )
    intra_length = sum(calculate_path_length(route) for route in optimized_routes.values())
    total_length = intra_length + inter_length
    
    return optimized_routes, connections, total_length, x_mid, y_mid
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

def generate_statistics_report(routes: Quadrants, total_length: int, execution_time: float) -> str:
    """
    Genera un reporte estadístico de los resultados.
    
    Args:
        routes: Diccionario con las rutas de cada cuadrante
        total_length: Longitud total de la solución
        execution_time: Tiempo de ejecución
        
    Returns:
        String con el reporte formateado
    """
    report = "-- RESULTADOS GSPH-FC --\n"
    report += f"Longitud total de la ruta: {total_length:.2f}\n"
    report += f"Tiempo de ejecucion      : {execution_time:.3f} segundos\n"
    report += f"Puntos por cuadrante:\n"
    
    for quad_name, route in routes.items():
        route_length = calculate_path_length(route)
        report += f"  {quad_name}: {len(route)} puntos, longitud: {route_length}\n"
    
    return report


def save_results_to_file(routes: Quadrants, total_length: int, execution_time: float, 
                        results_dir: str, instance_name: str = ""):
    """
    Guarda los resultados en un archivo de texto.
    
    Args:
        routes: Diccionario con las rutas de cada cuadrante
        total_length: Longitud total de la solución
        execution_time: Tiempo de ejecución
        results_dir: Directorio donde guardar los resultados
        instance_name: Nombre de la instancia (opcional)
    """
    os.makedirs(results_dir, exist_ok=True)
    report = generate_statistics_report(routes, total_length, execution_time)
    
    filename = f"resultados_{instance_name}.txt" if instance_name else "resultados.txt"
    with open(os.path.join(results_dir, filename), "w", encoding="utf-8") as f:
        f.write(f"Instancia: {instance_name}\n" if instance_name else "")
        f.write(report)


def print_results_summary(routes: Quadrants, total_length: int, execution_time: float, instance_name: str = ""):
    """
    Imprime un resumen de los resultados en consola.
    
    Args:
        routes: Diccionario con las rutas de cada cuadrante
        total_length: Longitud total de la solución
        execution_time: Tiempo de ejecución
        instance_name: Nombre de la instancia (opcional)
    """
    if instance_name:
        print(f"\n{'='*60}")
        print(f"RESULTADOS PARA {instance_name.upper()}")
        print(f"{'='*60}")
    else:
        print("\n-- RESULTADOS GSPH-FC --")
    
    print(f"Longitud total de la ruta: {total_length:.2f}")
    print(f"Tiempo de ejecucion      : {execution_time:.3f} segundos")
    print(f"Puntos por cuadrante:")
    
    for quad_name, route in routes.items():
        route_length = calculate_path_length(route)
        print(f"  {quad_name}: {len(route)} puntos, longitud: {route_length}")


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
        nodes = read_tsplib(tsp_file)
        num_points = len(nodes)
        
        start_time = time.time()
        routes, connections, total_length, x_mid, y_mid = gsph_fc_algorithm(
            nodes, eps_frontier, max_iter_local
        )
        execution_time = time.time() - start_time
        
        instance_dir = os.path.join(RESULTS_DIR, instance_name)
        
        save_results_to_file(routes, total_length, execution_time, instance_dir, instance_name)
        
        if save_plots:
            plot_path = os.path.join(instance_dir, f"grafico_{instance_name}.png")
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
        instance_name = get_instance_name(tsp_file)
        print(f"🎯 MODO INDIVIDUAL: Ejecutando en {instance_name}")
        print(f"Archivo: {tsp_file}")
        print(f"Configuración: eps_frontier={eps_frontier}, max_iter_local={max_iter_local}")
        
        nodes = read_tsplib(tsp_file)
        print(f"Instancia cargada: {len(nodes)} puntos")
        
        start_time = time.time()
        routes, connections, total_length, x_mid, y_mid = gsph_fc_algorithm(
            nodes, eps_frontier, max_iter_local
        )
        execution_time = time.time() - start_time
        
        save_results_to_file(routes, total_length, execution_time, RESULTS_DIR, instance_name)
        print_results_summary(routes, total_length, execution_time, instance_name)
        
        # Crear visualización
        if save_plots:
            plot_path = os.path.join(RESULTS_DIR, f"grafico_{instance_name}.png")
            create_complete_plot(
                routes, connections, x_mid, y_mid, total_length, 
                save_path=plot_path, show_plot=show_plots
            )
if __name__ == "__main__":
    main()