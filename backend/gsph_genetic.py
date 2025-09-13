import random
import numpy as np
import time
from math import hypot
from copy import deepcopy
from gsph_algorithm import euclidean_distance, eucl_float, subdivide_quadrants, best_frontier_pair, polish_fast

class Individual:
    """
    Representa un individuo en la población genética.
    
    Cromosoma: Lista de nodos representando el orden de visita en el tour
    Gen: Cada posición en el cromosoma (índice de un nodo)
    """
    def __init__(self, chromosome=None, nodes=None):
        self.chromosome = chromosome if chromosome else []  
        self.fitness = None
        self.tour_length = None
        self.nodes = nodes
        self.quadrant_structure = None 
        
    def calculate_fitness(self):
        """Función fitness: 1 / (longitud_del_tour + penalización)"""
        if not self.chromosome or len(self.chromosome) < 2:
            self.fitness = 0
            self.tour_length = float('inf')
            return
            
        total_length = 0
        for i in range(len(self.chromosome)):
            current = self.nodes[self.chromosome[i]]
            next_node = self.nodes[self.chromosome[(i + 1) % len(self.chromosome)]]
            total_length += euclidean_distance(current, next_node)
            
        penalty = self._gsph_penalty()
        
        self.tour_length = total_length + penalty
        self.fitness = 1.0 / (self.tour_length + 1) 
        
    def _gsph_penalty(self):
        """Penalización basada en violación de principios GSPH"""
        if not self.nodes or len(self.chromosome) < 4:
            return 0
            
        quads, xmid, ymid = subdivide_quadrants(self.nodes)
        
        quad_sequence = []
        for gene in self.chromosome:
            node = self.nodes[gene]
            x, y = node
            if x <= xmid and y > ymid:
                quad_sequence.append('Q1')
            elif x > xmid and y > ymid:
                quad_sequence.append('Q2')
            elif x <= xmid and y <= ymid:
                quad_sequence.append('Q3')
            else:
                quad_sequence.append('Q4')
        
        penalty = 0
        for i in range(len(quad_sequence) - 1):
            if quad_sequence[i] != quad_sequence[i + 1]:
                penalty += 50 
                
        return penalty
    
    def copy(self):
        """Crear una copia del individuo"""
        new_individual = Individual(self.chromosome[:], self.nodes)
        new_individual.fitness = self.fitness
        new_individual.tour_length = self.tour_length
        return new_individual

class GeneticGSPH:
    """
    Implementación del algoritmo genético orientado a GSPH
    """
    
    def __init__(self, nodes, population_size=100, elite_size=20, mutation_rate=0.01, 
                 generations=500, tournament_size=5):
        self.nodes = nodes
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.tournament_size = tournament_size
        self.population = []
        self.best_individual = None
        self.generation_stats = []
        
    def create_initial_population(self):
        """
        1) Creación de población inicial
        
        Estrategias:
        - Individuos aleatorios
        - Individuos basados en GSPH
        - Individuos con heurísticas greedy
        """
        self.population = []
        
        # 30% de la población: soluciones GSPH
        gsph_count = int(0.3 * self.population_size)
        for _ in range(gsph_count):
            individual = self._create_gsph_individual()
            self.population.append(individual)
        
        # 40% de la población: soluciones greedy
        greedy_count = int(0.4 * self.population_size)
        for _ in range(greedy_count):
            individual = self._create_greedy_individual()
            self.population.append(individual)
            
        # 30% de la población: soluciones completamente aleatorias
        remaining = self.population_size - len(self.population)
        for _ in range(remaining):
            individual = self._create_random_individual()
            self.population.append(individual)
            
        # Calcular fitness para toda la población
        for individual in self.population:
            individual.calculate_fitness()
            
        # Ordenar por fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        self.best_individual = self.population[0].copy()
        
    def _create_gsph_individual(self):
        """Crear individuo basado en principios GSPH"""
        quads, xmid, ymid = subdivide_quadrants(self.nodes)
        
        # Crear tours por cuadrante
        chromosome = []
        quad_order = ['Q1', 'Q2', 'Q4', 'Q3']
        
        for quad_name in quad_order:
            quad_nodes = quads[quad_name]
            if not quad_nodes:
                continue
                
            quad_indices = []
            for node in quad_nodes:
                for i, n in enumerate(self.nodes):
                    if n == node:
                        quad_indices.append(i)
                        break
            
            if len(quad_indices) > 1:
                sorted_indices = self._nearest_neighbor_sort(quad_indices)
                chromosome.extend(sorted_indices)
            elif quad_indices:
                chromosome.extend(quad_indices)
                
        return Individual(chromosome, self.nodes)
    
    def _create_greedy_individual(self):
        """Crear individuo usando heurística greedy (nearest neighbor)"""
        start_idx = random.randint(0, len(self.nodes) - 1)
        chromosome = self._nearest_neighbor_sort(list(range(len(self.nodes))), start_idx)
        return Individual(chromosome, self.nodes)
    
    def _create_random_individual(self):
        """Crear individuo completamente aleatorio"""
        chromosome = list(range(len(self.nodes)))
        random.shuffle(chromosome)
        return Individual(chromosome, self.nodes)
    
    def _nearest_neighbor_sort(self, indices, start_idx=None):
        """Algoritmo nearest neighbor para ordenar nodos"""
        if not indices:
            return []
            
        if start_idx is None:
            start_idx = indices[0]
        elif start_idx not in indices:
            start_idx = indices[0]
            
        result = [start_idx]
        remaining = set(indices)
        remaining.remove(start_idx)
        
        current = start_idx
        while remaining:
            nearest = min(remaining, key=lambda x: eucl_float(self.nodes[current], self.nodes[x]))
            result.append(nearest)
            remaining.remove(nearest)
            current = nearest
            
        return result
    
    def fitness_function(self, individual):
        """
        2) Función fitness
        
        Considera:
        - Longitud total del tour
        - Adherencia a principios GSPH
        - Penalizaciones por violaciones
        """
        individual.calculate_fitness()
        return individual.fitness
    
    def tournament_selection(self, k=None):
        """
        3) Algoritmo de selección - Selección por torneo
        
        Selecciona individuos para reproducción mediante torneos
        """
        if k is None:
            k = self.tournament_size
            
        selected = []
        for _ in range(self.population_size):
            tournament = random.sample(self.population, min(k, len(self.population)))
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner.copy())
            
        return selected
    
    def create_mating_pool(self):
        """
        4) Algoritmo de creación de pool de apareamiento
        
        Combina elitismo con selección por torneo
        """
        mating_pool = []
        
        elite = self.population[:self.elite_size]
        mating_pool.extend([ind.copy() for ind in elite])
        
        remaining_size = self.population_size - len(mating_pool)
        if remaining_size > 0:
            tournament_selected = self.tournament_selection()
            mating_pool.extend(tournament_selected[:remaining_size])
            
        return mating_pool[:self.population_size]
    
    def crossover_pmx(self, parent1, parent2):
        """
        5) Algoritmo de cruzamiento - Partially Mapped Crossover (PMX)
        
        Especializado para problemas de permutación como TSP
        """
        size = len(parent1.chromosome)
        
        start = random.randint(0, size - 2)
        end = random.randint(start + 1, size)
        
        child1_chrom = [-1] * size
        child2_chrom = [-1] * size
        
        child1_chrom[start:end] = parent1.chromosome[start:end]
        child2_chrom[start:end] = parent2.chromosome[start:end]
        
        def fill_child(child_chrom, other_parent_chrom):
            for i in range(size):
                if child_chrom[i] == -1:
                    candidate = other_parent_chrom[i]
                    while candidate in child_chrom:
                        pos = child_chrom.index(candidate)
                        candidate = other_parent_chrom[pos]
                    child_chrom[i] = candidate
        
        fill_child(child1_chrom, parent2.chromosome)
        fill_child(child2_chrom, parent1.chromosome)
        
        child1 = Individual(child1_chrom, self.nodes)
        child2 = Individual(child2_chrom, self.nodes)
        
        return child1, child2
    
    def crossover_order(self, parent1, parent2):
        """
        Algoritmo de cruzamiento - Order Crossover (OX)
        """
        size = len(parent1.chromosome)
        
        start = random.randint(0, size - 2)
        end = random.randint(start + 1, size)
        
        child1_chrom = [-1] * size
        child2_chrom = [-1] * size
        
        child1_chrom[start:end] = parent1.chromosome[start:end]
        child2_chrom[start:end] = parent2.chromosome[start:end]
        
        # Llenar posiciones restantes
        def fill_remaining(child_chrom, other_parent_chrom):
            remaining = [x for x in other_parent_chrom if x not in child_chrom]
            j = 0
            for i in range(size):
                if child_chrom[i] == -1:
                    child_chrom[i] = remaining[j]
                    j += 1
        
        fill_remaining(child1_chrom, parent2.chromosome)
        fill_remaining(child2_chrom, parent1.chromosome)
        
        child1 = Individual(child1_chrom, self.nodes)
        child2 = Individual(child2_chrom, self.nodes)
        
        return child1, child2
    
    def mutate_swap(self, individual):
        """
        6) Algoritmo de mutación - Intercambio de genes
        """
        if random.random() < self.mutation_rate:
            chromosome = individual.chromosome[:]
            if len(chromosome) > 1:
                i, j = random.sample(range(len(chromosome)), 2)
                chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
            return Individual(chromosome, self.nodes)
        return individual
    
    def mutate_inversion(self, individual):
        """
        Algoritmo de mutación - Inversión de segmento
        """
        if random.random() < self.mutation_rate:
            chromosome = individual.chromosome[:]
            if len(chromosome) > 2:
                start = random.randint(0, len(chromosome) - 2)
                end = random.randint(start + 1, len(chromosome))
                chromosome[start:end] = reversed(chromosome[start:end])
            return Individual(chromosome, self.nodes)
        return individual
    
    def mutate_gsph_aware(self, individual):
        """
        Mutación especializada para GSPH - Reordena dentro de cuadrantes
        """
        if random.random() < self.mutation_rate * 2:  # Mayor probabilidad para esta mutación
            chromosome = individual.chromosome[:]
            
            # Dividir en cuadrantes
            quads, xmid, ymid = subdivide_quadrants(self.nodes)
            
            # Agrupar genes por cuadrante
            quad_groups = {'Q1': [], 'Q2': [], 'Q3': [], 'Q4': []}
            for i, gene in enumerate(chromosome):
                node = self.nodes[gene]
                x, y = node
                if x <= xmid and y > ymid:
                    quad_groups['Q1'].append((i, gene))
                elif x > xmid and y > ymid:
                    quad_groups['Q2'].append((i, gene))
                elif x <= xmid and y <= ymid:
                    quad_groups['Q3'].append((i, gene))
                else:
                    quad_groups['Q4'].append((i, gene))
            
            # Seleccionar un cuadrante aleatorio para mutar
            non_empty_quads = [q for q in quad_groups if len(quad_groups[q]) > 1]
            if non_empty_quads:
                selected_quad = random.choice(non_empty_quads)
                quad_positions = quad_groups[selected_quad]
                
                if len(quad_positions) > 1:
                    # Reordenar aleatoriamente dentro del cuadrante
                    genes = [gene for _, gene in quad_positions]
                    random.shuffle(genes)
                    
                    for i, (pos, _) in enumerate(quad_positions):
                        chromosome[pos] = genes[i]
            
            return Individual(chromosome, self.nodes)
        return individual
    
    def replacement_generational(self, mating_pool):
        """
        7) Algoritmo de reemplazo - Reemplazo generacional con elitismo
        """
        new_population = []
        
        # Conservar élite
        elite = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:self.elite_size]
        new_population.extend([ind.copy() for ind in elite])
        
        # Generar nuevos individuos
        while len(new_population) < self.population_size:
            # Seleccionar padres
            parent1 = random.choice(mating_pool)
            parent2 = random.choice(mating_pool)
            
            # Cruzamiento
            if random.random() < 0.8:  # Probabilidad de cruzamiento
                child1, child2 = self.crossover_order(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutación
            child1 = self.mutate_gsph_aware(child1)
            child1 = self.mutate_inversion(child1)
            child1 = self.mutate_swap(child1)
            
            child2 = self.mutate_gsph_aware(child2)
            child2 = self.mutate_inversion(child2)
            child2 = self.mutate_swap(child2)
            
            # Calcular fitness
            child1.calculate_fitness()
            child2.calculate_fitness()
            
            # Agregar a nueva población
            new_population.extend([child1, child2])
        
        # Truncar si es necesario
        new_population = new_population[:self.population_size]
        
        # Ordenar por fitness
        new_population.sort(key=lambda x: x.fitness, reverse=True)
        
        return new_population
    
    def evolve(self):
        """
        Algoritmo principal de evolución
        """
        # Crear población inicial
        print("Creando población inicial...")
        self.create_initial_population()
        
        print(f"Generación 0: Mejor fitness = {self.best_individual.fitness:.6f}, "
              f"Longitud = {self.best_individual.tour_length:.2f}")
        
        for generation in range(self.generations):
            # Crear pool de apareamiento
            mating_pool = self.create_mating_pool()
            
            # Crear nueva población
            self.population = self.replacement_generational(mating_pool)
            
            # Actualizar mejor individuo
            current_best = self.population[0]
            if current_best.fitness > self.best_individual.fitness:
                self.best_individual = current_best.copy()
            
            # Estadísticas de generación
            avg_fitness = sum(ind.fitness for ind in self.population) / len(self.population)
            self.generation_stats.append({
                'generation': generation + 1,
                'best_fitness': current_best.fitness,
                'best_length': current_best.tour_length,
                'avg_fitness': avg_fitness
            })
            
            # Imprimir progreso
            if (generation + 1) % 50 == 0 or generation == self.generations - 1:
                print(f"Generación {generation + 1}: Mejor fitness = {current_best.fitness:.6f}, "
                      f"Longitud = {current_best.tour_length:.2f}, "
                      f"Promedio = {avg_fitness:.6f}")
        
        return self.best_individual
    
    def get_best_tour(self):
        """Obtener el mejor tour encontrado"""
        if self.best_individual:
            tour_nodes = [self.nodes[i] for i in self.best_individual.chromosome]
            return tour_nodes, self.best_individual.tour_length
        return [], float('inf')

def optimize_route_genetic(depot, orders, population_size=100, generations=300):
    """
    Función de interfaz para optimización genética con GSPH
    
    Entrada:
      - depot: {id, name, lat, lng}
      - orders: [{id, lat, lng}, ...]
      - population_size: tamaño de la población
      - generations: número de generaciones
    
    Salida:
      - optimized_coords: [[lat, lng], ...]
      - total_length: float
    """
    # Preparar nodos
    nodes = [(depot["lat"], depot["lng"])] + [(o["lat"], o["lng"]) for o in orders]
    
    # Crear y ejecutar algoritmo genético
    genetic_algo = GeneticGSPH(
        nodes=nodes,
        population_size=population_size,
        elite_size=max(10, population_size // 5),
        mutation_rate=0.02,
        generations=generations,
        tournament_size=5
    )
    
    # Evolucionar
    start_time = time.time()
    best_individual = genetic_algo.evolve()
    evolution_time = time.time() - start_time
    
    print(f"\nTiempo de evolución: {evolution_time:.2f} segundos")
    
    # Obtener resultado
    tour_nodes, total_length = genetic_algo.get_best_tour()
    
    # Convertir a formato de salida
    optimized_coords = [[float(lat), float(lng)] for lat, lng in tour_nodes]
    
    return optimized_coords, float(total_length)

if __name__ == "__main__":
    # Ejemplo de uso con datos de prueba
    depot = {"id": "D1", "name": "Depot Central", "lat": -33.447487, "lng": -70.673676}
    
    orders = [
        {"id": "O1", "lat": -33.441000, "lng": -70.680000},
        {"id": "O2", "lat": -33.450000, "lng": -70.670000},
        {"id": "O3", "lat": -33.455000, "lng": -70.665000},
        {"id": "O4", "lat": -33.445000, "lng": -70.685000},
        {"id": "O5", "lat": -33.440000, "lng": -70.675000},
        {"id": "O6", "lat": -33.452000, "lng": -70.668000},
        {"id": "O7", "lat": -33.448000, "lng": -70.672000},
        {"id": "O8", "lat": -33.443000, "lng": -70.678000},
    ]
    
    print("Ejecutando optimización genética GSPH...")
    coords, length = optimize_route_genetic(depot, orders, population_size=50, generations=100)
    
    print(f"\nLongitud final del tour: {length:.2f}")
    print("Coordenadas optimizadas:")
    for i, (lat, lng) in enumerate(coords):
        print(f"  {i+1}: ({lat:.6f}, {lng:.6f})")