import random
import numpy as np
import time
from math import hypot
from copy import deepcopy
from gsph_algorithm import euclidean_distance, eucl_float, subdivide_quadrants, best_frontier_pair, polish_fast

class Individual:
    def __init__(self, chromosome=None, nodes=None):
        self.chromosome = chromosome if chromosome else []  
        self.fitness = None
        self.tour_length = None
        self.nodes = nodes
        self.quadrant_structure = None 
        
    def calculate_fitness(self, depot=None):
        if not self.chromosome or len(self.chromosome) < 1:
            self.fitness = 0
            self.tour_length = float('inf')
            return
            
        total_length = 0
        
        if depot:
            first_order = self.nodes[self.chromosome[0]]
            total_length += eucl_float(depot, first_order)
        
        for i in range(len(self.chromosome) - 1):
            current = self.nodes[self.chromosome[i]]
            next_node = self.nodes[self.chromosome[i + 1]]
            total_length += eucl_float(current, next_node)
            
        if depot:
            last_order = self.nodes[self.chromosome[-1]]
            total_length += eucl_float(last_order, depot)
            
        penalty = self._gsph_penalty()
        
        self.tour_length = total_length + penalty
        self.fitness = 1.0 / (self.tour_length + 1) 
        
    def _gsph_penalty(self):
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
        new_individual = Individual(self.chromosome[:], self.nodes)
        new_individual.fitness = self.fitness
        new_individual.tour_length = self.tour_length
        return new_individual

class GeneticGSPH:
    def __init__(self, nodes, depot=None, population_size=100, elite_size=20, mutation_rate=0.01, 
                 generations=500, tournament_size=5):
        self.nodes = nodes
        self.depot = depot
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.tournament_size = tournament_size
        self.population = []
        self.best_individual = None
        self.generation_stats = []
        
    def create_initial_population(self):
        self.population = []
        
        gsph_count = int(0.3 * self.population_size)
        for _ in range(gsph_count):
            individual = self._create_gsph_individual()
            self.population.append(individual)
        
        greedy_count = int(0.4 * self.population_size)
        for _ in range(greedy_count):
            individual = self._create_greedy_individual()
            self.population.append(individual)
            
        remaining = self.population_size - len(self.population)
        for _ in range(remaining):
            individual = self._create_random_individual()
            self.population.append(individual)
            
        for individual in self.population:
            individual.calculate_fitness(self.depot)
            
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        self.best_individual = self.population[0].copy()
        
    def _create_gsph_individual(self):
        quads, xmid, ymid = subdivide_quadrants(self.nodes)
        
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
        start_idx = random.randint(0, len(self.nodes) - 1)
        chromosome = self._nearest_neighbor_sort(list(range(len(self.nodes))), start_idx)
        return Individual(chromosome, self.nodes)
    
    def _create_random_individual(self):
        chromosome = list(range(len(self.nodes)))
        random.shuffle(chromosome)
        return Individual(chromosome, self.nodes)
    
    def _nearest_neighbor_sort(self, indices, start_idx=None):
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
        
        individual.calculate_fitness(self.depot)
        return individual.fitness
    
    def tournament_selection(self, k=None):
        
        if k is None:
            k = self.tournament_size
            
        selected = []
        for _ in range(self.population_size):
            tournament = random.sample(self.population, min(k, len(self.population)))
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner.copy())
            
        return selected
    
    def create_mating_pool(self):
        
        mating_pool = []
        
        elite = self.population[:self.elite_size]
        mating_pool.extend([ind.copy() for ind in elite])
        
        remaining_size = self.population_size - len(mating_pool)
        if remaining_size > 0:
            tournament_selected = self.tournament_selection()
            mating_pool.extend(tournament_selected[:remaining_size])
            
        return mating_pool[:self.population_size]
    
    def crossover_pmx(self, parent1, parent2):
        
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
        
        size = len(parent1.chromosome)
        
        start = random.randint(0, size - 2)
        end = random.randint(start + 1, size)
        
        child1_chrom = [-1] * size
        child2_chrom = [-1] * size
        
        child1_chrom[start:end] = parent1.chromosome[start:end]
        child2_chrom[start:end] = parent2.chromosome[start:end]

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
        
        if random.random() < self.mutation_rate:
            chromosome = individual.chromosome[:]
            if len(chromosome) > 1:
                i, j = random.sample(range(len(chromosome)), 2)
                chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
            return Individual(chromosome, self.nodes)
        return individual
    
    def mutate_inversion(self, individual):
        
        if random.random() < self.mutation_rate:
            chromosome = individual.chromosome[:]
            if len(chromosome) > 2:
                start = random.randint(0, len(chromosome) - 2)
                end = random.randint(start + 1, len(chromosome))
                chromosome[start:end] = reversed(chromosome[start:end])
            return Individual(chromosome, self.nodes)
        return individual
    
    def mutate_gsph_aware(self, individual):
        
        if random.random() < self.mutation_rate * 2:
            chromosome = individual.chromosome[:]

            quads, xmid, ymid = subdivide_quadrants(self.nodes)

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

            non_empty_quads = [q for q in quad_groups if len(quad_groups[q]) > 1]
            if non_empty_quads:
                selected_quad = random.choice(non_empty_quads)
                quad_positions = quad_groups[selected_quad]
                
                if len(quad_positions) > 1:

                    genes = [gene for _, gene in quad_positions]
                    random.shuffle(genes)
                    
                    for i, (pos, _) in enumerate(quad_positions):
                        chromosome[pos] = genes[i]
            
            return Individual(chromosome, self.nodes)
        return individual
    
    def replacement_generational(self, mating_pool):
        
        new_population = []

        elite = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:self.elite_size]
        new_population.extend([ind.copy() for ind in elite])

        while len(new_population) < self.population_size:

            parent1 = random.choice(mating_pool)
            parent2 = random.choice(mating_pool)

            if random.random() < 0.8:
                child1, child2 = self.crossover_order(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            child1 = self.mutate_gsph_aware(child1)
            child1 = self.mutate_inversion(child1)
            child1 = self.mutate_swap(child1)
            
            child2 = self.mutate_gsph_aware(child2)
            child2 = self.mutate_inversion(child2)
            child2 = self.mutate_swap(child2)

            child1.calculate_fitness(self.depot)
            child2.calculate_fitness(self.depot)

            new_population.extend([child1, child2])

        new_population = new_population[:self.population_size]

        new_population.sort(key=lambda x: x.fitness, reverse=True)
        
        return new_population
    
    def evolve(self):
        
        print("Creando población inicial...")
        self.create_initial_population()
        
        print(f"Generación 0: Mejor fitness = {self.best_individual.fitness:.6f}, "
              f"Longitud = {self.best_individual.tour_length:.2f}")
        
        for generation in range(self.generations):
            mating_pool = self.create_mating_pool()
            
            self.population = self.replacement_generational(mating_pool)
            
            current_best = self.population[0]
            if current_best.fitness > self.best_individual.fitness:
                self.best_individual = current_best.copy()
            
            avg_fitness = sum(ind.fitness for ind in self.population) / len(self.population)
            self.generation_stats.append({
                'generation': generation + 1,
                'best_fitness': current_best.fitness,
                'best_length': current_best.tour_length,
                'avg_fitness': avg_fitness
            })
            
            if (generation + 1) % 50 == 0 or generation == self.generations - 1:
                print(f"Generación {generation + 1}: Mejor fitness = {current_best.fitness:.6f}, "
                      f"Longitud = {current_best.tour_length:.2f}, "
                      f"Promedio = {avg_fitness:.6f}")
        
        return self.best_individual
    
    def get_best_tour(self):
        
        if self.best_individual:
            tour_nodes = [self.nodes[i] for i in self.best_individual.chromosome]
            return tour_nodes, self.best_individual.tour_length
        return [], float('inf')

def optimize_route_genetic(depot, orders, population_size=100, generations=300):
    

    depot_coords = (depot["lat"], depot["lng"])
    order_coords = [(o["lat"], o["lng"]) for o in orders]
    order_ids = [o["id"] for o in orders]

    genetic_algo = GeneticGSPH(
        nodes=order_coords,
        depot=depot_coords,
        population_size=population_size,
        elite_size=max(10, population_size // 5),
        mutation_rate=0.02,
        generations=generations,
        tournament_size=5
    )
    
    start_time = time.time()
    best_individual = genetic_algo.evolve()
    evolution_time = time.time() - start_time
    
    print(f"\nTiempo de evolución: {evolution_time:.2f} segundos")

    best_sequence = best_individual.chromosome

    optimized_coords = [
        [float(depot["lat"]), float(depot["lng"])]
    ]

    for idx in best_sequence:
        order_coord = order_coords[idx]
        optimized_coords.append([float(order_coord[0]), float(order_coord[1])])

    optimized_coords.append([float(depot["lat"]), float(depot["lng"])])

    order_sequence = [order_ids[idx] for idx in best_sequence]

    total_length = 0
    for i in range(len(optimized_coords) - 1):
        curr = optimized_coords[i]
        next_coord = optimized_coords[i + 1]
        total_length += eucl_float(curr, next_coord)
    
    return optimized_coords, float(total_length), order_sequence

if __name__ == "__main__":
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
    coords, length, sequence = optimize_route_genetic(depot, orders, population_size=50, generations=100)
    
    print(f"\nLongitud final del tour: {length:.2f}")
    print("Coordenadas optimizadas:")
    for i, (lat, lng) in enumerate(coords):
        print(f"  {i+1}: ({lat:.6f}, {lng:.6f})")
    
    print("\nSecuencia de órdenes optimizada:")
    print(f"  {sequence}")