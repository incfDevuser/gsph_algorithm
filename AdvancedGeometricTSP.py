"""

Advanced Geometric TSP (SPARC) Solver with Intelligent Lookahead and Multi-Partitioning

::Abstract
This implementation presents an advanced geometric Traveling Salesman Problem (TSP) 
solver that combines intelligent lookahead strategies, adaptive multi-way space 
partitioning, and optimized local search techniques. The algorithm incorporates
novel anchor selection methods and connectivity-aware path construction to achieve
improved solution quality for geometric TSP instances.

::Literature Foundation and Related Work

This work builds upon several established algorithmic foundations:

Classical TSP Approaches:
- Christofides, N. (1976). "Worst-case analysis of a new heuristic for the travelling salesman problem." Management Sciences Research Report 388, Carnegie-Mellon University.
- Lin, S., & Kernighan, B. W. (1973). "An effective heuristic algorithm for the traveling-salesman problem." Operations Research, 21(2), 498-516.

Local Search Optimization:
- Croes, G. A. (1958). "A method for solving traveling-salesman problems." Operations Research, 6(6), 791-812.
- Or, I. (1976). "Traveling salesman-type combinatorial problems and their relation to the logistics of regional blood banking." PhD thesis, Northwestern University.

Geometric and Spatial Partitioning:
- Bentley, J. L. (1992). "Fast algorithms for geometric traveling salesman problems." ORSA Journal on Computing, 4(4), 387-411.
- Arora, S. (1998). "Polynomial time approximation schemes for Euclidean traveling salesman and other geometric problems." Journal of the ACM, 45(5), 753-782.

Divide-and-Conquer Methods:
- Karp, R. M. (1977). "Probabilistic analysis of partitioning algorithms for the traveling-salesman problem in the plane." Mathematics of Operations Research, 2(3), 209-224.



::Implementation Notes and Areas for Future Enhancement

:Clustering Coefficient Implementation Review

Current Status: The clustering coefficient calculation in `calculate_clustering_coefficient()` has the right conceptual approach but needs refinement:

"""

import math
import random
import time
from typing import List, Tuple, Optional, Set
from collections import defaultdict
import heapq
import matplotlib.pyplot as plt


class AdvancedGeometricTSP:
    """

    Advanced Geometric TSP with Intelligent Lookahead and Multiple Partitioning Strategies
    
    Key Innovations:
    1. Smart anchor selection (random + farthest strategy)
    2. Multi-level space partitioning (2-way, 4-way, 6-way, 8-way)
    3. Lookahead path planning (considers future 4-5 steps)
    4. Distance-aware selection with connectivity scoring
    5. Adaptive strategy selection based on point distribution
    6. Multiple optimization passes

    """
    
    def __init__(self, lookahead_depth=7, enable_adaptive=True):
        self.lookahead_depth = lookahead_depth
        self.enable_adaptive = enable_adaptive
        self.stats = {
            'partitions_created': 0,
            'lookahead_evaluations': 0,
            'strategy_switches': 0,
            'local_improvements': 0
        }
    
    def distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Euclidean distance between two points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def analyze_point_distribution(self, points: List[Tuple[float, float]]) -> dict:
        """Analyze point distribution to choose optimal strategy"""
        n = len(points)
        if n < 4:
            return {'type': 'simple', 'clustering': 1.0, 'spread': 1.0}
        
        # Calculate spread ratio
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                distances.append(self.distance(points[i], points[j]))
        
        max_dist = max(distances)
        avg_dist = sum(distances) / len(distances)
        spread_ratio = max_dist / avg_dist if avg_dist > 0 else 1.0
        
        # Calculate clustering coefficient
        clustering = self.calculate_clustering_coefficient(points)
          

        # Determine best partitioning strategy
        if spread_ratio > 3.0:
            partition_type = 'radial'  # Use 6-way or 8-way
        elif clustering > 0.7:
            partition_type = 'cluster'  # Use adaptive clustering (just for refernce)
        else:
            partition_type = 'geometric'  # Use 2-way or 4-way
        

        return {
            'type': partition_type,
            'clustering': clustering,
            'spread': spread_ratio,
            'optimal_partitions': min(8, max(2, n // 6))
        }
    
    def calculate_clustering_coefficient(self, points: List[Tuple[float, float]]) -> float:
        """Calculate clustering coefficient of points"""
        n = len(points)
        if n < 3:
            return 0.0
        
        # Use k-nearest neighbours approach
        k = min(3, n - 1)
        total_clustering = 0.0
        
        for i in range(n):
            # Find k nearest neighbours
            distances = [(self.distance(points[i], points[j]), j) for j in range(n) if j != i]
            distances.sort()
            neighbors = [j for _, j in distances[:k]]
            
            # Count edges between neighbours
            edges = 0
            possible_edges = k * (k - 1) // 2
            

            if possible_edges > 0:
                for j in range(len(neighbors)):
                    for l in range(j + 1, len(neighbors)):
                        if self.distance(points[neighbors[j]], points[neighbors[l]]) < distances[k-1][0] * 1.5:
                            edges += 1
                
                total_clustering += edges / possible_edges
        
        return total_clustering / n
    
    def smart_anchor_selection(self, points: List[Tuple[float, float]], indices: List[int]) -> Tuple[int, int]:
        """Smart anchor selection: random first, then farthest"""
        if len(indices) < 2:
            return indices[0] if indices else 0, indices[0] if indices else 0
        
        # Strategy 1: Random first point (your idea)
        first_anchor = random.choice(indices)
        
        # Strategy 2: Find farthest point from first anchor
        max_dist = -1
        second_anchor = indices[0]
        
        for idx in indices:
            if idx != first_anchor:
                dist = self.distance(points[first_anchor], points[idx])
                if dist > max_dist:
                    max_dist = dist
                    second_anchor = idx
        
        return first_anchor, second_anchor
    
    def multi_way_partition(self, points: List[Tuple[float, float]], indices: List[int], 
                           anchor_a: int, anchor_b: int, num_partitions: int = 4) -> List[List[int]]:
        """Multi-way space partitioning (2, 4, 6, or 8 way)"""
        if len(indices) <= 2:
            return [indices]
        
        A, B = points[anchor_a], points[anchor_b]
        midpoint = ((A[0] + B[0]) / 2, (A[1] + B[1]) / 2)
        
        # Direction vector and perpendicular
        direction = (B[0] - A[0], B[1] - A[1])
        dir_norm = math.sqrt(direction[0]**2 + direction[1]**2)
        
        if dir_norm < 1e-10:
            return [indices]
        
        unit_dir = (direction[0] / dir_norm, direction[1] / dir_norm)
        perp = (-unit_dir[1], unit_dir[0])
        
        if num_partitions == 2:
            return self.two_way_partition(points, indices, anchor_a, anchor_b, midpoint, unit_dir)
        elif num_partitions == 4:
            return self.four_way_partition(points, indices, anchor_a, anchor_b, midpoint, unit_dir, perp)
        elif num_partitions == 6:
            return self.six_way_partition(points, indices, anchor_a, anchor_b, midpoint)
        else:  # 8-way
            return self.eight_way_partition(points, indices, anchor_a, anchor_b, midpoint)
    
    def two_way_partition(self, points, indices, anchor_a, anchor_b, midpoint, unit_dir):
        """Two-way partitioning along main axis"""
        partition_1, partition_2 = [anchor_a], [anchor_b]
        
        for idx in indices:
            if idx in {anchor_a, anchor_b}:
                continue
            
            p = points[idx]
            relative_pos = (p[0] - midpoint[0], p[1] - midpoint[1])
            projection = relative_pos[0] * unit_dir[0] + relative_pos[1] * unit_dir[1]
            
            if projection >= 0:
                partition_2.append(idx)
            else:
                partition_1.append(idx)
        
        return [p for p in [partition_1, partition_2] if p]
    
    def four_way_partition(self, points, indices, anchor_a, anchor_b, midpoint, unit_dir, perp):
        """Four-way partitioning"""
        quadrants = [[] for _ in range(4)]
        
        for idx in indices:
            if idx in {anchor_a, anchor_b}:
                continue
            
            p = points[idx]
            relative_pos = (p[0] - midpoint[0], p[1] - midpoint[1])
            parallel = relative_pos[0] * unit_dir[0] + relative_pos[1] * unit_dir[1]
            perpendicular = relative_pos[0] * perp[0] + relative_pos[1] * perp[1]
            
            if parallel >= 0 and perpendicular >= 0:
                quadrants[0].append(idx)
            elif parallel < 0 and perpendicular >= 0:
                quadrants[1].append(idx)
            elif parallel < 0 and perpendicular < 0:
                quadrants[2].append(idx)
            else:
                quadrants[3].append(idx)
        
        # Distribute anchors
        quadrants[1].append(anchor_a)
        quadrants[0].append(anchor_b)
        
        return [q for q in quadrants if q]
    
    def six_way_partition(self, points, indices, anchor_a, anchor_b, center):
        """Six-way radial partitioning (hexagonal)"""
        sectors = [[] for _ in range(6)]
        
        for idx in indices:
            if idx in {anchor_a, anchor_b}:
                continue
            
            p = points[idx]
            # Calculate angle from center
            dx, dy = p[0] - center[0], p[1] - center[1]
            angle = math.atan2(dy, dx)
            if angle < 0:
                angle += 2 * math.pi
            
            # Assign to sector (60 degrees each)
            sector = int(angle / (math.pi / 3)) % 6
            sectors[sector].append(idx)
        
        # Distribute anchors
        sectors[0].append(anchor_a)
        sectors[3].append(anchor_b)
        
        return [s for s in sectors if s]
    
    def eight_way_partition(self, points, indices, anchor_a, anchor_b, center):
        """Eight-way radial partitioning (octagonal)"""
        sectors = [[] for _ in range(8)]
        
        for idx in indices:
            if idx in {anchor_a, anchor_b}:
                continue
            
            p = points[idx]
            # Calculate angle from center
            dx, dy = p[0] - center[0], p[1] - center[1]
            angle = math.atan2(dy, dx)
            if angle < 0:
                angle += 2 * math.pi
            
            # Assign to sector (45 degrees each)
            sector = int(angle / (math.pi / 4)) % 8
            sectors[sector].append(idx)
        
        # Distribute anchors
        sectors[0].append(anchor_a)
        sectors[4].append(anchor_b)
        
        return [s for s in sectors if s]
    
    def lookahead_path_evaluation(self, points: List[Tuple[float, float]], 
                                 current_pos: int, candidates: List[int], 
                                 remaining: Set[int]) -> int:
        """Evaluate candidates by looking ahead multiple steps"""
        if not candidates:
            return None
        
        best_candidate = candidates[0]
        best_score = float('inf')
        
        for candidate in candidates:
            # Score based on immediate distance
            immediate_cost = self.distance(points[current_pos], points[candidate])
            
            # Lookahead evaluation
            lookahead_cost = self.evaluate_lookahead_path(
                points, candidate, remaining - {candidate}, 
                min(self.lookahead_depth, len(remaining) - 1)
            )
            
            # Connectivity score (how well connected this choice leaves us)
            connectivity_score = self.calculate_connectivity_score(
                points, candidate, remaining - {candidate}
            )
            
            # Combined score
            total_score = immediate_cost + 0.5 * lookahead_cost + 0.3 * connectivity_score
            
            if total_score < best_score:
                best_score = total_score
                best_candidate = candidate
        
        self.stats['lookahead_evaluations'] += len(candidates)
        return best_candidate
    
    def evaluate_lookahead_path(self, points: List[Tuple[float, float]], 
                               start: int, remaining: Set[int], depth: int) -> float:
        """Recursively evaluate lookahead path quality"""
        if depth <= 0 or not remaining:
            return 0.0
        
        # For efficiency, only consider closest few points
        candidates = sorted(remaining, key=lambda x: self.distance(points[start], points[x]))[:3]
        
        min_cost = float('inf')
        for candidate in candidates:
            immediate = self.distance(points[start], points[candidate])
            future = self.evaluate_lookahead_path(
                points, candidate, remaining - {candidate}, depth - 1
            )
            total = immediate + 0.8 * future  # Discount future costs
            min_cost = min(min_cost, total)
        
        return min_cost
    
    def calculate_connectivity_score(self, points: List[Tuple[float, float]], 
                                   candidate: int, remaining: Set[int]) -> float:
        """Calculate how well connected a candidate leaves the remaining points"""
        if not remaining:
            return 0.0
        
        # Average distance from candidate to remaining points
        distances = [self.distance(points[candidate], points[r]) for r in remaining]
        avg_distance = sum(distances) / len(distances)
        
        # Penalty for leaving isolated clusters
        isolation_penalty = 0.0
        for r in remaining:
            # Find nearest neighbor distance in remaining set
            if len(remaining) > 1:
                nn_dist = min(self.distance(points[r], points[other]) 
                             for other in remaining if other != r)
                # If this point is far from its nearest neighbor, penalize
                if nn_dist > avg_distance * 1.5:
                    isolation_penalty += nn_dist
        
        return avg_distance + isolation_penalty
    
    def intelligent_path_construction(self, points: List[Tuple[float, float]], 
                                    indices: List[int], start: int, end: int) -> List[int]:
        """Construct path from start to end using lookahead strategy"""
        if len(indices) <= 2:
            return indices
        
        path = [start]
        remaining = set(indices) - {start}
        current = start
        
        while len(remaining) > 1:
            # Get candidates (nearby points + some random exploration)
            distances = [(self.distance(points[current], points[r]), r) for r in remaining]
            distances.sort()
            
            # Take closest 3-5 points as main candidates
            main_candidates = [r for _, r in distances[:min(5, len(distances))]]
            
            # Add one random candidate for exploration
            if len(remaining) > len(main_candidates):
                random_candidates = list(remaining - set(main_candidates))
                main_candidates.append(random.choice(random_candidates))
            
            # Use lookahead to select best candidate
            next_point = self.lookahead_path_evaluation(
                points, current, main_candidates, remaining
            )
            
            if next_point is None:
                next_point = min(remaining, key=lambda x: self.distance(points[current], points[x]))
            
            path.append(next_point)
            remaining.remove(next_point)
            current = next_point
        
        # Add last remaining point if exists
        if remaining:
            path.append(remaining.pop())
        
        return path
    
    def optimize_tour_with_2opt(self, points: List[Tuple[float, float]], tour: List[int]) -> List[int]:
        """Apply 2-opt improvements to tour"""
        if len(tour) < 4:
            return tour
        
    
        improved = True
        current_tour = tour[:]
        
        while improved:
            improved = False
            for i in range(len(current_tour)):
                for j in range(i + 2, len(current_tour)):
                    if j == len(current_tour) - 1 and i == 0:
                        continue
                    
                    # Calculate improvement
                    old_cost = (self.distance(points[current_tour[i]], points[current_tour[i + 1]]) +
                               self.distance(points[current_tour[j]], points[current_tour[(j + 1) % len(current_tour)]]))
                    
                    new_cost = (self.distance(points[current_tour[i]], points[current_tour[j]]) +
                               self.distance(points[current_tour[i + 1]], points[current_tour[(j + 1) % len(current_tour)]]))
                    
                    if new_cost < old_cost:
                        # Apply 2-opt swap
                        current_tour = current_tour[:i+1] + current_tour[i+1:j+1][::-1] + current_tour[j+1:]
                        improved = True
                        self.stats['local_improvements'] += 1
                        break
                if improved:
                    break
        
        return current_tour
    
    def merge_partitions_intelligently(self, points: List[Tuple[float, float]], 
                                     subtours: List[List[int]]) -> List[int]:
        """Merge partition solutions intelligently"""
        if not subtours:
            return []
        if len(subtours) == 1:
            return subtours[0]
        
        # Find optimal merging order based on geometric proximity
        merged = subtours[0]
        remaining_tours = subtours[1:]
        
        while remaining_tours:
            # Find closest tour to current merged tour
            best_tour_idx = 0
            best_cost = float('inf')
            best_connection = None
            
            for i, tour in enumerate(remaining_tours):
                # Try different connection points
                for end1 in [0, -1]:  # endpoints of merged tour
                    for end2 in [0, -1]:  # endpoints of candidate tour
                        cost = self.distance(points[merged[end1]], points[tour[end2]])
                        if cost < best_cost:
                            best_cost = cost
                            best_tour_idx = i
                            best_connection = (end1, end2)
            
            # Merge the best tour
            next_tour = remaining_tours.pop(best_tour_idx)
            merged = self.connect_tours(merged, next_tour, best_connection)
        
        return merged
    
    def connect_tours(self, tour1: List[int], tour2: List[int], 
                     connection: Tuple[int, int]) -> List[int]:
        """Connect two tours at specified endpoints"""
        end1, end2 = connection
        
        if end1 == 0 and end2 == 0:
            return tour1[::-1] + tour2
        elif end1 == 0 and end2 == -1:
            return tour1[::-1] + tour2[::-1]
        elif end1 == -1 and end2 == 0:
            return tour1 + tour2
        else:  # end1 == -1 and end2 == -1
            return tour1 + tour2[::-1]
    
    def calculate_tour_length(self, points: List[Tuple[float, float]], tour: List[int]) -> float:
        """Calculate total tour length"""
        if len(tour) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(tour)):
            total += self.distance(points[tour[i]], points[tour[(i + 1) % len(tour)]])
        return total
    
    def solve(self, points: List[Tuple[float, float]], threshold: int = 10) -> List[int]:
        """Main solving method with adaptive strategy"""
        if len(points) <= 1:
            return list(range(len(points)))
        
        # Reset stats
        self.stats = {key: 0 for key in self.stats}
        
        # Analyze point distribution
        analysis = self.analyze_point_distribution(points)
        
        # Solve recursively
        indices = list(range(len(points)))
        tour = self._solve_recursive(points, indices, threshold, analysis)
        
        # Apply final optimization
        tour = self.optimize_tour_with_2opt(points, tour)
        
        return tour
    
    def _solve_recursive(self, points: List[Tuple[float, float]], indices: List[int], 
                        threshold: int, analysis: dict) -> List[int]:
        """Recursive solving with adaptive strategy"""
        if len(indices) <= threshold:
            if len(indices) <= 3:
                return indices
            
            # Use intelligent path construction for small instances
            anchor_a, anchor_b = self.smart_anchor_selection(points, indices)
            return self.intelligent_path_construction(points, indices, anchor_a, anchor_b)
        
        # Smart anchor selection
        anchor_a, anchor_b = self.smart_anchor_selection(points, indices)
        
        # Adaptive partitioning based on analysis
        num_partitions = analysis.get('optimal_partitions', 4)
        partitions = self.multi_way_partition(points, indices, anchor_a, anchor_b, num_partitions)
        
        self.stats['partitions_created'] += len(partitions)
        
        # Solve each partition
        subtours = []
        for partition in partitions:
            if len(partition) > threshold:
                subtour = self._solve_recursive(points, partition, threshold, analysis)
            else:
                # Small partition - use intelligent construction
                if len(partition) >= 2:
                    start, end = partition[0], partition[-1]
                    subtour = self.intelligent_path_construction(points, partition, start, end)
                else:
                    subtour = partition
            subtours.append(subtour)
        
        # Intelligent merging
        return self.merge_partitions_intelligently(points, subtours)

def plot_tour(points: List[Tuple[float, float]], tour: List[int], 
            title: str = "TSP Tour", show: bool = True, 
            filename: Optional[str] = None, annotate: bool = True):
    """

    Visualize the TSP tour solution
    :param points: Original list of (x, y) coordinates
    :param tour: List of indices representing the tour order
    :param title: Plot title
    :param show: Display the plot immediately
    :param filename: Save plot to file (optional)
    :param annotate: Label points with indices

    """
    if not tour:
        print("Empty tour provided")
        return

    # Prepare coordinates in tour order
    x = [points[i][0] for i in tour]
    y = [points[i][1] for i in tour]
    x.append(x[0])  # Close the loop
    y.append(y[0])

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=1, label='Tour Path')
    plt.scatter(x, y, s=50, c='blue', alpha=0.7)
    plt.scatter([x[0]], [y[0]], s=100, c='red', marker='*', label='Start/End')

    # Add point labels if requested
    if annotate and len(tour) <= 100:
        for i, (px, py) in enumerate(points):
            plt.annotate(str(i), (px, py), xytext=(3, 3), 
                        textcoords='offset points', fontsize=8)

    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


SEED= 42
random_points = 200

if __name__ == "__main__":
    
    print("Advanced Geometric TSP Solver Demo")
    print("=" * 40)
    
    # Generate a small test case
    random.seed(SEED)
    demo_points = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(random_points)]
    
    solver = AdvancedGeometricTSP(lookahead_depth=3)
    start_time = time.time()
    tour = solver.solve(demo_points)
    solve_time = (time.time() - start_time) * 1000
    
    tour_length = solver.calculate_tour_length(demo_points, tour)
    
    print(f"Tour length: {tour_length:.2f}")
    print(f"Solve time: {solve_time:.1f} ms")
    print(f"Tour: {tour}")
    print("\nSolver Statistics:")
    for key, value in solver.stats.items():
        print(f"  {key}: {value}")
    
    plot_tour(demo_points,tour)