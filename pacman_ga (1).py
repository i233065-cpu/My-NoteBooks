"""
PACMAN MAP GENERATOR - COMPLETE IMPLEMENTATION
===============================================
Genetic Algorithm for evolving high-quality Pacman maps.

This module implements a hybrid AI approach combining:
- Genetic Algorithm for population-based optimization
- BFS for connectivity validation and path finding
- DFS for loop detection and structural analysis
- Heuristic scoring for multi-metric fitness evaluation

Authors: Muhammad Hamza Khan, Muhammad Irfan, Hassan Qureshi
Roll Numbers: 23I-3032, 23I-3065, 23I-3029
Section: A

Time Complexity: O(G × P × N) where G=generations, P=population size, N=grid cells
Space Complexity: O(P × N) for population storage
"""

import random
import copy
import time
import logging
from collections import deque
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
from enum import IntEnum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class CellType(IntEnum):
    """Cell type enumeration for grid encoding."""
    WALL = 0
    PATH = 1
    START = 2
    END = 3
    PELLET = 4      # Reserved for future enhancement
    GHOST = 5       # Reserved for future enhancement


class Direction:
    """Movement directions for BFS/DFS traversal."""
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)
    
    @classmethod
    def all(cls) -> List[Tuple[int, int]]:
        """Return all four directions."""
        return [cls.UP, cls.DOWN, cls.LEFT, cls.RIGHT]


# =============================================================================
# MAP ANALYZER - BFS AND DFS FOR VALIDATION AND ANALYSIS
# =============================================================================

class MapAnalyzer:
    """
    Analyzes map structure using BFS and DFS algorithms.
    
    BFS Usage:
        - Connectivity validation from start to end
        - Shortest path length calculation
        - Reachable cell identification
    
    DFS Usage:
        - Loop/cycle detection for gameplay complexity
        - Branch point identification for decision richness
        - Path depth analysis
    """
    
    def __init__(self, grid: List[List[int]]):
        """
        Initialize analyzer with a grid.
        
        Args:
            grid: 2D list representing the map (15×30)
        """
        self.grid = grid
        self.rows = len(grid) if grid else 0
        self.cols = len(grid[0]) if self.rows > 0 else 0
        
    def find_start_end(self) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """
        Locate start (value 2) and end (value 3) positions.
        
        Returns:
            Tuple of (start_position, end_position)
        """
        start = None
        end = None
        
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i][j] == CellType.START:
                    start = (i, j)
                elif self.grid[i][j] == CellType.END:
                    end = (i, j)
                    
        return start, end
    
    def bfs_reachability(self, start: Tuple[int, int]) -> Tuple[Set[Tuple[int, int]], Dict[Tuple[int, int], int]]:
        """
        Perform BFS to find all reachable cells from start position.
        
        Uses queue-based traversal with O(N) time complexity.
        Essential for validating map connectivity.
        
        Args:
            start: (row, col) coordinates of start position
            
        Returns:
            reachable: Set of coordinates reachable from start
            distances: Dict mapping coordinates to shortest path distance
        """
        if not start:
            return set(), {}
            
        reachable = set()
        distances = {}
        queue = deque()
        
        queue.append(start)
        reachable.add(start)
        distances[start] = 0
        
        while queue:
            current = queue.popleft()
            current_dist = distances[current]
            
            for dr, dc in Direction.all():
                nr, nc = current[0] + dr, current[1] + dc
                neighbor = (nr, nc)
                
                # Boundary check
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                
                # Wall check (cell value 0 = WALL)
                if self.grid[nr][nc] == CellType.WALL:
                    continue
                
                if neighbor not in reachable:
                    reachable.add(neighbor)
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
                    
        return reachable, distances
    
    def get_shortest_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path between start and end using BFS.
        
        Args:
            start: Starting position
            end: Target position
            
        Returns:
            List of coordinates forming the shortest path, or None if no path exists
        """
        if not start or not end:
            return None
            
        reachable, distances, parent = self.bfs_with_parent(start)
        
        if end not in reachable:
            return None
            
        # Reconstruct path from end to start
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = parent.get(current)
            
        return list(reversed(path))
    
    def bfs_with_parent(self, start: Tuple[int, int]) -> Tuple[Set, Dict, Dict]:
        """
        BFS that also tracks parent nodes for path reconstruction.
        
        Returns:
            reachable: Set of reachable coordinates
            distances: Dict of distances from start
            parent: Dict of parent nodes
        """
        reachable = set()
        distances = {}
        parent = {}
        queue = deque()
        
        queue.append(start)
        reachable.add(start)
        distances[start] = 0
        parent[start] = None
        
        while queue:
            current = queue.popleft()
            current_dist = distances[current]
            
            for dr, dc in Direction.all():
                nr, nc = current[0] + dr, current[1] + dc
                neighbor = (nr, nc)
                
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                if self.grid[nr][nc] == CellType.WALL:
                    continue
                    
                if neighbor not in reachable:
                    reachable.add(neighbor)
                    distances[neighbor] = current_dist + 1
                    parent[neighbor] = current
                    queue.append(neighbor)
                    
        return reachable, distances, parent
    
    def find_loops(self, start: Tuple[int, int], time_limit: float = 0.5) -> List[List[Tuple[int, int]]]:
        """
        Detect cycles/loops in the map using DFS with time limiting.
        
        Prevents exponential time explosion by imposing a time budget.
        Identifies valid cycles of length >= 4 for gameplay complexity.
        
        Args:
            start: Starting position for DFS traversal
            time_limit: Maximum execution time in seconds
            
        Returns:
            List of cycles (each cycle is list of coordinates)
        """
        if not start:
            return []
            
        start_time = time.time()
        cycles = []
        visited_in_stack = set()
        
        def dfs(current: Tuple[int, int], path: List[Tuple[int, int]]):
            """Recursive DFS with time limit check."""
            if time.time() - start_time > time_limit:
                return
            
            for neighbor in self._get_neighbors(current):
                # Back edge detection - cycle found
                if neighbor in path:
                    cycle_start_idx = path.index(neighbor)
                    if len(path) - cycle_start_idx >= 4:
                        cycle = path[cycle_start_idx:] + [neighbor]
                        if not self._is_duplicate_cycle(cycle, cycles):
                            cycles.append(cycle)
                elif neighbor not in visited_in_stack:
                    visited_in_stack.add(neighbor)
                    dfs(neighbor, path + [neighbor])
                    visited_in_stack.remove(neighbor)
        
        visited_in_stack.add(start)
        dfs(start, [start])
        
        return cycles
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring cells (non-wall, within bounds)."""
        neighbors = []
        for dr, dc in Direction.all():
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if self.grid[nr][nc] != CellType.WALL:
                    neighbors.append((nr, nc))
        return neighbors
    
    def _is_duplicate_cycle(self, cycle: List[Tuple[int, int]], existing: List[List[Tuple[int, int]]]) -> bool:
        """Check if a cycle already exists in the list."""
        cycle_set = set(cycle)
        for existing_cycle in existing:
            if set(existing_cycle) == cycle_set:
                return True
        return False
    
    def find_branch_points(self) -> List[Tuple[int, int]]:
        """
        Identify branching points (intersections) in the map.
        
        Branch points are cells with 3 or more non-wall neighbors.
        These create decision points for gameplay.
        
        Returns:
            List of coordinates where degree >= 3
        """
        branch_points = []
        
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i][j] != CellType.WALL:
                    neighbor_count = len(self._get_neighbors((i, j)))
                    if neighbor_count >= 3:
                        branch_points.append((i, j))
                        
        return branch_points
    
    def count_2x2_wall_blocks(self) -> int:
        """
        Count 2x2 blocks of walls.
        
        Large wall blocks are penalized as they reduce gameplay quality.
        
        Returns:
            Number of 2x2 wall blocks
        """
        count = 0
        for i in range(self.rows - 1):
            for j in range(self.cols - 1):
                if (self.grid[i][j] == CellType.WALL and 
                    self.grid[i+1][j] == CellType.WALL and
                    self.grid[i][j+1] == CellType.WALL and 
                    self.grid[i+1][j+1] == CellType.WALL):
                    count += 1
        return count
    
    def count_non_wall_cells(self) -> int:
        """Count total traversable cells (non-walls)."""
        count = 0
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i][j] != CellType.WALL:
                    count += 1
        return count
    
    def get_wall_density(self) -> float:
        """Calculate ratio of walls to total cells."""
        wall_count = 0
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i][j] == CellType.WALL:
                    wall_count += 1
        return wall_count / (self.rows * self.cols)


# =============================================================================
# FITNESS EVALUATOR - HEURISTIC SCORING FUNCTION
# =============================================================================

@dataclass
class FitnessWeights:
    """Configuration weights for fitness components."""
    path_weight: float = 1.2
    complexity_weight: float = 6.0
    loop_weight: float = 12.0
    variety_bonus: float = 20.0
    isolation_penalty: float = 2.0
    wall_penalty: float = 3.0
    loop_target: int = 3


class FitnessEvaluator:
    """
    Evaluates map quality using BFS and DFS analysis.
    
    The heuristic combines multiple metrics:
    
    REWARD COMPONENTS:
    - Path Score: Rewards longer shortest paths (up to 45 points)
    - Complexity Score: Rewards branch points/intersections (6 points each)
    - Loop Score: Rewards cycles in map (12 points each)
    - Variety Bonus: Rewards different forward/backward paths (20 points)
    
    PENALTY COMPONENTS:
    - Isolation Penalty: Punishes unreachable cells (2 points per cell)
    - Thick Wall Penalty: Punishes 2x2 wall blocks (3 points each)
    - Loop Deficit: Punishes maps with fewer than 3 loops (8 points per missing loop)
    """
    
    def __init__(self, weights: Optional[Dict] = None):
        """
        Initialize evaluator with configurable weights.
        
        Args:
            weights: Dictionary of weight factors (uses defaults if None)
        """
        if weights:
            self.weights = FitnessWeights(**weights)
        else:
            self.weights = FitnessWeights()
            
        self.rows = 15
        self.cols = 30
        
    def evaluate(self, grid: List[List[int]]) -> float:
        """
        Calculate comprehensive fitness score for a map.
        
        Args:
            grid: 2D list representing map (0=wall, 1=path, 2=start, 3=end)
            
        Returns:
            Fitness score between 0 and 100
        """
        # === VALIDATION PHASE ===
        analyzer = MapAnalyzer(grid)
        start, end = analyzer.find_start_end()
        
        # Invalid if missing start or end
        if not start or not end:
            logger.debug("Map invalid: missing start or end")
            return 0.0
        
        # === BFS ANALYSIS ===
        reachable, distances = analyzer.bfs_reachability(start)
        
        # Invalid if end is not reachable
        if end not in reachable:
            logger.debug("Map invalid: end not reachable from start")
            return 0.0
        
        # === METRIC CALCULATION ===
        # 1. Path Score - rewards longer shortest paths
        shortest_path_length = distances[end]
        path_score = min(shortest_path_length * self.weights.path_weight, 45.0)
        
        # 2. Complexity Score - rewards branching points
        branch_points = analyzer.find_branch_points()
        complexity_score = len(branch_points) * self.weights.complexity_weight
        
        # 3. Loop Score - rewards cycles (enhances gameplay)
        loops = analyzer.find_loops(start, time_limit=0.5)
        loop_score = len(loops) * self.weights.loop_weight
        
        # 4. Variety Bonus - different forward/backward paths
        variety_bonus = self._check_path_variety(analyzer, start, end)
        variety_score = self.weights.variety_bonus if variety_bonus else 0.0
        
        # 5. Isolation Penalty - punish unreachable cells
        total_traversable = analyzer.count_non_wall_cells()
        isolated_cells = total_traversable - len(reachable)
        isolation_penalty = min(isolated_cells * self.weights.isolation_penalty, 30.0)
        
        # 6. Thick Wall Penalty - punish 2x2 wall blocks
        thick_walls = analyzer.count_2x2_wall_blocks()
        wall_penalty = thick_walls * self.weights.wall_penalty
        
        # 7. Loop Deficit Penalty - enforce minimum loops
        loop_deficit = max(0, self.weights.loop_target - len(loops))
        loop_deficit_penalty = loop_deficit * 8.0
        
        # === AGGREGATION ===
        base_score = path_score + complexity_score + loop_score + variety_score
        penalties = isolation_penalty + wall_penalty + loop_deficit_penalty
        raw_score = base_score - penalties
        
        # Normalize to [0, 100]
        final_score = max(0.0, min(100.0, raw_score))
        
        return final_score
    
    def _check_path_variety(self, analyzer: MapAnalyzer, start: Tuple[int, int], 
                           end: Tuple[int, int]) -> bool:
        """
        Check if forward and backward paths are different.
        
        Different paths indicate better map design with multiple routes.
        """
        # Get forward path
        forward_path = analyzer.get_shortest_path(start, end)
        
        # Get backward path (from end to start)
        backward_path = analyzer.get_shortest_path(end, start)
        
        if not forward_path or not backward_path:
            return False
            
        # Compare paths (ignoring reverse order)
        return set(forward_path) != set(backward_path)
    
    def get_detailed_scores(self, grid: List[List[int]]) -> Dict[str, float]:
        """
        Get detailed breakdown of fitness components for analysis.
        
        Returns:
            Dictionary with individual component scores
        """
        analyzer = MapAnalyzer(grid)
        start, end = analyzer.find_start_end()
        
        if not start or not end:
            return {'error': 'Invalid map - missing start or end'}
        
        reachable, distances = analyzer.bfs_reachability(start)
        
        if end not in reachable:
            return {'error': 'Invalid map - end not reachable'}
        
        loops = analyzer.find_loops(start, time_limit=0.5)
        branch_points = analyzer.find_branch_points()
        
        return {
            'path_length': distances[end],
            'path_score': min(distances[end] * self.weights.path_weight, 45.0),
            'branch_points': len(branch_points),
            'complexity_score': len(branch_points) * self.weights.complexity_weight,
            'loop_count': len(loops),
            'loop_score': len(loops) * self.weights.loop_weight,
            'wall_density': analyzer.get_wall_density(),
            'thick_walls': analyzer.count_2x2_wall_blocks(),
            'reachable_percentage': (len(reachable) / analyzer.count_non_wall_cells()) * 100
        }


# =============================================================================
# MAP GENERATOR - INITIAL POPULATION CREATION
# =============================================================================

class MapGenerator:
    """Generates initial random maps with validity constraints."""
    
    def __init__(self, rows: int = 15, cols: int = 30, wall_density: float = 0.35):
        """
        Initialize map generator.
        
        Args:
            rows: Number of rows in grid (default: 15)
            cols: Number of columns in grid (default: 30)
            wall_density: Probability of wall placement (default: 0.35)
        """
        self.rows = rows
        self.cols = cols
        self.wall_density = wall_density
        
    def generate_random(self) -> List[List[int]]:
        """
        Generate a random map with walls, paths, start, and end.
        
        Ensures:
        - Border cells are walls (Pacman can't leave grid)
        - Exactly one start and one end position
        - Path connectivity between start and end
        
        Returns:
            2D grid representing the map
        """
        max_attempts = 50
        
        for attempt in range(max_attempts):
            # Initialize grid with paths (1)
            grid = [[CellType.PATH for _ in range(self.cols)] for _ in range(self.rows)]
            
            # Set borders as walls (0)
            for i in range(self.rows):
                grid[i][0] = CellType.WALL
                grid[i][self.cols-1] = CellType.WALL
            for j in range(self.cols):
                grid[0][j] = CellType.WALL
                grid[self.rows-1][j] = CellType.WALL
            
            # Add random interior walls
            for i in range(1, self.rows-1):
                for j in range(1, self.cols-1):
                    if random.random() < self.wall_density:
                        grid[i][j] = CellType.WALL
            
            # Place start and end positions
            start_pos = self._get_random_non_wall(grid)
            end_pos = self._get_random_non_wall(grid, exclude=[start_pos])
            grid[start_pos[0]][start_pos[1]] = CellType.START
            grid[end_pos[0]][end_pos[1]] = CellType.END
            
            # Validate connectivity
            if self._is_connected(grid, start_pos, end_pos):
                logger.debug(f"Generated valid map after {attempt + 1} attempts")
                return grid
        
        # Fallback: generate simple corridor map
        logger.warning("Using fallback corridor map after max attempts")
        return self._generate_corridor_map()
    
    def _get_random_non_wall(self, grid: List[List[int]], exclude: List[Tuple[int, int]] = None) -> Tuple[int, int]:
        """Get random non-wall cell position."""
        exclude = exclude or []
        while True:
            row = random.randint(1, self.rows - 2)
            col = random.randint(1, self.cols - 2)
            if grid[row][col] != CellType.WALL and (row, col) not in exclude:
                return (row, col)
    
    def _is_connected(self, grid: List[List[int]], start: Tuple[int, int], 
                      end: Tuple[int, int]) -> bool:
        """Check if end is reachable from start using BFS."""
        visited = set()
        queue = deque([start])
        visited.add(start)
        
        while queue:
            current = queue.popleft()
            if current == end:
                return True
            
            for dr, dc in Direction.all():
                nr, nc = current[0] + dr, current[1] + dc
                neighbor = (nr, nc)
                
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if grid[nr][nc] != CellType.WALL and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
        
        return False
    
    def _generate_corridor_map(self) -> List[List[int]]:
        """
        Generate a simple but guaranteed valid corridor map.
        
        Used when random generation fails repeatedly.
        """
        grid = [[CellType.WALL for _ in range(self.cols)] for _ in range(self.rows)]
        
        # Create a winding path
        path_cells = [
            (1,1), (1,2), (1,3), (2,3), (3,3), (3,4), (3,5), (4,5), (5,5),
            (5,6), (5,7), (6,7), (7,7), (7,8), (7,9), (8,9), (9,9), (9,10),
            (9,11), (10,11), (11,11), (11,12), (11,13), (12,13), (13,13)
        ]
        
        for i, j in path_cells:
            grid[i][j] = CellType.PATH
        
        grid[1][1] = CellType.START
        grid[13][13] = CellType.END
        
        return grid


# =============================================================================
# GENETIC ALGORITHM - CORE OPTIMIZATION ENGINE
# =============================================================================

class GeneticAlgorithm:
    """
    Genetic Algorithm for evolving Pacman maps.
    
    The algorithm maintains a population of candidate maps and iteratively
    improves them through selection, crossover, and mutation operations.
    Fitness is evaluated using a multi-metric heuristic function that
    considers connectivity, path length, loops, and structural complexity.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize GA with configuration parameters.
        
        Args:
            config: Dictionary containing:
                - population_size: Number of maps per generation (default: 10)
                - mutation_rate: Probability of mutation per cell (default: 0.15)
                - crossover_rate: Probability of crossover (default: 0.8)
                - elitism_count: Number of best maps to preserve (default: 1)
                - max_generations: Maximum evolution cycles (default: 50)
                - fitness_threshold: Target fitness to stop early (default: 85.0)
                - rows: Grid height (default: 15)
                - cols: Grid width (default: 30)
                - weights: Fitness component weights (optional)
        """
        self.population_size = config.get('population_size', 10)
        self.mutation_rate = config.get('mutation_rate', 0.15)
        self.crossover_rate = config.get('crossover_rate', 0.8)
        self.elitism_count = config.get('elitism_count', 1)
        self.max_generations = config.get('max_generations', 50)
        self.fitness_threshold = config.get('fitness_threshold', 85.0)
        self.rows = config.get('rows', 15)
        self.cols = config.get('cols', 30)
        
        # Core components
        weights = config.get('weights', {})
        self.evaluator = FitnessEvaluator(weights)
        self.generator = MapGenerator(self.rows, self.cols, wall_density=0.35)
        
        # State tracking
        self.population = []
        self.best_individual = None
        self.best_fitness = 0.0
        self.generation_history = []
        self.convergence_counter = 0
        
        logger.info(f"GA initialized: pop_size={self.population_size}, "
                   f"mut_rate={self.mutation_rate}, generations={self.max_generations}")
        
    def run(self) -> List[List[int]]:
        """
        Execute the main GA evolution loop.
        
        Returns:
            Best map found during evolution.
        """
        # Initialize population
        logger.info("Initializing population...")
        self.population = self._initialize_population()
        
        # Main evolution loop
        for generation in range(self.max_generations):
            # Evaluate all individuals
            fitness_scores = self._evaluate_population()
            
            # Track best solution
            current_best_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
            current_best_fitness = fitness_scores[current_best_idx]
            current_best = copy.deepcopy(self.population[current_best_idx])
            
            # Update global best if improved
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_individual = current_best
                self.convergence_counter = 0
                logger.info(f"Generation {generation}: New best fitness = {self.best_fitness:.2f}")
            else:
                self.convergence_counter += 1
            
            # Store history
            self._record_generation_stats(generation, fitness_scores)
            
            # Early termination conditions
            if self.best_fitness >= self.fitness_threshold:
                logger.info(f"Early termination at generation {generation} - Target reached!")
                break
            
            if self.convergence_counter >= 10:
                logger.info(f"Early termination at generation {generation} - No improvement for 10 generations")
                break
            
            # Create next generation
            new_population = []
            
            # Elitism: preserve best individuals
            if self.elitism_count > 0:
                elite_indices = sorted(range(len(fitness_scores)), 
                                      key=lambda i: fitness_scores[i])[-self.elitism_count:]
                for idx in elite_indices:
                    new_population.append(copy.deepcopy(self.population[idx]))
            
            # Generate remaining individuals
            while len(new_population) < self.population_size:
                # Select parents
                parent1 = self._tournament_select(fitness_scores)
                parent2 = self._tournament_select(fitness_scores)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
                
                # Mutation
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                # Add to new population
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            self.population = new_population
        
        logger.info(f"Evolution complete! Best fitness: {self.best_fitness:.2f}")
        return self.best_individual
    
    def run_one_generation(self) -> None:
        """
        Execute a single generation of evolution.
        
        Useful for real-time visualization or interactive mode.
        """
        fitness_scores = self._evaluate_population()
        
        current_best_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
        current_best_fitness = fitness_scores[current_best_idx]
        current_best = copy.deepcopy(self.population[current_best_idx])
        
        if current_best_fitness > self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_individual = current_best
            self.convergence_counter = 0
        else:
            self.convergence_counter += 1
        
        # Create next generation
        new_population = []
        
        if self.elitism_count > 0:
            elite_indices = sorted(range(len(fitness_scores)), 
                                  key=lambda i: fitness_scores[i])[-self.elitism_count:]
            for idx in elite_indices:
                new_population.append(copy.deepcopy(self.population[idx]))
        
        while len(new_population) < self.population_size:
            parent1 = self._tournament_select(fitness_scores)
            parent2 = self._tournament_select(fitness_scores)
            
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
            
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        self.population = new_population
    
    def _initialize_population(self) -> List[List[List[int]]]:
        """Generate initial random population."""
        population = []
        for _ in range(self.population_size):
            map_grid = self.generator.generate_random()
            population.append(map_grid)
        return population
    
    def _evaluate_population(self) -> List[float]:
        """Evaluate fitness for all individuals in population."""
        fitness_scores = []
        for individual in self.population:
            score = self.evaluator.evaluate(individual)
            fitness_scores.append(score)
        return fitness_scores
    
    def _tournament_select(self, fitness_scores: List[float], k: int = 3) -> List[List[int]]:
        """
        Tournament selection for parent choice.
        
        Args:
            fitness_scores: List of fitness values for population
            k: Tournament size (higher = more selective pressure)
        
        Returns:
            Selected parent map
        """
        tournament_indices = random.sample(range(len(self.population)), k)
        best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
        return copy.deepcopy(self.population[best_idx])
    
    def _crossover(self, parent1: List[List[int]], parent2: List[List[int]]) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Block-based crossover operation.
        
        Splits maps into 5x5 blocks and swaps them between parents.
        Preserves start/end positions by not crossing over those cells.
        
        Args:
            parent1, parent2: Two parent maps
        
        Returns:
            Two child maps
        """
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        block_size = 5
        
        for i in range(0, self.rows, block_size):
            for j in range(0, self.cols, block_size):
                if random.random() < self.crossover_rate:
                    for di in range(min(block_size, self.rows - i)):
                        for dj in range(min(block_size, self.cols - j)):
                            # Don't swap start/end positions (values 2 and 3)
                            if parent1[i+di][j+dj] not in [CellType.START, CellType.END] and \
                               parent2[i+di][j+dj] not in [CellType.START, CellType.END]:
                                child1[i+di][j+dj], child2[i+di][j+dj] = \
                                    child2[i+di][j+dj], child1[i+di][j+dj]
        
        return child1, child2
    
    def _mutate(self, individual: List[List[int]]) -> List[List[int]]:
        """
        Multi-strategy mutation operator.
        
        Applies one of three mutation strategies:
        1. Cell flip: Randomly flip wall <-> path
        2. Region mutation: Randomly regenerate a block
        3. Path repair: Fix connectivity issues
        
        Returns:
            Mutated map
        """
        mutated = copy.deepcopy(individual)
        
        # Choose mutation strategy
        strategy = random.choice(['cell_flip', 'region', 'path_repair'])
        
        if strategy == 'cell_flip':
            # Flip individual cells with probability mutation_rate
            for i in range(self.rows):
                for j in range(self.cols):
                    if random.random() < self.mutation_rate:
                        # Protect borders and start/end positions
                        if i == 0 or i == self.rows-1 or j == 0 or j == self.cols-1:
                            continue
                        if mutated[i][j] in [CellType.START, CellType.END]:
                            continue
                        # Flip between wall (0) and path (1)
                        mutated[i][j] = CellType.PATH if mutated[i][j] == CellType.WALL else CellType.WALL
        
        elif strategy == 'region':
            # Regenerate a random block
            region_size = random.randint(3, 6)
            start_row = random.randint(1, self.rows - region_size - 1)
            start_col = random.randint(1, self.cols - region_size - 1)
            
            for i in range(start_row, start_row + region_size):
                for j in range(start_col, start_col + region_size):
                    if random.random() < 0.3:
                        if mutated[i][j] not in [CellType.START, CellType.END]:
                            mutated[i][j] = CellType.PATH if mutated[i][j] == CellType.WALL else CellType.WALL
        
        else:  # path_repair
            # Ensure map remains connected
            mutated = self._repair_connectivity(mutated)
        
        return mutated
    
    def _repair_connectivity(self, grid: List[List[int]]) -> List[List[int]]:
        """
        Repair connectivity issues in a map.
        
        Ensures start and end are connected by creating a path if needed.
        """
        analyzer = MapAnalyzer(grid)
        start, end = analyzer.find_start_end()
        
        if not start or not end:
            return grid
        
        reachable, _ = analyzer.bfs_reachability(start)
        
        if end in reachable:
            return grid
        
        # Create a simple path from start to end
        # This is a simplified repair - creates a straight line corridor
        repaired = copy.deepcopy(grid)
        
        # If start and end are in same row
        if start[0] == end[0]:
            for c in range(min(start[1], end[1]), max(start[1], end[1]) + 1):
                if repaired[start[0]][c] == CellType.WALL:
                    repaired[start[0]][c] = CellType.PATH
        # If start and end are in same column
        elif start[1] == end[1]:
            for r in range(min(start[0], end[0]), max(start[0], end[0]) + 1):
                if repaired[r][start[1]] == CellType.WALL:
                    repaired[r][start[1]] = CellType.PATH
        else:
            # L-shaped path
            for c in range(min(start[1], end[1]), max(start[1], end[1]) + 1):
                if repaired[start[0]][c] == CellType.WALL:
                    repaired[start[0]][c] = CellType.PATH
            for r in range(min(start[0], end[0]), max(start[0], end[0]) + 1):
                if repaired[r][end[1]] == CellType.WALL:
                    repaired[r][end[1]] = CellType.PATH
        
        return repaired
    
    def _record_generation_stats(self, generation: int, fitness_scores: List[float]) -> None:
        """Record statistics for the current generation."""
        self.generation_history.append({
            'generation': generation,
            'best_fitness': max(fitness_scores),
            'avg_fitness': sum(fitness_scores) / len(fitness_scores),
            'min_fitness': min(fitness_scores),
            'std_fitness': self._calculate_std(fitness_scores)
        })
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation of values."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return variance ** 0.5
    
    def get_statistics(self) -> Dict:
        """Get evolution statistics."""
        if not self.generation_history:
            return {}
        
        return {
            'final_best_fitness': self.best_fitness,
            'total_generations': len(self.generation_history),
            'initial_best_fitness': self.generation_history[0]['best_fitness'],
            'improvement': self.best_fitness - self.generation_history[0]['best_fitness'],
            'convergence_generations': self.convergence_counter,
            'history': self.generation_history
        }


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

def visualize_map(grid: List[List[int]]) -> str:
    """
    Create ASCII representation of map for console display.
    
    Character mapping:
    ■ = Wall (0)
    . = Path (1)
    P = Pacman Start (2)
    E = End/Goal (3)
    
    Args:
        grid: 2D list representing the map
        
    Returns:
        ASCII string representation of the map
    """
    symbols = {
        CellType.WALL: '■',
        CellType.PATH: '.',
        CellType.START: 'P',
        CellType.END: 'E'
    }
    
    if not grid or not grid[0]:
        return "Empty grid"
    
    rows = len(grid)
    cols = len(grid[0])
    
    result = []
    result.append("┌" + "─" * (cols * 2) + "┐")
    
    for row in grid:
        line = "│"
        for cell in row:
            line += symbols.get(cell, '?') + " "
        line += "│"
        result.append(line)
    
    result.append("└" + "─" * (cols * 2) + "┘")
    
    return "\n".join(result)


def visualize_map_colored(grid: List[List[int]]) -> str:
    """
    Create colored ASCII representation (ANSI colors).
    
    Requires terminal that supports ANSI escape codes.
    """
    colors = {
        CellType.WALL: '\033[44m',   # Blue background
        CellType.PATH: '\033[47m',   # White background
        CellType.START: '\033[42m',  # Green background
        CellType.END: '\033[41m'     # Red background
    }
    reset = '\033[0m'
    
    symbols = {
        CellType.WALL: ' ',
        CellType.PATH: ' ',
        CellType.START: 'P',
        CellType.END: 'E'
    }
    
    result = []
    for row in grid:
        line = ""
        for cell in row:
            color = colors.get(cell, '\033[47m')
            symbol = symbols.get(cell, ' ')
            line += f"{color} {symbol}{reset}"
        result.append(line)
    
    return "\n".join(result)


def print_fitness_breakdown(grid: List[List[int]]) -> None:
    """
    Print detailed fitness breakdown for analysis.
    """
    evaluator = FitnessEvaluator()
    scores = evaluator.get_detailed_scores(grid)
    
    print("\n" + "="*50)
    print("FITNESS BREAKDOWN")
    print("="*50)
    
    if 'error' in scores:
        print(f"Error: {scores['error']}")
        return
    
    for key, value in scores.items():
        if isinstance(value, float):
            print(f"  {key.replace('_', ' ').title():25}: {value:8.2f}")
        else:
            print(f"  {key.replace('_', ' ').title():25}: {value}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main entry point for the Pacman Map Generator."""
    print("\n" + "="*80)
    print(" " * 25 + "PACMAN MAP GENERATOR")
    print(" " * 20 + "Genetic Algorithm with BFS/DFS Validation")
    print("="*80)
    print("\nAuthors: Muhammad Hamza Khan, Muhammad Irfan, Hassan Qureshi")
    print("Roll Numbers: 23I-3032, 23I-3065, 23I-3029")
    print("Section: A\n")
    
    # Configuration
    config = {
        'population_size': 10,
        'max_generations': 30,
        'mutation_rate': 0.15,
        'crossover_rate': 0.8,
        'elitism_count': 1,
        'fitness_threshold': 85.0,
        'rows': 15,
        'cols': 30,
        'weights': {
            'path_weight': 1.2,
            'complexity_weight': 6.0,
            'loop_weight': 12.0,
            'variety_bonus': 20.0,
            'isolation_penalty': 2.0,
            'wall_penalty': 3.0,
            'loop_target': 3
        }
    }
    
    # Run Genetic Algorithm
    print("Starting Genetic Algorithm Evolution...")
    print("-" * 60)
    
    start_time = time.time()
    ga = GeneticAlgorithm(config)
    best_map = ga.run()
    elapsed_time = time.time() - start_time
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print(f"\nEvolution Statistics:")
    print(f"  Total Time: {elapsed_time:.2f} seconds")
    print(f"  Generations: {len(ga.generation_history)}")
    print(f"  Best Fitness: {ga.best_fitness:.2f}/100")
    
    if ga.generation_history:
        initial_fitness = ga.generation_history[0]['best_fitness']
        improvement = ga.best_fitness - initial_fitness
        print(f"  Improvement: {improvement:+.1f} points")
    
    print("\nBest Generated Map:")
    print(visualize_map(best_map))
    
    print_fitness_breakdown(best_map)
    
    print("\n" + "="*80)
    print("Evolution Complete!")
    print("="*80)


if __name__ == "__main__":
    main()