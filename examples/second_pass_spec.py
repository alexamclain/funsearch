"""Second-pass specification to further improve no-three-in-line solutions.
On every iteration, improve the priority_v# function for adding more points
to an existing solution.
"""
import itertools
import numpy as np
import funsearch

def are_collinear(p1, p2, p3):
    """Returns True if the three points are collinear."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (y1 - y2) * (x1 - x3) == (y1 - y3) * (x1 - x2)

def load_first_pass_solution(n):
    """Load the solution from the first pass."""
    all_points = np.array(list(itertools.product(np.arange(n), repeat=2)), dtype=np.int32)
    grid_set = np.empty(shape=(0, 2), dtype=np.int32)
    
    for point in all_points:
        x, y = point
        if (x * n + y) % 3 == 0:
            # Check if it creates a line with existing points
            valid = True
            for i in range(len(grid_set)):
                for j in range(i+1, len(grid_set)):
                    if are_collinear(point, grid_set[i], grid_set[j]):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                grid_set = np.concatenate([grid_set, point.reshape(1, -1)], axis=0)
    
    return grid_set

@funsearch.run
def evaluate(n: int) -> int:
    """Returns the size of a point set on the nxn grid that does not contain 3 points in a line."""
    first_solution = load_first_pass_solution(n)
    grid_set = solve_second_pass(n, first_solution)
    return len(grid_set)

def solve_second_pass(n: int, initial_solution) -> np.ndarray:
    """Returns a larger no-three-in-line subset by adding more points to the initial solution."""
    all_points = np.array(list(itertools.product(np.arange(n), repeat=2)), dtype=np.int32)
    grid_set = initial_solution
    
    # Create masks for points that are already in the grid_set
    used_points_mask = np.zeros(len(all_points), dtype=bool)
    for point in grid_set:
        for i, candidate in enumerate(all_points):
            if np.array_equal(point, candidate):
                used_points_mask[i] = True
                break
    
    remaining_points = all_points[~used_points_mask]
    
    # Precompute priorities for remaining points
    if len(remaining_points) > 0:
        priorities = np.array([priority(tuple(point), n, grid_set) for point in remaining_points])
        
        # Try to add more points greedily
        while np.any(priorities != -np.inf):
            max_index = np.argmax(priorities)
            new_point = remaining_points[max_index]
            priorities[max_index] = -np.inf
            
            # Check if adding this point would create a three-in-line configuration
            is_valid = True
            for i in range(len(grid_set)):
                for j in range(i+1, len(grid_set)):
                    if are_collinear(new_point, grid_set[i], grid_set[j]):
                        is_valid = False
                        break
                if not is_valid:
                    break
            
            if is_valid:
                grid_set = np.concatenate([grid_set, new_point.reshape(1, -1)], axis=0)
    
    return grid_set

@funsearch.evolve
def priority(el: tuple[int, int], n: int, current_solution: np.ndarray) -> float:
    """
    Returns the priority for adding a point to the existing solution.
    
    Args:
        el: The point being evaluated (x, y)
        n: Grid size
        current_solution: The current set of points
        
    Returns:
        A float indicating priority (higher means higher priority)
    """
    # This function will be evolved by funsearch
    return 0.0