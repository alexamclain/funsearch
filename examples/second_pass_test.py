"""
Script to test the second pass approach for the no-three-in-line problem.
This loads a specified good solution and tries to add more points using a second priority function.
"""
import numpy as np
import itertools
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

def are_collinear(p1, p2, p3):
    """Returns True if the three points are collinear."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (y1 - y2) * (x1 - x3) == (y1 - y3) * (x1 - x2)

def generate_test_solution(n, size=None):
    """
    Generate a test solution of specified size.
    If size is None, returns a reasonably good solution.
    """
    all_points = np.array(list(itertools.product(range(n), repeat=2)))
    if size is None:
        # Simple strategy: take points near the boundary
        # This isn't optimal but gives us something to work with
        points = []
        boundary = 3  # Take points within this distance of the boundary
        
        for point in all_points:
            x, y = point
            if x < boundary or x >= n - boundary or y < boundary or y >= n - boundary:
                if not any(are_collinear(point, p1, p2) for i, p1 in enumerate(points) for j, p2 in enumerate(points) if i < j):
                    points.append(point)
                    if len(points) >= n:  # Cap at n points for simplicity
                        break
        
        return np.array(points)
    else:
        # Generate a random valid solution of the given size
        points = []
        indices = list(range(len(all_points)))
        random.shuffle(indices)
        
        for idx in indices:
            point = all_points[idx]
            if not any(are_collinear(point, p1, p2) for i, p1 in enumerate(points) for j, p2 in enumerate(points) if i < j):
                points.append(point)
                if len(points) >= size:
                    break
        
        return np.array(points)

def first_priority(el, n):
    """Priority function that prioritizes points where (x * n + y) % 3 == 0."""
    x, y = el
    return float((x * n + y) % 3 == 0)

def second_priority_random(el, n):
    """A random priority function for second pass."""
    return random.random()

def second_priority_corners(el, n):
    """Prioritize corners and areas less favored by first priority."""
    x, y = el
    # Prioritize corners
    corner_distance = min(
        (x**2 + y**2),  # top-left
        (x**2 + (n-1-y)**2),  # bottom-left
        ((n-1-x)**2 + y**2),  # top-right
        ((n-1-x)**2 + (n-1-y)**2)  # bottom-right
    )
    return 1.0 / (corner_distance + 1)  # Higher priority for points close to corners

def solve_two_pass(n, initial_solution=None, first_priority_func=None, second_priority_func=None):
    """Returns a large no-three-in-line subset on the nxn grid using a two-pass approach."""
    all_points = np.array(list(itertools.product(range(n), repeat=2)), dtype=np.int32)
    
    # First pass: either use provided solution or build one with first_priority
    if initial_solution is None and first_priority_func is not None:
        print("Building initial solution using first priority function")
        # Precompute all priorities for first pass
        priorities = np.array([first_priority_func(tuple(point), n) for point in all_points])
        
        # Build `grid_set` greedily, using priorities for prioritization
        grid_set = np.empty(shape=(0, 2), dtype=np.int32)
        while np.any(priorities != -np.inf):
            max_index = np.argmax(priorities)
            new_point = all_points[max_index]
            priorities[max_index] = -np.inf
            
            # Block those points in all_points which lie on a line spanned by a point in grid_set and new_point
            for index in range(len(all_points)):
                if all_points[index][0] == new_point[0] and all_points[index][1] == new_point[1]:
                    continue  # Skip the new point itself
                if grid_set.size == 0:
                    continue
                elif any(are_collinear(new_point, all_points[index], cap) for cap in grid_set):
                    priorities[index] = -np.inf
            
            grid_set = np.concatenate([grid_set, new_point.reshape(1, -1)], axis=0)
    else:
        grid_set = initial_solution if initial_solution is not None else np.empty(shape=(0, 2), dtype=np.int32)
        print(f"Starting with provided solution of size {len(grid_set)}")
    
    # Second pass with second_priority_func
    if second_priority_func is not None:
        print("Applying second pass")
        # Create masks for points that are already in the grid_set
        used_points_mask = np.zeros(len(all_points), dtype=bool)
        for point in grid_set:
            for i, candidate in enumerate(all_points):
                if np.array_equal(point, candidate):
                    used_points_mask[i] = True
                    break
        
        remaining_points = all_points[~used_points_mask]
        print(f"Remaining points to consider: {len(remaining_points)}")
        
        # Track points added in second pass
        added_in_second_pass = 0
        
        # Precompute priorities for remaining points
        if len(remaining_points) > 0:
            priorities = np.array([second_priority_func(tuple(point), n) for point in remaining_points])
            
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
                    added_in_second_pass += 1
        
        print(f"Added {added_in_second_pass} points in second pass")
    
    return grid_set

def visualize_grid(grid_set, n, title="No-Three-in-Line Grid"):
    """Visualize the grid with points."""
    # Create a grid to represent points
    grid = np.zeros((n, n))
    for point in grid_set:
        x, y = point
        grid[y, x] = 1  # Note: y is the row, x is the column
    
    # Create a custom colormap
    colors = [(1, 1, 1), (0.8, 0, 0)]  # white to red
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap=cmap, interpolation='nearest')
    plt.grid(True, color='black', linestyle='-', linewidth=0.5)
    plt.xticks(np.arange(-0.5, n, 1), [])
    plt.yticks(np.arange(-0.5, n, 1), [])
    plt.title(f"{title} - {len(grid_set)} points")
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    plt.savefig(f"output/{title.replace(' ', '_')}.png")
    plt.close()

def run_experiments(n=47, num_trials=5):
    """Run multiple experiments with different second pass strategies."""
    print(f"Running experiments for {n}x{n} grid")
    
    # Generate initial solution using the first priority function
    print("Building solution using the modulo-3 priority function...")
    initial_solution = solve_two_pass(n, first_priority_func=first_priority)
    print(f"Generated initial solution with {len(initial_solution)} points")
    
    # Visualize initial solution
    visualize_grid(initial_solution, n, "Initial Solution")
    
    # Try different second pass strategies
    strategies = {
        "Random": second_priority_random,
        "Corners": second_priority_corners
    }
    
    best_solution = initial_solution
    best_score = len(initial_solution)
    
    for strategy_name, strategy_func in strategies.items():
        print(f"\nTesting strategy: {strategy_name}")
        
        for trial in range(num_trials):
            if strategy_name == "Random":
                # Set a different seed for each random trial
                random.seed(trial)
            
            solution = solve_two_pass(n, initial_solution=initial_solution, second_priority_func=strategy_func)
            score = len(solution)
            
            print(f"  Trial {trial+1}: Score = {score}")
            
            if score > best_score:
                best_score = score
                best_solution = solution
                print(f"  New best score: {best_score}")
                visualize_grid(solution, n, f"{strategy_name} Trial {trial+1}")
    
    # Visualize best overall solution
    visualize_grid(best_solution, n, "Best Overall Solution")
    print(f"\nBest overall score: {best_score}")
    return best_solution

if __name__ == "__main__":
    # Set the grid size
    n = 47
    
    # Run the experiments
    best_solution = run_experiments(n)