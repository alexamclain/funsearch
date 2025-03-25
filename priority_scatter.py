import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import wandb
import argparse
from pathlib import Path
import pickle

def are_collinear(p1, p2, p3):
    """Returns True if the three points are collinear."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (y1 - y2) * (x1 - x3) == (y1 - y3) * (x1 - x2)

# This will be dynamically replaced if using a function from a backup file
def priority(el, n):
    """Default priority function (favors points near the edges)."""
    x, y = el
    center_x, center_y = n/2, n/2
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    return dist_from_center

def solve(n):
    """Returns a large subset on the nxn grid that does not contain 3 points in a line."""
    all_points = np.array(list(itertools.product(np.arange(n), repeat=2)), dtype=np.int32)
    
    # Precompute all priorities
    priorities = np.array([priority(tuple(point), n) for point in all_points])

    # Build grid_set greedily
    grid_set = np.empty(shape=(0, 2), dtype=np.int32)
    while np.any(priorities != -np.inf):
        # Add point with maximum priority
        max_index = np.argmax(priorities)
        new_point = all_points[max_index]
        priorities[max_index] = -np.inf

        # Skip if this would create collinear points
        if grid_set.size > 0:
            for index in range(len(all_points)):
                if priorities[index] == -np.inf:
                    continue
                    
                point = all_points[index]
                skip = False
                for existing_point in grid_set:
                    if are_collinear(new_point, point, existing_point):
                        priorities[index] = -np.inf
                        skip = True
                        break
                if skip:
                    break

        # Add the point
        grid_set = np.vstack([grid_set, new_point]) if grid_set.size > 0 else np.array([new_point])

    return grid_set

def visualize(points, n, priority_func=None, output_dir=None, log_to_wandb=False):
    """Create visualizations for the no-three-in-line problem."""
    print(f"Creating visualizations for {len(points)} points in {n}x{n} grid...")
    
    # Create output directory if specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Create scatter plot of points
    plt.figure(figsize=(10, 10))
    plt.scatter(points[:, 0], points[:, 1], s=20, c='red')
    plt.xlim(-0.5, n-0.5)
    plt.ylim(-0.5, n-0.5)
    plt.title(f"No-three-in-line subset of size {len(points)} for a {n}×{n} grid")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    
    if output_dir:
        plt.savefig(output_path / f"no_three_in_line_points_n{n}.png", dpi=300)
    
    if log_to_wandb:
        wandb.log({"points_plot": wandb.Image(plt.gcf())})
    
    plt.close()
    
    # Create heatmap if priority function is available
    if priority_func:
        # Generate heatmap
        priorities = np.zeros((n, n))
        for x in range(n):
            for y in range(n):
                priorities[y, x] = priority_func((x, y), n)
        
        plt.figure(figsize=(12, 10))
        im = plt.imshow(priorities, origin='lower', cmap='viridis', extent=[0, n, 0, n])
        plt.title(f"Priority heatmap for n={n}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.colorbar(label='Priority')
        
        if output_dir:
            plt.savefig(output_path / f"priority_heatmap_n{n}.png", dpi=300)
        
        if log_to_wandb:
            wandb.log({"priority_heatmap": wandb.Image(plt.gcf())})
        
        plt.close()
        
        # Create overlay visualization
        plt.figure(figsize=(12, 10))
        im = plt.imshow(priorities, origin='lower', cmap='viridis', extent=[0, n, 0, n])
        plt.scatter(points[:, 0], points[:, 1], s=15, c='red', marker='o', edgecolors='white')
        plt.title(f"No-three-in-line subset (size {len(points)}) with priority heatmap")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.colorbar(label='Priority')
        
        if output_dir:
            plt.savefig(output_path / f"overlay_n{n}.png", dpi=300)
        
        if log_to_wandb:
            wandb.log({"overlay": wandb.Image(plt.gcf())})
        
        plt.close()
    
    # Save the points as CSV
    if output_dir:
        np.savetxt(output_path / f"no_three_in_line_points_n{n}.csv", points, delimiter=",", fmt="%d", header="x,y")
    
    # Log the data as a table
    if log_to_wandb:
        import pandas as pd
        points_df = pd.DataFrame(points, columns=["x", "y"])
        wandb.log({"points_table": wandb.Table(dataframe=points_df)})
        wandb.log({"n": n, "num_points": len(points)})
    
    print(f"Found {len(points)} points in {n}×{n} grid")
    if output_dir:
        print(f"Saved visualizations to {output_path}")

def extract_best_program(db_file):
    """Extract the best priority function from a database backup file."""
    try:
        with open(db_file, "rb") as f:
            data = pickle.load(f)
        
        # Find the best program across all islands
        if isinstance(data, dict) and "_best_program_per_island" in data and "_best_score_per_island" in data:
            # Get the best program across all islands
            best_score = float('-inf')
            best_program = None
            best_island = -1
            
            for i, (program, score) in enumerate(zip(data["_best_program_per_island"], data["_best_score_per_island"])):
                if score > best_score and program is not None:
                    best_score = score
                    best_program = program
                    best_island = i
            
            if best_program:
                print(f"Found best program in database from island {best_island}. Score: {best_score}")
                return str(best_program)
            else:
                print("No best program found in database file.")
                return None
        else:
            print("Database file does not contain the expected structure.")
            return None
    except Exception as e:
        print(f"Error extracting best program: {e}")
        return None

def create_priority_function(function_code):
    """Create a callable priority function from code string."""
    try:
        # Create a namespace for executing the code
        namespace = {}
        
        # Execute the function code in the namespace
        exec(function_code, namespace)
        
        # Return the priority function from the namespace
        if 'priority' in namespace:
            return namespace['priority']
        else:
            print("Warning: Could not find 'priority' function in the extracted code")
            return None
    except Exception as e:
        print(f"Error creating priority function: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Visualize no-three-in-line points and priority heatmap')
    parser.add_argument('--n', type=int, default=32, help='Size of the grid (n×n)')
    parser.add_argument('--output', type=str, default='./visualizations', help='Output directory')
    parser.add_argument('--wandb', action='store_true', help='Log to Weights & Biases')
    parser.add_argument('--project', type=str, default='funsearch', help='W&B project name')
    parser.add_argument('--run_name', type=str, help='W&B run name')
    parser.add_argument('--db_file', type=str, help='Path to database backup file')
    
    args = parser.parse_args()
    
    # Initialize W&B if requested
    if args.wandb:
        run_name = args.run_name or f"no-three-in-line-n{args.n}"
        wandb.init(project=args.project, name=run_name)
    
    # Try to extract the best priority function if a database file is provided
    priority_func = priority  # Default
    if args.db_file:
        best_program_code = extract_best_program(args.db_file)
        if best_program_code:
            # Save the extracted function
            if args.output:
                output_path = Path(args.output)
                output_path.mkdir(parents=True, exist_ok=True)
                with open(output_path / "best_priority.py", 'w') as f:
                    f.write(best_program_code)
            
            # Create a dynamic priority function
            dynamic_priority = create_priority_function(best_program_code)
            if dynamic_priority:
                print("Successfully loaded extracted priority function.")
                # Override the global priority function
                global priority
                priority = dynamic_priority
                priority_func = dynamic_priority
            else:
                print("Failed to load extracted priority function. Using default.")
    
    # Solve and visualize
    points = solve(args.n)
    visualize(
        points=points,
        n=args.n,
        priority_func=priority_func,
        output_dir=args.output,
        log_to_wandb=args.wandb
    )
    
    # Clean up W&B
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()