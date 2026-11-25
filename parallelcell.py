import os
from concurrent.futures import ThreadPoolExecutor
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# Cell states
EMPTY = 0
TREE = 1
BURNING = 2


def initialize_forest_grid(size=128, tree_density=0.62, seed=42):
    """Initializes the 2D cellular grid G with trees, empty soil, and initial sparks."""
    rng = np.random.default_rng(seed)
    grid = np.where(rng.random((size, size)) < tree_density, TREE, EMPTY)

    # Ignite a few initial fires near the left/center to observe spread
    grid[size // 2 - 2 : size // 2 + 2, 15:19] = BURNING
    grid[30:33, 40:43] = BURNING
    return grid


def forest_row_rule(args):
    """Step 4a: Parallel cellular transition rule for a single row (Moore neighborhood)."""
    i, G, p_grow, p_lightning, step_seed = args
    rows, cols = G.shape
    new_row = np.copy(G[i])

    # Deterministic per-row random generator for thread-safe stochastic transitions
    rng = np.random.default_rng(step_seed + i)
    rand_vals = rng.random(cols)

    for j in range(cols):
        state = G[i, j]

        if state == BURNING:
            # Rule 1: A burning tree turns into empty ash in the next step
            new_row[j] = EMPTY

        elif state == TREE:
            # Extract 3x3 Moore neighborhood safely at grid boundaries
            r_min, r_max = max(0, i - 1), min(rows, i + 2)
            c_min, c_max = max(0, j - 1), min(cols, j + 2)
            neighborhood = G[r_min:r_max, c_min:c_max]

            # Rule 2: Tree catches fire if at least one neighbor is burning
            if np.any(neighborhood == BURNING) or (rand_vals[j] < p_lightning):
                new_row[j] = BURNING

        elif state == EMPTY:
            # Rule 3: New tree grows on empty ground with probability p_grow
            if rand_vals[j] < p_grow:
                new_row[j] = TREE

    return i, new_row


def run_parallel_forest_fire_ca():
    # 1. Start & 2. Initialize cellular grid G
    size = 128
    G = initialize_forest_grid(size=size, tree_density=0.62)
    initial_grid = np.copy(G)

    rows, _ = G.shape
    T = 60  # Total simulation time steps
    p_grow = 0.015  # Probability of tree regrowth
    p_lightning = 0.0002  # Probability of spontaneous ignition
    num_workers = min(rows, os.cpu_count() or 4)

    # 3. Create temporary grid New
    New = np.copy(G)
    snapshots = {0: initial_grid}
    capture_steps = {15, 35, T}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 4. For t = 1 to T do
        for t in range(1, T + 1):
            tasks = [(i, G, p_grow, p_lightning, t * 1000) for i in range(rows)]

            # 4a. Parallel execution per row & 4b. Implicit barrier synchronization
            for i, updated_row in executor.map(forest_row_rule, tasks):
                New[i] = updated_row

            # 4c. Synchronous Grid Update: G = New
            G = np.copy(New)

            if t in capture_steps:
                snapshots[t] = np.copy(G)
                trees = np.sum(G == TREE)
                fires = np.sum(G == BURNING)
                print(
                    f"Step t = {t:02d} synced | Trees: {trees:5d} | Burning: {fires:4d}"
                )

    # 5. Visualize evolution across time steps & 6. Stop
    cmap = mcolors.ListedColormap(
        ["#2b2118", "#2e8b57", "#ff4500"]
    )  # Soil, Tree, Fire
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (step, grid_state) in zip(axes, sorted(snapshots.items())):
        ax.imshow(grid_state, cmap=cmap, norm=norm)
        ax.set_title(f"Step t = {step}", fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("parallel_forest_fire_ca.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    run_parallel_forest_fire_ca()
