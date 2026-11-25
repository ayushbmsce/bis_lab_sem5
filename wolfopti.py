import numpy as np

np.random.seed(42)


# Real-life objective: Minimize daily delivery + land zoning costs
def warehouse_cost(pos):
    x, y = pos
    # Coordinates of 3 major retail hubs and their daily truckload weights
    hubs = np.array([[10.0, 5.0], [15.0, 10.0], [12.0, 9.0]])
    weights = np.array([2.0, 3.0, 5.0])

    # Weighted Euclidean distance (transport cost in $1,000s)
    distances = np.sqrt(np.sum((hubs - pos) ** 2, axis=1))
    transport_cost = np.sum(weights * distances)

    # Zoning penalty if built too far from the industrial corridor (12.5, 8.0)
    zoning_cost = 0.5 * ((x - 12.5) ** 2 + (y - 8.0) ** 2)
    return transport_cost + zoning_cost


def grey_wolf_optimizer(obj_func, dim, bounds, num_wolves=15, max_iter=50):
    lb, ub = bounds
    positions = np.random.uniform(lb, ub, (num_wolves, dim))

    # Initialize Alpha (best), Beta (2nd), and Delta (3rd) wolves
    alpha_pos, beta_pos, delta_pos = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    alpha_score, beta_score, delta_score = (
        float('inf'),
        float('inf'),
        float('inf'),
    )

    for t in range(max_iter):
        for i in range(num_wolves):
            positions[i] = np.clip(positions[i], lb, ub)
            fitness = obj_func(positions[i])

            # Update pack leadership hierarchy
            if fitness < alpha_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = alpha_score, alpha_pos.copy()
                alpha_score, alpha_pos = fitness, positions[i].copy()
            elif fitness < beta_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = fitness, positions[i].copy()
            elif fitness < delta_score:
                delta_score, delta_pos = fitness, positions[i].copy()

        # Parameter 'a' decreases linearly from 2 to 0 (exploration -> exploitation)
        a = 2 - t * (2 / max_iter)

        # Update each wolf's position guided by Alpha, Beta, and Delta
        for i in range(num_wolves):
            for j in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                X1 = alpha_pos[j] - (2 * a * r1 - a) * abs(
                    2 * r2 * alpha_pos[j] - positions[i][j]
                )

                r1, r2 = np.random.rand(), np.random.rand()
                X2 = beta_pos[j] - (2 * a * r1 - a) * abs(
                    2 * r2 * beta_pos[j] - positions[i][j]
                )

                r1, r2 = np.random.rand(), np.random.rand()
                X3 = delta_pos[j] - (2 * a * r1 - a) * abs(
                    2 * r2 * delta_pos[j] - positions[i][j]
                )

                positions[i][j] = (X1 + X2 + X3) / 3.0

    return alpha_pos, alpha_score


# Run optimization on a 25km x 25km city grid
best_coords, min_cost = grey_wolf_optimizer(
    warehouse_cost, dim=2, bounds=(0, 25)
)
print(
    f'Optimal Warehouse Coordinates: (X = {best_coords[0]:.2f} km, Y ='
    f' {best_coords[1]:.2f} km)'
)
print(f'Minimum Daily Logistics Cost:  ${min_cost * 1000:,.2f}')
