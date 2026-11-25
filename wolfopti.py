import numpy as np

np.random.seed(42)


def warehouse_cost(pos):
    x, y = pos
    hubs = np.array([[10.0, 5.0], [15.0, 10.0], [12.0, 9.0]])
    weights = np.array([2.0, 3.0, 5.0])

    distances = np.sqrt(np.sum((hubs - pos) ** 2, axis=1))
    transport_cost = np.sum(weights * distances)

    zoning_cost = 0.5 * ((x - 12.5) ** 2 + (y - 8.0) ** 2)
    return transport_cost + zoning_cost


def grey_wolf_optimizer(obj_func, dim, bounds, num_wolves=15, max_iter=50):
    lb, ub = bounds
    positions = np.random.uniform(lb, ub, (num_wolves, dim))

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

            if fitness < alpha_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = alpha_score, alpha_pos.copy()
                alpha_score, alpha_pos = fitness, positions[i].copy()
            elif fitness < beta_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = fitness, positions[i].copy()
            elif fitness < delta_score:
                delta_score, delta_pos = fitness, positions[i].copy()

        a = 2 - t * (2 / max_iter)

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


best_coords, min_cost = grey_wolf_optimizer(
    warehouse_cost, dim=2, bounds=(0, 25)
)
print(
    f'Optimal Warehouse Coordinates: (X = {best_coords[0]:.2f} km, Y ='
    f' {best_coords[1]:.2f} km)'
)
print(f'Minimum Daily Logistics Cost:  ${min_cost * 1000:,.2f}')
