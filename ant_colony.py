import numpy as np
import matplotlib.pyplot as plt
import random

# 1. Setup Environment (15 Waypoints) & Distance Matrix
random.seed(42); np.random.seed(42)
num_wp, num_drones, max_iter = 15, 10, 80
alpha, beta, rho, Q = 1.0, 2.0, 0.5, 100.0
waypoints = np.random.rand(num_wp, 2) * 100

dist_matrix = np.zeros((num_wp, num_wp))
for i in range(num_wp):
    for j in range(num_wp):
        dist_matrix[i][j] = np.linalg.norm(waypoints[i] - waypoints[j]) if i != j else np.inf

# Initialize Pheromones
pheromones = np.full((num_wp, num_wp), 0.1)
best_tour, best_dist = None, np.inf

# Helper to select next waypoint
def next_waypoint(curr, visited):
    unvisited = [i for i in range(num_wp) if i not in visited]
    probs = []
    for node in unvisited:
        h = 1.0 / dist_matrix[curr][node]
        probs.append((pheromones[curr][node] ** alpha) * (h ** beta))
    return random.choices(unvisited, weights=probs, k=1)[0]

# 2. Swarm Optimization Main Loop
for iteration in range(max_iter):
    tours, tours_dist = [], []
    for d in range(num_drones):
        tour = [random.randint(0, num_wp - 1)]
        while len(tour) < num_wp:
            tour.append(next_waypoint(tour[-1], tour))
        
        # Complete the loop and calculate flight distance
        d_sum = sum(dist_matrix[tour[i]][tour[i+1]] for i in range(num_wp - 1)) + dist_matrix[tour[-1]][tour[0]]
        tours.append(tour)
        tours_dist.append(d_sum)
        
        if d_sum < best_dist:
            best_dist, best_tour = d_sum, tour
            
    # Update Pheromones (Evaporation + Deposit)
    pheromones *= (1 - rho)
    for tour, dist in zip(tours, tours_dist):
        for i in range(num_wp):
            u, v = tour[i], tour[(i + 1) % num_wp]
            pheromones[u][v] += Q / dist

print(f"Optimized Swarm Route Distance: {best_dist:.2f}m\nRoute: {best_tour}")

# 3. Route Visualization
plt.figure(figsize=(8, 6))
plt.scatter(waypoints[:, 0], waypoints[:, 1], color='red', s=80, label='Waypoints')
for idx, (x, y) in enumerate(waypoints):
    plt.text(x + 1, y + 1, f"WP{idx}", fontsize=8)

# Plot the optimized drone route
route_pts = np.array([waypoints[node] for node in best_tour] + [waypoints[best_tour[0]]])
plt.plot(route_pts[:, 0], route_pts[:, 1], color='dodgerblue', linewidth=2, label='Drone Path')
plt.scatter(waypoints[best_tour[0]][0], waypoints[best_tour[0]][1], color='green', s=150, marker='*', label='Launch Base')
plt.title('Autonomous Drone Swarm Navigation (< 100 LOC)')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()
