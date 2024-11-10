import numpy as np
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons

n_clusters = 3  # Number of clusters
n_points = 100  # Number of points


# 1. Generate uniformly distributed points
points = np.random.rand(n_points, 2) * 100

# Save to file
with open("input-1.txt", "w") as file:
    file.write(f"{n_clusters}\n{n_points}\n")
    for point in points:
        file.write(f"{point[0]} {point[1]}\n")


# 2. Generate clearly separated clusters
points, _ = make_blobs(
    n_samples=n_points, centers=n_clusters, n_features=2, random_state=42
)

# Save to file
with open("input-2.txt", "w") as file:
    file.write(f"{n_clusters}\n{n_points}\n")
    for point in points:
        file.write(f"{point[0]} {point[1]}\n")


n_clusters = 4
n_points = 100

# 3. Generate overlapping clusters
points, _ = make_blobs(
    n_samples=n_points, centers=n_clusters, cluster_std=2.5, random_state=42
)

# Save to file
with open("input-3.txt", "w") as file:
    file.write(f"{n_clusters}\n{n_points}\n")
    for point in points:
        file.write(f"{point[0]} {point[1]}\n")


n_clusters = 5
n_points = 1000  # Large number of points

# 4. Generate random points
points = np.random.rand(n_points, 2) * 100

# Save to file
with open("input-4.txt", "w") as file:
    file.write(f"{n_clusters}\n{n_points}\n")
    for point in points:
        file.write(f"{point[0]} {point[1]}\n")


n_clusters = 3
n_points = 100

# 5. Generate clusters with varying densities
points, _ = make_blobs(
    n_samples=n_points, centers=n_clusters, cluster_std=[1.0, 2.5, 0.5], random_state=42
)

# Save to file
with open("input-5.txt", "w") as file:
    file.write(f"{n_clusters}\n{n_points}\n")
    for point in points:
        file.write(f"{point[0]} {point[1]}\n")


n_clusters = 2  # Non-convex shapes typically have 2 clusters
n_points = 100

# 6. Generate non-convex shaped data
points, _ = make_moons(n_samples=n_points, noise=0.05)

# Save to file
with open("input-6.txt", "w") as file:
    file.write(f"{n_clusters}\n{n_points}\n")
    for point in points:
        file.write(f"{point[0]} {point[1]}\n")


n_clusters = 3
n_points = 100
n_outliers = 10

# 7. Generate data with clusters
points, _ = make_blobs(n_samples=n_points, centers=n_clusters, random_state=42)

# Add outliers
outliers = np.random.uniform(low=-10, high=10, size=(n_outliers, 2))
points = np.vstack([points, outliers])

# Save to file
with open("input-7.txt", "w") as file:
    file.write(f"{n_clusters}\n{n_points + n_outliers}\n")
    for point in points:
        file.write(f"{point[0]} {point[1]}\n")
