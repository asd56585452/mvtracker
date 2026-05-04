import numpy as np
from sklearn.cluster import MiniBatchKMeans

pts = np.random.rand(1000000, 3) * 10
colors = np.random.rand(1000000, 3)

# 1. Voxel Downsample
voxel_size = 0.05
coords = np.round(pts / voxel_size)
_, unique_indices = np.unique(coords, axis=0, return_index=True)
pts_voxel = pts[unique_indices]
colors_voxel = colors[unique_indices]
print(f"Voxel downsampled from {len(pts)} to {len(pts_voxel)}")

# 2. KMeans Grouping
num_queries = 250000
chunk_size = 10000
num_groups = max(1, num_queries // chunk_size)

kmeans = MiniBatchKMeans(n_clusters=num_groups, random_state=42, n_init="auto")
labels = kmeans.fit_predict(pts_voxel)

# 3. Sample from each group
sampled_pts = []
sampled_colors = []
pts_per_group = chunk_size

for g in range(num_groups):
    mask = (labels == g)
    g_pts = pts_voxel[mask]
    g_colors = colors_voxel[mask]
    
    if len(g_pts) > pts_per_group:
        idx = np.random.choice(len(g_pts), pts_per_group, replace=False)
        g_pts = g_pts[idx]
        g_colors = g_colors[idx]
    
    sampled_pts.append(g_pts)
    sampled_colors.append(g_colors)

sampled_pts = np.concatenate(sampled_pts, axis=0)
sampled_colors = np.concatenate(sampled_colors, axis=0)
print(f"Sampled {len(sampled_pts)} points grouped into {num_groups} groups.")
