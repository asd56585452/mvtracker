#!/usr/bin/env python3
"""
Visualize and compare voxel point clouds under min vs mean confidence fusion.
Generates top-down (X-Z) and front (X-Y) projection comparisons.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData

ARTIFACT_DIR = "mvtracker_results/da3_confidence_inspect"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def load_ply_pts(path):
    p = PlyData.read(path)
    vx = np.array(p['vertex']['x'], dtype=np.float32)
    vy = np.array(p['vertex']['y'], dtype=np.float32)
    vz = np.array(p['vertex']['z'], dtype=np.float32)
    r = np.array(p['vertex']['red'], dtype=np.float32) if 'red' in p['vertex'] else np.zeros_like(vx)
    g = np.array(p['vertex']['green'], dtype=np.float32) if 'green' in p['vertex'] else np.zeros_like(vx)
    b = np.array(p['vertex']['blue'], dtype=np.float32) if 'blue' in p['vertex'] else np.zeros_like(vx)
    pts = np.stack([vx, vy, vz], axis=-1)
    cols = np.stack([r, g, b], axis=-1) / 255.0
    return pts, cols

def main():
    p_min_25 = "datasets/coffee_martini/colmap_25/test_voxel_conf_min_thresh2.5.ply"
    p_min_50 = "datasets/coffee_martini/colmap_25/test_voxel_conf_min_thresh5.0.ply"
    p_mean_25 = "datasets/coffee_martini/colmap_25/test_voxel_conf_mean_thresh2.5.ply"
    p_mean_50 = "datasets/coffee_martini/colmap_25/test_voxel_conf_mean_thresh5.0.ply"

    pts_min25, col_min25 = load_ply_pts(p_min_25)
    pts_min50, col_min50 = load_ply_pts(p_min_50)
    pts_mean25, col_mean25 = load_ply_pts(p_mean_25)
    pts_mean50, col_mean50 = load_ply_pts(p_mean_50)

    print(f"min_2.5: {len(pts_min25):,} pts")
    print(f"min_5.0: {len(pts_min50):,} pts")
    print(f"mean_2.5: {len(pts_mean25):,} pts")
    print(f"mean_5.0: {len(pts_mean50):,} pts")

    fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=180)

    # Subplot 1: Top-down (X-Z) min 2.5 vs min 5.0
    ax = axes[0, 0]
    sub_min25 = pts_min25[::max(1, len(pts_min25)//20000)]
    sub_min50 = pts_min50[::max(1, len(pts_min50)//20000)]
    ax.scatter(sub_min25[:, 0], sub_min25[:, 2], s=1.5, c='gray', alpha=0.3, label=f"min thresh 2.5 ({len(pts_min25):,} pts)")
    ax.scatter(sub_min50[:, 0], sub_min50[:, 2], s=1.5, c='#1f77b4', alpha=0.5, label=f"min thresh 5.0 ({len(pts_min50):,} pts)")
    ax.set_title("Top-Down View (X-Z): MIN Aggregation (2.5 vs 5.0)", fontsize=13, fontweight='bold')
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Depth Z (meters)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.5)

    # Subplot 2: Top-down (X-Z) mean 2.5 vs mean 5.0
    ax = axes[0, 1]
    sub_mean25 = pts_mean25[::max(1, len(pts_mean25)//20000)]
    sub_mean50 = pts_mean50[::max(1, len(pts_mean50)//20000)]
    ax.scatter(sub_mean25[:, 0], sub_mean25[:, 2], s=1.5, c='gray', alpha=0.3, label=f"mean thresh 2.5 ({len(pts_mean25):,} pts)")
    ax.scatter(sub_mean50[:, 0], sub_mean50[:, 2], s=1.5, c='#ff7f0e', alpha=0.5, label=f"mean thresh 5.0 ({len(pts_mean50):,} pts)")
    ax.set_title("Top-Down View (X-Z): MEAN Aggregation (2.5 vs 5.0)", fontsize=13, fontweight='bold')
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Depth Z (meters)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.5)

    # Subplot 3: Direct Comparison: MIN 5.0 vs MEAN 5.0
    ax = axes[1, 0]
    ax.scatter(sub_mean50[:, 0], sub_mean50[:, 2], s=2.0, c='#ff7f0e', alpha=0.4, label=f"MEAN thresh 5.0 ({len(pts_mean50):,} pts)")
    ax.scatter(sub_min50[:, 0], sub_min50[:, 2], s=1.5, c='#1f77b4', alpha=0.6, label=f"MIN thresh 5.0 ({len(pts_min50):,} pts)")
    ax.set_title("Top-Down View (X-Z): MIN 5.0 vs MEAN 5.0", fontsize=13, fontweight='bold')
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Depth Z (meters)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.5)

    # Subplot 4: Front View (X-Y) MIN 5.0
    ax = axes[1, 1]
    ax.scatter(sub_min50[:, 0], sub_min50[:, 1], s=1.5, c='#1f77b4', alpha=0.5, label=f"MIN thresh 5.0")
    ax.set_title("Front View (X-Y): MIN thresh 5.0 Structure", fontsize=13, fontweight='bold')
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.invert_yaxis()
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    out_img = os.path.join(ARTIFACT_DIR, "voxel_conf_fusion_comparison.png")
    plt.savefig(out_img, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to: {out_img}")

if __name__ == "__main__":
    main()
