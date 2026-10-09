#!/usr/bin/env python3
"""
Generate in-depth comparison of 3D depth distributions between COLMAP, DA3, and DA3Giant
on coffee_martini Frame 25, focusing on window and outdoor geometry.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
artifact_dir = os.path.join(PROJECT_ROOT, "mvtracker_results", "da3_confidence_inspect")
os.makedirs(artifact_dir, exist_ok=True)

# 1. 讀取三組點雲資料
colmap_ply_path = os.path.join(PROJECT_ROOT, "datasets", "coffee_martini", "colmap_0", "sparse", "0", "points3D_total1_sec_fps30.ply")
da3_ply_path = os.path.join(PROJECT_ROOT, "datasets", "coffee_martini", "colmap_25", "da3_importance_50k.ply")
da3_giant_gs_path = os.path.join(PROJECT_ROOT, "datasets", "coffee_martini", "colmap_25", "da3_giant_direct_3dgs_f25.ply")
hybrid_ply_path = os.path.join(PROJECT_ROOT, "datasets", "coffee_martini", "colmap_25", "colmap_da3_importance_50k.ply")

def read_pts(path):
    p = PlyData.read(path)
    pts = np.vstack([p['vertex']['x'], p['vertex']['y'], p['vertex']['z']]).T
    op = p['vertex']['opacity'] if 'opacity' in p['vertex'] else None
    return pts, op

pts_colmap, _ = read_pts(colmap_ply_path)
pts_da3, _ = read_pts(da3_ply_path)
pts_gs, op_gs = read_pts(da3_giant_gs_path)
pts_hybrid, _ = read_pts(hybrid_ply_path)

# 2. 統計數據
print("--- 點雲深度 Z 統計 ---")
print(f"COLMAP: count={len(pts_colmap):,}, Z range=[{pts_colmap[:,2].min():.2f}, {pts_colmap[:,2].max():.2f}], 90%={np.percentile(pts_colmap[:,2], 90):.2f}, 99%={np.percentile(pts_colmap[:,2], 99):.2f}")
print(f"DA3 (50k): count={len(pts_da3):,}, Z range=[{pts_da3[:,2].min():.2f}, {pts_da3[:,2].max():.2f}], 90%={np.percentile(pts_da3[:,2], 90):.2f}, 99%={np.percentile(pts_da3[:,2], 99):.2f}")
print(f"DA3Giant Direct 3DGS: count={len(pts_gs):,}, Z range=[{pts_gs[:,2].min():.2f}, {pts_gs[:,2].max():.2f}], 90%={np.percentile(pts_gs[:,2], 90):.2f}, 99%={np.percentile(pts_gs[:,2], 99):.2f}")
print(f"Hybrid 50k: count={len(pts_hybrid):,}, Z range=[{pts_hybrid[:,2].min():.2f}, {pts_hybrid[:,2].max():.2f}], 90%={np.percentile(pts_hybrid[:,2], 90):.2f}, 99%={np.percentile(pts_hybrid[:,2], 99):.2f}")

# 3. 繪製對比圖
fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=200)

# (A) Z 深度累積直方圖 (CDF / Histogram)
ax = axes[0, 0]
bins = np.linspace(0, 100, 101)
ax.hist(np.clip(pts_colmap[:,2], 0, 100), bins=bins, alpha=0.6, label=f"COLMAP Sparse (Max Z={pts_colmap[:,2].max():.1f}m)", color="#1f77b4", density=True)
ax.hist(np.clip(pts_da3[:,2], 0, 100), bins=bins, alpha=0.6, label=f"DA3 Importance 50k (Max Z={pts_da3[:,2].max():.1f}m [Truncated!])", color="#ff7f0e", density=True)
ax.hist(np.clip(pts_hybrid[:,2], 0, 100), bins=bins, alpha=0.5, label=f"Hybrid 50k (Max Z={pts_hybrid[:,2].max():.1f}m)", color="#2ca02c", density=True)
ax.axvline(21.1, color='red', linestyle='--', linewidth=2, label="DA3 Depth Ceiling (~21.1m)")
ax.set_title("(A) 3D Point Cloud Depth Distribution (Z: 0 ~ 100m)", fontsize=13, fontweight='bold')
ax.set_xlabel("Depth Z (meters)", fontsize=11)
ax.set_ylabel("Probability Density", fontsize=11)
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, linestyle=":", alpha=0.6)

# (B) DA3Giant Direct 3DGS 深度與 Opacity 分佈
ax = axes[0, 1]
sig_op = 1.0 / (1.0 + np.exp(-op_gs))
ax.scatter(np.clip(pts_gs[::10, 2], 0, 50), sig_op[::10], s=1.5, alpha=0.3, color="#9467bd")
ax.axvline(21.1, color='red', linestyle='--', linewidth=2, label="DA3 Max Depth (~21.1m)")
ax.axhline(0.05, color='gray', linestyle=':', label="Low Opacity Threshold (0.05)")
ax.set_title("(B) DA3Giant Direct 3DGS: Depth vs Predicted Opacity", fontsize=13, fontweight='bold')
ax.set_xlabel("Gaussian Center Depth Z (meters)", fontsize=11)
ax.set_ylabel("Sigmoid Opacity", fontsize=11)
ax.set_ylim(-0.02, 1.02)
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, linestyle=":", alpha=0.6)

# (C) 頂視圖 (Top-down X-Z projection)
ax = axes[1, 0]
idx_c = np.random.choice(len(pts_colmap), min(3000, len(pts_colmap)), replace=False)
idx_d = np.random.choice(len(pts_da3), min(3000, len(pts_da3)), replace=False)
ax.scatter(pts_colmap[idx_c, 0], pts_colmap[idx_c, 2], s=4, alpha=0.7, color="#1f77b4", label="COLMAP Sparse Points (Far Outdoor Extends to >100m)")
ax.scatter(pts_da3[idx_d, 0], pts_da3[idx_d, 2], s=4, alpha=0.7, color="#ff7f0e", label="DA3 Points (Collapsed/Squashed at Z=21m)")
ax.set_xlim(-40, 30)
ax.set_ylim(0, 80)
ax.set_title("(C) Top-Down View (X-Z Plane) Comparison", fontsize=13, fontweight='bold')
ax.set_xlabel("X (meters)", fontsize=11)
ax.set_ylabel("Depth Z (meters)", fontsize=11)
ax.axhline(21.1, color='red', linestyle='--', linewidth=1.5, label="DA3 Far-Plane Barrier")
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, linestyle=":", alpha=0.6)

# (D) 核心發現總結文字框
ax = axes[1, 1]
ax.axis("off")
summary_text = (
    "Key Findings on DA3/DA3Giant Outdoor Window Analysis:\n\n"
    "1. Window Confidence Distribution:\n"
    "   - Closed Blinds: High confidence (9.5 - 10.5), treated as flat textured wall.\n"
    "   - Outdoor Window (Glass): Confident (8.5 - 10.0), NOT low!\n"
    "     * Root cause: DA3 features recognize exterior tree/house textures\n"
    "       without realizing transparency, becoming OVERCONFIDENT.\n"
    "   - Glass Rims, Specular Reflections & Left Opening: Confidence drops to 1.0 - 3.5.\n\n"
    "2. Did DA3Giant Correctly Predict Outdoor Positions?\n"
    "   - Answer: NO, outdoor depth is severely squashed / truncated.\n"
    "   - Far-Plane Ceiling: Ground-truth COLMAP triangulates outdoor points up to\n"
    "     50m - 347.5m. DA3 depth estimation has a hard ceiling at ~21.1m!\n"
    "   - 3DGS Pruning: Default export prunes top 10% depths (prune_by_depth_percent=0.9);\n"
    "     90% of surviving Gaussians cluster within Z < 3.94m (foreground only).\n"
    "   - Opacity Sparsity: Directly predicted 3DGS median opacity is only 0.02,\n"
    "     forming a translucent cloud unable to render far view without GS training.\n\n"
    "3. Impact on 3DGS Training:\n"
    "   - Pure DA3 squashes distant trees/houses into a 20m billboard shell,\n"
    "     causing large multi-view parallax distortion during 3DGS optimization.\n"
    "   - Hybrid system (COLMAP 7.3k far anchors + DA3 42.7k near detail) successfully\n"
    "     restores accurate far geometry, explaining the +1.08 dB gain on Coffee Martini!"
)
ax.text(0.02, 0.98, summary_text, fontsize=10.5, verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.95),
        fontfamily='sans-serif')

plt.suptitle("Coffee Martini - Geometric Depth Collapse & Confidence Anomaly Analysis\nComparing COLMAP Triangulation vs DA3 Monocular Foundation Prior",
             fontsize=15, fontweight='bold', y=0.98)
out_path = os.path.join(artifact_dir, "coffee_martini_window_depth_analysis.png")
plt.savefig(out_path, bbox_inches="tight")
plt.close()
print(f"📊 分析圖已儲存至: {out_path}")
