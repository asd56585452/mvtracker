#!/usr/bin/env python3
"""
Visualize Point Cloud Error Maps for sear_steak Frame 0-49.
Compares 4 candidate point clouds against the target reference (colmap_init_STG_Strategy ckpt_10999):
  1. COLMAP (100k)
  2. DA3 Importance (100k)
  3. DA3 Uniform (100k)
  4. DA3-init Trained 3DGS (ckpt_10999)

Features:
  - Joint normalization across all 4 point clouds for fair side-by-side comparison.
  - Chamfer Dist L1 point-wise error maps.
  - EMD per-point transport cost (subsampled 4096 points).
  - Spatial Voxel JSD & KL divergence point-wise error heatmaps (voxel size: 0.2m).
  - Exports colored PLY point clouds (clean headers, no bogus normals, outlier-bounded).
  - Generates publication-grade multi-view comparison figures with shared colorbars.
"""

import os
import time
import argparse
from collections import Counter
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
import ot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm


def load_ply_coords(path: str) -> np.ndarray:
    ply = PlyData.read(path)
    v = ply["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)


def load_ckpt_coords(path: str) -> np.ndarray:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["splats"]["means"].numpy().astype(np.float32)


def export_colored_ply(path: str, points: np.ndarray, colors_rgb: np.ndarray,
                       scalar_val: np.ndarray, scalar_name: str = "error"):
    """
    Export points with RGB colors [0, 255] and a scalar attribute.
    Clean vertex format without bogus normals so all viewers render it properly.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    N = len(points)
    vertex_dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        (scalar_name, 'f4')
    ]
    vertex_data = np.empty(N, dtype=vertex_dtype)
    vertex_data['x'] = points[:, 0]
    vertex_data['y'] = points[:, 1]
    vertex_data['z'] = points[:, 2]
    vertex_data['red'] = np.clip(colors_rgb[:, 0], 0, 255).astype(np.uint8)
    vertex_data['green'] = np.clip(colors_rgb[:, 1], 0, 255).astype(np.uint8)
    vertex_data['blue'] = np.clip(colors_rgb[:, 2], 0, 255).astype(np.uint8)
    vertex_data[scalar_name] = scalar_val.astype(np.float32)

    el = PlyElement.describe(vertex_data, 'vertex')
    PlyData([el], text=False).write(path)
    print(f"  💾 Exported: {path} ({N:,} points)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="sear_steak")
    parser.add_argument("--chunk", type=str, default="0-49")
    parser.add_argument("--results_dir", type=str, default="/media/tsaichenghan/2TBSSD/results")
    parser.add_argument("--datasets_dir", type=str, default="datasets")
    parser.add_argument("--output_dir", type=str, default="visualizations/sear_steak_f0")
    parser.add_argument("--voxel_size", type=float, default=0.2)
    parser.add_argument("--emd_samples", type=int, default=4096)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 70)
    print(f" 點雲誤差空間可視化生成: {args.scene} Frame {args.chunk}")
    print("=" * 70)

    # 1. 讀取 Ground Truth Reference (Colmap-init 10999 ckpt)
    ref_ckpt = os.path.join(args.results_dir, f"{args.scene}_colmap_init_STG_Strategy", f"Frame{args.chunk}", "ckpts", "ckpt_10999_rank0.pt")
    pts_ref = load_ckpt_coords(ref_ckpt)
    print(f"🎯 Reference 3DGS (colmap_init 10999): {len(pts_ref):,} 點")

    # 2. 讀取 4 種候選點雲
    # (A) COLMAP 100k
    colmap_ply = os.path.join(args.datasets_dir, args.scene, "colmap_0", "sparse", "0", "points3D_total50_sec_fps30.ply")
    p_col_all = load_ply_coords(colmap_ply)
    rng = np.random.RandomState(42)
    idx_col = rng.choice(len(p_col_all), min(100000, len(p_col_all)), replace=False)
    pts_colmap = p_col_all[idx_col]
    print(f"📦 [1] COLMAP 100k: {len(pts_colmap):,} 點")

    # (B) DA3 Importance 100k
    imp_ply = os.path.join(args.datasets_dir, args.scene, f"{args.chunk}frame", "init_pointcloud_da3_large_4d_importance_100kpts_surfel_s1.0_min0.01_ang25_res672.ply")
    pts_da3_imp = load_ply_coords(imp_ply)
    print(f"✨ [2] DA3 Importance 100k: {len(pts_da3_imp):,} 點")

    # (C) DA3 Uniform 100k
    full_ply = os.path.join(args.datasets_dir, args.scene, f"{args.chunk}frame", "init_pointcloud_da3_large_4d_full_surfel_s1.0_min0.01_ang25_res672.ply")
    p_full_all = load_ply_coords(full_ply)
    idx_unif = rng.choice(len(p_full_all), min(100000, len(p_full_all)), replace=False)
    pts_da3_unif = p_full_all[idx_unif]
    print(f"🎲 [3] DA3 Uniform 100k: {len(pts_da3_unif):,} 點")

    # (D) DA3-init Trained 3DGS (ckpt 10999)
    da3_ckpt = os.path.join(args.results_dir, f"{args.scene}_da3_importance_100k_init_STG_Strategy", f"Frame{args.chunk}", "ckpts", "ckpt_10999_rank0.pt")
    pts_da3_trained = load_ckpt_coords(da3_ckpt)
    print(f"🚀 [4] DA3-Init Trained 10999 ckpt: {len(pts_da3_trained):,} 點")

    candidates = {
        "COLMAP_100k": pts_colmap,
        "DA3_Importance_100k": pts_da3_imp,
        "DA3_Uniform_100k": pts_da3_unif,
        "DA3_Init_Trained_10999": pts_da3_trained,
    }

    # =========================================================================
    # 一、Chamfer Dist L1 逐點誤差計算與【跨 4 種方法全域聯合歸一化】
    # =========================================================================
    print("\n" + "-" * 60)
    print("📏 計算 Chamfer L1 逐點距離 (Point-to-Reference Nearest Neighbor)...")
    print("-" * 60)

    tree_ref = cKDTree(pts_ref)
    chamfer_dists = {}
    for name, pts in candidates.items():
        dists, _ = tree_ref.query(pts, k=1, workers=4)
        chamfer_dists[name] = dists
        print(f"  👉 {name:24s} | Mean: {dists.mean():.4f}m | Median: {np.median(dists):.4f}m | 95%: {np.percentile(dists, 95):.4f}m")

    # 聯合歸一化：合併 4 種點雲的所有距離，求取共用顏色範圍
    all_cd_dists = np.concatenate(list(chamfer_dists.values()))
    cd_vmin = 0.0
    # 取 98 百分位數作為全域上限截斷，避免極少數極端離群點使色條壓縮失去對比
    cd_vmax = float(np.percentile(all_cd_dists, 97.5))
    cd_vmax = round(cd_vmax, 2)  # 例如 ~1.5m
    print(f"\n🎨 Chamfer L1 全域統一色標範圍 (Shared Normalization): [0.0m, {cd_vmax}m]")

    cmap_cd = plt.cm.get_cmap("turbo")
    norm_cd = Normalize(vmin=cd_vmin, vmax=cd_vmax, clip=True)

    # 匯出 Chamfer L1 著色點雲 PLY
    print("\n💾 匯出 Chamfer L1 著色點雲 PLY (Turbo Colormap)...")
    for name, pts in candidates.items():
        dists = chamfer_dists[name]
        colors_rgba = cmap_cd(norm_cd(dists))
        colors_rgb = (colors_rgba[:, :3] * 255.0).astype(np.uint8)

        # 過濾 COLMAP 極端外太空雜訊點，維持適當包圍盒以便 3D 檢視器直接置中
        mask_box = (np.abs(pts[:, 0]) < 35) & (np.abs(pts[:, 1]) < 30) & (pts[:, 2] > 0) & (pts[:, 2] < 50)
        out_ply = os.path.join(args.output_dir, f"chamfer_{name.lower()}.ply")
        export_colored_ply(out_ply, pts[mask_box], colors_rgb[mask_box], dists[mask_box], "chamfer_l1_m")

    # 匯出 Reference 基準點雲 PLY
    ref_rgb = np.ones((len(pts_ref), 3), dtype=np.uint8) * 180
    export_colored_ply(os.path.join(args.output_dir, "ref_colmap_init_10999.ply"), pts_ref, ref_rgb, np.zeros(len(pts_ref)), "ref")

    # =========================================================================
    # 二、EMD 搬運距離 (m) 抽樣點計算與【跨 4 種方法全域聯合歸一化】
    # =========================================================================
    print("\n" + "-" * 60)
    print(f"🚚 計算 EMD 最佳傳輸搬運成本 (抽樣 {args.emd_samples} 點 Optimal Transport)...")
    print("-" * 60)

    idx_ref_sub = rng.choice(len(pts_ref), args.emd_samples, replace=False)
    pts_ref_sub = pts_ref[idx_ref_sub]
    b = np.ones(args.emd_samples) / args.emd_samples

    emd_costs = {}
    emd_sub_pts = {}
    for name, pts in candidates.items():
        t0 = time.time()
        idx_sub = rng.choice(len(pts), args.emd_samples, replace=False)
        sub_pts = pts[idx_sub]
        a = np.ones(args.emd_samples) / args.emd_samples

        cost_matrix = ot.dist(sub_pts, pts_ref_sub, metric="euclidean")
        gamma = ot.emd(a, b, cost_matrix, numItermax=1000000)
        # 每個點 i 的搬運距離 = sum_j gamma_ij * cost_ij / a_i
        per_pt_cost = (np.sum(gamma * cost_matrix, axis=1) / a).astype(np.float32)

        emd_costs[name] = per_pt_cost
        emd_sub_pts[name] = sub_pts
        print(f"  👉 {name:24s} | Mean EMD: {per_pt_cost.mean():.4f}m | Median: {np.median(per_pt_cost):.4f}m ({time.time()-t0:.2f}s)")

    # 聯合歸一化 EMD
    all_emd_costs = np.concatenate(list(emd_costs.values()))
    emd_vmin = 0.0
    emd_vmax = float(np.percentile(all_emd_costs, 98))
    emd_vmax = round(emd_vmax, 2)
    print(f"\n🎨 EMD 搬運距離全域統一色標範圍: [0.0m, {emd_vmax}m]")

    norm_emd = Normalize(vmin=emd_vmin, vmax=emd_vmax, clip=True)
    cmap_emd = plt.cm.get_cmap("plasma")

    # 匯出 EMD PLY
    for name, sub_pts in emd_sub_pts.items():
        costs = emd_costs[name]
        c_rgba = cmap_emd(norm_emd(costs))
        c_rgb = (c_rgba[:, :3] * 255.0).astype(np.uint8)
        out_ply = os.path.join(args.output_dir, f"emd_{name.lower()}_sub4096.ply")
        export_colored_ply(out_ply, sub_pts, c_rgb, costs, "emd_cost_m")

    # =========================================================================
    # 三、空間體素 KL 散度 & JSD 散度【體素化 3D 誤差分佈】
    # =========================================================================
    print("\n" + "-" * 60)
    print(f"🧊 計算空間體素 KL & JSD 密度誤差 (Voxel Size: {args.voxel_size}m)...")
    print("-" * 60)

    # 建立統一空間包圍盒
    all_points_stacked = np.vstack([pts_ref] + list(candidates.values()))
    # 限制在場景主要區域內以建立體素網格
    mask_inliers = (np.abs(all_points_stacked[:, 0]) < 25) & (np.abs(all_points_stacked[:, 1]) < 20) & (all_points_stacked[:, 2] > 2) & (all_points_stacked[:, 2] < 30)
    inlier_pts = all_points_stacked[mask_inliers]
    min_bound = np.min(inlier_pts, axis=0) - args.voxel_size

    def to_voxel(pts):
        return np.floor((pts - min_bound) / args.voxel_size).astype(np.int32)

    v_ref = to_voxel(pts_ref)
    c_ref = Counter(map(tuple, v_ref))

    voxel_jsd_data = {}
    voxel_kl_data = {}
    voxel_coords = {}

    eps = 1e-7
    for name, pts in candidates.items():
        v_cand = to_voxel(pts)
        c_cand = Counter(map(tuple, v_cand))

        # 兩者活躍體素聯集
        active_v = list(set(c_ref.keys()) | set(c_cand.keys()))
        V_tot = len(active_v)
        Nr = len(pts_ref)
        Nc = len(pts)

        p_r = np.array([(c_ref.get(v, 0) + eps) / (Nr + eps * V_tot) for v in active_v], dtype=np.float64)
        p_c = np.array([(c_cand.get(v, 0) + eps) / (Nc + eps * V_tot) for v in active_v], dtype=np.float64)
        p_r /= p_r.sum()
        p_c /= p_c.sum()

        # 各體素 JSD 貢獻 (嚴格 >= 0, 物理界限清楚)
        m = 0.5 * (p_r + p_c)
        jsd_v = (0.5 * p_r * np.log(p_r / m) + 0.5 * p_c * np.log(p_c / m)).astype(np.float32)

        # 各體素 KL 貢獻
        kl_v = (p_r * np.log(p_r / p_c)).astype(np.float32)

        # 體素中心坐標
        v_arr = np.array(active_v, dtype=np.float32)
        v_centers = min_bound + (v_arr + 0.5) * args.voxel_size

        voxel_jsd_data[name] = jsd_v
        voxel_kl_data[name] = kl_v
        voxel_coords[name] = v_centers
        print(f"  👉 {name:24s} | 活躍體素數: {V_tot:,} | 總 JSD: {jsd_v.sum():.4f} | 總 KL: {kl_v.sum():.4f}")

    # 聯合歸一化 Voxel JSD
    all_jsd_v = np.concatenate(list(voxel_jsd_data.values()))
    # 只看具有誤差的體素
    pos_jsd = all_jsd_v[all_jsd_v > 1e-6]
    jsd_vmax = float(np.percentile(pos_jsd, 99))
    print(f"\n🎨 體素 JSD 誤差全域統一色標範圍: [0.0, {jsd_vmax:.6f}]")

    norm_jsd = Normalize(vmin=0.0, vmax=jsd_vmax, clip=True)
    cmap_jsd = plt.cm.get_cmap("magma")

    # 匯出 Voxel JSD 3D 點雲 PLY
    for name in candidates.keys():
        v_centers = voxel_coords[name]
        jsd_v = voxel_jsd_data[name]
        # 只保留有顯著機率密度的體素，避免大量接近 0 的空體素干擾視覺
        sig_mask = jsd_v > (jsd_vmax * 0.05)
        sub_centers = v_centers[sig_mask]
        sub_jsd = jsd_v[sig_mask]

        c_rgba = cmap_jsd(norm_jsd(sub_jsd))
        c_rgb = (c_rgba[:, :3] * 255.0).astype(np.uint8)
        out_ply = os.path.join(args.output_dir, f"voxel_jsd_{name.lower()}.ply")
        export_colored_ply(out_ply, sub_centers, c_rgb, sub_jsd, "jsd_density_error")

    # =========================================================================
    # 四、繪製多重視角 publication-grade 對比大圖 (PNG Dashboard)
    # =========================================================================
    print("\n" + "-" * 60)
    print("📊 繪製綜合誤差多視角視覺化圖表面板 (High-Resolution Dashboard)...")
    print("-" * 60)

    method_titles = [
        ("COLMAP_100k", "COLMAP (100k pts)"),
        ("DA3_Importance_100k", "DA3 Importance (100k pts)"),
        ("DA3_Uniform_100k", "DA3 Uniform (100k pts)"),
        ("DA3_Init_Trained_10999", "DA3-Init Trained (ckpt 10999)"),
    ]

    fig, axes = plt.subplots(4, 4, figsize=(22, 18), dpi=150)
    plt.subplots_adjust(top=0.93, bottom=0.06, left=0.04, right=0.92, hspace=0.30, wspace=0.15)
    fig.suptitle(f"Point Cloud Error Distribution Benchmarking: {args.scene} Frame {args.chunk}\n(Target Reference: colmap_init_STG_Strategy 10999 ckpt | Globally Normalized)",
                 fontsize=18, fontweight='bold')

    # 主視窗視野限制 (牛排工作台核心區)
    xlim = (-12, 14)
    ylim = (-7, 8)
    zlim = (7, 20)

    # Row 0: Chamfer L1 俯視圖 (X-Z 平面 Top-Down View)
    for col_idx, (m_key, title) in enumerate(method_titles):
        ax = axes[0, col_idx]
        pts = candidates[m_key]
        dists = chamfer_dists[m_key]

        # 隨機子取樣 40k 點以繪製平滑散佈圖
        plot_idx = rng.choice(len(pts), min(40000, len(pts)), replace=False)
        p_sub = pts[plot_idx]
        d_sub = dists[plot_idx]

        sc = ax.scatter(p_sub[:, 0], p_sub[:, 2], c=d_sub, cmap="turbo", norm=norm_cd, s=1.2, alpha=0.7)
        ax.set_xlim(xlim); ax.set_ylim(zlim)
        ax.set_title(f"{title}\nChamfer L1 (Top-Down X-Z)", fontsize=12, fontweight='bold')
        ax.set_xlabel("X (meters)", fontsize=10)
        if col_idx == 0:
            ax.set_ylabel("Z Depth (meters)", fontsize=11, fontweight='bold')
        ax.grid(True, linestyle="--", alpha=0.3)

    # 放置 Row 0 Colorbar
    cbar_ax0 = fig.add_axes([0.93, 0.725, 0.015, 0.18])
    cbar0 = fig.colorbar(cm.ScalarMappable(norm=norm_cd, cmap="turbo"), cax=cbar_ax0)
    cbar0.set_label("Chamfer L1 Distance (meters)", fontsize=11, fontweight='bold')

    # Row 1: Chamfer L1 正視圖 (X-Y 平面 Front View)
    for col_idx, (m_key, title) in enumerate(method_titles):
        ax = axes[1, col_idx]
        pts = candidates[m_key]
        dists = chamfer_dists[m_key]

        plot_idx = rng.choice(len(pts), min(40000, len(pts)), replace=False)
        p_sub = pts[plot_idx]
        d_sub = dists[plot_idx]

        ax.scatter(p_sub[:, 0], p_sub[:, 1], c=d_sub, cmap="turbo", norm=norm_cd, s=1.2, alpha=0.7)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_title(f"Chamfer L1 (Front X-Y)", fontsize=11)
        ax.set_xlabel("X (meters)", fontsize=10)
        if col_idx == 0:
            ax.set_ylabel("Y Height (meters)", fontsize=11, fontweight='bold')
        ax.grid(True, linestyle="--", alpha=0.3)

    # Row 2: EMD Transport Cost 搬運距離 (抽樣 4096 點)
    for col_idx, (m_key, title) in enumerate(method_titles):
        ax = axes[2, col_idx]
        sub_pts = emd_sub_pts[m_key]
        costs = emd_costs[m_key]

        ax.scatter(sub_pts[:, 0], sub_pts[:, 2], c=costs, cmap="plasma", norm=norm_emd, s=6.0, alpha=0.85, edgecolors='none')
        ax.set_xlim(xlim); ax.set_ylim(zlim)
        ax.set_title(f"EMD Cost (Top-Down X-Z)", fontsize=11)
        ax.set_xlabel("X (meters)", fontsize=10)
        if col_idx == 0:
            ax.set_ylabel("Z Depth (meters)", fontsize=11, fontweight='bold')
        ax.grid(True, linestyle="--", alpha=0.3)

    # 放置 Row 2 Colorbar
    cbar_ax2 = fig.add_axes([0.93, 0.285, 0.015, 0.18])
    cbar2 = fig.colorbar(cm.ScalarMappable(norm=norm_emd, cmap="plasma"), cax=cbar_ax2)
    cbar2.set_label("EMD Transport Distance (m)", fontsize=11, fontweight='bold')

    # Row 3: 空間體素 JSD 密度誤差熱力圖 (Voxel Heatmap)
    for col_idx, (m_key, title) in enumerate(method_titles):
        ax = axes[3, col_idx]
        v_centers = voxel_coords[m_key]
        jsd_v = voxel_jsd_data[m_key]

        # 篩選活躍體素
        v_mask = (jsd_v > jsd_vmax * 0.02) & (v_centers[:, 0] > xlim[0]) & (v_centers[:, 0] < xlim[1]) & (v_centers[:, 2] > zlim[0]) & (v_centers[:, 2] < zlim[1])
        sub_c = v_centers[v_mask]
        sub_j = jsd_v[v_mask]

        ax.scatter(sub_c[:, 0], sub_c[:, 2], c=sub_j, cmap="magma", norm=norm_jsd, s=12.0, alpha=0.9, marker='s', edgecolors='none')
        ax.set_xlim(xlim); ax.set_ylim(zlim)
        ax.set_title(f"Voxel JSD Divergence Error", fontsize=11)
        ax.set_xlabel("X (meters)", fontsize=10)
        if col_idx == 0:
            ax.set_ylabel("Z Depth (meters)", fontsize=11, fontweight='bold')
        ax.grid(True, linestyle="--", alpha=0.3)

    # 放置 Row 3 Colorbar
    cbar_ax3 = fig.add_axes([0.93, 0.065, 0.015, 0.18])
    cbar3 = fig.colorbar(cm.ScalarMappable(norm=norm_jsd, cmap="magma"), cax=cbar_ax3)
    cbar3.set_label("Voxel JSD Error Contribution", fontsize=11, fontweight='bold')

    dash_path = os.path.join(args.output_dir, "pointcloud_error_comparison_dashboard.png")
    plt.savefig(dash_path, bbox_inches='tight')
    plt.close()
    print(f"\n🖼️  綜合誤差視覺化面板已儲存至: {dash_path}")

    # =========================================================================
    # 五、繪製 Chamfer 與 EMD 誤差分佈直方圖對比 (Distribution Histograms)
    # =========================================================================
    fig_hist, (ax_h1, ax_h2) = plt.subplots(1, 2, figsize=(16, 5), dpi=150)

    colors_list = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]
    labels_list = ["COLMAP (100k)", "DA3 Importance (100k)", "DA3 Uniform (100k)", "DA3-Init Trained (10999)"]

    for (m_key, label), col in zip(method_titles, colors_list):
        dists = chamfer_dists[m_key]
        ax_h1.hist(dists, bins=80, range=(0, 2.0), density=True, histtype='step', linewidth=2.2, label=f"{label} (Mean={dists.mean():.3f}m)", color=col)

    ax_h1.set_title("Chamfer L1 Distance Error Distribution (Shared Scale)", fontsize=12, fontweight='bold')
    ax_h1.set_xlabel("Chamfer L1 Distance to Reference (meters)", fontsize=10)
    ax_h1.set_ylabel("Probability Density", fontsize=10)
    ax_h1.grid(True, linestyle="--", alpha=0.4)
    ax_h1.legend(fontsize=9)

    for (m_key, label), col in zip(method_titles, colors_list):
        costs = emd_costs[m_key]
        ax_h2.hist(costs, bins=80, range=(0, 10.0), density=True, histtype='step', linewidth=2.2, label=f"{label} (Mean={costs.mean():.3f}m)", color=col)

    ax_h2.set_title("EMD Per-Point Transport Distance Distribution", fontsize=12, fontweight='bold')
    ax_h2.set_xlabel("Optimal Transport Distance (meters)", fontsize=10)
    ax_h2.set_ylabel("Probability Density", fontsize=10)
    ax_h2.grid(True, linestyle="--", alpha=0.4)
    ax_h2.legend(fontsize=9)

    hist_path = os.path.join(args.output_dir, "error_distribution_histograms.png")
    plt.savefig(hist_path, bbox_inches='tight')
    plt.close()
    print(f"📈 誤差分佈直方圖已儲存至: {hist_path}")

    print("\n" + "=" * 70)
    print("🎉 視覺化產出與檔案清單完成！")
    print(f"📂 輸出目錄: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
