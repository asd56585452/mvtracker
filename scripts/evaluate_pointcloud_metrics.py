#!/usr/bin/env python3
"""
Evaluate Point Cloud Distribution Metrics against 3DGS Converged Model (ckpt_10999_rank0.pt).

Compares:
  1. DA3 Importance Sample (100k pts)
  2. DA3 Uniform Sample (100k pts)
  3. COLMAP (100k pts)

Metrics:
  - Chamfer Distance (L1 and L2)
  - Earth Mover's Distance (EMD) via Exact OT (subsampled) & Sliced Wasserstein Distance (full 100k)
  - KL Divergence & Jensen-Shannon Divergence (Spatial Voxel Density Distribution)
"""

import os
import sys
import glob
import json
import argparse
import time
from collections import Counter
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial import cKDTree
import ot


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Point Cloud Metrics against 3DGS Checkpoints")
    parser.add_argument("--scenes", nargs="+", default=["sear_steak"],
                        help="List of scene names or 'all' for all 6 N3D scenes")
    parser.add_argument("--chunks", nargs="+", default=["all"],
                        help="List of chunks (e.g., '0-49', '50-99') or 'all'")
    parser.add_argument("--results_dir", type=str, default="/media/tsaichenghan/2TBSSD/results",
                        help="Base results directory containing models")
    parser.add_argument("--target_strategy", type=str, default="da3_importance_100k_init_STG_Strategy",
                        help="Target 3DGS strategy suffix (e.g. 'da3_importance_100k_init_STG_Strategy' or 'colmap_init_STG_Strategy')")
    parser.add_argument("--datasets_dir", type=str, default="datasets",
                        help="Base datasets directory containing point clouds")
    parser.add_argument("--num_samples", type=int, default=100000,
                        help="Number of points to sample for uniform and colmap baselines (default: 100,000)")
    parser.add_argument("--voxel_size", type=float, default=0.2,
                        help="Voxel size in meters for spatial probability density calculation (default: 0.2)")
    parser.add_argument("--emd_subsample", type=int, default=4096,
                        help="Subsample size for exact EMD Hungarian/Network-Simplex (default: 4096)")
    parser.add_argument("--swd_projections", type=int, default=100,
                        help="Number of random 1D projections for Sliced Wasserstein Distance (default: 100)")
    parser.add_argument("--output_json", "--output-json", type=str, default="pointcloud_benchmark_results.json",
                        help="Output path for detailed metrics JSON")
    parser.add_argument("--output_csv", "--output-csv", type=str, default="pointcloud_benchmark_results.csv",
                        help="Output path for summary CSV table")
    return parser.parse_args()


ALL_SCENES = [
    "coffee_martini",
    "cook_spinach",
    "cut_roasted_beef",
    "flame_salmon_1",
    "flame_steak",
    "sear_steak"
]

ALL_CHUNKS = [
    ("0-49", 0),
    ("50-99", 50),
    ("100-149", 100),
    ("150-199", 150),
    ("200-249", 200),
    ("250-299", 250)
]


def load_ply_points(path: str) -> np.ndarray:
    """Read XYZ coordinates from a PLY file."""
    ply = PlyData.read(path)
    v = ply["vertex"]
    pts = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    return pts


def load_ckpt_points(path: str) -> np.ndarray:
    """Read 3DGS Gaussian means from a checkpoint .pt file."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    means = ckpt["splats"]["means"].numpy().astype(np.float32)
    return means


def compute_chamfer_distance(p_src: np.ndarray, p_ref: np.ndarray) -> Tuple[float, float]:
    """Compute symmetric L1 and L2 Chamfer Distances."""
    tree_ref = cKDTree(p_ref)
    tree_src = cKDTree(p_src)
    d_src_to_ref, _ = tree_ref.query(p_src, k=1, workers=4)
    d_ref_to_src, _ = tree_src.query(p_ref, k=1, workers=4)
    cd_l1 = float(0.5 * (np.mean(d_src_to_ref) + np.mean(d_ref_to_src)))
    cd_l2 = float(0.5 * (np.mean(d_src_to_ref ** 2) + np.mean(d_ref_to_src ** 2)))
    return cd_l1, cd_l2


def compute_swd_and_emd(p_src: np.ndarray, p_ref: np.ndarray,
                        n_projections: int = 100, emd_subsample: int = 4096) -> Tuple[float, float]:
    """Compute Sliced Wasserstein Distance on full sets and exact EMD on subsamples."""
    # 1. Sliced Wasserstein Distance across full points
    swd = float(ot.sliced_wasserstein_distance(p_src, p_ref, n_projections=n_projections, seed=42))

    # 2. Exact EMD on subsample
    np.random.seed(42)
    idx_src = np.random.choice(len(p_src), min(emd_subsample, len(p_src)), replace=False)
    idx_ref = np.random.choice(len(p_ref), min(emd_subsample, len(p_ref)), replace=False)
    sub_src = p_src[idx_src]
    sub_ref = p_ref[idx_ref]

    cost_matrix = ot.dist(sub_src, sub_ref, metric="euclidean")
    a = np.ones(len(sub_src)) / len(sub_src)
    b = np.ones(len(sub_ref)) / len(sub_ref)
    emd_exact = float(ot.emd2(a, b, cost_matrix, numItermax=500000))

    return swd, emd_exact


def compute_kl_and_jsd(p_src: np.ndarray, p_ref: np.ndarray,
                       voxel_size: float = 0.2, eps: float = 1e-7) -> Tuple[float, float, float]:
    """
    Compute spatial voxel probability distributions and evaluate KL(ref || src),
    symmetric KL, and Jensen-Shannon Divergence.
    """
    all_pts = np.vstack([p_src, p_ref])
    min_bound = np.min(all_pts, axis=0) - voxel_size

    v_src = np.floor((p_src - min_bound) / voxel_size).astype(np.int32)
    v_ref = np.floor((p_ref - min_bound) / voxel_size).astype(np.int32)

    c_src = Counter(map(tuple, v_src))
    c_ref = Counter(map(tuple, v_ref))
    all_voxels = list(set(c_src.keys()) | set(c_ref.keys()))

    N_s = len(p_src)
    N_r = len(p_ref)
    V_total = len(all_voxels)

    p_s = np.array([(c_src.get(v, 0) + eps) / (N_s + eps * V_total) for v in all_voxels], dtype=np.float64)
    p_r = np.array([(c_ref.get(v, 0) + eps) / (N_r + eps * V_total) for v in all_voxels], dtype=np.float64)

    p_s /= p_s.sum()
    p_r /= p_r.sum()

    # KL(ref || src): How well does candidate src model/cover reference 3DGS ref
    kl_r_s = float(np.sum(p_r * np.log(p_r / p_s)))
    kl_s_r = float(np.sum(p_s * np.log(p_s / p_r)))
    sym_kl = float(0.5 * (kl_r_s + kl_s_r))

    # Jensen-Shannon Divergence (bounded in [0, ln 2])
    m = 0.5 * (p_r + p_s)
    jsd = float(0.5 * np.sum(p_r * np.log(p_r / m)) + 0.5 * np.sum(p_s * np.log(p_s / m)))

    return kl_r_s, sym_kl, jsd


def evaluate_chunk(scene: str, chunk_name: str, start_frame: int,
                   args: argparse.Namespace) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"🎬 評估場景: {scene} | 區間: Frame {chunk_name}")
    print(f"=======================================================")

    # 1. 讀取 Ground Truth 3DGS Target (ckpt_10999_rank0.pt)
    target_strategy = args.target_strategy
    ckpt_path = os.path.join(args.results_dir, f"{scene}_{target_strategy}", f"Frame{chunk_name}", "ckpts", "ckpt_10999_rank0.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"找不到 3DGS checkpoint 檔案: {ckpt_path}")
    p_target = load_ckpt_points(ckpt_path)
    print(f"🎯 Target 3DGS ({target_strategy}) 高斯點數: {len(p_target):,} (來自 ckpt_10999_rank0.pt)")

    # 2. 讀取 DA3 Importance Sampling (100k pts)
    da3_imp_files = glob.glob(os.path.join(args.datasets_dir, scene, f"{chunk_name}frame", "init_pointcloud_*_importance_100kpts_*.ply"))
    if not da3_imp_files:
        da3_imp_files = glob.glob(os.path.join(args.datasets_dir, scene, f"{chunk_name}frame", "init_pointcloud_*_importance*.ply"))
    if not da3_imp_files:
        raise FileNotFoundError(f"在 {args.datasets_dir}/{scene}/{chunk_name}frame 找不到 DA3 重要度採樣點雲 PLY！")
    p_da3_imp = load_ply_points(da3_imp_files[0])
    print(f"✨ DA3 Importance 點數: {len(p_da3_imp):,} ({os.path.basename(da3_imp_files[0])})")

    # 3. 讀取 DA3 Uniform Sampling (100k pts, 均勻隨機抽樣自 Full 4D 點雲)
    da3_full_files = glob.glob(os.path.join(args.datasets_dir, scene, f"{chunk_name}frame", "init_pointcloud_*_full_*.ply"))
    if not da3_full_files:
        raise FileNotFoundError(f"在 {args.datasets_dir}/{scene}/{chunk_name}frame 找不到 DA3 Full 4D 點雲 PLY！")
    p_da3_full = load_ply_points(da3_full_files[0])
    np.random.seed(42)
    if len(p_da3_full) > args.num_samples:
        idx_unif = np.random.choice(len(p_da3_full), args.num_samples, replace=False)
        p_da3_unif = p_da3_full[idx_unif]
    else:
        p_da3_unif = p_da3_full
    print(f"🎲 DA3 Uniform 點數: {len(p_da3_unif):,} (自 {len(p_da3_full):,} 點 Full 4D 均勻下採樣)")

    # 4. 讀取 COLMAP (100k pts, 抽樣自 COLMAP SfM 點雲)
    colmap_path = os.path.join(args.datasets_dir, scene, f"colmap_{start_frame}", "sparse", "0", "points3D_total50_sec_fps30.ply")
    if not os.path.isfile(colmap_path):
        colmap_path = os.path.join(args.results_dir, f"{scene}_colmap_init_STG_Strategy", "init_pcd.ply")
    if not os.path.isfile(colmap_path):
        raise FileNotFoundError(f"找不到 COLMAP 點雲檔案: {colmap_path}")
    p_colmap_all = load_ply_points(colmap_path)
    np.random.seed(42)
    if len(p_colmap_all) > args.num_samples:
        idx_col = np.random.choice(len(p_colmap_all), args.num_samples, replace=False)
        p_colmap = p_colmap_all[idx_col]
    else:
        p_colmap = p_colmap_all
    print(f"🏛️  COLMAP 點數: {len(p_colmap):,} (自 {len(p_colmap_all):,} 點 SfM 點雲下採樣)")

    candidates = {
        "DA3_Importance_100k": p_da3_imp,
        "DA3_Uniform_100k": p_da3_unif,
        "COLMAP_100k": p_colmap,
    }

    # 若存在對照策略的收斂 3DGS checkpoint，加入比較以驗證不同初始化的收斂幾何差異
    counterpart_strategy = "colmap_init_STG_Strategy" if "da3" in target_strategy else "da3_importance_100k_init_STG_Strategy"
    counterpart_ckpt = os.path.join(args.results_dir, f"{scene}_{counterpart_strategy}", f"Frame{chunk_name}", "ckpts", "ckpt_10999_rank0.pt")
    if os.path.isfile(counterpart_ckpt):
        tag = "Colmap_Init_Trained_10999" if "colmap" in counterpart_strategy else "DA3_Init_Trained_10999"
        candidates[tag] = load_ckpt_points(counterpart_ckpt)

    results = {}
    for name, pts in candidates.items():
        t0 = time.time()
        cd_l1, cd_l2 = compute_chamfer_distance(pts, p_target)
        swd, emd = compute_swd_and_emd(pts, p_target, n_projections=args.swd_projections, emd_subsample=args.emd_subsample)
        kl, sym_kl, jsd = compute_kl_and_jsd(pts, p_target, voxel_size=args.voxel_size)
        elapsed = time.time() - t0

        results[name] = {
            "num_points": int(len(pts)),
            "chamfer_l1_m": cd_l1,
            "chamfer_l2_m2": cd_l2,
            "emd_exact_m": emd,
            "sliced_wasserstein_m": swd,
            "kl_divergence": kl,
            "symmetric_kl": sym_kl,
            "jensen_shannon_div": jsd,
            "eval_time_sec": elapsed,
        }
        print(f"  👉 [{name:20s}] CD-L1: {cd_l1:.4f}m | SWD: {swd:.4f}m | EMD: {emd:.4f}m | KL: {kl:.4f} | JSD: {jsd:.4f} ({elapsed:.2f}s)")

    return {
        "scene": scene,
        "chunk": chunk_name,
        "start_frame": start_frame,
        "target_strategy": target_strategy,
        "target_points": int(len(p_target)),
        "metrics": results,
    }


def main():
    args = parse_args()

    scenes = ALL_SCENES if "all" in [s.lower() for s in args.scenes] else args.scenes
    chunks = ALL_CHUNKS if "all" in [c.lower() for c in args.chunks] else [c for c in ALL_CHUNKS if c[0] in args.chunks]

    all_chunk_results = []
    print("=" * 70)
    print(" 4D 點雲分佈指標綜合評估 (Chamfer Distance, EMD, KL Divergence)")
    print(f" 目標訓練策略: {args.target_strategy}")
    print(f" 評估場景列表: {scenes}")
    print(f" 評估片段列表: {[c[0] for c in chunks]}")
    print("=" * 70)

    for scene in scenes:
        for chunk_name, start_frame in chunks:
            try:
                res = evaluate_chunk(scene, chunk_name, start_frame, args)
                all_chunk_results.append(res)
            except Exception as e:
                print(f"❌ 評估 {scene} Frame {chunk_name} 時發生錯誤: {e}")

    # 儲存詳細結果 JSON
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_chunk_results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 詳細評估指標 JSON 已儲存至: {args.output_json}")

    # 彙總並輸出 CSV
    import csv
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Scene", "Chunk", "Target_Strategy", "Method", "Num_Points", "Target_Points",
            "Chamfer_L1_m", "Chamfer_L2_m2", "Sliced_Wasserstein_m", "EMD_Exact_m",
            "KL_Divergence", "Jensen_Shannon_Div"
        ])
        for r in all_chunk_results:
            sc = r["scene"]
            ck = r["chunk"]
            ts = r.get("target_strategy", args.target_strategy)
            tp = r["target_points"]
            for m_name, m_vals in r["metrics"].items():
                writer.writerow([
                    sc, ck, ts, m_name, m_vals["num_points"], tp,
                    f"{m_vals['chamfer_l1_m']:.4f}",
                    f"{m_vals['chamfer_l2_m2']:.4f}",
                    f"{m_vals['sliced_wasserstein_m']:.4f}",
                    f"{m_vals['emd_exact_m']:.4f}",
                    f"{m_vals['kl_divergence']:.4f}",
                    f"{m_vals['jensen_shannon_div']:.4f}",
                ])
    print(f"📊 彙總表格 CSV 已儲存至: {args.output_csv}")

    # 計算平均值
    summary_by_method = {}
    for r in all_chunk_results:
        for m_name, m_vals in r["metrics"].items():
            if m_name not in summary_by_method:
                summary_by_method[m_name] = {
                    "cd_l1": [], "cd_l2": [], "swd": [], "emd": [], "kl": [], "jsd": []
                }
            summary_by_method[m_name]["cd_l1"].append(m_vals["chamfer_l1_m"])
            summary_by_method[m_name]["cd_l2"].append(m_vals["chamfer_l2_m2"])
            summary_by_method[m_name]["swd"].append(m_vals["sliced_wasserstein_m"])
            summary_by_method[m_name]["emd"].append(m_vals["emd_exact_m"])
            summary_by_method[m_name]["kl"].append(m_vals["kl_divergence"])
            summary_by_method[m_name]["jsd"].append(m_vals["jensen_shannon_div"])

    print("\n" + "=" * 80)
    print("📈 全評估片段平均指標總結 (Average Across Evaluated Chunks)")
    print("=" * 80)
    print(f"| {'方法 (Method)':<22} | {'Chamfer L1 (m)':<15} | {'SWD (m)':<10} | {'EMD (m)':<10} | {'KL 散度':<10} | {'JSD 散度':<10} |")
    print(f"| {':---':<22} | {':---:':<15} | {':---:':<10} | {':---:':<10} | {':---:':<10} | {':---:':<10} |")
    for m_name, vals in summary_by_method.items():
        print(f"| **{m_name:<20}** | {np.mean(vals['cd_l1']):<15.4f} | {np.mean(vals['swd']):<10.4f} | {np.mean(vals['emd']):<10.4f} | {np.mean(vals['kl']):<10.4f} | {np.mean(vals['jsd']):<10.4f} |")
    print("=" * 80)


if __name__ == "__main__":
    main()
