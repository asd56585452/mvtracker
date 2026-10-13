#!/usr/bin/env python3
"""
3DGS 切片導向之 3D 空間特徵權重凸優化求解器 (optimize_spatial_weights_3dgs.py)

功能：
1. 讀取指定資料集（預設 datasets/sear_steak）frame 0 多視角影像與相機幾何
2. 提取 2D 特徵（Sobel, DCT, ChromaGrad），實施「跨視角共享全域尺度校準 Q_global」，防禦單圖白牆噪點放大
3. 將多視角 RGB、深度與特徵反投影至 3D 空間
4. 呼叫 spatial_voxelizer 執行特徵向量 Max-Pooling 體素化（支援均勻與動態足跡體素）
5. 讀取 3DGS 切片 PLY（datasets/sear_steak/ckpt_10999_rank0_frame0.ply），建構真實高斯機率分佈 P_ideal
6. 透過 SLSQP 最小化 KL 散度，求解最佳特徵組合權重 [w0*, w1*, w2*, w3*]
7. 評估各特徵單獨與組合之 KL 散度及 Top-K% 高斯捕捉率，並匯出結果為 JSON
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import json
import time
import struct
import argparse
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import plyfile
from PIL import Image
from scipy.spatial import cKDTree
from scipy.optimize import minimize

from mvtracker.utils.feature_scores import RGBTextureScorer, depth_to_surface_normals
from mvtracker.utils.spatial_voxelizer import (
    voxel_downsample_unified_with_features,
    unproject_camera_points_with_features
)
from mvtracker.utils.camera_selection import (
    select_cams_by_da3_surfel,
    select_cams_by_fps
)


def load_colmap_points(colmap_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    載入 COLMAP 3D 稀疏點雲。支援 .bin (points3D.bin), .ply, .txt
    回傳: (points: [N, 3] float32, colors: [N, 3] uint8)
    """
    if colmap_path.endswith('.bin'):
        points, colors = [], []
        with open(colmap_path, 'rb') as f:
            num_points = struct.unpack('<Q', f.read(8))[0]
            for _ in range(num_points):
                binary_point = struct.unpack('<QdddBBBd', f.read(43))
                xyz = binary_point[1:4]
                rgb = binary_point[4:7]
                track_length = struct.unpack('<Q', f.read(8))[0]
                f.read(8 * track_length)
                points.append(xyz)
                colors.append(rgb)
        return np.array(points, dtype=np.float32), np.array(colors, dtype=np.uint8)
    elif colmap_path.endswith('.ply'):
        ply = plyfile.PlyData.read(colmap_path)
        vx = np.array(ply['vertex']['x'], dtype=np.float32)
        vy = np.array(ply['vertex']['y'], dtype=np.float32)
        vz = np.array(ply['vertex']['z'], dtype=np.float32)
        r = np.array(ply['vertex']['red'], dtype=np.uint8) if 'red' in ply['vertex'] else np.zeros_like(vx, dtype=np.uint8)
        g = np.array(ply['vertex']['green'], dtype=np.uint8) if 'green' in ply['vertex'] else np.zeros_like(vx, dtype=np.uint8)
        b = np.array(ply['vertex']['blue'], dtype=np.uint8) if 'blue' in ply['vertex'] else np.zeros_like(vx, dtype=np.uint8)
        return np.stack([vx, vy, vz], axis=-1), np.stack([r, g, b], axis=-1)
    elif colmap_path.endswith('.txt'):
        points, colors = [], []
        with open(colmap_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                parts = line.split()
                xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
                rgb = [int(parts[4]), int(parts[5]), int(parts[6])]
                points.append(xyz)
                colors.append(rgb)
        return np.array(points, dtype=np.float32), np.array(colors, dtype=np.uint8)
    else:
        raise ValueError(f"不支援的 COLMAP 點雲格式: {colmap_path}")


def assign_normals_to_colmap_points(
    colmap_pts: np.ndarray,
    da3_pts: Optional[np.ndarray],
    da3_normals: Optional[np.ndarray],
    cams_pos: np.ndarray,
    max_nn_dist: float = 0.15,
) -> np.ndarray:
    """
    為無 normal 的 COLMAP 點雲自適應指派表面單位法向量：
    1. 近場點 (與最近 DA3 surfel 距離 < max_nn_dist)：繼承該 DA3 surfel 之單位法向，保持表面幾何共面性。
    2. 遠場/孤立點 (距 DA3 表面 > max_nn_dist，例如戶外窗外景物)：以相機中心至該點的射線反向 (面向相機) 作為 billboard 法向。
    """
    N = len(colmap_pts)
    normals = np.zeros((N, 3), dtype=np.float32)
    
    # 預設：面向最近相機的視線法向量 (Billboard Normal)
    dists_to_cams = np.linalg.norm(colmap_pts[:, None, :] - cams_pos[None, :, :], axis=-1)  # [N, num_cams]
    nearest_cam_idx = np.argmin(dists_to_cams, axis=-1)  # [N]
    nearest_cam_centers = cams_pos[nearest_cam_idx]  # [N, 3]
    view_dirs = nearest_cam_centers - colmap_pts  # 朝向相機中心
    view_lens = np.linalg.norm(view_dirs, axis=-1, keepdims=True)
    billboard_normals = (view_dirs / np.clip(view_lens, 1e-6, None)).astype(np.float32)
    normals[:] = billboard_normals

    # 近場轉移：若有 DA3 點與法向量，透過 KD-Tree 轉移近場表面法向量
    if da3_pts is not None and len(da3_pts) > 0 and da3_normals is not None and len(da3_normals) > 0:
        if len(da3_pts) > 100_000:
            step = len(da3_pts) // 100_000
            tree_pts = da3_pts[::step]
            tree_norms = da3_normals[::step]
        else:
            tree_pts = da3_pts
            tree_norms = da3_normals

        tree = cKDTree(tree_pts)
        nn_dists, nn_idx = tree.query(colmap_pts, k=1)
        is_near = nn_dists < max_nn_dist
        if np.any(is_near):
            normals[is_near] = tree_norms[nn_idx[is_near]]

    norm_len = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = (normals / np.clip(norm_len, 1e-6, None)).astype(np.float32)
    return normals


def parse_cam_idx(cam_name: str) -> int:
    digits = "".join(filter(str.isdigit, os.path.basename(cam_name)))
    return int(digits) if digits else 0


def get_sorted_frame_paths(view_dir: str) -> List[str]:
    frames = sorted(
        glob.glob(os.path.join(view_dir, '*.png')) + glob.glob(os.path.join(view_dir, '*.jpg')),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0]) if os.path.splitext(os.path.basename(x))[0].isdigit() else os.path.basename(x)
    )
    if not frames:
        frames = sorted(
            glob.glob(os.path.join(view_dir, 'images', '*.png')) + glob.glob(os.path.join(view_dir, 'images', '*.jpg')),
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0]) if os.path.splitext(os.path.basename(x))[0].isdigit() else os.path.basename(x)
        )
    return frames


def llff_to_opencv_w2c(pose_llff: np.ndarray, actual_W: int, actual_H: int) -> Tuple[np.ndarray, np.ndarray]:
    H_bound, W_bound, f_bound = pose_llff[:, 4]
    scale_x = actual_W / W_bound
    scale_y = actual_H / H_bound

    K = np.array([
        [f_bound * scale_x, 0, actual_W / 2.0],
        [0, f_bound * scale_y, actual_H / 2.0],
        [0, 0, 1]
    ], dtype=np.float32)

    R_llff = pose_llff[:, :3]
    t_llff = pose_llff[:, 3]

    R_cv = np.zeros_like(R_llff)
    R_cv[:, 0] = R_llff[:, 1]
    R_cv[:, 1] = R_llff[:, 0]
    R_cv[:, 2] = -R_llff[:, 2]

    c2w_cv = np.eye(4, dtype=np.float32)
    c2w_cv[:3, :3] = R_cv
    c2w_cv[:3, 3] = t_llff

    w2c_cv = np.linalg.inv(c2w_cv)[:3, :4]
    return w2c_cv, K


def compute_robust_quantile(tensor: torch.Tensor, q: float = 0.99, max_samples: int = 500_000) -> float:
    """安全的分位數計算函數，抽樣避免 PyTorch 大張量 RuntimeError"""
    pos_mask = (tensor > 0)
    if not torch.any(pos_mask):
        return 1.0
    valid_vals = tensor[pos_mask]
    numel = valid_vals.numel()
    if numel > max_samples:
        step = numel // max_samples
        sample = valid_vals[::step]
    else:
        sample = valid_vals
    return float(torch.quantile(sample, q))


def get_or_compute_frame_depths(
    image_paths: List[str],
    all_w2c: np.ndarray,
    all_intrs: np.ndarray,
    model_id: str = "depth-anything/DA3-GIANT-1.1",
    process_res: int = 728,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """透過 DA3 即時計算多視角深度圖與信心圖 (不使用硬碟快取，保持原生 process_res 輸出解析度)，回傳 (depths, confs, W_proc, H_proc)"""
    print(f"🚀 啟動 {model_id} 即時預測多視角深度與信心 (process_res={process_res})...")
    from depth_anything_3.api import DepthAnything3
    from mvtracker.datasets.generic_scene_dataset import enable_da3_layer_offload

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "GIANT" in model_id.upper() and device.type == "cuda":
        total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        if total_vram_gb <= 16.0:
            print(f"⚡ 啟用 DA3-GIANT 動態分層推論 (Sequential Block Offload) 以防 VRAM 溢出...")
            model = DepthAnything3.from_pretrained(model_id)
            model = enable_da3_layer_offload(model, device=device)
        else:
            model = DepthAnything3.from_pretrained(model_id).to(device)
    else:
        model = DepthAnything3.from_pretrained(model_id).to(device)
    model.eval()

    images_pil = [Image.open(p).convert("RGB") for p in image_paths]
    num_views = len(images_pil)
    extr_4x4 = np.zeros((num_views, 4, 4), dtype=np.float32)
    extr_4x4[:, :3, :4] = all_w2c
    extr_4x4[:, 3, 3] = 1.0

    with torch.no_grad():
        pred = model.inference(
            image=images_pil,
            extrinsics=extr_4x4,
            intrinsics=all_intrs,
            align_to_input_ext_scale=True,
            infer_gs=False,
            process_res=process_res,
        )

    depths_raw = pred.depth.astype(np.float32)  # [V, H_proc, W_proc]
    confs_raw = pred.conf.astype(np.float32) if (hasattr(pred, 'conf') and pred.conf is not None) else np.ones_like(depths_raw, dtype=np.float32) * 10.0
    _, H_proc, W_proc = depths_raw.shape
    print(f"📐 DA3 原生輸出解析度: {W_proc}x{H_proc} (無額外插值)")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return depths_raw, confs_raw, W_proc, H_proc


def main():
    parser = argparse.ArgumentParser(description="3DGS 切片導向之 3D 空間特徵權重凸優化求解器")
    parser.add_argument("--dataset_dir", type=str, default="datasets/sear_steak", help="資料集路徑")
    parser.add_argument("--ply_path", type=str, default="datasets/sear_steak/ckpt_10999_rank0_frame0.ply", help="3DGS 切片 PLY 檔案路徑")
    parser.add_argument("--frame_idx", type=int, default=0, help="評估影格索引 (預設 0)")
    parser.add_argument("--voxel_size", type=float, default=0.1, help="體素大小 (預設 0.1)")
    parser.add_argument("--voxel_mode", type=str, choices=["uniform", "dynamic", "surfel"], default="surfel", help="體素化模式: uniform (均勻), dynamic (動態足跡), surfel (多尺度表面基元融合 - 方案 B)")
    parser.add_argument("--use_dynamic_voxel", action="store_true", default=None, help="啟用動態足跡體素化 (已整合至 --voxel_mode dynamic)")
    parser.add_argument("--surfel_normal_scale", type=float, default=5.0, help="Surfel 模式法向容差倍率 (預設 5.0)")
    parser.add_argument("--surfel_angle_deg", type=float, default=25.0, help="Surfel 模式表面法向夾角門檻 (預設 25.0 度)")
    parser.add_argument("--min_voxel_size", type=float, default=0.01, help="最小體素大小 (預設 0.01)")
    parser.add_argument("--dynamic_voxel_scale", type=float, default=1.0, help="動態足跡縮放係數 (預設 1.0)")
    parser.add_argument("--num_cams", type=int, default=5, help="參與相機數量 (預設 5)")
    parser.add_argument("--cam_select_mode", type=str, choices=["da3_surfel", "fps"], default="fps", help="相機挑選模式: fps (外參最遠點採樣，預設) 或 da3_surfel (基於第0幀DA3表面最大覆蓋)")
    parser.add_argument("--include_cam00", action="store_true", help="是否包含 cam00 (預設排除 index 0 測試視角)")
    parser.add_argument("--da3_model", type=str, default="depth-anything/DA3-GIANT-1.1", help="DA3 模型版本 (預設 depth-anything/DA3-GIANT-1.1)")
    parser.add_argument("--process_res", type=int, default=728, help="DA3 推論解析度 (預設 728)")
    parser.add_argument("--conf_thresh", type=float, default=5.0, help="體素信心門檻 (預設 5.0)")
    parser.add_argument("--conf_filter_mode", type=str, choices=["2d", "3d"], default="2d", help="信心度過濾階段 (預設 2d)")
    parser.add_argument("--conf_agg", type=str, choices=["min", "mean"], default="min", help="體素信心聚合方式: min (多視角最小信心) 或 mean (多視角平均信心)")
    parser.add_argument("--colmap_strategy", type=str, choices=["decoupled", "coupled"], default="decoupled", help="COLMAP 融合策略 (decoupled: 保留原錨點不參與體素融合與權重求解)")
    parser.add_argument("--target_budget", type=int, default=0, help="輸出點雲目標點數預算 (預設 0，僅當明確指定 --output_ply 時生效)")
    parser.add_argument("--output_ply", type=str, default=None, help="輸出點雲 PLY 路徑 (預設 None，不輸出 PLY，僅產出權重 JSON)")
    parser.add_argument("--colmap_path", type=str, default=None, help="COLMAP 點雲路徑 (.bin, .ply, .txt，預設自動偵測)")
    parser.add_argument("--output_json", type=str, default=None, help="輸出權重 JSON 路徑 (預設依體素模式自動命名)")
    args = parser.parse_args()

    if args.use_dynamic_voxel is not None and args.use_dynamic_voxel:
        args.voxel_mode = "dynamic"

    if args.output_json is None:
        dataset_name = os.path.basename(os.path.normpath(args.dataset_dir))
        strategy_tag = "decoupled" if args.colmap_strategy == "decoupled" else "colmap"
        suffix = f"_{args.voxel_mode}_{strategy_tag}.json"
        args.output_json = os.path.join(args.dataset_dir, f"weights_{dataset_name}_f{args.frame_idx}{suffix}")

    print("\n" + "=" * 70)
    print(" 3D 空間特徵 Max Voxelize 融合與 3DGS 分佈對齊權重求解器")
    print("=" * 70)

    # 1. 檢驗相機與資料夾
    view_dirs = sorted([f for f in glob.glob(os.path.join(args.dataset_dir, 'cam*')) if os.path.isdir(f)], key=parse_cam_idx)
    if not view_dirs:
        view_dirs = sorted([f for f in glob.glob(os.path.join(args.dataset_dir, 'view_*')) if os.path.isdir(f)], key=parse_cam_idx)
    if not view_dirs:
        raise FileNotFoundError(f"在 {args.dataset_dir} 找不到相機資料夾！")

    poses_path = os.path.join(args.dataset_dir, 'poses_bounds.npy')
    if not os.path.isfile(poses_path):
        raise FileNotFoundError(f"找不到相機參數檔: {poses_path}")
    poses_bounds = np.load(poses_path)
    all_poses_llff = poses_bounds[:, :-2].reshape([-1, 3, 5])

    sample_frames = get_sorted_frame_paths(view_dirs[0])
    test_img = Image.open(sample_frames[args.frame_idx])
    W_orig, H_orig = test_img.size
    print(f"📷 相機總數: {len(view_dirs)} | 原圖: {W_orig}x{H_orig} | DA3 process_res: {args.process_res}")

    all_w2c_list, all_intrs_list = [], []
    for i in range(len(all_poses_llff)):
        w2c, K = llff_to_opencv_w2c(all_poses_llff[i], W_orig, H_orig)
        all_w2c_list.append(w2c)
        all_intrs_list.append(K)
    all_w2c_np = np.stack(all_w2c_list)
    all_intrs_np = np.stack(all_intrs_list)

    # 決定使用的相機列表
    exclude_idx = [] if args.include_cam00 else [0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.num_cams is not None and args.num_cams > 0 and args.num_cams < len(view_dirs) - len(exclude_idx):
        if args.cam_select_mode == "da3_surfel":
            selected_cam_indices = select_cams_by_da3_surfel(
                view_dirs=view_dirs,
                all_w2c_np=all_w2c_np,
                all_intrs_np=all_intrs_np,
                num_to_select=args.num_cams,
                voxel_size=0.2,
                exclude_indices=exclude_idx,
                da3_model=args.da3_model,
                process_res=args.process_res,
                device=device,
            )
        else:
            selected_cam_indices = select_cams_by_fps(all_w2c_np, num_to_select=args.num_cams, alpha=0.8, exclude_indices=exclude_idx)
    else:
        selected_cam_indices = [i for i in range(len(view_dirs)) if i not in exclude_idx]

    selected_cams = [view_dirs[i] for i in selected_cam_indices]
    cams_w2c = all_w2c_np[selected_cam_indices]
    cams_intrs = all_intrs_np[selected_cam_indices]
    cams_c2w = [np.linalg.inv(np.vstack([w2c, [0, 0, 0, 1]])) for w2c in cams_w2c]

    print(f"🎯 選定相機視角 (共 {len(selected_cams)} 個): {[os.path.basename(c) for c in selected_cams]}")

    # 2. 取得多視角 RGB 影像與原生解析度深度 (無快取、無額外插值)
    frame_img_paths = [get_sorted_frame_paths(d)[args.frame_idx] for d in selected_cams]
    depths, confs, W_proc, H_proc = get_or_compute_frame_depths(
        image_paths=frame_img_paths,
        all_w2c=cams_w2c,
        all_intrs=cams_intrs,
        model_id=args.da3_model,
        process_res=args.process_res,
    )

    # 縮放相機內參至 DA3 輸出解析度 W_proc, H_proc
    scale_x = W_proc / W_orig
    scale_y = H_proc / H_orig
    cams_intrs_scaled = cams_intrs.copy()
    cams_intrs_scaled[:, 0, :] *= scale_x
    cams_intrs_scaled[:, 1, :] *= scale_y

    # 若為 surfel 模式，在 GPU 上計算各視角相機表面單位法向量 (OpenCV LINEMOD 最小差分標準)
    if args.voxel_mode == "surfel":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        d_raw_t = torch.from_numpy(depths).unsqueeze(1).to(device)  # [V, 1, H_proc, W_proc]
        intrs_raw = torch.from_numpy(cams_intrs_scaled).clone().float().to(device)

        with torch.no_grad():
            normals_raw, _ = depth_to_surface_normals(
                d_raw_t,
                intrinsics=intrs_raw,
                method="min_gradient"
            )
            normals_target = F.normalize(normals_raw, dim=1, eps=1e-6)
            normals_np = normals_target.permute(0, 2, 3, 1).cpu().numpy()  # [V, H_proc, W_proc, 3]
    else:
        normals_np = None

    # 3. 提取 2D 特徵並執行跨視角全域尺度校準 (Cross-View Global Scaling)
    print(f"\n🔍 正在計算多視角 2D 特徵 ({W_proc}x{H_proc}, Noise Coring + 跨視角 Q_global 校準)...")
    scorer = RGBTextureScorer(normalize=False)

    raw_sobels, raw_dcts, raw_chromas, rgb_imgs = [], [], [], []
    for img_p in frame_img_paths:
        img_pil = Image.open(img_p).convert("RGB").resize((W_proc, H_proc), Image.Resampling.BILINEAR)
        img_np = np.array(img_pil, dtype=np.uint8)
        rgb_imgs.append(img_np)
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        with torch.no_grad():
            res = scorer(img_t)
            raw_sobels.append(res['s_sobel'].squeeze().cpu())
            raw_dcts.append(res['s_dct'].squeeze().cpu())
            raw_chromas.append(res['s_chroma'].squeeze().cpu())

    # 跨視角聯合計算 99% 分位數作為全域共享基準 Q_global (防禦白牆噪點放大)
    all_sobel_flat = torch.cat([s.flatten() for s in raw_sobels])
    all_dct_flat = torch.cat([s.flatten() for s in raw_dcts])
    all_chroma_flat = torch.cat([s.flatten() for s in raw_chromas])

    q_sobel = compute_robust_quantile(all_sobel_flat, q=0.99)
    q_dct = compute_robust_quantile(all_dct_flat, q=0.99)
    q_chroma = compute_robust_quantile(all_chroma_flat, q=0.99)

    print(f"📊 跨視角全域尺度基準 Q_global:")
    print(f"   Sobel  (Q99): {q_sobel:.4f}")
    print(f"   DCT    (Q99): {q_dct:.4f}")
    print(f"   Chroma (Q99): {q_chroma:.4f}")

    # 縮放至 [0, 1] 物理機率區間
    scaled_feats_list = []
    for i in range(len(selected_cams)):
        s_sobel = np.clip(raw_sobels[i].numpy() / max(q_sobel, 1e-4), 0.0, 1.0)
        s_dct = np.clip(raw_dcts[i].numpy() / max(q_dct, 1e-4), 0.0, 1.0)
        s_chroma = np.clip(raw_chromas[i].numpy() / max(q_chroma, 1e-4), 0.0, 1.0)
        feat_map = np.stack([s_sobel, s_dct, s_chroma], axis=-1)  # [H, W, 3]
        scaled_feats_list.append(feat_map)

    # 4. 多視角 3D 逆投影 (保留各像素信心度攜帶至 3D 點)
    print(f"\n🌐 執行多視角 3D 逆投影 (攜帶信心度，準備體素化 {args.conf_agg.upper()} 融合過濾)...")
    pts_all, colors_all, feats_all, footprints_all, norms_all, confs_all = [], [], [], [], [], []
    for i in range(len(selected_cams)):
        norm_c_v = normals_np[i] if normals_np is not None else None
        valid_mask_i = (depths[i] > 0.1) & (depths[i] < 150.0) & np.isfinite(depths[i])
        if args.conf_filter_mode == "2d":
            valid_mask_i = valid_mask_i & (confs[i] >= args.conf_thresh)

        res_unproj = unproject_camera_points_with_features(
            rgb=rgb_imgs[i],
            depth=depths[i],
            features=scaled_feats_list[i],
            K=cams_intrs_scaled[i],
            c2w=cams_c2w[i],
            depth_min=0.1,
            depth_max=150.0,
            valid_mask=valid_mask_i,
            normals=norm_c_v,
            return_normals=(args.voxel_mode == "surfel"),
            confidences=confs[i],
            return_confidences=True,
        )
        if args.voxel_mode == "surfel":
            pts_w, cols, feats, fps, norm_w, conf_w = res_unproj
            norms_all.append(norm_w)
        else:
            pts_w, cols, feats, fps, conf_w = res_unproj

        pts_all.append(pts_w)
        colors_all.append(cols)
        feats_all.append(feats)
        footprints_all.append(fps)
        confs_all.append(conf_w)
        low_cnt = np.sum(conf_w < args.conf_thresh)
        print(f"   視角 {i} ({os.path.basename(selected_cams[i])}): 逆投影 {len(pts_w):,} 點 | 信心均值: {conf_w.mean():.2f} | 信心 < {args.conf_thresh} 點數: {low_cnt:,} ({low_cnt/len(pts_w)*100:.1f}%)")

    pts_cat = np.concatenate(pts_all, axis=0)
    colors_cat = np.concatenate(colors_all, axis=0)
    feats_cat = np.concatenate(feats_all, axis=0)
    footprints_cat = np.concatenate(footprints_all, axis=0) if args.voxel_mode != "uniform" else None
    norms_cat = np.concatenate(norms_all, axis=0) if args.voxel_mode == "surfel" else None
    confs_cat = np.concatenate(confs_all, axis=0)

    print(f"   DA3 深度有效逆投影總點數: {len(pts_cat):,}")

    # 4.1 載入 COLMAP 稀疏點雲並執行幾何融合
    if args.colmap_path is None:
        candidates = [
            os.path.join(args.dataset_dir, f"colmap_{args.frame_idx}", "sparse", "0", "points3D.bin"),
            os.path.join(args.dataset_dir, f"colmap_{args.frame_idx}", "sparse", "0", "points3D.ply"),
            os.path.join(args.dataset_dir, f"colmap_{args.frame_idx}", "sparse", "0", "points3D.txt"),
            os.path.join(args.dataset_dir, "colmap_0", "sparse", "0", "points3D.bin"),
            os.path.join(args.dataset_dir, "colmap_0", "sparse", "0", "points3D_total1_sec_fps30.ply"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                args.colmap_path = c
                break

    if args.colmap_path and os.path.isfile(args.colmap_path):
        colmap_pts, colmap_cols = load_colmap_points(args.colmap_path)
        print(f"📌 成功載入 COLMAP 稀疏點雲: {len(colmap_pts):,} 點 ({args.colmap_path})")
    else:
        raise FileNotFoundError(f"找不到 COLMAP 點雲檔案，請透過 --colmap_path 指定！")

    # 計算相機光心位置 (用於視向法向量與足跡計算)
    cams_pos = np.stack([c2w[:3, 3] for c2w in cams_c2w])
    dists_col_cams = np.linalg.norm(colmap_pts[:, None, :] - cams_pos[None, :, :], axis=-1)
    min_dists = dists_col_cams.min(axis=-1)
    avg_focal = 0.5 * (cams_intrs_scaled[:, 0, 0].mean() + cams_intrs_scaled[:, 1, 1].mean())
    footprints_col = np.clip(min_dists / avg_focal, args.min_voxel_size, args.voxel_size).astype(np.float32)

    if args.voxel_mode == "surfel":
        norms_col = assign_normals_to_colmap_points(
            colmap_pts=colmap_pts,
            da3_pts=pts_cat,
            da3_normals=norms_cat,
            cams_pos=cams_pos,
            max_nn_dist=args.voxel_size * 1.5
        )
    else:
        norms_col = None

    pts_col_normed = (colmap_cols / 255.0).astype(np.float32) if colmap_cols.max() > 1.0 else colmap_cols.astype(np.float32)
    feats_col = np.zeros((len(colmap_pts), 3), dtype=np.float32)
    confs_col = np.full(len(colmap_pts), 20.0, dtype=np.float32)

    if args.colmap_strategy == "decoupled":
        pts_merged = pts_cat
        colors_merged = colors_cat
        feats_merged = feats_cat
        footprints_merged = footprints_cat
        norms_merged = norms_cat
        confs_merged = confs_cat
        print(f"   純淨 DA3 輸入體素化點數 (COLMAP 獨立解耦): {len(pts_merged):,}")
    else:
        # 合併 DA3 與 COLMAP 點雲進入體素化
        pts_merged = np.concatenate([pts_cat, colmap_pts], axis=0)
        colors_merged = np.concatenate([colors_cat, pts_col_normed], axis=0)
        feats_merged = np.concatenate([feats_cat, feats_col], axis=0)
        footprints_merged = np.concatenate([footprints_cat, footprints_col], axis=0) if footprints_cat is not None else None
        norms_merged = np.concatenate([norms_cat, norms_col], axis=0) if norms_cat is not None else None
        confs_merged = np.concatenate([confs_cat, confs_col], axis=0)
        print(f"   總輸入體素化點數 (DA3 + COLMAP): {len(pts_merged):,}")

    # 5. 體素化特徵聚合 (Max-Pooling) 與信心度融合
    if args.voxel_mode == "surfel":
        mode_str = f"多尺度表面基元融合 (Surfel h_norm={args.surfel_normal_scale}x, ang={args.surfel_angle_deg}°)"
    elif args.voxel_mode == "dynamic":
        mode_str = f"動態足跡體素化 (Dynamic Footprint scale={args.dynamic_voxel_scale})"
    else:
        mode_str = f"均勻體素化 (Uniform voxel_size={args.voxel_size})"

    print(f"\n🧱 執行 {mode_str} 並對特徵向量實施 Max-Pooling (信心度 {args.conf_agg.upper()} 融合)...")
    t_vox_start = time.time()
    v_pts, v_cols, v_feats_max, _, v_confs = voxel_downsample_unified_with_features(
        pts=pts_merged,
        colors=colors_merged,
        features=feats_merged,
        normals=norms_merged,
        voxel_size=args.voxel_size,
        footprints=footprints_merged,
        min_voxel_size=args.min_voxel_size,
        scale=args.dynamic_voxel_scale,
        mode=args.voxel_mode,
        surfel_normal_scale=args.surfel_normal_scale,
        surfel_angle_deg=args.surfel_angle_deg,
        confidences=confs_merged,
        conf_agg=args.conf_agg,
        return_confidences=True,
    )
    t_vox = time.time() - t_vox_start
    raw_voxels = len(v_pts)
    print(f"   體素化完成！耗時: {t_vox:.2f}s | 初始體素數: {raw_voxels:,}")
    print(f"   體素信心分佈: min={v_confs.min():.2f}, p10={np.percentile(v_confs, 10):.2f}, mean={v_confs.mean():.2f}, max={v_confs.max():.2f}")

    # 體素信心度過濾 (刪除低於門檻之體素，消除窗戶/室外虛假偽影)
    valid_vox_mask = (v_confs >= args.conf_thresh)
    del_cnt = np.sum(~valid_vox_mask)
    print(f"🧹 依據體素信心度門檻 (conf_thresh >= {args.conf_thresh}, agg={args.conf_agg.upper()}): 刪除 {del_cnt:,} 個體素 ({del_cnt/raw_voxels*100:.1f}%)")

    v_pts = v_pts[valid_vox_mask]
    v_cols = v_cols[valid_vox_mask]
    v_feats_max = v_feats_max[valid_vox_mask]
    v_confs = v_confs[valid_vox_mask]
    num_voxels = len(v_pts)
    print(f"   過濾後保留體素數: {num_voxels:,}")

    # 6. 讀取 3DGS 切片 Ground Truth 並建立理想分佈 P_ideal
    if not os.path.isfile(args.ply_path):
        raise FileNotFoundError(f"找不到 3DGS 切片 PLY 檔案: {args.ply_path}")

    print(f"\n⭐ 讀取 3DGS 切片: {args.ply_path}")
    ply_data = plyfile.PlyData.read(args.ply_path)
    vx = np.array(ply_data['vertex']['x'], dtype=np.float32)
    vy = np.array(ply_data['vertex']['y'], dtype=np.float32)
    vz = np.array(ply_data['vertex']['z'], dtype=np.float32)
    gt_pts = np.stack([vx, vy, vz], axis=-1)
    num_gt = len(gt_pts)
    print(f"   3DGS 高斯數量: {num_gt:,}")

    # 建構 cKDTree 匹配高斯至體素
    print("🌲 建構 cKDTree 將高斯中心投影映射至最接近體素...")
    tree = cKDTree(v_pts)
    search_radius = max(args.voxel_size * 2.5, 0.5)
    dists, nn_indices = tree.query(gt_pts, k=1, distance_upper_bound=search_radius)

    valid_gt_mask = dists < search_radius
    matched_vox_idx = nn_indices[valid_gt_mask]
    print(f"   有效匹配高斯數: {np.sum(valid_gt_mask):,} / {num_gt:,} ({np.sum(valid_gt_mask)/num_gt*100:.1f}%)")

    # 統計各體素包含之高斯數量
    voxel_gaussian_counts = np.bincount(matched_vox_idx, minlength=num_voxels).astype(np.float64)
    total_matched = np.sum(voxel_gaussian_counts)
    if total_matched == 0:
        raise RuntimeError("未能成功將任何高斯匹配至 3D 體素！請檢查座標系統或體素尺寸設定。")

    p_ideal = voxel_gaussian_counts / total_matched  # [M]

    # 7. 統計各體素內包含之 COLMAP 點數量，作為獨立特徵分數 S_colmap
    print("🌲 統計各體素內所包含的 COLMAP 點雲數量 (S_colmap)...")
    search_r_col = max(args.voxel_size * 1.5, 0.15)
    dists_col, matched_vox_col = tree.query(colmap_pts, k=1, distance_upper_bound=search_r_col)
    valid_col_mask = dists_col < search_r_col
    colmap_counts = np.bincount(matched_vox_col[valid_col_mask], minlength=num_voxels).astype(np.float64)
    print(f"   COLMAP 匹配成功: {np.sum(valid_col_mask):,} / {len(colmap_pts):,} 點 ({np.sum(valid_col_mask)/len(colmap_pts)*100:.1f}%)")
    print(f"   包含 COLMAP 點的體素數: {np.sum(colmap_counts > 0):,} / {num_voxels:,} ({np.sum(colmap_counts > 0)/num_voxels*100:.2f}%)")
    print(f"   單一體素最大 COLMAP 點數: {int(np.max(colmap_counts))}")

    # 8. 建構候選特徵預測分佈 (Base, Sobel, DCT, Chroma, Colmap)
    p_base = np.full(num_voxels, 1.0 / num_voxels, dtype=np.float64)
    s_sobel_v = v_feats_max[:, 0].astype(np.float64)
    s_dct_v = v_feats_max[:, 1].astype(np.float64)
    s_chroma_v = v_feats_max[:, 2].astype(np.float64)
    s_colmap_v = colmap_counts

    p_sobel = s_sobel_v / max(np.sum(s_sobel_v), 1e-12)
    p_dct = s_dct_v / max(np.sum(s_dct_v), 1e-12)
    p_chroma = s_chroma_v / max(np.sum(s_chroma_v), 1e-12)
    p_colmap = s_colmap_v / max(np.sum(s_colmap_v), 1e-12)

    # 9. 凸優化求解最佳特徵組合權重
    eps = 1e-12

    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        valid = (p > 0)
        return float(np.sum(p[valid] * np.log(p[valid] / (q[valid] + eps))))

    def calc_recall(p_score: np.ndarray, percentiles=[0.10, 0.20, 0.30, 0.50]):
        sort_idx = np.argsort(-p_score)
        recalls = {}
        for pct in percentiles:
            k = int(num_voxels * pct)
            recalls[f"top_{int(pct*100)}%"] = float(np.sum(p_ideal[sort_idx[:k]]))
        return recalls

    if args.colmap_strategy == "decoupled":
        print("\n⚡ 執行 SLSQP 凸優化求解 4 項最佳特徵組合權重 (COLMAP 作為解耦固定錨點)...")

        def objective(w: np.ndarray) -> float:
            p_pred = (
                w[0] * p_base +
                w[1] * p_sobel +
                w[2] * p_dct +
                w[3] * p_chroma
            )
            return kl_divergence(p_ideal, p_pred)

        bounds = [(0.0, 1.0)] * 4
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        init_w = np.array([0.25, 0.25, 0.25, 0.25])

        opt_res = minimize(objective, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
        w_opt = opt_res.x
        w_opt = np.clip(w_opt, 0.0, 1.0)
        w_opt = w_opt / np.sum(w_opt)

        p_opt = (
            w_opt[0] * p_base +
            w_opt[1] * p_sobel +
            w_opt[2] * p_dct +
            w_opt[3] * p_chroma
        )

        kl_base = kl_divergence(p_ideal, p_base)
        kl_sobel = kl_divergence(p_ideal, 0.05 * p_base + 0.95 * p_sobel)
        kl_dct = kl_divergence(p_ideal, 0.05 * p_base + 0.95 * p_dct)
        kl_chroma = kl_divergence(p_ideal, 0.05 * p_base + 0.95 * p_chroma)
        kl_opt = kl_divergence(p_ideal, p_opt)

        recall_base = calc_recall(p_base)
        recall_opt = calc_recall(p_opt)
        recall_sobel = calc_recall(p_sobel)
        recall_dct = calc_recall(p_dct)
        recall_chroma = calc_recall(p_chroma)

        print("\n" + "=" * 70)
        print(" 評估結果摘要 (4 項特徵 SLSQP 優化 - COLMAP 解耦固定錨點)")
        print("=" * 70)
        print(f"最佳權重求解結果:")
        print(f"  w0*       (平滑底色基準)   : {w_opt[0]:.4f}")
        print(f"  w_sobel*  (S_sobel 邊界)   : {w_opt[1]:.4f}")
        print(f"  w_dct*    (S_dct 材質紋理) : {w_opt[2]:.4f}")
        print(f"  w_chroma* (S_chroma 色彩)  : {w_opt[3]:.4f}")
        print("-" * 70)
        print(f"KL 散度 (越小代表越吻合 3DGS 高斯分佈):")
        print(f"  Uniform Base : {kl_base:.4f}")
        print(f"  S_sobel 單項 : {kl_sobel:.4f}")
        print(f"  S_dct   單項 : {kl_dct:.4f}")
        print(f"  S_chroma單項 : {kl_chroma:.4f}")
        print(f"  最佳 4 項組合: {kl_opt:.4f} (相比均勻基準降低: {kl_base - kl_opt:.4f})")
        print("-" * 70)
        print(f"Top-K% 體素高斯捕捉率 (Gaussian Capture Recall):")
        print(f"  指標              | Top 10% | Top 20% | Top 30% | Top 50%")
        print(f"  均勻隨機 (Base)   | {recall_base['top_10%']*100:6.2f}% | {recall_base['top_20%']*100:6.2f}% | {recall_base['top_30%']*100:6.2f}% | {recall_base['top_50%']*100:6.2f}%")
        print(f"  S_sobel 邊界      | {recall_sobel['top_10%']*100:6.2f}% | {recall_sobel['top_20%']*100:6.2f}% | {recall_sobel['top_30%']*100:6.2f}% | {recall_sobel['top_50%']*100:6.2f}%")
        print(f"  S_dct 材質        | {recall_dct['top_10%']*100:6.2f}% | {recall_dct['top_20%']*100:6.2f}% | {recall_dct['top_30%']*100:6.2f}% | {recall_dct['top_50%']*100:6.2f}%")
        print(f"  S_chroma 色彩     | {recall_chroma['top_10%']*100:6.2f}% | {recall_chroma['top_20%']*100:6.2f}% | {recall_chroma['top_30%']*100:6.2f}% | {recall_chroma['top_50%']*100:6.2f}%")
        print(f"  最佳 4 項組合     | {recall_opt['top_10%']*100:6.2f}% | {recall_opt['top_20%']*100:6.2f}% | {recall_opt['top_30%']*100:6.2f}% | {recall_opt['top_50%']*100:6.2f}%")
        print("=" * 70)

        opt_weights_dict = {
            "w_0": float(w_opt[0]),
            "w_sobel": float(w_opt[1]),
            "w_dct": float(w_opt[2]),
            "w_chroma": float(w_opt[3]),
        }
        kl_dict = {
            "uniform_base": float(kl_base),
            "sobel_alone": float(kl_sobel),
            "dct_alone": float(kl_dct),
            "chroma_alone": float(kl_chroma),
            "optimal": float(kl_opt)
        }
        recall_dict = {
            "uniform_base": recall_base,
            "sobel_alone": recall_sobel,
            "dct_alone": recall_dct,
            "chroma_alone": recall_chroma,
            "optimal": recall_opt
        }

        # 僅當明確指定 --output_ply 時才匯出點雲 (預設僅求解權重 JSON)
        if args.output_ply:
            target_total = args.target_budget if args.target_budget > 0 else 50000
            num_colmap = len(colmap_pts)
            needed_da3 = max(0, target_total - num_colmap)
            p_torch = torch.from_numpy(p_opt).float()
            torch.manual_seed(42)
            np.random.seed(42)
            sampled_da3_idx = torch.multinomial(p_torch, num_samples=min(needed_da3, num_voxels), replacement=False).numpy()
            da3_sampled_pts = v_pts[sampled_da3_idx]
            da3_sampled_cols = (np.clip(v_cols[sampled_da3_idx], 0.0, 1.0) * 255).astype(np.uint8) if v_cols.max() <= 1.0 else v_cols[sampled_da3_idx].astype(np.uint8)
            colmap_cols_u8 = (np.clip(colmap_cols, 0.0, 1.0) * 255).astype(np.uint8) if colmap_cols.max() <= 1.0 else colmap_cols.astype(np.uint8)
            final_pts = np.concatenate([colmap_pts, da3_sampled_pts], axis=0)
            final_cols = np.concatenate([colmap_cols_u8, da3_sampled_cols], axis=0)

            dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
            el_data = np.empty(len(final_pts), dtype=dtype)
            el_data['x'] = final_pts[:, 0]
            el_data['y'] = final_pts[:, 1]
            el_data['z'] = final_pts[:, 2]
            el_data['red'] = final_cols[:, 0]
            el_data['green'] = final_cols[:, 1]
            el_data['blue'] = final_cols[:, 2]
            os.makedirs(os.path.dirname(os.path.abspath(args.output_ply)), exist_ok=True)
            plyfile.PlyData([plyfile.PlyElement.describe(el_data, 'vertex')], text=False).write(args.output_ply)
            print(f"💾 已匯出最終解耦點雲: {args.output_ply} ({len(final_pts):,} 點: {num_colmap} COLMAP + {len(da3_sampled_pts)} DA3)")

    else:
        print("\n⚡ 執行 SLSQP 凸優化求解 5 項最佳特徵組合權重...")

        def objective(w: np.ndarray) -> float:
            p_pred = (
                w[0] * p_base +
                w[1] * p_sobel +
                w[2] * p_dct +
                w[3] * p_chroma +
                w[4] * p_colmap
            )
            return kl_divergence(p_ideal, p_pred)

        bounds = [(0.0, 1.0)] * 5
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        init_w = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        opt_res = minimize(objective, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
        w_opt = opt_res.x
        w_opt = np.clip(w_opt, 0.0, 1.0)
        w_opt = w_opt / np.sum(w_opt)

        p_opt = (
            w_opt[0] * p_base +
            w_opt[1] * p_sobel +
            w_opt[2] * p_dct +
            w_opt[3] * p_chroma +
            w_opt[4] * p_colmap
        )

        kl_base = kl_divergence(p_ideal, p_base)
        kl_sobel = kl_divergence(p_ideal, 0.05 * p_base + 0.95 * p_sobel)
        kl_dct = kl_divergence(p_ideal, 0.05 * p_base + 0.95 * p_dct)
        kl_chroma = kl_divergence(p_ideal, 0.05 * p_base + 0.95 * p_chroma)
        kl_colmap = kl_divergence(p_ideal, 0.05 * p_base + 0.95 * p_colmap)
        kl_opt = kl_divergence(p_ideal, p_opt)

        recall_base = calc_recall(p_base)
        recall_opt = calc_recall(p_opt)
        recall_sobel = calc_recall(p_sobel)
        recall_dct = calc_recall(p_dct)
        recall_chroma = calc_recall(p_chroma)
        recall_colmap = calc_recall(p_colmap)

        print("\n" + "=" * 70)
        print(" 評估結果摘要 (5 項特徵 SLSQP 優化)")
        print("=" * 70)
        print(f"最佳權重求解結果:")
        print(f"  w0*       (平滑底色基準)   : {w_opt[0]:.4f}")
        print(f"  w_sobel*  (S_sobel 邊界)   : {w_opt[1]:.4f}")
        print(f"  w_dct*    (S_dct 材質紋理) : {w_opt[2]:.4f}")
        print(f"  w_chroma* (S_chroma 色彩)  : {w_opt[3]:.4f}")
        print(f"  w_colmap* (S_colmap 稀疏點): {w_opt[4]:.4f}")
        print("-" * 70)
        print(f"KL 散度 (越小代表越吻合 3DGS 高斯分佈):")
        print(f"  Uniform Base : {kl_base:.4f}")
        print(f"  S_sobel 單項 : {kl_sobel:.4f}")
        print(f"  S_dct   單項 : {kl_dct:.4f}")
        print(f"  S_chroma單項 : {kl_chroma:.4f}")
        print(f"  S_colmap單項 : {kl_colmap:.4f}")
        print(f"  最佳 5 項組合: {kl_opt:.4f} (相比均勻基準降低: {kl_base - kl_opt:.4f})")
        print("-" * 70)

        opt_weights_dict = {
            "w_0": float(w_opt[0]),
            "w_sobel": float(w_opt[1]),
            "w_dct": float(w_opt[2]),
            "w_chroma": float(w_opt[3]),
            "w_colmap": float(w_opt[4])
        }
        kl_dict = {
            "uniform_base": float(kl_base),
            "sobel_alone": float(kl_sobel),
            "dct_alone": float(kl_dct),
            "chroma_alone": float(kl_chroma),
            "colmap_alone": float(kl_colmap),
            "optimal": float(kl_opt)
        }
        recall_dict = {
            "uniform_base": recall_base,
            "sobel_alone": recall_sobel,
            "dct_alone": recall_dct,
            "chroma_alone": recall_chroma,
            "colmap_alone": recall_colmap,
            "optimal": recall_opt
        }

    # 11. 匯出結果至 JSON
    output_dict = {
        "dataset": args.dataset_dir,
        "frame_idx": args.frame_idx,
        "voxel_size": args.voxel_size,
        "voxel_mode": args.voxel_mode,
        "use_dynamic_voxel": (args.voxel_mode in ["dynamic", "surfel"]),
        "surfel_normal_scale": args.surfel_normal_scale if args.voxel_mode == "surfel" else None,
        "surfel_angle_deg": args.surfel_angle_deg if args.voxel_mode == "surfel" else None,
        "min_voxel_size": args.min_voxel_size,
        "dynamic_voxel_scale": args.dynamic_voxel_scale,
        "conf_thresh": args.conf_thresh,
        "conf_filter_mode": args.conf_filter_mode,
        "colmap_strategy": "decoupled_fixed_anchors" if args.colmap_strategy == "decoupled" else "coupled",
        "colmap_path": args.colmap_path,
        "da3_model": args.da3_model,
        "process_res": args.process_res,
        "target_w": int(W_proc),
        "target_h": int(H_proc),
        "num_voxels": int(num_voxels),
        "num_gt_gaussians": int(num_gt),
        "matched_gt_gaussians": int(np.sum(valid_gt_mask)),
        "num_colmap_points": int(len(colmap_pts)),
        "matched_colmap_points": int(np.sum(valid_col_mask)),
        "q_global": {
            "s_sobel": float(q_sobel),
            "s_dct": float(q_dct),
            "s_chroma": float(q_chroma)
        },
        "optimal_weights": opt_weights_dict,
        "kl_divergence": kl_dict,
        "recall": recall_dict
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=2, ensure_ascii=False)

    print(f"\n🎉 權重求解完成！JSON 已匯出至: {args.output_json}")

    # 若為 surfel 模式，同步寫入/覆蓋 weights_{scene}_f{idx}_surfel.json 供統一調用
    if args.voxel_mode == "surfel":
        dataset_name = os.path.basename(os.path.normpath(args.dataset_dir))
        dyn_fallback = os.path.join(args.dataset_dir, f"weights_{dataset_name}_f{args.frame_idx}_surfel.json")
        try:
            with open(dyn_fallback, "w", encoding="utf-8") as f:
                json.dump(output_dict, f, indent=2, ensure_ascii=False)
            print(f"📄 已同步更新 surfel 預設設定檔: {dyn_fallback}")
        except Exception as e:
            print(f"⚠️ 無法寫入備用設定檔: {e}")


if __name__ == "__main__":
    main()
