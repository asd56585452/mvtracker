#!/usr/bin/env python3
"""
生成 N3D 6 個資料集在 6 個均勻幀 (f25, f75, f125, f175, f225, f275) 下的
DA3 空間特徵重要度採樣點雲 (7,330, 50,000, 250,000 點)。
"""

import os
import sys
import argparse
import time
import glob
import json
import re
import struct

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mvtracker.utils.feature_scores import RGBTextureScorer, depth_to_surface_normals
from mvtracker.utils.spatial_voxelizer import (
    unproject_camera_points_with_features,
    voxel_downsample_unified_with_features,
)
from mvtracker.utils.camera_selection import select_cams_by_da3_surfel, select_cams_by_fps
from mvtracker.datasets.generic_scene_dataset import enable_da3_layer_offload

ALL_SCENES = [
    "coffee_martini",
    "cook_spinach",
    "cut_roasted_beef",
    "flame_salmon_1",
    "flame_steak",
    "sear_steak",
]

TARGET_FRAMES = [25, 75, 125, 175, 225, 275]
DEFAULT_BUDGETS = [7330, 50000, 250000]


def load_colmap_points(colmap_path: str):
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
        ply = PlyData.read(colmap_path)
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
    da3_pts: np.ndarray,
    da3_normals: np.ndarray,
    cams_pos: np.ndarray,
    max_nn_dist: float = 0.15,
) -> np.ndarray:
    """
    為無 normal 的 COLMAP 點雲自適應指派表面單位法向量：
    1. 近場點 (與最近 DA3 surfel 距離 < max_nn_dist)：繼承該 DA3 surfel 之單位法向。
    2. 遠場/孤立點 (距 DA3 表面 > max_nn_dist)：以相機中心至該點的射線反向 (面向相機) 作為 billboard 法向。
    """
    N = len(colmap_pts)
    normals = np.zeros((N, 3), dtype=np.float32)

    dists_to_cams = np.linalg.norm(colmap_pts[:, None, :] - cams_pos[None, :, :], axis=-1)
    nearest_cam_idx = np.argmin(dists_to_cams, axis=-1)
    nearest_cam_centers = cams_pos[nearest_cam_idx]
    view_dirs = nearest_cam_centers - colmap_pts
    view_lens = np.linalg.norm(view_dirs, axis=-1, keepdims=True)
    billboard_normals = (view_dirs / np.clip(view_lens, 1e-6, None)).astype(np.float32)
    normals[:] = billboard_normals

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


def llff_to_opencv_w2c(pose_llff, actual_W, actual_H):
    """將 LLFF 轉換為 OpenCV，並進行解析度焦距自適應縮放"""
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

def get_sorted_frame_paths(cam_dir):
    frames = glob.glob(os.path.join(cam_dir, '*.png')) + glob.glob(os.path.join(cam_dir, '*.jpg'))
    def parse_frame_idx(path):
        basename = os.path.splitext(os.path.basename(path))[0]
        match = re.search(r'\d+', basename)
        return int(match.group()) if match else 0
    frames.sort(key=parse_frame_idx)
    return frames

def save_ply(path, pts, colors):
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ]
    elements = np.empty(len(pts), dtype=dtype)
    elements['x'] = pts[:, 0]
    elements['y'] = pts[:, 1]
    elements['z'] = pts[:, 2]
    elements['red'] = colors[:, 0]
    elements['green'] = colors[:, 1]
    elements['blue'] = colors[:, 2]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    PlyData([PlyElement.describe(elements, 'vertex')], text=False).write(path)

def process_scene_frame(
    scene_name: str,
    frame_idx: int,
    da3_model,
    device: str,
    budgets: list,
    weights_path: str = None,
    num_cams: int = 5,
    cam_select_mode: str = "fps",
    da3_model_name: str = "depth-anything/DA3-GIANT-1.1",
    process_res: int = 728,
    conf_thresh: float = 5.0,
    conf_filter_mode: str = "2d",
    conf_agg: str = "min",
    colmap_strategy: str = "decoupled",
    overwrite: bool = False,
    export_full: bool = False
):
    scene_dir = os.path.join(PROJECT_ROOT, "datasets", scene_name)
    colmap_frame_dir = os.path.join(scene_dir, f"colmap_{frame_idx}")
    if not os.path.exists(colmap_frame_dir):
        print(f"⚠️ 找不到 {colmap_frame_dir}，跳過！")
        return

    # 檢查目標點雲是否都已存在
    needed_budgets = []
    for b in budgets:
        b_str = f"{b // 1000}k" if (b % 1000 == 0 and b >= 1000) else str(b)
        if colmap_strategy == "decoupled":
            out_ply_target = os.path.join(colmap_frame_dir, f"da3_giant_colmap_importance_{b_str}.ply")
            if b in (14000, 7330):
                out_7k = os.path.join(colmap_frame_dir, "da3_giant_colmap_importance_7k.ply")
                out_uni = os.path.join(colmap_frame_dir, f"da3_giant_colmap_uniform_{b_str}.ply")
                if overwrite or not (os.path.exists(out_ply_target) and os.path.exists(out_7k) and os.path.exists(out_uni)):
                    needed_budgets.append(b)
                continue
        else:
            out_ply_target = os.path.join(colmap_frame_dir, f"da3_importance_{b_str}.ply")
        if overwrite or not os.path.exists(out_ply_target):
            needed_budgets.append(b)

    if not needed_budgets:
        print(f"⏩ [{scene_name} | Frame {frame_idx:03d}] 所有目標點雲已存在，跳過。")
        return

    # 載入相機參數與視角
    poses_path = os.path.join(scene_dir, 'poses_bounds.npy')
    poses_bounds = np.load(poses_path)
    all_poses_llff = poses_bounds[:, :-2].reshape([-1, 3, 5])

    view_dirs = sorted([
        os.path.join(scene_dir, d) for d in os.listdir(scene_dir)
        if os.path.isdir(os.path.join(scene_dir, d)) and d.startswith('cam')
    ], key=lambda x: int("".join(filter(str.isdigit, os.path.basename(x)))))
    num_views = len(view_dirs)

    first_frame_path = get_sorted_frame_paths(view_dirs[0])[0]
    with Image.open(first_frame_path) as im:
        W_img, H_img = im.size

    all_w2c, all_intrs = [], []
    for i in range(len(all_poses_llff)):
        w2c, K = llff_to_opencv_w2c(all_poses_llff[i], W_img, H_img)
        all_w2c.append(w2c)
        all_intrs.append(K)
    all_w2c_np = np.stack(all_w2c)
    all_intrs_np = np.stack(all_intrs)

    # 選擇訓練相機 (每個場景僅在第 0 幀計算一次並快取，排除 cam00 測試視角)
    global _SCENE_CAMS_CACHE
    if "_SCENE_CAMS_CACHE" not in globals():
        _SCENE_CAMS_CACHE = {}

    cache_key = (scene_name, cam_select_mode, num_cams)
    if cache_key in _SCENE_CAMS_CACHE:
        selected_cams = _SCENE_CAMS_CACHE[cache_key]
    else:
        if cam_select_mode == "da3_surfel":
            selected_cams = select_cams_by_da3_surfel(
                view_dirs=view_dirs,
                all_w2c_np=all_w2c_np,
                all_intrs_np=all_intrs_np,
                num_to_select=num_cams,
                voxel_size=0.2,
                exclude_indices=[0],
                da3_model=da3_model_name,
                process_res=process_res,
                device=device,
            )
        else:
            selected_cams = select_cams_by_fps(all_w2c_np, num_to_select=num_cams, alpha=0.8, exclude_indices=[0])
        _SCENE_CAMS_CACHE[cache_key] = selected_cams

    # 載入權重設定 (支援自動探索)
    if weights_path is None or not os.path.isfile(weights_path):
        candidate_weights = [
            os.path.join(scene_dir, f"weights_{scene_name}_f{frame_idx}_giant_2d5.0_4feat.json"),
            os.path.join(scene_dir, f"weights_{scene_name}_f25_giant_2d5.0_4feat.json"),
            os.path.join(scene_dir, f"weights_{scene_name}_f{frame_idx}_surfel.json"),
            os.path.join(scene_dir, f"weights_{scene_name}_f25_surfel.json"),
        ]
        weights_path = None
        for cw in candidate_weights:
            if os.path.isfile(cw):
                weights_path = cw
                break

    if weights_path and os.path.isfile(weights_path):
        print(f"📖 [{scene_name} | Frame {frame_idx:03d}] 讀取特徵權重檔: {weights_path}")
        with open(weights_path, 'r') as f:
            w_cfg = json.load(f)
        feat_weights = w_cfg.get("optimal_weights", {
            "w_0": 0.1181, "w_sobel": 0.4352, "w_dct": 0.2486, "w_chroma": 0.1981
        })
        q_global = w_cfg.get("q_global", {
            "s_sobel": 0.2878, "s_dct": 3.1797, "s_chroma": 0.0616
        })
    else:
        print(f"⚙️ [{scene_name} | Frame {frame_idx:03d}] 使用預設 DA3-GIANT 4 項最佳特徵權重")
        feat_weights = {
            "w_0": 0.1181, "w_sobel": 0.4352, "w_dct": 0.2486, "w_chroma": 0.1981
        }
        q_global = {
            "s_sobel": 0.2878, "s_dct": 3.1797, "s_chroma": 0.0616
        }

    # 讀取當前幀影像
    cam_imgs_full = []
    cam_w2c = []
    cam_intr_orig = []

    for cam_idx in selected_cams:
        cam_dir = view_dirs[cam_idx]
        frame_files = get_sorted_frame_paths(cam_dir)
        if frame_idx >= len(frame_files):
            raise IndexError(f"Frame {frame_idx} 超出相機 {cam_dir} 的總幀數 ({len(frame_files)})")
        img_full = Image.open(frame_files[frame_idx]).convert('RGB')
        cam_imgs_full.append(img_full)
        cam_w2c.append(all_w2c_np[cam_idx])
        cam_intr_orig.append(all_intrs_np[cam_idx].copy())

    # 執行 DA3 深度推論 (使用完整原生解析度影像與原始相機內參，嚴禁提前縮圖破壞信心度)
    extr_chunk = np.stack(cam_w2c, axis=0)
    intr_chunk_orig = np.stack(cam_intr_orig, axis=0)
    extr_4x4 = np.zeros((num_cams, 4, 4), dtype=np.float32)
    extr_4x4[:, :3, :4] = extr_chunk
    extr_4x4[:, 3, 3] = 1.0

    print(f"🚀 [{scene_name} | Frame {frame_idx:03d}] 執行 DA3 深度推論 (原生解析度 {W_img}x{H_img}, 5 視角, res={process_res})...")
    with torch.no_grad():
        pred = da3_model.inference(
            image=cam_imgs_full,
            extrinsics=extr_4x4,
            intrinsics=intr_chunk_orig,
            process_res=process_res,
            align_to_input_ext_scale=True,
            infer_gs=False,
        )
        depths_raw = pred.depth.astype(np.float32)  # [V, H_proc, W_proc]
        confs_raw = pred.conf.astype(np.float32) if (hasattr(pred, 'conf') and pred.conf is not None) else np.ones_like(depths_raw, dtype=np.float32) * 10.0

    _, H_proc, W_proc = depths_raw.shape
    print(f"📐 [{scene_name} | Frame {frame_idx:03d}] DA3 原生輸出解析度: {W_proc}x{H_proc} (無額外插值)")

    depths_np = depths_raw
    confs_np = confs_raw

    scale_x = W_proc / W_img
    scale_y = H_proc / H_img
    cam_intr_scaled = []
    cam_imgs_target = []
    for v in range(num_cams):
        K_s = cam_intr_orig[v].copy()
        K_s[0, :] *= scale_x
        K_s[1, :] *= scale_y
        cam_intr_scaled.append(K_s)
        cam_imgs_target.append(np.array(cam_imgs_full[v].resize((W_proc, H_proc), Image.Resampling.BILINEAR)))

    # 特徵分數計算與法向量提取 (全流程原生 W_proc x H_proc，無任何深度插值失真)
    imgs_tensor = torch.from_numpy(np.stack(cam_imgs_target)).permute(0, 3, 1, 2).float() / 255.0  # [V, 3, H_proc, W_proc]
    feature_scorer = RGBTextureScorer(normalize=False)

    with torch.no_grad():
        f_res = feature_scorer(imgs_tensor)
        s_sob = f_res['s_sobel'].squeeze(1).cpu()
        s_dct = f_res['s_dct'].squeeze(1).cpu()
        s_chr = f_res['s_chroma'].squeeze(1).cpu()

        sob_scaled = torch.clamp(s_sob / max(q_global["s_sobel"], 1e-4), 0.0, 1.0).numpy()
        dct_scaled = torch.clamp(s_dct / max(q_global["s_dct"], 1e-4), 0.0, 1.0).numpy()
        chr_scaled = torch.clamp(s_chr / max(q_global["s_chroma"], 1e-4), 0.0, 1.0).numpy()

        intrs_proc_tensor = torch.from_numpy(np.stack(cam_intr_scaled)).clone().float().to(device)
        pred_depth_tensor = torch.from_numpy(depths_np).unsqueeze(1).to(device)
        normals_raw, _ = depth_to_surface_normals(pred_depth_tensor, intrinsics=intrs_proc_tensor, method="min_gradient")
        normals_torch = F.normalize(normals_raw, dim=1, eps=1e-6)
        normals_np = normals_torch.permute(0, 2, 3, 1).cpu().numpy()  # [V, H_proc, W_proc, 3]

    # 多視角 3D 逆投影 (支援 2D 前置信心度過濾)
    pts_all, cols_all, feats_all, fps_all, norms_all, confs_all = [], [], [], [], [], []
    for v in range(num_cams):
        rgb_v = cam_imgs_target[v]
        d_v = depths_np[v]
        c_v = confs_np[v]
        feat_v = np.stack([sob_scaled[v], dct_scaled[v], chr_scaled[v]], axis=-1)
        K_v = cam_intr_scaled[v]
        c2w_v = np.linalg.inv(extr_4x4[v])
        norm_v = normals_np[v]

        valid_mask_v = (d_v > 0.1) & (d_v < 150.0) & np.isfinite(d_v)
        if conf_filter_mode == "2d":
            valid_mask_v = valid_mask_v & (c_v >= conf_thresh)

        pts_w, cols_w, feats_w, fps_w, norms_w, confs_w = unproject_camera_points_with_features(
            rgb=rgb_v,
            depth=d_v,
            features=feat_v,
            K=K_v,
            c2w=c2w_v,
            depth_min=0.1,
            depth_max=150.0,
            valid_mask=valid_mask_v,
            normals=norm_v,
            return_normals=True,
            confidences=c_v,
            return_confidences=True,
        )
        pts_all.append(pts_w)
        cols_all.append(cols_w)
        feats_all.append(feats_w)
        fps_all.append(fps_w)
        norms_all.append(norms_w)
        confs_all.append(confs_w)

    pts_cat = np.concatenate(pts_all, axis=0)
    cols_cat = np.concatenate(cols_all, axis=0)
    feats_cat = np.concatenate(feats_all, axis=0)
    fps_cat = np.concatenate(fps_all, axis=0)
    norms_cat = np.concatenate(norms_all, axis=0)
    confs_cat = np.concatenate(confs_all, axis=0)

    # 載入 COLMAP 稀疏點雲 (解耦架構下保留原汁原味，不參與體素融合)
    colmap_candidates = [
        os.path.join(scene_dir, f"colmap_{frame_idx}", "sparse", "0", "points3D.bin"),
        os.path.join(scene_dir, f"colmap_{frame_idx}", "sparse", "0", "points3D.ply"),
        os.path.join(scene_dir, f"colmap_{frame_idx}", "sparse", "0", "points3D.txt"),
        os.path.join(scene_dir, "colmap_0", "sparse", "0", "points3D.bin"),
        os.path.join(scene_dir, "colmap_0", "sparse", "0", "points3D_total1_sec_fps30.ply"),
    ]
    colmap_path = None
    for cp in colmap_candidates:
        if os.path.isfile(cp):
            colmap_path = cp
            break

    colmap_pts = np.zeros((0, 3), dtype=np.float32)
    colmap_cols = np.zeros((0, 3), dtype=np.uint8)
    has_colmap = False
    if colmap_path is not None:
        try:
            colmap_pts, colmap_cols = load_colmap_points(colmap_path)
            has_colmap = True
            print(f"📌 [{scene_name} | Frame {frame_idx:03d}] 成功載入 COLMAP 稀疏錨點: {len(colmap_pts):,} 點 ({colmap_path})")
        except Exception as e:
            print(f"⚠️ 載入 COLMAP 點雲失敗 ({colmap_path}): {e}")

    # 體素融合
    if colmap_strategy == "decoupled":
        # 純淨 DA3 點雲進行體素化
        v_pts, v_cols, v_feats, _, v_confs = voxel_downsample_unified_with_features(
            pts=pts_cat,
            colors=cols_cat,
            features=feats_cat,
            normals=norms_cat,
            voxel_size=0.1,
            footprints=fps_cat,
            min_voxel_size=0.01,
            scale=1.0,
            mode="surfel",
            surfel_normal_scale=5.0,
            surfel_angle_deg=25.0,
            confidences=confs_cat,
            conf_agg=conf_agg,
            return_confidences=True,
        )
        if conf_filter_mode == "3d":
            valid_vox_mask = (v_confs >= conf_thresh)
            v_pts = v_pts[valid_vox_mask]
            v_cols = v_cols[valid_vox_mask]
            v_feats = v_feats[valid_vox_mask]
    else:
        # Legacy 耦合模式
        if has_colmap:
            cams_pos = np.stack([np.linalg.inv(extr)[:3, 3] for extr in extr_4x4])
            dists_col_cams = np.linalg.norm(colmap_pts[:, None, :] - cams_pos[None, :, :], axis=-1)
            min_dists = dists_col_cams.min(axis=-1)
            avg_focal = 0.5 * (intr_chunk[:, 0, 0].mean() + intr_chunk[:, 1, 1].mean())
            footprints_col = np.clip(min_dists / avg_focal, 0.01, 0.1).astype(np.float32)
            norms_col = assign_normals_to_colmap_points(
                colmap_pts=colmap_pts,
                da3_pts=pts_cat,
                da3_normals=norms_cat,
                cams_pos=cams_pos,
                max_nn_dist=0.15
            )
            pts_col_normed = (colmap_cols / 255.0).astype(np.float32) if colmap_cols.max() > 1.0 else colmap_cols.astype(np.float32)
            feats_col = np.zeros((len(colmap_pts), 3), dtype=np.float32)
            confs_col = np.full(len(colmap_pts), 20.0, dtype=np.float32)
            pts_cat = np.concatenate([pts_cat, colmap_pts], axis=0)
            cols_cat = np.concatenate([cols_cat, pts_col_normed], axis=0)
            feats_cat = np.concatenate([feats_cat, feats_col], axis=0)
            fps_cat = np.concatenate([fps_cat, footprints_col], axis=0)
            norms_cat = np.concatenate([norms_cat, norms_col], axis=0)
            confs_cat = np.concatenate([confs_cat, confs_col], axis=0)

        v_pts, v_cols, v_feats, _, v_confs = voxel_downsample_unified_with_features(
            pts=pts_cat,
            colors=cols_cat,
            features=feats_cat,
            normals=norms_cat,
            voxel_size=0.1,
            footprints=fps_cat,
            min_voxel_size=0.01,
            scale=1.0,
            mode="surfel",
            surfel_normal_scale=5.0,
            surfel_angle_deg=25.0,
            confidences=confs_cat,
            conf_agg=conf_agg,
            return_confidences=True,
        )
        valid_vox_mask = (v_confs >= conf_thresh)
        v_pts = v_pts[valid_vox_mask]
        v_cols = v_cols[valid_vox_mask]
        v_feats = v_feats[valid_vox_mask]

    M = len(v_pts)
    print(f"📊 [{scene_name} | Frame {frame_idx:03d}] 純淨 DA3 體素候選點數: {M:,}")

    # 機率分佈計算
    s_sob_v = v_feats[:, 0]
    s_dct_v = v_feats[:, 1]
    s_chr_v = v_feats[:, 2]

    p_base = 1.0 / M
    p_sob = s_sob_v / max(np.sum(s_sob_v), 1e-12)
    p_dct = s_dct_v / max(np.sum(s_dct_v), 1e-12)
    p_chr = s_chr_v / max(np.sum(s_chr_v), 1e-12)

    w_0 = feat_weights.get("w_0", 0.1181)
    w_sob = feat_weights.get("w_sobel", 0.4352)
    w_dct = feat_weights.get("w_dct", 0.2486)
    w_chr = feat_weights.get("w_chroma", 0.1981)

    if colmap_strategy == "decoupled":
        p_k = (
            w_0 * p_base +
            w_sob * p_sob +
            w_dct * p_dct +
            w_chr * p_chr
        )
    else:
        tree_v = cKDTree(v_pts)
        search_r_col = 0.15
        dists_col, matched_vox_col = tree_v.query(colmap_pts, k=1, distance_upper_bound=search_r_col)
        valid_col = dists_col < search_r_col
        colmap_counts = np.bincount(matched_vox_col[valid_col], minlength=M).astype(np.float32)
        p_col = colmap_counts / max(np.sum(colmap_counts), 1e-12)
        w_col = feat_weights.get("w_colmap", 0.4 if has_colmap else 0.0)
        p_k = (
            w_0 * p_base +
            w_sob * p_sob +
            w_dct * p_dct +
            w_chr * p_chr +
            w_col * p_col
        )

    p_sum = p_k.sum()
    if p_sum > 0:
        p_k /= p_sum
    else:
        p_k = np.full(M, 1.0 / M, dtype=np.float32)

    p_torch = torch.from_numpy(p_k).float()
    v_cols_u8 = (np.clip(v_cols, 0.0, 1.0) * 255).astype(np.uint8) if v_cols.max() <= 1.0 else v_cols.astype(np.uint8)

    # 匯出完整候選點雲 (選用，預設不匯出以節省硬碟空間)
    if export_full:
        out_ply_full = os.path.join(colmap_frame_dir, "da3_full.ply")
        if overwrite or not os.path.exists(out_ply_full):
            save_ply(out_ply_full, v_pts, v_cols_u8)
            print(f"  ✅ 成功匯出 (Full Surfel): {out_ply_full} ({M:,} 點)")

    # 匯出各預算點雲
    for b in budgets:
        b_str = f"{b // 1000}k" if (b % 1000 == 0 and b >= 1000) else str(b)

        if colmap_strategy == "decoupled":
            # 解耦架構: COLMAP 100% 保留，剩餘預算由 DA3 體素補充
            num_col = len(colmap_pts)
            needed_da3 = max(0, b - num_col)
            k_sample = min(needed_da3, M)
            sampled_indices = torch.multinomial(p_torch, num_samples=k_sample, replacement=False).numpy()
            da3_sampled_pts = v_pts[sampled_indices]
            da3_sampled_cols = v_cols_u8[sampled_indices]

            colmap_cols_u8 = (np.clip(colmap_cols, 0.0, 1.0) * 255).astype(np.uint8) if colmap_cols.max() <= 1.0 else colmap_cols.astype(np.uint8)

            final_pts = np.concatenate([colmap_pts, da3_sampled_pts], axis=0)
            final_cols = np.concatenate([colmap_cols_u8, da3_sampled_cols], axis=0)

            out_ply_giant = os.path.join(colmap_frame_dir, f"da3_giant_colmap_importance_{b_str}.ply")
            if overwrite or not os.path.exists(out_ply_giant):
                save_ply(out_ply_giant, final_pts, final_cols)
                print(f"  ✅ 成功匯出 (Decoupled {b_str}): {out_ply_giant} ({len(final_pts):,} 點: {num_col} COLMAP + {k_sample} DA3)")

            # 同步匯出 Uniform 消融組: COLMAP 全量 + k_sample DA3 Uniform 採樣
            out_ply_giant_uni = os.path.join(colmap_frame_dir, f"da3_giant_colmap_uniform_{b_str}.ply")
            if overwrite or not os.path.exists(out_ply_giant_uni):
                uni_indices = np.random.choice(M, size=k_sample, replace=False)
                final_pts_uni = np.concatenate([colmap_pts, v_pts[uni_indices]], axis=0)
                final_cols_uni = np.concatenate([colmap_cols_u8, v_cols_u8[uni_indices]], axis=0)
                save_ply(out_ply_giant_uni, final_pts_uni, final_cols_uni)
                print(f"  ✅ 成功匯出 (Decoupled Uniform {b_str}): {out_ply_giant_uni} ({len(final_pts_uni):,} 點)")

            # 若預算為 14000 (14k)，同步產出 Group B: COLMAP 砍半 (~3.5k) + 3.5k DA3 Importance (總計約 7k 點)
            if b == 14000 or b == 7330:
                col_half_pts = colmap_pts[::2]
                col_half_cols = colmap_cols_u8[::2]
                k_half = max(0, 7000 - len(col_half_pts))
                sampled_half_indices = torch.multinomial(p_torch, num_samples=k_half, replacement=False).numpy()
                final_pts_half = np.concatenate([col_half_pts, v_pts[sampled_half_indices]], axis=0)
                final_cols_half = np.concatenate([col_half_cols, v_cols_u8[sampled_half_indices]], axis=0)
                out_ply_giant_7k = os.path.join(colmap_frame_dir, "da3_giant_colmap_importance_7k.ply")
                if overwrite or not os.path.exists(out_ply_giant_7k):
                    save_ply(out_ply_giant_7k, final_pts_half, final_cols_half)
                    print(f"  ✅ 成功匯出 (Decoupled 7k, COLMAP砍半): {out_ply_giant_7k} ({len(final_pts_half):,} 點: {len(col_half_pts)} COLMAP + {k_half} DA3)")

            # 同步更新 da3_importance_{b_str}.ply 作為相容輸出
            out_ply_imp = os.path.join(colmap_frame_dir, f"da3_importance_{b_str}.ply")
            if overwrite or not os.path.exists(out_ply_imp):
                save_ply(out_ply_imp, final_pts, final_cols)
        else:
            k_sample = min(b, M)
            out_ply_imp = os.path.join(colmap_frame_dir, f"da3_importance_{b_str}.ply")
            if overwrite or not os.path.exists(out_ply_imp):
                sampled_indices = torch.multinomial(p_torch, num_samples=k_sample, replacement=False).numpy()
                sampled_pts = v_pts[sampled_indices]
                sampled_cols = v_cols_u8[sampled_indices]
                save_ply(out_ply_imp, sampled_pts, sampled_cols)
                print(f"  ✅ 成功匯出 (Importance): {out_ply_imp} ({len(sampled_pts):,} 點)")

            out_ply_uni = os.path.join(colmap_frame_dir, f"da3_uniform_{b_str}.ply")
            if overwrite or not os.path.exists(out_ply_uni):
                uni_indices = np.random.choice(M, size=k_sample, replace=False)
                uni_pts = v_pts[uni_indices]
                uni_cols = v_cols_u8[uni_indices]
                save_ply(out_ply_uni, uni_pts, uni_cols)
                print(f"  ✅ 成功匯出 (Uniform):    {out_ply_uni} ({len(uni_pts):,} 點)")


def main():
    parser = argparse.ArgumentParser(description="生成 N3D 6 個場景在 6 個均勻幀下的 DA3 重要度採樣點雲")
    parser.add_argument("--scenes", nargs="+", default=ALL_SCENES, help="目標場景名稱")
    parser.add_argument("--frames", nargs="+", type=int, default=TARGET_FRAMES, help="目標幀列表 (預設 25 75 125 175 225 275)")
    parser.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS, help="點數預算 (預設 7330 50000 250000)")
    parser.add_argument("--cam_select_mode", type=str, choices=["da3_surfel", "fps"], default="fps", help="相機挑選模式 (預設 fps)")
    parser.add_argument("--overwrite", action="store_true", help="強制重新生成已存在的點雲")
    parser.add_argument("--weights_json", type=str, default=None, help="特徵權重 JSON 檔 (預設自動搜尋最佳權重檔)")
    parser.add_argument("--conf_thresh", type=float, default=5.0, help="體素信心度門檻 (預設 5.0)")
    parser.add_argument("--conf_filter_mode", type=str, choices=["2d", "3d"], default="2d", help="信心度過濾階段 (預設 2d)")
    parser.add_argument("--conf_agg", type=str, choices=["min", "mean"], default="min", help="體素信心聚合方式: min (多視角最小信心，推薦) 或 mean (多視角平均信心)")
    parser.add_argument("--colmap_strategy", type=str, choices=["decoupled", "coupled", "none"], default="decoupled", help="COLMAP 融合策略 (decoupled: 保留原錨點不參與體素融合)")
    parser.add_argument("--da3_model", type=str, default="depth-anything/DA3-GIANT-1.1", help="DA3 模型權重")
    parser.add_argument("--process_res", type=int, default=728, help="DA3 推理解析度 (GIANT 預設 728)")
    parser.add_argument("--export_full", action="store_true", help="額外匯出未採樣之完整候選體素點雲 da3_full.ply (預設不匯出以節省硬碟空間)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("="*70)
    print(f"🎯 開始生成 DA3 空間特徵重要度採樣點雲")
    print(f"   場景: {args.scenes}")
    print(f"   幀數: {args.frames}")
    print(f"   點數預算: {args.budgets}")
    print(f"   COLMAP 策略: {args.colmap_strategy.upper()} (信心過濾: {args.conf_filter_mode.upper()} >= {args.conf_thresh})")
    print(f"   DA3 模型: {args.da3_model} (解析度: {args.process_res})")
    print("="*70)

    from depth_anything_3.api import DepthAnything3
    print("⏳ 載入 DA3 模型至 GPU...")
    if "GIANT" in args.da3_model.upper() and device == "cuda":
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if total_vram_gb <= 16.0:
            print("⚡ 啟用 DA3-GIANT 動態分層推論 (Sequential Block Offload) 以防顯存溢出...")
            da3_model = DepthAnything3.from_pretrained(args.da3_model)
            da3_model = enable_da3_layer_offload(da3_model, device=torch.device(device))
        else:
            da3_model = DepthAnything3.from_pretrained(args.da3_model).to(device)
    else:
        da3_model = DepthAnything3.from_pretrained(args.da3_model).to(device)
    da3_model.eval()

    np.random.seed(72)
    torch.manual_seed(72)

    total_start = time.time()
    count = 0
    total_tasks = len(args.scenes) * len(args.frames)

    for s_idx, scene in enumerate(args.scenes, 1):
        for f_idx, frame in enumerate(args.frames, 1):
            count += 1
            print(f"\n[{count}/{total_tasks}] 處理資料集: {scene} | 幀數: {frame}")
            try:
                process_scene_frame(
                    scene_name=scene,
                    frame_idx=frame,
                    da3_model=da3_model,
                    device=device,
                    budgets=args.budgets,
                    weights_path=os.path.join(PROJECT_ROOT, args.weights_json) if args.weights_json else None,
                    cam_select_mode=args.cam_select_mode,
                    da3_model_name=args.da3_model,
                    process_res=args.process_res,
                    conf_thresh=args.conf_thresh,
                    conf_filter_mode=args.conf_filter_mode,
                    conf_agg=args.conf_agg,
                    colmap_strategy=args.colmap_strategy,
                    overwrite=args.overwrite,
                    export_full=args.export_full
                )
            except Exception as e:
                print(f"❌ 處理 {scene} f{frame} 失敗: {e}")
                import traceback
                traceback.print_exc()

    elapsed = time.time() - total_start
    print("\n" + "="*70)
    print(f"🎉 全部點雲生成完成！總耗時: {elapsed/60:.2f} 分鐘")
    print("="*70)


if __name__ == "__main__":
    main()
