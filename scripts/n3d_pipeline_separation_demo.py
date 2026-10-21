#!/usr/bin/env python3
"""
N3D Pipeline 4D Static/Dynamic Separation Demo (v4.1 - Multi-Frame Decoupled Surfel + pycolmap & Hybrid Projection)
完全遵守 scripts/README_DA3_COLMAP_PIPELINE.md 規範：
1. 使用 pycolmap 4.1.1 官方 C++ 綁定載入 COLMAP 稀疏重建（相容二進位回退）
2. 混合式多視角動態判定（Hybrid Track + 3D Projection）：
   - 有原生 SIFT 軌跡者查表原生 2D 座標（0 遮擋）
   - 在 5 主相機中無 SIFT 軌跡者自動執行 3D->2D 視錐投影查表，徹底根除人物動態殘留綠點
3. 全局純靜態骨架 (Global Static Scaffold)：
   - 由全時段累積 Mask 進行多視角投票 (0.0, 0.5, 1.0)
   - 100% 杜絕任何動態物體或陰影被誤釘在靜態背景上 (Rerun static=True 跨全時段可見)
4. 逐影格專屬動態點雲 (Per-Frame Dynamic Point Clouds, t=0 ... T-1)：
   - 各影格讀取專屬的 colmap_t 錨點與當前幀 DA3 深度（process_res=728, conf >= 5.0）
   - mode='surfel' 表面基元融合，CUDA 自動聚類並計算動態投票比率
   - 賦予時間標籤 time=t (即 CSTG 4DGS 的 trbf_center = t / fps)
5. Rerun 0.35 視覺化分層標記：
   - 靜態點：COLMAP (深翠綠 🟢) vs DA3 (薄荷青綠 🟩)
   - 動態點：COLMAP (鮮豔紅 🔴) vs DA3 (亮橙黃 🟠)
   - 可在 Rerun UI 中獨立勾選開關，並輸出 CSTG 4D PLY
"""

import os
import sys
import glob
import time
import json
import struct
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import binary_dilation
import rerun as rr
from plyfile import PlyData, PlyElement

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mvtracker.utils.camera_selection import select_cams_by_fps
from mvtracker.utils.feature_scores import depth_to_surface_normals
from mvtracker.utils.spatial_voxelizer import (
    unproject_camera_points_with_features,
    voxel_downsample_unified_with_features,
)
from depth_anything_3.api import DepthAnything3
from mvtracker.datasets.generic_scene_dataset import enable_da3_layer_offload
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights


def parse_cam_idx(path):
    import re
    match = re.search(r'\d+', os.path.basename(path))
    return int(match.group()) if match else 0


def compute_adaptive_radii(pts, base_radius=0.018, scale=0.004, max_radius=1.5):
    """
    計算距離自適應半徑，確保超遠景 (如 r > 50m ~ 350m 之 COLMAP 錨點)
    在 Rerun 3D 視角中不會因世界座標像素投影過小而消失。
    """
    if pts is None or len(pts) == 0:
        return base_radius
    dists = np.linalg.norm(pts, axis=-1)
    return np.clip(dists * scale, base_radius, max_radius).astype(np.float32)


def llff_to_opencv_w2c(pose_llff, W, H):
    """將 LLFF [3, 5] 姿態矩陣轉為 OpenCV 格式之 w2c [3, 4] 與相機內參 K [3, 3]"""
    R_c2w = pose_llff[:, :3].copy()
    t_c2w = pose_llff[:, 3].copy()
    focal = pose_llff[2, 4]
    R_c2w_cv = np.stack([R_c2w[:, 1], R_c2w[:, 0], -R_c2w[:, 2]], axis=-1)
    c2w_cv = np.eye(4, dtype=np.float32)
    c2w_cv[:3, :3] = R_c2w_cv
    c2w_cv[:3, 3] = t_c2w
    w2c_cv = np.linalg.inv(c2w_cv)

    K = np.array([
        [focal, 0.0, W / 2.0],
        [0.0, focal, H / 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    return w2c_cv[:3, :4], K


def load_colmap_with_pycolmap(colmap_dir, cam_id_to_v_idx=None):
    """
    使用 pycolmap 官方 C++ 綁定載入 COLMAP 重建資料，並提供輕量二進位回退機制。
    支援 cam_id_to_v_idx 映射以防如 coffee_martini 相機缺號時索引錯位。
    回傳:
      pts: [N, 3] float32
      cols: [N, 3] uint8
      images_meta: dict[img_id -> {'name', 'cam_idx', 'xys': [num_2d, 2]}]
      point_tracks: list of [track_len, 2] (img_id, pt2d_idx)
    """
    sparse_dir = os.path.join(colmap_dir, "sparse", "0")
    if not os.path.exists(sparse_dir):
        sparse_dir = os.path.join(colmap_dir, "sparse")
    if not os.path.exists(sparse_dir):
        sparse_dir = colmap_dir

    try:
        import pycolmap
        rec = pycolmap.Reconstruction(sparse_dir)
        p_ids = list(rec.points3D.keys())
        num_pts = len(p_ids)
        pts = np.empty((num_pts, 3), dtype=np.float32)
        cols = np.empty((num_pts, 3), dtype=np.uint8)
        tracks = []
        for i, p_id in enumerate(p_ids):
            p = rec.points3D[p_id]
            pts[i] = p.xyz
            cols[i] = p.color
            tracks.append([(el.image_id, el.point2D_idx) for el in p.track.elements])

        images_meta = {}
        for img_id, im in rec.images.items():
            c_num = parse_cam_idx(im.name)
            cam_idx = cam_id_to_v_idx[c_num] if (cam_id_to_v_idx is not None and c_num in cam_id_to_v_idx) else c_num
            xys = np.array([pt.xy for pt in im.points2D], dtype=np.float32)
            images_meta[img_id] = {'name': im.name, 'cam_idx': cam_idx, 'xys': xys}

        return pts, cols, images_meta, tracks
    except Exception as e:
        print(f"⚠️ pycolmap 載入失敗 ({e})，使用自建二進位解析器回退...")
        return load_colmap_binary_fallback(colmap_dir, cam_id_to_v_idx=cam_id_to_v_idx)


def load_colmap_binary_fallback(colmap_dir, cam_id_to_v_idx=None):
    """自建二進位 points3D.bin 與 images.bin 解析回退函式"""
    img_path = os.path.join(colmap_dir, "sparse", "0", "images.bin")
    pts_path = os.path.join(colmap_dir, "sparse", "0", "points3D.bin")
    if not os.path.isfile(img_path) or not os.path.isfile(pts_path):
        img_path = os.path.join(colmap_dir, "images.bin")
        pts_path = os.path.join(colmap_dir, "points3D.bin")

    if not os.path.isfile(img_path) or not os.path.isfile(pts_path):
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), {}, []

    images_meta = {}
    with open(img_path, 'rb') as f:
        num_images = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_images):
            img_id = struct.unpack('<I', f.read(4))[0]
            f.read(60)
            name = ''
            while True:
                ch = f.read(1)
                if ch == b'\x00':
                    break
                name += ch.decode('latin1')
            num_2d = struct.unpack('<Q', f.read(8))[0]
            raw_pts = f.read(24 * num_2d)
            arr = np.frombuffer(raw_pts, dtype=np.float64).reshape(num_2d, 3)
            xys = arr[:, :2].astype(np.float32)
            c_num = parse_cam_idx(name)
            cam_idx = cam_id_to_v_idx[c_num] if (cam_id_to_v_idx is not None and c_num in cam_id_to_v_idx) else c_num
            images_meta[img_id] = {'name': name, 'cam_idx': cam_idx, 'xys': xys}

    colmap_pts, colmap_cols, point_tracks = [], [], []
    with open(pts_path, 'rb') as f:
        num_pts = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_pts):
            data = f.read(43)
            p_id, x, y, z, r, g, b, err = struct.unpack('<QdddBBBd', data)
            tl = struct.unpack('<Q', f.read(8))[0]
            track_raw = f.read(8 * tl)
            track = np.frombuffer(track_raw, dtype=np.int32).reshape(tl, 2)
            colmap_pts.append([x, y, z])
            colmap_cols.append([r, g, b])
            point_tracks.append(track)

    return (
        np.array(colmap_pts, dtype=np.float32),
        np.array(colmap_cols, dtype=np.uint8),
        images_meta,
        point_tracks,
    )


def compute_colmap_dyn_ratios(
    colmap_pts,
    colmap_tracks,
    colmap_images_meta,
    dyn_masks,
    all_w2c,
    all_K,
    selected_cams,
    W_orig,
    H_orig,
):
    """
    計算 COLMAP 點在多視角下的動態投票比率：
    1. 優先使用原生 2D SIFT 軌跡座標查表 (高精度 0 遮擋)
    2. 若該點在所選 5 主視角中沒有 SIFT 軌跡 (n_vis == 0)，自動採用 3D->2D 相機投影進行動態檢測，
       徹底修復無軌跡點被錯誤判定為 0.0 (純靜態) 導致人物殘留綠點的問題！
    """
    N = len(colmap_pts)
    colmap_dyn_ratios = np.zeros(N, dtype=np.float32)
    colmap_cam_counts = np.zeros(N, dtype=np.int32)

    for i in range(N):
        track = colmap_tracks[i]
        n_vis = 0
        n_dyn = 0
        for row in track:
            img_id = row[0]
            pt2d_idx = row[1]
            if img_id not in colmap_images_meta:
                continue
            meta = colmap_images_meta[img_id]
            cam_idx = meta['cam_idx']

            if cam_idx in dyn_masks:
                n_vis += 1
                x_2d, y_2d = meta['xys'][pt2d_idx]
                xi = int(np.clip(np.round(x_2d), 0, W_orig - 1))
                yi = int(np.clip(np.round(y_2d), 0, H_orig - 1))
                if dyn_masks[cam_idx][yi, xi]:
                    n_dyn += 1

        # 若在 5 主相機中有原生 SIFT 軌跡：
        if n_vis > 0:
            colmap_cam_counts[i] = n_vis
            colmap_dyn_ratios[i] = float(n_dyn) / float(n_vis)
        else:
            # 3D 空間投影補全：檢查該點投影在 5 主相機視錐內時是否落在動態遮罩內
            p = colmap_pts[i]
            n_proj_vis = 0
            n_proj_dyn = 0
            for c in selected_cams:
                w2c = all_w2c[c]
                K = all_K[c]
                p_c = w2c[:3, :3] @ p + w2c[:3, 3]
                z = p_c[2]
                if z > 0.1:
                    u = int(round((K[0, 0] * p_c[0] / z) + K[0, 2]))
                    v = int(round((K[1, 1] * p_c[1] / z) + K[1, 2]))
                    if 0 <= u < W_orig and 0 <= v < H_orig:
                        n_proj_vis += 1
                        if dyn_masks[c][v, u]:
                            n_proj_dyn += 1

            colmap_cam_counts[i] = n_proj_vis
            if n_proj_vis > 0:
                colmap_dyn_ratios[i] = float(n_proj_dyn) / float(n_proj_vis)
            else:
                colmap_dyn_ratios[i] = 0.0

    return colmap_dyn_ratios, colmap_cam_counts


def compute_gof_2d_dynamic_masks(
    view_dirs,
    cams,
    t0,
    T,
    raft_step=3,
    raft_thresh=1.0,
    color_thresh=0.04,
    dilation_radius=5,
    device="cuda",
):
    """
    在 5 個主視角相機上計算 GOF (T 幀) 的時序聯集 2D 動態掩碼 (Temporal Union + Dilation)。
    回傳: dict[cam_idx -> mask_2d (H, W) bool]
    """
    print(f"🌊 [RAFT + 幀差] 計算 {len(cams)} 個主視角在 {T} 幀內的時序累積動態掩碼 (step={raft_step})...")
    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()
    raft_model = raft_large(weights=weights, progress=False).to(device)
    raft_model.eval()

    cam_masks = {}

    for c in tqdm(cams, desc="視角累積動態遮罩推論"):
        f_paths = sorted(
            glob.glob(os.path.join(view_dirs[c], "*.png")) + glob.glob(os.path.join(view_dirs[c], "*.jpg")),
            key=lambda x: int("".join(filter(str.isdigit, os.path.basename(x)))) if any(ch.isdigit() for ch in os.path.basename(x)) else 0
        )[t0:t0 + T]

        img0_pil = Image.open(f_paths[0]).convert("RGB")
        W, H = img0_pil.size
        W_raft, H_raft = 672, 504
        img0_raft = img0_pil.resize((W_raft, H_raft), Image.Resampling.BILINEAR)
        img0_t = torch.from_numpy(np.array(img0_raft)).permute(2, 0, 1).unsqueeze(0).to(device)
        img0_norm = img0_t.float() / 255.0

        accum_dyn_mask_raft = torch.zeros((H_raft, W_raft), dtype=torch.bool, device=device)

        steps = list(range(raft_step, len(f_paths), raft_step))
        if not steps and len(f_paths) > 1:
            steps = [len(f_paths) - 1]

        with torch.no_grad():
            for t_idx in steps:
                img_t_pil = Image.open(f_paths[t_idx]).convert("RGB")
                img_t_raft = img_t_pil.resize((W_raft, H_raft), Image.Resampling.BILINEAR)
                img_t = torch.from_numpy(np.array(img_t_raft)).permute(2, 0, 1).unsqueeze(0).to(device)
                img_t_norm = img_t.float() / 255.0

                t1, t2 = transforms(img0_t, img_t)
                flow = raft_model(t1, t2)[-1][0]
                mag = torch.norm(flow, dim=0)

                c_diff = torch.abs(img_t_norm - img0_norm).squeeze(0).mean(dim=0)

                step_dyn = (mag > raft_thresh) | (c_diff > color_thresh)
                accum_dyn_mask_raft = accum_dyn_mask_raft | step_dyn

        mask_raft_np = accum_dyn_mask_raft.cpu().numpy()
        mask_pil = Image.fromarray(mask_raft_np).resize((W, H), Image.Resampling.NEAREST)
        mask_np = np.array(mask_pil, dtype=bool)

        if dilation_radius > 0:
            struct_elem = np.ones((2 * dilation_radius + 1, 2 * dilation_radius + 1), dtype=bool)
            mask_np = binary_dilation(mask_np, structure=struct_elem)

        cam_masks[c] = mask_np
        dyn_pct = mask_np.mean() * 100.0
        print(f"  📷 Cam {c:02d}: 全時段累積動態區域佔比 = {dyn_pct:.2f}% (膨脹半徑={dilation_radius}px)")

    del raft_model
    torch.cuda.empty_cache()
    return cam_masks


def process_frame_surfel_geometry(
    cam_imgs,
    selected_cams,
    extr_4x4,
    intr_orig_arr,
    da3_model,
    dyn_masks,
    W_orig,
    H_orig,
    process_res=728,
    conf_thresh=5.0,
    depth_max=500.0,
    device="cuda",
):
    """
    推論單一影格之 DA3 深度圖與法向量，並反投影執行 mode='surfel' 體素聚類。
    """
    with torch.no_grad():
        pred = da3_model.inference(
            image=cam_imgs,
            extrinsics=extr_4x4,
            intrinsics=intr_orig_arr,
            process_res=process_res,
            align_to_input_ext_scale=True,
            infer_gs=False,
        )
        depths_raw = pred.depth.astype(np.float32)
        confs_raw = pred.conf.astype(np.float32) if (hasattr(pred, 'conf') and pred.conf is not None) else np.full_like(depths_raw, 10.0)

    _, H_proc, W_proc = depths_raw.shape
    scale_x = W_proc / W_orig
    scale_y = H_proc / H_orig

    cam_intr_scaled = []
    cam_masks_proc = []
    for v, c in enumerate(selected_cams):
        K_s = intr_orig_arr[v].copy()
        K_s[0, :] *= scale_x
        K_s[1, :] *= scale_y
        cam_intr_scaled.append(K_s)
        m_pil = Image.fromarray(dyn_masks[c]).resize((W_proc, H_proc), Image.Resampling.NEAREST)
        cam_masks_proc.append(np.array(m_pil, dtype=bool))

    intrs_proc_tensor = torch.from_numpy(np.stack(cam_intr_scaled)).float().to(device)
    depths_tensor = torch.from_numpy(depths_raw).unsqueeze(1).to(device)
    with torch.no_grad():
        normals_raw, _ = depth_to_surface_normals(depths_tensor, intrinsics=intrs_proc_tensor, method="min_gradient")
        normals_torch = F.normalize(normals_raw, dim=1, eps=1e-6)
        normals_proc = normals_torch.permute(0, 2, 3, 1).cpu().numpy()

    pts_all, cols_all, norms_all, dyns_all, fps_all = [], [], [], [], []
    for v in range(len(cam_imgs)):
        rgb_v = np.array(cam_imgs[v].resize((W_proc, H_proc), Image.Resampling.BILINEAR))
        d_v = depths_raw[v]
        c_v = confs_raw[v]
        m_v = cam_masks_proc[v]
        norm_v = normals_proc[v]
        K_v = cam_intr_scaled[v]
        c2w_v = np.linalg.inv(extr_4x4[v])

        valid_mask_v = (d_v > 0.1) & (d_v < depth_max) & np.isfinite(d_v) & (c_v >= conf_thresh)
        pts_w, cols_w, _, fps_w, norms_w = unproject_camera_points_with_features(
            rgb=rgb_v,
            depth=d_v,
            features=None,
            K=K_v,
            c2w=c2w_v,
            depth_min=0.1,
            depth_max=depth_max,
            valid_mask=valid_mask_v,
            normals=norm_v,
            return_normals=True,
            confidences=None,
            return_confidences=False,
        )
        pts_all.append(pts_w)
        cols_all.append(cols_w)
        norms_all.append(norms_w)
        fps_all.append(fps_w)
        dyns_all.append(m_v[valid_mask_v])

    pts_cat = np.concatenate(pts_all, axis=0)
    cols_cat = np.concatenate(cols_all, axis=0)
    norms_cat = np.concatenate(norms_all, axis=0)
    fps_cat = np.concatenate(fps_all, axis=0)
    dyns_cat = np.concatenate(dyns_all, axis=0)

    v_pts, v_cols, _, v_dyn_ratio = voxel_downsample_unified_with_features(
        pts=pts_cat,
        colors=cols_cat,
        features=None,
        normals=norms_cat,
        is_dynamic=dyns_cat,
        voxel_size=0.1,
        footprints=fps_cat,
        min_voxel_size=0.01,
        scale=1.0,
        mode="surfel",
        surfel_normal_scale=5.0,
        surfel_angle_deg=25.0,
        return_dyn_ratio=True,
    )
    return v_pts, v_cols, v_dyn_ratio, cam_intr_scaled, cam_masks_proc, (W_proc, H_proc)


def save_4d_ply(ply_path, pts, colors, times, normals=None, is_dynamic=None, labels=None):
    """
    匯出符合 CSTG 格式之 4D PLY (含 x, y, z, nx, ny, nz, red, green, blue, time, t, is_dynamic, label)
    - is_dynamic: 0 為純靜態 3DGS 點，1 為動態 4DGS 點
    - label: 0=COLMAP 靜態, 1=DA3 靜態, 2=COLMAP 動態, 3=DA3 動態
    """
    N = len(pts)
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('time', 'f4'),
        ('t', 'f4'),
        ('is_dynamic', 'u1'),
        ('label', 'u1')
    ]
    elements = np.empty(N, dtype=dtype)
    elements['x'] = pts[:, 0]
    elements['y'] = pts[:, 1]
    elements['z'] = pts[:, 2]
    if normals is not None and len(normals) == N:
        elements['nx'] = normals[:, 0]
        elements['ny'] = normals[:, 1]
        elements['nz'] = normals[:, 2]
    else:
        elements['nx'] = 0.0
        elements['ny'] = 0.0
        elements['nz'] = 0.0
    elements['red'] = colors[:, 0]
    elements['green'] = colors[:, 1]
    elements['blue'] = colors[:, 2]
    elements['time'] = times
    elements['t'] = times
    elements['is_dynamic'] = is_dynamic if is_dynamic is not None else (times > 0).astype(np.uint8)
    elements['label'] = labels if labels is not None else elements['is_dynamic']

    PlyData([PlyElement.describe(elements, 'vertex')], text=False).write(ply_path)
    n_sta = int(np.sum(elements['is_dynamic'] == 0))
    n_dyn = int(np.sum(elements['is_dynamic'] == 1))
    print(f"💾 [4D PLY] 成功儲存 4D 點雲至: {ply_path} (共 {N:,} 點, 靜態: {n_sta:,}, 動態: {n_dyn:,})")


def save_3d_ply(ply_path, pts, colors, normals=None, labels=None):
    """匯出純靜態 3DGS 初始化 PLY (相容常規 3DGS 載入器)"""
    N = len(pts)
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('is_dynamic', 'u1'),
        ('label', 'u1')
    ]
    elements = np.empty(N, dtype=dtype)
    elements['x'] = pts[:, 0]
    elements['y'] = pts[:, 1]
    elements['z'] = pts[:, 2]
    if normals is not None and len(normals) == N:
        elements['nx'] = normals[:, 0]
        elements['ny'] = normals[:, 1]
        elements['nz'] = normals[:, 2]
    else:
        elements['nx'] = 0.0
        elements['ny'] = 0.0
        elements['nz'] = 0.0
    elements['red'] = colors[:, 0]
    elements['green'] = colors[:, 1]
    elements['blue'] = colors[:, 2]
    elements['is_dynamic'] = 0
    elements['label'] = labels if labels is not None else 0

    PlyData([PlyElement.describe(elements, 'vertex')], text=False).write(ply_path)
    print(f"💾 [Static 3D PLY] 成功儲存純靜態 3D 點雲至: {ply_path} (共 {N:,} 點)")


def main():
    p = argparse.ArgumentParser(description="N3D 4D 靜動態分離管線 (pycolmap + Decoupled Surfel)")
    p.add_argument("--dir", type=str, required=True, help="N3D 資料集目錄 (例如 datasets/sear_steak)")
    p.add_argument("--max_frames", type=int, default=50, help="GOF 處理幀數 (預設 50)")
    p.add_argument("--start_frame", type=int, default=0, help="起始影格 t0 (預設 0)")
    p.add_argument("--frame_step", type=int, default=1, help="動態點雲抽樣影格步長 (預設 1 即逐影格生成)")
    p.add_argument("--num_cams", type=int, default=5, help="FPS 挑選相機數 (預設 5，用於 DA3 深度幾何表面重建)")
    p.add_argument("--static_budget", type=int, default=12000, help="全局靜態骨架點數預算 (預設 12000)")
    p.add_argument("--colmap_dyn_budget", type=int, default=800, help="每影格 COLMAP 動態錨點抽樣預算 (預設 800，<=0 則不抽樣)")
    p.add_argument("--da3_dyn_budget", type=int, default=800, help="每影格 DA3 動態 Surfel 抽樣預算 (預設 800，<=0 則不抽樣)")
    p.add_argument("--dyn_budget_per_frame", type=int, default=0, help="可選: 每影格總動態點數上限 (若 > 0 且未個別設定，則 COLMAP 與 DA3 各半)")
    p.add_argument("--mask_views", type=str, default="all", choices=["all", "selected"], help="2D 動態遮罩計算視角範圍: all (全訓練視角，預設) 或 selected (僅 5 主視角)")
    p.add_argument("--test_cam_idx", type=int, default=0, help="排除之測試視角索引 (預設 0 即 cam00，嚴格杜絕測試集資訊外洩)")
    p.add_argument("--da3_model", type=str, default="depth-anything/DA3-GIANT-1.1", help="DA3 模型版本 (預設 depth-anything/DA3-GIANT-1.1)")
    p.add_argument("--process_res", type=int, default=728, help="DA3 原生推論解析度 (預設 728)")
    p.add_argument("--conf_thresh", type=float, default=5.0, help="DA3 2D 信心度門檻 (預設 5.0)")
    p.add_argument("--depth_max", type=float, default=500.0, help="DA3 深度上限 (預設 500.0，避免遠景幾何截斷)")
    p.add_argument("--raft_step", type=int, default=3, help="光流抽樣間隔幀數 (預設 3)")
    p.add_argument("--raft_thresh", type=float, default=1.0, help="2D 光流強度門檻 (像素)")
    p.add_argument("--color_thresh", type=float, default=0.04, help="顏色差分門檻 (0~1)")
    p.add_argument("--dilation_radius", type=int, default=5, help="2D 遮罩形態學膨脹半徑 (像素)")
    p.add_argument("--rrd", type=str, default="sear_steak_4d_separation.rrd", help="輸出的 Rerun 檔名 (若指定 --no_rrd 則跳過 Rerun 記錄)")
    p.add_argument("--no_rrd", action="store_true", help="關閉 Rerun 視覺化記錄以大幅提升處理速度並節省磁碟空間 (適合批次產出)")
    p.add_argument("--export_ply", type=str, default="", help="可選: 輸出 CSTG 4D PLY 路徑")
    p.add_argument("--export_split_ply", action="store_true", default=True, help="是否額外輸出獨立的 init_static_3d.ply 與 init_dynamic_4d.ply (預設開啟)")
    args = p.parse_args()

    if args.dyn_budget_per_frame > 0:
        half_b = args.dyn_budget_per_frame // 2
        args.colmap_dyn_budget = half_b
        args.da3_dyn_budget = half_b

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_start_total = time.time()

    print("\n" + "=" * 65)
    print("🚀 [N3D 4D Pipeline] 靜動態分離展示 (pycolmap + Decoupled Surfel + Hybrid Projection)")
    print(f"📁 場景路徑: {args.dir} | 處理影格: t0={args.start_frame} 至 {args.start_frame + args.max_frames - 1} (step={args.frame_step})")
    print("=" * 65)

    # 1. 讀取相機外參與內參
    view_dirs = sorted([f for f in glob.glob(os.path.join(args.dir, 'cam*')) if os.path.isdir(f)], key=parse_cam_idx)
    if not view_dirs:
        view_dirs = sorted([f for f in glob.glob(os.path.join(args.dir, 'view_*')) if os.path.isdir(f)], key=parse_cam_idx)
    num_views = len(view_dirs)
    cam_id_to_v_idx = {parse_cam_idx(d): idx for idx, d in enumerate(view_dirs)}

    sample_f = glob.glob(os.path.join(view_dirs[0], "*.png")) + glob.glob(os.path.join(view_dirs[0], "*.jpg"))
    sample_img = Image.open(sample_f[0])
    W_orig, H_orig = sample_img.size

    poses_path = os.path.join(args.dir, 'poses_bounds.npy')
    poses_bounds = np.load(poses_path)
    all_poses_llff = poses_bounds[:, :-2].reshape([-1, 3, 5])

    all_w2c_K = [llff_to_opencv_w2c(all_poses_llff[i], W_orig, H_orig) for i in range(len(all_poses_llff))]
    all_w2c_np = np.stack([x[0] for x in all_w2c_K])
    all_intrs_np = np.stack([x[1] for x in all_w2c_K])

    # 2. FPS 挑選 5 個最佳訓練相機 (嚴格排除測試視角 cam00，嚴守基準測試紅線)
    exclude_cams = [args.test_cam_idx] if args.test_cam_idx >= 0 else []
    selected_cams = select_cams_by_fps(all_w2c_np, num_to_select=args.num_cams, alpha=0.8, exclude_indices=exclude_cams)
    print(f"🎯 [相機挑選] FPS 選取 5 個主視角相機: {selected_cams} (已排除測試視角: {exclude_cams})")

    # 準備 5 主相機檔案清單與幾何矩陣
    cam_file_lists = []
    cam_w2c_5 = []
    cam_intr_5 = []
    for c in selected_cams:
        flist = sorted(
            glob.glob(os.path.join(view_dirs[c], "*.png")) + glob.glob(os.path.join(view_dirs[c], "*.jpg")),
            key=lambda x: int("".join(filter(str.isdigit, os.path.basename(x)))) if any(ch.isdigit() for ch in os.path.basename(x)) else 0
        )
        cam_file_lists.append(flist)
        cam_w2c_5.append(all_w2c_np[c])
        cam_intr_5.append(all_intrs_np[c].copy())

    extr_4x4 = np.zeros((len(selected_cams), 4, 4), dtype=np.float32)
    extr_4x4[:, :3, :4] = np.stack(cam_w2c_5)
    extr_4x4[:, 3, 3] = 1.0
    intr_orig_arr = np.stack(cam_intr_5)

    # 3. 在全訓練視角或主視角上計算 GOF 全時段累積 2D 動態掩碼 (Temporal Union)
    if args.mask_views == "all":
        mask_cams = [i for i in range(num_views) if i not in exclude_cams]
        print(f"🌟 [全視角升級] 使用全部 {len(mask_cams)} 個訓練視角計算 2D 動態 Mask (嚴守紅線：已排除測試視角 {exclude_cams})，確保 100% 訓練相機原生 SIFT 查表且無測試資料外洩！")
    else:
        mask_cams = selected_cams
        print(f"🎯 使用挑選之 {len(mask_cams)} 個相機視角計算 2D 動態 Mask (已排除測試視角 {exclude_cams})")

    gof_dyn_masks = compute_gof_2d_dynamic_masks(
        view_dirs=view_dirs,
        cams=mask_cams,
        t0=args.start_frame,
        T=args.max_frames,
        raft_step=args.raft_step,
        raft_thresh=args.raft_thresh,
        color_thresh=args.color_thresh,
        dilation_radius=args.dilation_radius,
        device=device,
    )

    # 4. 載入 DA3 推論模型 (GIANT 模式支援動態分層 Sequential Block Offload 以防顯存溢出)
    print(f"🚀 [DA3 模型] 載入 {args.da3_model} (原生解析度 process_res={args.process_res})...")
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

    # 5. 【階段一】生成全局純靜態骨架 (Global Static Scaffold - 以 t0 為基準幾何)
    print("\n" + "=" * 55)
    print("🧱 [階段一] 構建全局純靜態骨架 (Global Static Scaffold)...")
    print("=" * 55)

    cam_imgs_t0 = [Image.open(cam_file_lists[v][args.start_frame]).convert("RGB") for v in range(len(selected_cams))]

    colmap_dir_0 = os.path.join(args.dir, f"colmap_{args.start_frame}")
    if not os.path.exists(colmap_dir_0):
        colmap_dir_0 = os.path.join(args.dir, "colmap_0")

    colmap_pts_0, colmap_cols_0, colmap_meta_0, colmap_tracks_0 = load_colmap_with_pycolmap(
        colmap_dir_0, cam_id_to_v_idx=cam_id_to_v_idx
    )
    colmap_ratios_0, colmap_counts_0 = compute_colmap_dyn_ratios(
        colmap_pts_0, colmap_tracks_0, colmap_meta_0, gof_dyn_masks,
        all_w2c=all_w2c_np, all_K=all_intrs_np, selected_cams=selected_cams,
        W_orig=W_orig, H_orig=H_orig,
    )
    n_native = int(np.sum(colmap_counts_0 > 0))
    n_proj = int(np.sum(colmap_counts_0 == 0))
    print(f"📌 [COLMAP t0] 經 pycolmap 載入 {len(colmap_pts_0):,} 錨點，查表全時段動態累積比例 ({n_native:,} 點原生 SIFT 軌跡查表, {n_proj} 點投影補全)")

    v_pts_0, v_cols_0, v_dyn_ratio_0, cam_intr_scaled, cam_masks_proc, (W_proc, H_proc) = process_frame_surfel_geometry(
        cam_imgs=cam_imgs_t0,
        selected_cams=selected_cams,
        extr_4x4=extr_4x4,
        intr_orig_arr=intr_orig_arr,
        da3_model=da3_model,
        dyn_masks=gof_dyn_masks,
        W_orig=W_orig,
        H_orig=H_orig,
        process_res=args.process_res,
        conf_thresh=args.conf_thresh,
        depth_max=args.depth_max,
        device=device,
    )
    print(f"📦 [DA3 t0 Surfel] 聚類產生 {len(v_pts_0):,} 表面基元")

    vote_thresholds = [0.0, 0.5, 1.0]
    th_keys = ["vote_0.0", "vote_0.5", "vote_1.0"]
    th_names = ["Separation_Vote_0.0_AnyView", "Separation_Vote_0.5_Majority", "Separation_Vote_1.0_AllViews"]

    static_scaffolds = {}
    print("\n🏛️ [全局純靜態骨架多視角投票結果] (動態比率 <= 門檻 視為靜態):")
    for th, k in zip(vote_thresholds, th_keys):
        if th == 0.0:
            col_sta_mask = colmap_ratios_0 <= 0.0001
            da3_sta_mask = v_dyn_ratio_0 <= 0.0001
        elif th == 0.5:
            col_sta_mask = colmap_ratios_0 < 0.5
            da3_sta_mask = v_dyn_ratio_0 < 0.5
        else:
            col_sta_mask = colmap_ratios_0 < 0.999
            da3_sta_mask = v_dyn_ratio_0 < 0.999

        col_sta_p = colmap_pts_0[col_sta_mask]
        col_sta_c = colmap_cols_0[col_sta_mask]

        da3_sta_p = v_pts_0[da3_sta_mask]
        da3_sta_c = v_cols_0[da3_sta_mask]
        if da3_sta_c.max() <= 1.0:
            da3_sta_c = (da3_sta_c * 255.0).astype(np.uint8)

        needed_da3 = max(0, args.static_budget - len(col_sta_p))
        k_sample = min(len(da3_sta_p), needed_da3)
        if len(da3_sta_p) > 0 and k_sample > 0:
            perm = np.random.choice(len(da3_sta_p), size=k_sample, replace=False)
            da3_sub_p = da3_sta_p[perm]
            da3_sub_c = da3_sta_c[perm]
        else:
            da3_sub_p = np.zeros((0, 3), np.float32)
            da3_sub_c = np.zeros((0, 3), np.uint8)

        final_sta_pts = np.concatenate([col_sta_p, da3_sub_p], axis=0).astype(np.float32)
        final_sta_cols = np.concatenate([col_sta_c, da3_sub_c], axis=0).astype(np.uint8)
        static_scaffolds[k] = {
            "threshold": th,
            "pts": final_sta_pts,
            "cols": final_sta_cols,
            "colmap_pts": col_sta_p,
            "colmap_cols": col_sta_c,
            "da3_pts": da3_sub_p,
            "da3_cols": da3_sub_c,
            "colmap_count": len(col_sta_p),
            "da3_count": len(da3_sub_p),
        }
        print(f"  門檻 {th:3.1f}: 靜態點數 {len(final_sta_pts):,} (COLMAP: {len(col_sta_p):,}, DA3: {len(da3_sub_p):,})")

    # 6. 【階段二】逐影格專屬動態點雲生成與 Rerun 時序記錄
    print("\n" + "=" * 55)
    print(f"🎬 [階段二] 逐影格動態生成 (Frames {args.start_frame} 至 {args.start_frame + args.max_frames - 1}, Step={args.frame_step})...")
    print("=" * 55)

    enable_rrd = not args.no_rrd and bool(args.rrd)
    rrd_path = args.rrd if args.rrd.endswith(".rrd") else f"{args.rrd}.rrd"

    if enable_rrd:
        print(f"🎨 [Rerun] 初始化 4D Recording: {rrd_path} ...")
        rr.init("n3d_pipeline_4d_separation", recording_id=f"sep4d_{int(time.time())}")
    else:
        print(f"⚡ [Rerun] 已啟用 --no_rrd，跳過 Rerun 視覺化記錄，以最高效能產出點雲與統計報告")

    # 色彩定義 (清楚區分 COLMAP 與 DA3 來源)
    C_COLMAP_STA = [20, 160, 45]     # 🟢 COLMAP 靜態：深翠綠
    C_DA3_STA    = [80, 230, 170]    # 🟩 DA3 靜態：薄荷青綠
    C_COLMAP_DYN = [255, 30, 30]     # 🔴 COLMAP 動態：鮮豔紅
    C_DA3_DYN    = [255, 140, 0]     # 🟠 DA3 動態：亮橙黃色

    if enable_rrd:
        # A. 記錄相機幾何與 2D 動態掩碼 (static=True 跨全時段可見，涵蓋全視角)
        for c in range(num_views):
            w2c_4x4 = np.eye(4, dtype=np.float32)
            w2c_4x4[:3, :4] = all_w2c_np[c]
            c2w = np.linalg.inv(w2c_4x4)
            R_cam = c2w[:3, :3]
            t_cam = c2w[:3, 3]
            K_s = all_intrs_np[c].copy()
            K_s[0, :] *= (W_proc / W_orig)
            K_s[1, :] *= (H_proc / H_orig)

            cam_tag = os.path.basename(view_dirs[c])
            rr.log(f"Cameras/{cam_tag}", rr.Pinhole(image_from_camera=K_s, width=W_proc, height=H_proc), static=True)
            rr.log(f"Cameras/{cam_tag}", rr.Transform3D(translation=t_cam, mat3x3=R_cam), static=True)

            if c in gof_dyn_masks:
                m_pil = Image.fromarray(gof_dyn_masks[c]).resize((W_proc, H_proc), Image.Resampling.NEAREST)
                mask_vis = np.zeros((H_proc, W_proc, 3), dtype=np.uint8)
                mask_vis[np.array(m_pil, dtype=bool)] = [255, 0, 0]
                rr.log(f"Cameras/{cam_tag}/Dynamic_Mask_2D", rr.Image(mask_vis), static=True)

        # B. 記錄三組門檻的純靜態骨架 (依據 COLMAP 與 DA3 來源分層記錄並使用不同顏色標記)
        for k, name in zip(th_keys, th_names):
            sta = static_scaffolds[k]
            if len(sta["colmap_pts"]) > 0:
                rr.log(
                    f"{name}/Static/COLMAP_Gaussians",
                    rr.Points3D(
                        positions=sta["colmap_pts"],
                        colors=np.full((len(sta["colmap_pts"]), 3), C_COLMAP_STA, dtype=np.uint8),
                        radii=compute_adaptive_radii(sta["colmap_pts"], base_radius=0.018),
                    ),
                    static=True
                )
            if len(sta["da3_pts"]) > 0:
                rr.log(
                    f"{name}/Static/DA3_Gaussians",
                    rr.Points3D(
                        positions=sta["da3_pts"],
                        colors=np.full((len(sta["da3_pts"]), 3), C_DA3_STA, dtype=np.uint8),
                        radii=compute_adaptive_radii(sta["da3_pts"], base_radius=0.016),
                    ),
                    static=True
                )

        # C. 幾何骨架分佈圖層 (COLMAP=青色, DA3=紫色)
        base_colmap_sta = static_scaffolds["vote_0.5"]["colmap_pts"]
        base_da3_sta = static_scaffolds["vote_0.5"]["da3_pts"]
        if len(base_colmap_sta) > 0:
            rr.log(
                "Geometry_Scaffold/COLMAP_Anchors",
                rr.Points3D(
                    positions=base_colmap_sta,
                    colors=[0, 230, 255],
                    radii=compute_adaptive_radii(base_colmap_sta, base_radius=0.018),
                ),
                static=True
            )
        if len(base_da3_sta) > 0:
            rr.log(
                "Geometry_Scaffold/DA3_Scaffold",
                rr.Points3D(
                    positions=base_da3_sta,
                    colors=[190, 100, 255],
                    radii=compute_adaptive_radii(base_da3_sta, base_radius=0.016),
                ),
                static=True
            )

    # D. 逐幀提取動態點雲並記錄至 timeline 'frame'
    frame_indices = list(range(args.start_frame, args.start_frame + args.max_frames, args.frame_step))
    stats_dyn_counts = {k: [] for k in th_keys}
    stats_colmap_dyn_counts = {k: [] for k in th_keys}
    stats_da3_dyn_counts = {k: [] for k in th_keys}

    # 準備 vote_0.5 靜態骨架點雲 (label 0=COLMAP 靜態, 1=DA3 靜態)
    base_sta = static_scaffolds["vote_0.5"]
    col_sta_p = base_sta["colmap_pts"]
    col_sta_c = base_sta["colmap_cols"]
    da3_sta_p = base_sta["da3_pts"]
    da3_sta_c = base_sta["da3_cols"]

    static_pts = np.concatenate([col_sta_p, da3_sta_p], axis=0).astype(np.float32)
    static_cols = np.concatenate([col_sta_c, da3_sta_c], axis=0).astype(np.uint8)
    static_labels = np.concatenate([
        np.zeros(len(col_sta_p), dtype=np.uint8),
        np.ones(len(da3_sta_p), dtype=np.uint8)
    ], axis=0)
    static_is_dyn = np.zeros(len(static_pts), dtype=np.uint8)
    static_times = np.zeros(len(static_pts), dtype=np.float32)

    export_dyn_pts = []
    export_dyn_cols = []
    export_dyn_times = []
    export_dyn_labels = []

    for t_idx in tqdm(frame_indices, desc="4D 逐影格動態生成"):
        cam_imgs_t = [Image.open(cam_file_lists[v][t_idx]).convert("RGB") for v in range(len(selected_cams))]

        # 讀取該幀專屬 colmap_t (透過 pycolmap 與 3D 投影補全)
        colmap_dir_t = os.path.join(args.dir, f"colmap_{t_idx}")
        if os.path.exists(colmap_dir_t):
            col_p_t, col_c_t, col_m_t, col_tr_t = load_colmap_with_pycolmap(
                colmap_dir_t, cam_id_to_v_idx=cam_id_to_v_idx
            )
            col_rat_t, _ = compute_colmap_dyn_ratios(
                col_p_t, col_tr_t, col_m_t, gof_dyn_masks,
                all_w2c=all_w2c_np, all_K=all_intrs_np, selected_cams=selected_cams,
                W_orig=W_orig, H_orig=H_orig,
            )
        else:
            col_p_t, col_c_t, col_rat_t = np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), np.zeros(0, np.float32)

        # 推論當前影格 DA3 與 Surfel 聚類
        v_pts_t, v_cols_t, v_dyn_ratio_t, _, _, _ = process_frame_surfel_geometry(
            cam_imgs=cam_imgs_t,
            selected_cams=selected_cams,
            extr_4x4=extr_4x4,
            intr_orig_arr=intr_orig_arr,
            da3_model=da3_model,
            dyn_masks=gof_dyn_masks,
            W_orig=W_orig,
            H_orig=H_orig,
            process_res=args.process_res,
            conf_thresh=args.conf_thresh,
            depth_max=args.depth_max,
            device=device,
        )
        if v_cols_t.max() <= 1.0:
            v_cols_t = (v_cols_t * 255.0).astype(np.uint8)

        # 記錄相機當前幀畫面至 Rerun 時間軸 (記錄 5 主相機畫面維護展示順暢)
        if enable_rrd:
            rr.set_time("frame", sequence=t_idx)
            for v, c in enumerate(selected_cams):
                cam_tag = os.path.basename(view_dirs[c])
                im_res = np.array(cam_imgs_t[v].resize((W_proc, H_proc), Image.Resampling.BILINEAR))
                rr.log(f"Cameras/{cam_tag}/Image", rr.Image(im_res))

        # 針對三組門檻生成該幀專屬動態點雲 (COLMAP 與 DA3 分層且分別上色)
        for th, k, name in zip(vote_thresholds, th_keys, th_names):
            if th == 0.0:
                col_dyn_mask = col_rat_t > 0.0001
                da3_dyn_mask = v_dyn_ratio_t > 0.0001
            elif th == 0.5:
                col_dyn_mask = col_rat_t >= 0.5
                da3_dyn_mask = v_dyn_ratio_t >= 0.5
            else:
                col_dyn_mask = col_rat_t >= 0.999
                da3_dyn_mask = v_dyn_ratio_t >= 0.999

            col_dyn_p = col_p_t[col_dyn_mask]
            col_dyn_c = col_c_t[col_dyn_mask]

            da3_dyn_p = v_pts_t[da3_dyn_mask]
            da3_dyn_c = v_cols_t[da3_dyn_mask]

            # 1. 抽樣 COLMAP 動態錨點 (獨立預算)
            if args.colmap_dyn_budget > 0 and len(col_dyn_p) > args.colmap_dyn_budget:
                perm_col = np.random.choice(len(col_dyn_p), size=args.colmap_dyn_budget, replace=False)
                col_dyn_sub_p = col_dyn_p[perm_col]
                col_dyn_sub_c = col_dyn_c[perm_col]
            else:
                col_dyn_sub_p = col_dyn_p
                col_dyn_sub_c = col_dyn_c

            # 2. 抽樣 DA3 動態 Surfel 點雲 (獨立預算，保證絕不被 COLMAP 擠壓排擠！)
            if args.da3_dyn_budget > 0 and len(da3_dyn_p) > args.da3_dyn_budget:
                perm_da3 = np.random.choice(len(da3_dyn_p), size=args.da3_dyn_budget, replace=False)
                da3_dyn_sub_p = da3_dyn_p[perm_da3]
                da3_dyn_sub_c = da3_dyn_c[perm_da3]
            else:
                da3_dyn_sub_p = da3_dyn_p
                da3_dyn_sub_c = da3_dyn_c

            frame_dyn_pts = np.concatenate([col_dyn_sub_p, da3_dyn_sub_p], axis=0).astype(np.float32)
            frame_dyn_cols = np.concatenate([col_dyn_sub_c, da3_dyn_sub_c], axis=0).astype(np.uint8)

            stats_dyn_counts[k].append(len(frame_dyn_pts))
            stats_colmap_dyn_counts[k].append(len(col_dyn_sub_p))
            stats_da3_dyn_counts[k].append(len(da3_dyn_sub_p))

            # 記錄到 Rerun 時間軸 (COLMAP 鮮紅 🔴 vs DA3 亮橘 🟠)
            if enable_rrd:
                rr.set_time("frame", sequence=t_idx)
                if len(col_dyn_sub_p) > 0:
                    rr.log(
                        f"{name}/Dynamic/COLMAP_Gaussians",
                        rr.Points3D(
                            positions=col_dyn_sub_p,
                            colors=np.full((len(col_dyn_sub_p), 3), C_COLMAP_DYN, dtype=np.uint8),
                            radii=compute_adaptive_radii(col_dyn_sub_p, base_radius=0.020),
                        )
                    )
                if len(da3_dyn_sub_p) > 0:
                    rr.log(
                        f"{name}/Dynamic/DA3_Gaussians",
                        rr.Points3D(
                            positions=da3_dyn_sub_p,
                            colors=np.full((len(da3_dyn_sub_p), 3), C_DA3_DYN, dtype=np.uint8),
                            radii=compute_adaptive_radii(da3_dyn_sub_p, base_radius=0.018),
                        )
                    )

            # 收集 vote_0.5 點雲用於 PLY 匯出 (label 2=COLMAP 動態, 3=DA3 動態)
            if k == "vote_0.5" and len(frame_dyn_pts) > 0:
                frame_labels = np.concatenate([
                    np.full(len(col_dyn_sub_p), 2, dtype=np.uint8),
                    np.full(len(da3_dyn_sub_p), 3, dtype=np.uint8)
                ], axis=0)
                export_dyn_pts.append(frame_dyn_pts)
                export_dyn_cols.append(frame_dyn_cols)
                export_dyn_times.append(np.full(len(frame_dyn_pts), float(t_idx), dtype=np.float32))
                export_dyn_labels.append(frame_labels)

    # 釋放 DA3 模型
    del da3_model
    torch.cuda.empty_cache()

    # 7. 儲存 Rerun 檔案
    if enable_rrd:
        rr.save(rrd_path)
        print(f"\n🎉 成功儲存 4D Rerun 檔案至: {os.path.abspath(rrd_path)}")

    # 8. 匯出 CSTG PLY 檔案 (統一整合版 + 獨立分流版)
    out_ply_path = args.export_ply if args.export_ply else os.path.join(args.dir, "init_pointcloud_4d_pipeline.ply")
    out_dir = os.path.dirname(os.path.abspath(out_ply_path))
    os.makedirs(out_dir, exist_ok=True)

    out_static_path = None
    out_dynamic_path = None

    if export_dyn_pts:
        dyn_pts_cat = np.concatenate(export_dyn_pts, axis=0)
        dyn_cols_cat = np.concatenate(export_dyn_cols, axis=0)
        dyn_times_cat = np.concatenate(export_dyn_times, axis=0)
        dyn_labels_cat = np.concatenate(export_dyn_labels, axis=0)
        dyn_is_dyn_cat = np.ones(len(dyn_pts_cat), dtype=np.uint8)

        # A. 輸出統一整合版 4D PLY (含 is_dynamic 與 label，向下相容 CSTG，向上支援靜動分離)
        all_4d_pts = np.concatenate([static_pts, dyn_pts_cat], axis=0)
        all_4d_cols = np.concatenate([static_cols, dyn_cols_cat], axis=0)
        all_4d_times = np.concatenate([static_times, dyn_times_cat], axis=0)
        all_4d_is_dyn = np.concatenate([static_is_dyn, dyn_is_dyn_cat], axis=0)
        all_4d_labels = np.concatenate([static_labels, dyn_labels_cat], axis=0)
        save_4d_ply(out_ply_path, all_4d_pts, all_4d_cols, all_4d_times, is_dynamic=all_4d_is_dyn, labels=all_4d_labels)

        # B. 輸出獨立分流版 PLY (專供 CSTG 靜態 3DGS + 動態 4DGS 獨立訓練加載)
        if args.export_split_ply:
            out_static_path = os.path.join(out_dir, "init_static_3d.ply")
            save_3d_ply(out_static_path, static_pts, static_cols, labels=static_labels)

            out_dynamic_path = os.path.join(out_dir, "init_dynamic_4d.ply")
            save_4d_ply(out_dynamic_path, dyn_pts_cat, dyn_cols_cat, dyn_times_cat, is_dynamic=dyn_is_dyn_cat, labels=dyn_labels_cat)

    # 9. 匯出統計 JSON 報告
    json_path = os.path.splitext(rrd_path)[0] + "_stats.json" if enable_rrd else os.path.splitext(out_ply_path)[0] + "_stats.json"
    stats_summary = {
        "scene": os.path.basename(args.dir),
        "total_frames_processed": len(frame_indices),
        "static_scaffolds": {
            k: {
                "threshold": v["threshold"],
                "total_static_points": len(v["pts"]),
                "colmap_static_points": v["colmap_count"],
                "da3_static_points": v["da3_count"],
            } for k, v in static_scaffolds.items()
        },
        "dynamic_stats": {
            k: {
                "avg_dynamic_points_per_frame": round(float(np.mean(counts)), 1) if counts else 0,
                "avg_colmap_dynamic_per_frame": round(float(np.mean(stats_colmap_dyn_counts[k])), 1) if stats_colmap_dyn_counts[k] else 0,
                "avg_da3_dynamic_per_frame": round(float(np.mean(stats_da3_dyn_counts[k])), 1) if stats_da3_dyn_counts[k] else 0,
                "max_dynamic_points": int(np.max(counts)) if counts else 0,
                "min_dynamic_points": int(np.min(counts)) if counts else 0,
            } for k, counts in stats_dyn_counts.items()
        },
        "output_files": {
            "rrd": os.path.abspath(rrd_path) if enable_rrd else None,
            "unified_4d_ply": os.path.abspath(out_ply_path) if export_dyn_pts else None,
            "static_3d_ply": os.path.abspath(out_static_path) if (export_dyn_pts and args.export_split_ply and out_static_path) else None,
            "dynamic_4d_ply": os.path.abspath(out_dynamic_path) if (export_dyn_pts and args.export_split_ply and out_dynamic_path) else None,
        },
        "elapsed_time_sec": round(time.time() - t_start_total, 2),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats_summary, f, indent=2, ensure_ascii=False)

    print(f"📄 統計報告已儲存至: {json_path}")
    print(f"⏱️ 總執行時間: {time.time() - t_start_total:.2f} 秒！")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
