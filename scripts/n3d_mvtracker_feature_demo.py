#!/usr/bin/env python3
"""
MVTracker 3D 空間特徵動態點雲串流視覺化 Demo (n3d_mvtracker_feature_demo.py)

在原版 n3d_mvtracker_demo.py 的基礎上擴充：
1. 100% 相容原腳本所有 CLI 參數與輸出結果 (追蹤軌跡、漂移分析圖、相機姿態)
2. 整合 3DGS 分佈對齊凸優化權重 (w0*, w1*, w2*, w3*)
3. 支援 Uniform Voxelization 與 Dynamic Footprint Voxelization
4. 逐幀多視角 2D 特徵提取 -> 跨視角 Q_global 尺度校準 -> 3D 反投影 -> Voxel Max-Pooling -> 特徵融合
5. 於 Rerun 逐幀寫入 3D 空間特徵體素點雲串流:
   - sequence-0/n3d/spatial_feature_pointcloud (主要融合特徵點雲)
   - sequence-0/n3d/spatial_feature_heatmap (Turbo 偽彩色熱力圖)
   - sequence-0/n3d/spatial_feature_dominant (四色機率優勢特徵分類圖)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import glob
import json
import re
import time
import warnings
import gc

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from mvtracker.datasets.generic_scene_dataset import (
    _ensure_da3_aligned_cache_and_load,
    _ensure_vggt_aligned_cache_and_load,
    _ensure_vggt_raw_cache_and_load,
)
from mvtracker.datasets.utils import align_depth_sparse_to_dense
from mvtracker.models.core.model_utils import init_pointcloud_from_rgbd
from mvtracker.utils.feature_scores import RGBTextureScorer, depth_to_surface_normals
from mvtracker.utils.spatial_voxelizer import (
    unproject_camera_points_with_features,
    voxel_downsample_unified_with_features,
)
from mvtracker.utils.visualizer_rerun import (
    log_pointclouds_to_rerun,
    log_tracks_to_rerun,
)
from mvtracker.utils.camera_selection import (
    select_cams_by_fps,
    select_cams_by_da3_surfel,
)


def get_sorted_frame_paths(cam_dir):
    frames = glob.glob(os.path.join(cam_dir, '*.png')) + glob.glob(os.path.join(cam_dir, '*.jpg'))
    def parse_frame_idx(path):
        basename = os.path.splitext(os.path.basename(path))[0]
        match = re.search(r'\d+', basename)
        return int(match.group()) if match else 0
    frames.sort(key=parse_frame_idx)
    return frames


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


def select_cams_by_fps(all_w2c_np, num_to_select, alpha=0.8, exclude_indices=[0]):
    all_indices = [i for i in range(len(all_w2c_np)) if i not in exclude_indices]
    if num_to_select >= len(all_indices):
        return sorted(all_indices)

    all_c2w = [np.linalg.inv(np.vstack([w2c, [0, 0, 0, 1]])) for w2c in all_w2c_np]
    centers = np.array([c2w[:3, 3] for c2w in all_c2w])
    dirs = np.array([c2w[:3, 2] for c2w in all_c2w])

    pos_dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    max_pos_dist = np.max(pos_dists) if np.max(pos_dists) > 0 else 1.0

    selected_indices = [all_indices[0]]

    while len(selected_indices) < num_to_select:
        combined_min_dists = []
        for i in all_indices:
            if i in selected_indices:
                combined_min_dists.append(-1.0)
                continue

            scores = []
            for s in selected_indices:
                d_pos = np.linalg.norm(centers[i] - centers[s]) / max_pos_dist
                cos_sim = np.dot(dirs[i], dirs[s])
                d_rot = (1.0 - cos_sim) / 2.0
                score = alpha * d_pos + (1.0 - alpha) * d_rot
                scores.append(score)

            combined_min_dists.append(np.min(scores))

        next_idx = int(np.argmax(combined_min_dists))
        selected_indices.append(next_idx)

    return sorted(selected_indices)


def main():
    p = argparse.ArgumentParser(description="MVTracker 3D 空間特徵動態點雲串流視覺化 Demo")
    p.add_argument("--dir", type=str, required=True, help="n3d 資料集路徑")
    p.add_argument("--max_frames", type=int, default=300, help="限制載入的最大幀數 (避免 MVTracker OOM)")
    p.add_argument("--num_queries", type=int, default=512, help="要追蹤的點數量")
    p.add_argument("--selected_cams", type=int, nargs="+", default=None, help="選擇的相機 index")
    p.add_argument("--num_cams", type=int, default=5, help="當未指定 --selected_cams 時，使用演算法自動挑選的訓練相機數量")
    p.add_argument("--cam_select_mode", type=str, choices=["da3_surfel", "fps"], default="da3_surfel", help="相機自動挑選模式: da3_surfel (第0幀DA3表面最大覆蓋) 或 fps (外參最遠點採樣)")
    p.add_argument("--fps_alpha", type=float, default=0.8, help="FPS 篩選權重: 3D 位置比例 (預設 0.8)")
    p.add_argument("--use_dynamic_vggt_cameras", action="store_true", help="使用動態 VGGT 預測的內外參")
    p.add_argument("--use_gt_cameras", action="store_true", help="使用 GT 的相機內外參，並利用 GT 縮放 VGGT 深度")
    p.add_argument("--use_da3", action="store_true", help="使用 Depth Anything 3 取代 VGGT")
    p.add_argument("--da3_chunk_size", type=int, default=1, help="DA3 跨時間處理的 chunk size (靜態相機視角建議設為 1)")
    p.add_argument("--da3_model", type=str, default="depth-anything/DA3-LARGE-1.1", help="DA3 模型版本")
    p.add_argument("--da3_layer_offload", action="store_true", default=None, help="強制啟用 DA3 動態分層推論")
    p.add_argument("--da3_process_res", type=int, default=672, help="DA3 推論解析度 (預設 672)")
    p.add_argument("--mask_img", type=str, default="mask.jpg", help="提供白色像素標註追蹤點的遮罩圖片路徑")
    p.add_argument("--mode", type=str, choices=["viz", "full", "both", "prove"], default="viz", help="運行模式")
    p.add_argument("--full_num_queries", type=int, default=200000, help="正式測試模式要追蹤的總點數量")
    p.add_argument("--rerun", choices=["save", "spawn", "stream"], default="save")
    p.add_argument("--lightweight", action="store_true", help="使用輕量級視覺化")
    p.add_argument("--rrd", default="n3d_mvtracker_feature_demo.rrd", help="輸出的 Rerun 檔名")
    p.add_argument("--start_frame", type=int, default=0, help="開始追蹤的幀數 (設定 t0)")

    # 新增 3D 空間特徵體素點雲串流相關參數
    p.add_argument("--voxel_size", type=float, default=0.1, help="體素尺寸 (預設 0.1)")
    p.add_argument("--voxel_mode", type=str, choices=["uniform", "dynamic", "surfel"], default="surfel", help="體素化模式: uniform (均勻), dynamic (動態足跡), surfel (多尺度表面基元融合 - 方案 B)")
    p.add_argument("--use_dynamic_voxel", action="store_true", default=None, help="啟用動態足跡體素化 (已整合至 --voxel_mode dynamic)")
    p.add_argument("--use_uniform_voxel", action="store_true", default=None, help="強制使用均勻體素化 (已整合至 --voxel_mode uniform)")
    p.add_argument("--surfel_normal_scale", type=float, default=5.0, help="Surfel 模式法向容差倍率 (預設 2.5)")
    p.add_argument("--surfel_angle_deg", type=float, default=25.0, help="Surfel 模式表面法向夾角門檻 (預設 25.0 度)")
    p.add_argument("--min_voxel_size", type=float, default=0.01, help="最小體素尺寸 (預設 0.01)")
    p.add_argument("--dynamic_voxel_scale", type=float, default=1.0, help="動態足跡縮放係數 (預設 1.0)")
    p.add_argument("--feature_weights_json", type=str, default=None, help="特徵權重與全域尺度 JSON 檔路徑 (預設依體素模式自動匹配)")
    p.add_argument("--w_0", type=float, default=None, help="覆寫 w0 (平滑底分)")
    p.add_argument("--w_sobel", type=float, default=None, help="覆寫 w1 (Sobel 邊界)")
    p.add_argument("--w_dct", type=float, default=None, help="覆寫 w2 (DCT 材質)")
    p.add_argument("--w_chroma", type=float, default=None, help="覆寫 w3 (Chroma 色彩)")
    p.add_argument("--feature_vis_mode", type=str, choices=["heatmap", "dominant", "both", "rgb", "all"], default="both", help="特徵點雲呈現模式 (heatmap, dominant, both, rgb, all)")
    p.add_argument("--no_voxel_rgb", action="store_true", help="停用體素化後的真實 RGB 點雲 (預設會記錄 sequence-0/n3d/spatial_voxel_rgb)")
    p.add_argument("--log_raw_pointcloud", action="store_true", help="是否記錄各視角未體素化合併前的原始 RGB 點雲 (在 Rerun 中依 sequence-0/n3d/point_cloud/view-v 獨立分層呈現)")
    p.add_argument("--raw_pointcloud_max_frames", type=int, default=None, help="限制原始各視角點雲記錄的最大幀數 (預設 None 即記錄所有幀，建議長影片設為 1 或 30 以免檔案過大)")
    p.add_argument("--no_importance_sampling", action="store_true", help="停用基於特徵機率分佈的 3D 重要度採樣點雲串流")
    p.add_argument("--importance_total_points", type=int, default=100000, help="全序列重要度採樣總點數預算 (預設 100,000，自動按幀數攤提)")
    p.add_argument("--importance_points_per_frame", type=int, default=None, help="每幀重要度採樣點數 (若指定則覆寫總點數預算)")
    p.add_argument("--importance_vis_style", type=str, choices=["highlight", "rgb", "feature"], default="highlight", help="重要度採樣點的視覺化著色樣式 (highlight: 金黃強調, rgb: 真實顏色, feature: 熱力圖色)")
    args = p.parse_args()

    np.random.seed(72)
    torch.manual_seed(72)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # 1. 讀取影像與相機姿態
    # ==========================================
    view_dirs = [f for f in glob.glob(os.path.join(args.dir, 'cam*')) if os.path.isdir(f)]
    if not view_dirs:
        view_dirs = [f for f in glob.glob(os.path.join(args.dir, 'view_*')) if os.path.isdir(f)]
    if not view_dirs:
        raise ValueError(f"在 {args.dir} 找不到任何 cam* 或 view_* 相機資料夾！")

    def parse_cam_idx(path):
        match = re.search(r'\d+', os.path.basename(path))
        return int(match.group()) if match else 0

    view_dirs.sort(key=parse_cam_idx)

    sample_frames = get_sorted_frame_paths(view_dirs[0])
    num_frames = min(len(sample_frames), args.max_frames)
    sample_img = Image.open(sample_frames[0])
    W_img, H_img = sample_img.size
    print(f"📷 處理圖片: {W_img}x{H_img} | 取樣幀數: {num_frames} 幀 | 相機視角數: {len(view_dirs)}")

    poses_path = os.path.join(args.dir, 'poses_bounds.npy')
    poses_bounds = np.load(poses_path)
    all_poses_llff = poses_bounds[:, :-2].reshape([-1, 3, 5])

    all_w2c, all_intrs = [], []
    for i in range(len(all_poses_llff)):
        w2c, K = llff_to_opencv_w2c(all_poses_llff[i], W_img, H_img)
        all_w2c.append(w2c)
        all_intrs.append(K)
    all_w2c_np = np.stack(all_w2c)
    all_intrs_np = np.stack(all_intrs)

    if args.selected_cams is not None:
        SELECTED_CAMS = sorted(args.selected_cams)
        print(f"🎯 使用手動指定的相機視角 Index: {SELECTED_CAMS}")
    elif args.cam_select_mode == "da3_surfel":
        SELECTED_CAMS = select_cams_by_da3_surfel(
            view_dirs=view_dirs,
            all_w2c_np=all_w2c_np,
            all_intrs_np=all_intrs_np,
            num_to_select=args.num_cams,
            voxel_size=0.2,
            exclude_indices=[0],
            da3_model=args.da3_model,
            process_res=args.da3_process_res,
            device=device,
        )
    else:
        SELECTED_CAMS = select_cams_by_fps(all_w2c_np, num_to_select=args.num_cams, alpha=args.fps_alpha, exclude_indices=[0])
        print(f"🎯 自動使用 FPS 挑選出 {len(SELECTED_CAMS)} 個訓練視角: {SELECTED_CAMS}")

    V = len(SELECTED_CAMS)

    TARGET_W, TARGET_H = 768, 576
    scale_x = TARGET_W / W_img
    scale_y = TARGET_H / H_img
    extrs_list, intrs_list = [], []
    for cam_idx in SELECTED_CAMS:
        extrs_list.append(torch.from_numpy(all_w2c_np[cam_idx]).float())
        K_original = all_intrs_np[cam_idx].copy()
        K_original[0, :] *= scale_x
        K_original[1, :] *= scale_y
        intrs_list.append(torch.from_numpy(K_original).float())

    extrs_gt = torch.stack(extrs_list).unsqueeze(1).repeat(1, num_frames, 1, 1)
    intrs_gt = torch.stack(intrs_list).unsqueeze(1).repeat(1, num_frames, 1, 1)

    # ==========================================
    # 2. 收集所有幀的 RGB 與深度
    # ==========================================
    print("🚀 讀取 RGB 影像並整理資料...")
    rgbs_list = []
    for cam_idx in tqdm(SELECTED_CAMS, desc="RGB 讀取進度"):
        frames = get_sorted_frame_paths(view_dirs[cam_idx])[:num_frames]
        cam_rgbs = []
        for t in range(num_frames):
            img = Image.open(frames[t]).convert('RGB')
            img_resized = img.resize((TARGET_W, TARGET_H), Image.Resampling.BILINEAR)
            cam_rgbs.append((to_tensor(img_resized) * 255).byte())
        rgbs_list.append(torch.stack(cam_rgbs))

    rgbs_tensor = torch.stack(rgbs_list)  # [V, T, 3, H, W]

    depths_raw_tensor = None
    if args.use_da3:
        print(f"🚀 提取 DA3 深度 (帶有 GT 位姿約束)...")
        with torch.no_grad():
            depths_tensor, _, intrs_model, extrs_model, *extra_da3 = _ensure_da3_aligned_cache_and_load(
                rgbs=rgbs_tensor,
                seq_name="n3d_da3_aligned",
                dataset_root=args.dir,
                extrs_gt=extrs_gt,
                intrs_gt=intrs_gt,
                skip_if_cached=True,
                model_id=args.da3_model,
                temporal_chunk_size=args.da3_chunk_size,
                enable_layer_offload=args.da3_layer_offload,
                process_res=args.da3_process_res,
            )
        if len(extra_da3) > 0 and extra_da3[0] is not None:
            depths_raw_tensor = extra_da3[0]
            print(f"✅ 取得 DA3 原始推理解析度深度圖: {list(depths_raw_tensor.shape)}")
        intrs_model = intrs_gt.clone()
        extrs_model = extrs_gt.clone()
    else:
        seq_name = "n3d_gt_init_4d" if args.use_gt_cameras else "n3d_gt_init_raw"
        print(f"🚀 提取 VGGT 深度 (模式: {seq_name})...")
        with torch.no_grad():
            depths_tensor, _, intrs_vggt, extrs_vggt = _ensure_vggt_aligned_cache_and_load(
                rgbs=rgbs_tensor,
                seq_name=seq_name,
                dataset_root=args.dir,
                extrs_gt=extrs_gt,
                vggt_cache_subdir="vggt_cache",
                skip_if_cached=True,
                model_id="facebook/VGGT-1B",
            )
        if args.use_gt_cameras:
            print("🚀 執行 SL(4) Global Homography Depth Alignment...")
            depths_tensor = align_depth_sparse_to_dense(rgbs_tensor, depths_tensor, intrs_vggt, extrs_vggt, intrs_gt, extrs_gt).cpu()
            intrs_model = intrs_gt.clone()
            extrs_model = extrs_gt.clone()
        else:
            intrs_model = intrs_vggt[:, 0:1].expand_as(intrs_vggt).clone()
            extrs_model = extrs_vggt[:, 0:1].expand_as(extrs_vggt).clone()

    # ==========================================
    # 3. 取樣 Query Points
    # ==========================================
    t0 = args.start_frame
    viz_pts_sampled = None
    full_pts_sampled = None
    len_viz = 0
    len_full = 0
    viz_pixel_coords = None
    full_pixel_coords = None

    if args.mode in ["viz", "both"]:
        print(f"🎯 [VIZ 模式] 從 t={t0} 幀深度圖取樣 {args.num_queries} 個追蹤點...")
        xyz_viz, _ = init_pointcloud_from_rgbd(
            fmaps=rgbs_tensor[0:1, t0:t0+1].unsqueeze(0).float(),
            depths=depths_tensor[0:1, t0:t0+1].unsqueeze(0),
            intrs=intrs_model[0:1, t0:t0+1].unsqueeze(0),
            extrs=extrs_model[0:1, t0:t0+1].unsqueeze(0),
            stride=1,
            level=0,
        )
        pts_viz = xyz_viz[0]

        if args.mask_img and os.path.exists(args.mask_img):
            print(f"🖼️ 讀取選點遮罩: {args.mask_img}")
            mask_pil = Image.open(args.mask_img).convert('L')
            mask_resized = mask_pil.resize((TARGET_W, TARGET_H), Image.Resampling.BILINEAR)
            mask_np = np.array(mask_resized)
            ys, xs = np.where(mask_np > 250)

            indices = []
            for y, x in zip(ys, xs):
                indices.append(y * TARGET_W + x)

            if len(indices) == 0:
                print("⚠️ 遮罩中未找到明顯的白色像素標註，將改回隨機取樣。")
                valid_mask = depths_tensor[0:1, t0, 0].flatten() > 0
                pool = pts_viz[valid_mask]
                viz_pts_sampled = pool[torch.randperm(pool.shape[0])[:args.num_queries]]
            else:
                indices = list(set(indices))
                pool = pts_viz[indices]
                if pool.shape[0] > args.num_queries:
                    rand_idx = torch.randperm(pool.shape[0])[:args.num_queries]
                    viz_pts_sampled = pool[rand_idx]
                    chosen_idx = torch.tensor(indices)[rand_idx]
                else:
                    viz_pts_sampled = pool
                    chosen_idx = torch.tensor(indices)
                print(f"✅ 成功從圖片中讀取 {len(viz_pts_sampled)} 個手動標註點！")

                viz_pixel_coords = torch.stack([
                    torch.zeros_like(chosen_idx),
                    chosen_idx // TARGET_W,
                    chosen_idx % TARGET_W
                ], dim=1)
        else:
            valid_mask = depths_tensor[0:1, t0, 0].flatten() > 0
            valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
            pool = pts_viz[valid_mask]
            assert pool.shape[0] > 0, "沒有找到有效的深度點來產生 Queries！"
            rand_idx = torch.randperm(pool.shape[0])[:args.num_queries]
            viz_pts_sampled = pool[rand_idx]
            chosen_idx = valid_indices[rand_idx]

            viz_pixel_coords = torch.stack([
                torch.zeros_like(chosen_idx),
                chosen_idx // TARGET_W,
                chosen_idx % TARGET_W
            ], dim=1)

    if args.mode in ["full", "both", "prove"]:
        print(f"🎯 [FULL 模式] 從 t={t0} 幀「全部相機」隨機取樣 {args.full_num_queries} 個追蹤點...")
        with torch.no_grad():
            xyz_full, rgb_full = init_pointcloud_from_rgbd(
                fmaps=rgbs_tensor[:, t0:t0+1].unsqueeze(0).float(),
                depths=depths_tensor[:, t0:t0+1].unsqueeze(0),
                intrs=intrs_model[:, t0:t0+1].unsqueeze(0),
                extrs=extrs_model[:, t0:t0+1].unsqueeze(0),
                stride=1,
                level=0,
            )
        pts_all = xyz_full[0]
        valid_mask = depths_tensor[:, t0, 0].flatten() > 0
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
        pts_pool = pts_all[valid_mask]
        rand_idx = torch.randperm(pts_pool.shape[0])[:args.full_num_queries]
        full_pts_sampled = pts_pool[rand_idx]
        chosen_indices = valid_indices[rand_idx]
        v_idx = chosen_indices // (TARGET_H * TARGET_W)
        rem = chosen_indices % (TARGET_H * TARGET_W)
        y_idx = rem // TARGET_W
        x_idx = rem % TARGET_W
        full_pixel_coords = torch.stack([v_idx, y_idx, x_idx], dim=1)

    if args.mode == "viz":
        pts_sampled = viz_pts_sampled
        pixel_coords = viz_pixel_coords
    elif args.mode in ["full", "prove"]:
        pts_sampled = full_pts_sampled
        pixel_coords = full_pixel_coords
    else:
        len_viz = len(viz_pts_sampled)
        len_full = len(full_pts_sampled)
        pts_sampled = torch.cat([viz_pts_sampled, full_pts_sampled], dim=0)
        pixel_coords = torch.cat([viz_pixel_coords, full_pixel_coords], dim=0)

    ts = torch.full((pts_sampled.shape[0], 1), float(t0), device=pts_sampled.device)
    query_points = torch.cat([ts, pts_sampled], dim=1).float()

    # ==========================================
    # 4. 載入並執行 MVTracker
    # ==========================================
    print("🧠 載入 MVTracker 並開始追蹤...")
    mvtracker = torch.hub.load(".", "mvtracker", source="local", pretrained=True, device=device)
    torch.set_float32_matmul_precision("high")
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

    start_time = time.time()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=amp_dtype):
        results = mvtracker(
            rgbs=rgbs_tensor[None].cpu(),
            depths=depths_tensor[None].cpu(),
            intrs=intrs_model[None].to(device),
            extrs=extrs_model[None].to(device),
            query_points_3d=query_points[None].to(device),
        )
    end_time = time.time()
    pred_tracks = results["traj_e"].cpu()
    pred_vis = results["vis_e"].cpu()
    print(f"✅ 追蹤完成！ (花費時間: {end_time - start_time:.2f} 秒)")

    # 釋放 MVTracker 模型與中間張量，節省 1.5~2.0 GB RAM 給 Rerun 串流
    del results, mvtracker
    torch.cuda.empty_cache()
    gc.collect()

    # ==========================================
    # 5. 視覺化 (匯出至 Rerun)
    # ==========================================
    if args.mode == "full":
        print("⏭️  [FULL 模式] 略過視覺化，測試結束。")
        return

    if args.mode == "both":
        idx_viz = torch.arange(len_viz)
        num_viz_from_full = min(args.num_queries, len_full)
        idx_full = torch.randperm(len_full)[:num_viz_from_full] + len_viz
        viz_indices = torch.cat([idx_viz, idx_full])
        viz_pred_tracks = pred_tracks[:, viz_indices]
        viz_pred_vis = pred_vis[:, viz_indices]
        viz_query_points = query_points[viz_indices]
        viz_final_pixel_coords = pixel_coords[viz_indices]
    else:
        viz_pred_tracks = pred_tracks
        viz_pred_vis = pred_vis
        viz_query_points = query_points
        viz_final_pixel_coords = pixel_coords

    print("🎨 正在打包 Rerun 視覺化檔案...")
    rr.init("n3d_tracking", recording_id="v0.16")
    if args.rerun == "save":
        # 🌟 核心關鍵：Rerun 官方規範需在 logging 前呼叫 rr.save(path) 以開啟 Disk Sink 直接串流至磁碟
        # 徹底避免在 RAM 中累積 300 幀點雲緩衝導致 OOM！
        rr.save(args.rrd)
    elif args.rerun == "stream":
        rr.connect_tcp()
    elif args.rerun == "spawn":
        rr.spawn()

    gt_centers_all = -extrs_model[:, 0, :3, :3].transpose(-1, -2) @ extrs_model[:, 0, :3, 3].unsqueeze(-1)
    scene_center = gt_centers_all.mean(dim=0).flatten().cpu().numpy()

    # 原版相機視錐 (停用未體素化的 220 萬點原始點雲，由後續 3D 特徵體素點雲全場景呈現)
    log_pointclouds_to_rerun(
        dataset_name="n3d",
        datapoint_idx=0,
        rgbs=rgbs_tensor[None],
        depths=depths_tensor[None],
        intrs=intrs_model[None],
        extrs=extrs_model[None],
        depths_conf=None,
        conf_thrs=[5.0],
        log_only_confident_pc=False,
        radii=-2.45,
        fps=30,
        bbox_crop=None,
        sphere_radius_crop=None,
        sphere_center_crop=scene_center,
        log_rgb_image=False,
        log_depthmap_as_image_v1=False,
        log_depthmap_as_image_v2=False,
        log_camera_frustrum=True,
        log_rgb_pointcloud=args.log_raw_pointcloud,
        raw_pointcloud_max_frames=args.raw_pointcloud_max_frames,
    )

    # 記錄全部候選相機姿態 (標示選取與未選取狀態)
    all_cam_centers = []
    all_cam_colors = []
    all_cam_labels = []
    for c_idx in range(len(all_w2c_np)):
        w2c = all_w2c_np[c_idx]
        w2c_4x4 = np.eye(4)
        w2c_4x4[:3, :4] = w2c
        c2w = np.linalg.inv(w2c_4x4)
        center = c2w[:3, 3]
        all_cam_centers.append(center)
        cam_name = os.path.basename(view_dirs[c_idx])
        if c_idx in SELECTED_CAMS:
            all_cam_colors.append([40, 220, 100])
            all_cam_labels.append(f"{cam_name} [Selected #{SELECTED_CAMS.index(c_idx)+1}]")
        else:
            all_cam_colors.append([140, 140, 150])
            all_cam_labels.append(f"{cam_name} [Candidate]")

    rr.log(
        "sequence-0/n3d/candidate_cameras",
        rr.Points3D(
            np.array(all_cam_centers),
            colors=np.array(all_cam_colors, dtype=np.uint8),
            labels=all_cam_labels,
            radii=0.06,
        ),
    )

    # 🌟【核心新功能】逐幀 3D 空間特徵體素點雲串流
    if args.voxel_mode is not None:
        voxel_mode = args.voxel_mode
    elif args.use_uniform_voxel:
        voxel_mode = "uniform"
    elif args.use_dynamic_voxel:
        voxel_mode = "dynamic"
    else:
        voxel_mode = "surfel"

    use_dynamic = (voxel_mode in ["dynamic", "surfel"])

    # 預設權重基底 (依照使用 dynamic 或 uniform 模式提供初始值)
    if use_dynamic:
        weights = {"w_0": 0.3229, "w_sobel": 0.1900, "w_dct": 0.2233, "w_chroma": 0.2639}
    else:
        weights = {"w_0": 0.1784, "w_sobel": 0.3949, "w_dct": 0.1140, "w_chroma": 0.3127}
    q_global = {"s_sobel": 0.2122, "s_dct": 1.9949, "s_chroma": 0.0485}

    weights_file = args.feature_weights_json
    if weights_file is None:
        dataset_name = os.path.basename(os.path.normpath(args.dir))
        sur_file = os.path.join(args.dir, f"weights_{dataset_name}_f0_surfel.json")
        dyn_file = os.path.join(args.dir, f"weights_{dataset_name}_f0_dynamic.json")
        uni_file = os.path.join(args.dir, f"weights_{dataset_name}_f0.json")
        fallback_sur = "datasets/sear_steak/weights_sear_steak_f0_surfel.json"
        fallback_dyn = "datasets/sear_steak/weights_sear_steak_f0_dynamic.json"
        fallback_uni = "datasets/sear_steak/weights_sear_steak_f0.json"
        if voxel_mode == "surfel":
            if os.path.isfile(sur_file): weights_file = sur_file
            elif os.path.isfile(fallback_sur): weights_file = fallback_sur
            elif os.path.isfile(dyn_file): weights_file = dyn_file
            elif os.path.isfile(fallback_dyn): weights_file = fallback_dyn
            elif os.path.isfile(fallback_uni): weights_file = fallback_uni
        elif use_dynamic:
            if os.path.isfile(dyn_file): weights_file = dyn_file
            elif os.path.isfile(fallback_dyn): weights_file = fallback_dyn
            elif os.path.isfile(fallback_uni): weights_file = fallback_uni
        else:
            if os.path.isfile(uni_file): weights_file = uni_file
            elif os.path.isfile(fallback_uni): weights_file = fallback_uni
            elif os.path.isfile(fallback_dyn): weights_file = fallback_dyn

    if weights_file and os.path.isfile(weights_file):
        print(f"📄 載入 3D 特徵權重與全域尺度配置: {weights_file}")
        with open(weights_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            if "optimal_weights" in cfg:
                weights.update(cfg["optimal_weights"])
            if "q_global" in cfg:
                q_global.update(cfg["q_global"])

    if args.w_0 is not None: weights["w_0"] = args.w_0
    if args.w_sobel is not None: weights["w_sobel"] = args.w_sobel
    if args.w_dct is not None: weights["w_dct"] = args.w_dct
    if args.w_chroma is not None: weights["w_chroma"] = args.w_chroma

    w_sum = sum(weights.values())
    for k in weights:
        weights[k] /= w_sum

    if voxel_mode == "surfel":
        mode_str = f"多尺度表面基元融合 (Surfel Point-to-Plane, scale={args.dynamic_voxel_scale}, min_s={args.min_voxel_size}, h_norm={args.surfel_normal_scale}x, ang={args.surfel_angle_deg}°)"
    elif voxel_mode == "dynamic":
        mode_str = f"動態足跡體素化 (Dynamic Footprint, scale={args.dynamic_voxel_scale}, min_s={args.min_voxel_size})"
    else:
        mode_str = f"均勻體素化 (Uniform voxel_size={args.voxel_size})"
    print(f"🔮 逐幀生成 3D 空間特徵體素點雲 [{mode_str}] (權重: w0={weights['w_0']:.3f}, sob={weights['w_sobel']:.3f}, dct={weights['w_dct']:.3f}, chr={weights['w_chroma']:.3f})...")
    if args.importance_points_per_frame is not None:
        importance_k = args.importance_points_per_frame
    else:
        importance_k = max(1, args.importance_total_points // max(1, num_frames))
    if not args.no_importance_sampling:
        print(f"🎯 啟用 3D 空間特徵重要度採樣 (Importance Sampling): 每幀採樣 {importance_k:,} 點 (預算模式, 樣式: {args.importance_vis_style})")

    feature_scorer = RGBTextureScorer(normalize=False)

    for t in tqdm(range(num_frames), desc="3D 特徵點雲串流"):
        rr.set_time("frame", duration=t / 30.0)

        # 提取當前幀多視角 2D 特徵
        rgbs_t = rgbs_tensor[:, t].float() / 255.0  # [V, 3, H, W]
        with torch.no_grad():
            f_res = feature_scorer(rgbs_t)
            s_sob = f_res['s_sobel'].squeeze(1).cpu()
            s_dct = f_res['s_dct'].squeeze(1).cpu()
            s_chr = f_res['s_chroma'].squeeze(1).cpu()

        # 應用跨視角全域尺度基準 Q_global (防禦白牆噪點放大)
        sob_scaled = torch.clamp(s_sob / max(q_global["s_sobel"], 1e-4), 0.0, 1.0).numpy()
        dct_scaled = torch.clamp(s_dct / max(q_global["s_dct"], 1e-4), 0.0, 1.0).numpy()
        chr_scaled = torch.clamp(s_chr / max(q_global["s_chroma"], 1e-4), 0.0, 1.0).numpy()

        # 若為 surfel 模式，在 GPU 上批次計算各視角相機表面單位法向量 (OpenCV LINEMOD 最小差分標準)
        if voxel_mode == "surfel":
            with torch.no_grad():
                if depths_raw_tensor is not None:
                    # 使用 DA3 原始無階梯噪點的連續深度圖計算法向量，再雙線性平滑插值回原始影像解析度
                    d_raw = depths_raw_tensor[:, t, 0:1].to(device)  # [V, 1, H_raw, W_raw]
                    _, _, H_raw, W_raw = d_raw.shape

                    intrs_raw = intrs_model[:, t].clone().to(device)
                    intrs_raw[:, 0, :] *= (float(W_raw) / float(TARGET_W))
                    intrs_raw[:, 1, :] *= (float(H_raw) / float(TARGET_H))

                    normals_raw, _ = depth_to_surface_normals(
                        d_raw,
                        intrinsics=intrs_raw,
                        method="min_gradient"
                    )
                    normals_torch = F.interpolate(
                        normals_raw,
                        size=(TARGET_H, TARGET_W),
                        mode="bilinear",
                        align_corners=False
                    )
                    normals_torch = F.normalize(normals_torch, dim=1, eps=1e-6)
                else:
                    normals_torch, _ = depth_to_surface_normals(
                        depths_tensor[:, t, 0:1].to(device),
                        intrinsics=intrs_model[:, t].to(device),
                        method="min_gradient"
                    )
                # normals_torch: [V, 3, H, W] -> [V, H, W, 3]
                normals_np = normals_torch.permute(0, 2, 3, 1).cpu().numpy()
        else:
            normals_np = None

        # 多視角 3D 反投影
        pts_t, cols_t, feats_t, fps_t, norms_t = [], [], [], [], []
        for v in range(V):
            rgb_np = rgbs_tensor[v, t].permute(1, 2, 0).numpy()
            d_np = depths_tensor[v, t, 0].numpy()
            feat_map = np.stack([sob_scaled[v], dct_scaled[v], chr_scaled[v]], axis=-1)
            K_v = intrs_model[v, t].numpy()
            w2c_v = extrs_model[v, t].numpy()
            c2w_v = np.linalg.inv(np.vstack([w2c_v, [0, 0, 0, 1]]))
            norm_c_v = normals_np[v] if normals_np is not None else None

            res_unproject = unproject_camera_points_with_features(
                rgb=rgb_np,
                depth=d_np,
                features=feat_map,
                K=K_v,
                c2w=c2w_v,
                depth_min=0.1,
                depth_max=150.0,
                normals=norm_c_v,
                return_normals=(voxel_mode == "surfel"),
            )
            if voxel_mode == "surfel":
                pts_w, cols, feats, footprints, norm_w = res_unproject
                norms_t.append(norm_w)
            else:
                pts_w, cols, feats, footprints = res_unproject

            pts_t.append(pts_w)
            cols_t.append(cols)
            feats_t.append(feats)
            fps_t.append(footprints)

        pts_cat = np.concatenate(pts_t, axis=0)
        cols_cat = np.concatenate(cols_t, axis=0)
        feats_cat = np.concatenate(feats_t, axis=0)
        fps_cat = np.concatenate(fps_t, axis=0) if use_dynamic else None
        norms_cat = np.concatenate(norms_t, axis=0) if voxel_mode == "surfel" else None

        # 體素化特徵 Max-Pooling (支援 uniform, dynamic, surfel)
        v_pts, v_cols, v_feats, _ = voxel_downsample_unified_with_features(
            pts=pts_cat,
            colors=cols_cat,
            features=feats_cat,
            normals=norms_cat,
            voxel_size=args.voxel_size,
            footprints=fps_cat,
            min_voxel_size=args.min_voxel_size,
            scale=args.dynamic_voxel_scale,
            mode=voxel_mode,
            surfel_normal_scale=args.surfel_normal_scale,
            surfel_angle_deg=args.surfel_angle_deg,
        )

        M = len(v_pts)
        if M == 0:
            continue

        s_sobel_v = v_feats[:, 0]
        s_dct_v = v_feats[:, 1]
        s_chroma_v = v_feats[:, 2]

        # 空間線性加權: S_spatial = w0* + w1*S_sobel + w2*S_dct + w3*S_chroma
        s_spatial = (
            weights["w_0"] +
            weights["w_sobel"] * s_sobel_v +
            weights["w_dct"] * s_dct_v +
            weights["w_chroma"] * s_chroma_v
        )

        # 模式 1: Turbo 熱力圖色彩映射 (99% 分位數動態截斷)
        q99_score = float(np.percentile(s_spatial, 99)) if len(s_spatial) > 0 else 1.0
        score_norm = np.clip(s_spatial / max(q99_score, 1e-4), 0.0, 1.0)
        turbo_colors = (plt.cm.turbo(score_norm)[:, :3] * 255).astype(np.uint8)

        # 模式 2: 機率空間優勢特徵分類地圖 (Dominant Feature Map)
        p_base_v = 1.0 / M
        p_sob_v = s_sobel_v / max(np.sum(s_sobel_v), 1e-12)
        p_dct_v = s_dct_v / max(np.sum(s_dct_v), 1e-12)
        p_chr_v = s_chroma_v / max(np.sum(s_chroma_v), 1e-12)

        weighted_comps = np.stack([
            np.full(M, weights["w_0"] * p_base_v),
            weights["w_sobel"] * p_sob_v,
            weights["w_dct"] * p_dct_v,
            weights["w_chroma"] * p_chr_v,
        ], axis=-1)

        dominant_idx = np.argmax(weighted_comps, axis=-1)
        palette = np.array([
            [170, 175, 185],  # 0: Base 平滑 (灰藍)
            [235, 45, 45],    # 1: Sobel 邊界 (鮮紅)
            [40, 200, 75],    # 2: DCT 材質 (亮綠)
            [25, 180, 240],   # 3: Chroma 色彩 (青藍)
        ], dtype=np.uint8)
        dominant_colors = palette[dominant_idx]

        radii_val = -2.45
        # 體素化真實 RGB 點雲 (融合後之均勻/動態體素表面)
        if (not args.no_voxel_rgb and args.feature_vis_mode != "dominant" and args.feature_vis_mode != "heatmap") or args.feature_vis_mode in ["rgb", "all"]:
            v_cols_u8 = (np.clip(v_cols, 0.0, 1.0) * 255).astype(np.uint8) if v_cols.max() <= 1.0 else v_cols.astype(np.uint8)
            rr.log(
                "sequence-0/n3d/spatial_voxel_rgb",
                rr.Points3D(v_pts, colors=v_cols_u8, radii=radii_val),
            )
        if args.feature_vis_mode in ["heatmap", "both", "all"]:
            rr.log(
                "sequence-0/n3d/spatial_feature_heatmap",
                rr.Points3D(v_pts, colors=turbo_colors, radii=radii_val),
            )
        if args.feature_vis_mode in ["dominant", "both", "all"]:
            rr.log(
                "sequence-0/n3d/spatial_feature_dominant",
                rr.Points3D(v_pts, colors=dominant_colors, radii=radii_val),
            )

        # 模式 3: 3D 空間特徵重要度採樣點雲 (Importance Sampled Points)
        if not args.no_importance_sampling and M > 0:
            # 綜合採樣機率分佈: p_k = w0*p_base + w1*p_sob + w2*p_dct + w3*p_chr
            p_k = (
                weights["w_0"] * p_base_v +
                weights["w_sobel"] * p_sob_v +
                weights["w_dct"] * p_dct_v +
                weights["w_chroma"] * p_chr_v
            )
            p_sum = p_k.sum()
            if p_sum > 0:
                p_k /= p_sum
            else:
                p_k = np.full(M, 1.0 / M, dtype=np.float32)

            k_sample = min(importance_k, M)
            p_torch = torch.from_numpy(p_k).float()
            sampled_indices = torch.multinomial(p_torch, num_samples=k_sample, replacement=False).numpy()

            sampled_pts = v_pts[sampled_indices]
            if args.importance_vis_style == "highlight":
                # 金黃色醒目強調 (Gold: [255, 215, 0])
                sampled_cols = np.tile(np.array([255, 215, 0], dtype=np.uint8), (k_sample, 1))
            elif args.importance_vis_style == "rgb":
                v_c = (np.clip(v_cols[sampled_indices], 0.0, 1.0) * 255).astype(np.uint8) if v_cols.max() <= 1.0 else v_cols[sampled_indices].astype(np.uint8)
                sampled_cols = v_c
            elif args.importance_vis_style == "feature":
                sampled_cols = turbo_colors[sampled_indices]
            else:
                sampled_cols = np.tile(np.array([255, 215, 0], dtype=np.uint8), (k_sample, 1))

            # 採樣點半徑設為 -3.5 (直徑比普通體素點大，方便清晰辨識聚集分佈)
            rr.log(
                "sequence-0/n3d/importance_sampled_points",
                rr.Points3D(sampled_pts, colors=sampled_cols, radii=-3.5),
            )

    # 記錄相機與軌跡
    if args.use_gt_cameras and not args.use_da3:
        for v in range(V):
            for t in range(num_frames):
                rr.set_time("frame", duration=t / 30.0)
                w2c = extrs_vggt[v, t].cpu().numpy()
                K = intrs_vggt[v, t].cpu().numpy()
                w2c_4x4 = np.eye(4)
                w2c_4x4[:3, :4] = w2c
                c2w_4x4 = np.linalg.inv(w2c_4x4)
                rot = c2w_4x4[:3, :3]
                trans = c2w_4x4[:3, 3]
                U, _, Vh = np.linalg.svd(rot)
                rot_ortho = U @ Vh
                if np.linalg.det(rot_ortho) < 0:
                    U[:, -1] *= -1
                    rot_ortho = U @ Vh
                rr.log(f"sequence-0/n3d/image_vggt/view-{v}",
                       rr.Pinhole(image_from_camera=K, width=TARGET_W, height=TARGET_H))
                rr.log(f"sequence-0/n3d/image_vggt/view-{v}",
                       rr.Transform3D(translation=trans, mat3x3=rot_ortho))

    log_tracks_to_rerun(
        dataset_name="n3d",
        datapoint_idx=0,
        predictor_name="MVTracker",
        gt_trajectories_3d_worldspace=None,
        gt_visibilities_any_view=None,
        query_points_3d=viz_query_points[None],
        pred_trajectories=viz_pred_tracks,
        pred_visibilities=viz_pred_vis,
        per_track_results=None,
        radii_scale=1.0,
        fps=30,
        sphere_radius_crop=None,
        sphere_center_crop=scene_center,
        log_per_interval_results=False,
        max_tracks_to_log=100 if args.lightweight else None,
        track_batch_size=50,
        method_id=None,
        color_per_method_id=None,
        memory_lightweight_logging=args.lightweight,
    )

    if args.rerun == "save":
        print(f"\n🎉 成功串流包含 3D 特徵體素點雲的 Rerun 檔案至: {os.path.abspath(args.rrd)}")

    # ==========================================
    # 6. 漂移驗證 (Drift Analysis)
    # ==========================================
    print("🔍 進行漂移驗證 (Drift Analysis)...")
    cam_idx = viz_final_pixel_coords[:, 0]
    y = viz_final_pixel_coords[:, 1]
    x = viz_final_pixel_coords[:, 2]

    T_frames = depths_tensor.shape[1]
    N_pts = len(viz_final_pixel_coords)

    vggt_tracks = torch.zeros((T_frames, N_pts, 3), device=device)
    depths_tensor_dev = depths_tensor.to(device)
    intrs_model_dev = intrs_model.to(device)
    extrs_model_dev = extrs_model.to(device)

    pixel_homo = torch.stack([x.float(), y.float(), torch.ones_like(x).float()], dim=-1).to(device)

    for t in range(T_frames):
        d_t = depths_tensor_dev[cam_idx, t, 0, y, x]
        K_t = intrs_model_dev[cam_idx, t]
        E_t = extrs_model_dev[cam_idx, t]

        K_inv = torch.inverse(K_t)
        cam_ray = torch.bmm(K_inv, pixel_homo.unsqueeze(-1)).squeeze(-1)
        cam_xyz = cam_ray * d_t.unsqueeze(-1)

        R = E_t[:, :3, :3]
        Tr = E_t[:, :3, 3]

        X_c_minus_Tr = cam_xyz - Tr
        R_inv = R.transpose(1, 2)
        world_xyz = torch.bmm(R_inv, X_c_minus_Tr.unsqueeze(-1)).squeeze(-1)
        vggt_tracks[t] = world_xyz

    pred_tracks_dev = viz_pred_tracks.to(device)
    if pred_tracks_dev.dim() == 4:
        pred_tracks_dev = pred_tracks_dev[0]

    mv_drift = torch.norm(pred_tracks_dev - pred_tracks_dev[t0:t0+1], dim=-1)
    vggt_drift = torch.norm(vggt_tracks - vggt_tracks[t0:t0+1], dim=-1)

    mv_drift_mean = mv_drift.mean(dim=1).cpu().numpy()
    vggt_drift_mean = vggt_drift.mean(dim=1).cpu().numpy()

    plt.figure(figsize=(12, 6))
    plt.plot(mv_drift_mean, label='MVTracker Mean Drift', color='blue', linewidth=2)
    plt.plot(vggt_drift_mean, label='Depth Backproj Mean Drift', color='orange', linewidth=2, linestyle='--')
    plt.xlabel('Frame')
    plt.ylabel(f'Mean Displacement from t={t0} (World Units)')
    plt.title(f'Drift Analysis Over Time (Static Scene Assumption) - Start t={t0}\nMode: {args.mode}, Points: {N_pts}')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(args.dir, f"drift_analysis_demo_{args.mode}.png")
    plt.savefig(plot_path)
    plt.close()

    threshold = 0.5
    mv_bad_pct = (mv_drift[-1] > threshold).float().mean().item() * 100
    vggt_bad_pct = (vggt_drift[-1] > threshold).float().mean().item() * 100

    print(f"📊 漂移分析圖表已儲存至 {plot_path}")
    print(f"   - MVTracker 最後一幀平均漂移: {mv_drift_mean[-1]:.4f} 單位")
    print(f"   - Depth Backproj 最後一幀平均漂移: {vggt_drift_mean[-1]:.4f} 單位")
    print(f"   - MVTracker > {threshold} 單位移動的點比例: {mv_bad_pct:.2f}%")
    print(f"   - Depth Backproj > {threshold} 單位移動的點比例: {vggt_bad_pct:.2f}%")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*DtypeTensor constructors are no longer.*", module="pointops.query")
    warnings.filterwarnings("ignore", message=".*Plan failed with a cudnnException.*", module="torch.nn.modules.conv")
    main()
