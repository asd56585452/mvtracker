import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import re
import time
import glob
import gc

import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
from plyfile import PlyData, PlyElement

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
from mvtracker.utils.camera_selection import (
    select_cams_by_fps,
    select_cams_by_da3_surfel,
)
from sklearn.cluster import MiniBatchKMeans
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

def get_sorted_frame_paths(cam_dir):
    frames = glob.glob(os.path.join(cam_dir, '*.png')) + glob.glob(os.path.join(cam_dir, '*.jpg'))
    def parse_frame_idx(path):
        basename = os.path.splitext(os.path.basename(path))[0]
        match = re.search(r'\d+', basename)
        return int(match.group()) if match else 0
    frames.sort(key=parse_frame_idx)
    return frames

def reorder_by_3d_grid_round_robin(pts, max_points):
    """
    根據目標點數自動決定網格大小，並進行 3D 均勻輪詢排序。
    """
    if len(pts) <= 1:
        return np.arange(len(pts))
        
    # 自動計算合適的 grid_size：目標點數乘以 2 (預留空氣網格的空間) 後開立方根
    grid_size = max(5, int(np.ceil((max_points * 2) ** (1/3))))
    # 如果 max_points 是 512，這裡算出來的 grid_size 會是 10
    
    # 1. 計算 3D 空間的邊界
    min_bound = np.min(pts, axis=0)
    max_bound = np.max(pts, axis=0)
    
    # 2. 計算每個 Voxel (體素) 的大小
    grid_dims = np.array([grid_size, grid_size, grid_size])
    voxel_size = (max_bound - min_bound) / grid_dims
    voxel_size = np.maximum(voxel_size, 1e-8) # 避免除以零
    
    # 3. 將點映射到 3D 網格索引
    voxel_indices = np.floor((pts - min_bound) / voxel_size).astype(np.int32)
    voxel_indices = np.clip(voxel_indices, 0, grid_size - 1)
    
    # 產生 1D Hash 鍵
    voxel_keys = voxel_indices[:, 0] * (grid_size**2) + voxel_indices[:, 1] * grid_size + voxel_indices[:, 2]
    
    # 4. 根據 Hash 鍵將點分組
    sort_idx = np.argsort(voxel_keys)
    sorted_keys = voxel_keys[sort_idx]
    
    _, unique_indices, counts = np.unique(sorted_keys, return_index=True, return_counts=True)
    voxels_points = [sort_idx[start:start+count].tolist() for start, count in zip(unique_indices, counts)]
    
    # 打亂每個 Voxel 內部的點，增加局部隨機性
    active_voxels = [vp for vp in voxels_points if len(vp) > 0]
    for vp in active_voxels:
        np.random.shuffle(vp)
        
    # 5. 輪詢提取 (Round-Robin)
    reordered_indices = []
    while len(active_voxels) > 0:
        next_active_voxels = []
        for vp in active_voxels:
            reordered_indices.append(vp.pop())
            # 如果這個 Voxel 裡面還有點，留到下一輪繼續抽
            if len(vp) > 0:
                next_active_voxels.append(vp)
        active_voxels = next_active_voxels
        
    return np.array(reordered_indices)

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

def voxel_downsample_unified(
    pts,
    colors,
    is_dynamic,
    voxel_size,
    footprints=None,
    min_voxel_size=0.01,
    scale=1.0,
    mode="dynamic",
    normals=None,
    surfel_normal_scale=5.0,
    surfel_angle_deg=25.0,
):
    """
    統一的 Voxel 融合下採樣函數 (委託至 spatial_voxelizer.voxel_downsample_unified_with_features)
    """
    effective_mode = "uniform" if footprints is None and mode == "uniform" else mode
    res_pts, res_cols, _, res_dyn = voxel_downsample_unified_with_features(
        pts=pts,
        colors=colors,
        features=None,
        is_dynamic=is_dynamic,
        normals=normals,
        voxel_size=voxel_size,
        footprints=footprints,
        min_voxel_size=min_voxel_size,
        scale=scale,
        mode=effective_mode,
        surfel_normal_scale=surfel_normal_scale,
        surfel_angle_deg=surfel_angle_deg,
    )
    if isinstance(res_pts, torch.Tensor):
        res_pts = res_pts.cpu().numpy()
        res_cols = res_cols.cpu().numpy()
        if res_dyn is not None:
            res_dyn = res_dyn.cpu().numpy()
    return res_pts, res_cols, res_dyn

def select_cams_by_fps(all_w2c_np, num_to_select, alpha=0.8, exclude_indices=[0]):
    """
    使用 FPS (最遠點採樣) 根據 3D 位置與視線方向篩選訓練視角
    - exclude_indices: 排除的視角 (預設排除 index 0，即 cam00 / view_0 測試視角)
    - alpha: 評分權重 (0.8 代表 80% 位置距離 + 20% 角度差距)
    """
    num_total = len(all_w2c_np)
    valid_candidates = [i for i in range(num_total) if i not in exclude_indices]
    
    if num_to_select >= len(valid_candidates):
        print(f"⚠️ 要求的相機數 ({num_to_select}) 大於等於可用訓練視角數 ({len(valid_candidates)})，回傳所有可用訓練視角。")
        return sorted(valid_candidates)

    centers, dirs = [], []
    for w2c in all_w2c_np:
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        c = -R.T @ t                 # 世界座標下的相機中心
        d = R.T @ np.array([0, 0, 1]) # 相機 Look-at 方向
        centers.append(c)
        dirs.append(d / (np.linalg.norm(d) + 1e-8))

    centers = np.stack(centers)
    dirs = np.stack(dirs)

    max_pos_dist = np.max([np.linalg.norm(centers[i] - centers[j]) for i in valid_candidates for j in valid_candidates])
    max_pos_dist = max(max_pos_dist, 1e-6)

    selected_indices = [valid_candidates[0]]
    
    for _ in range(1, num_to_select):
        combined_min_dists = []
        for i in range(num_total):
            if i in selected_indices or i in exclude_indices:
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

    full_start_time = time.time()
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=str, required=True, help="n3d 資料集路徑")
    p.add_argument("--max_frames", type=int, default=300, help="限制載入的最大幀數 (避免 MVTracker OOM)")
    p.add_argument("--selected_cams", type=int, nargs="+", default=None, help="手動選擇的相機 ID 列表 (若未提供，則使用演算法自動挑選)")
    p.add_argument("--num_cams", type=int, default=5, help="當未指定 --selected_cams 時，自動挑選的訓練相機數量")
    p.add_argument("--cam_select_mode", type=str, choices=["da3_surfel", "fps"], default="da3_surfel", help="相機自動挑選模式: da3_surfel (基於第0幀DA3表面最大覆蓋) 或 fps (外參最遠點採樣)")
    p.add_argument("--fps_alpha", type=float, default=0.8, help="FPS 篩選權重: 3D 位置比例 (預設 0.8，即 80%% 位置 + 20%% 視線方向)")
    p.add_argument("--track_chunk_size", type=int, default=4096, help="MVTracker 追蹤時的分批大小，避免 OOM")
    p.add_argument("--use_dynamic_vggt_cameras", action="store_true", help="使用動態 VGGT 預測的內外參；若不加此參數，則將第1幀的內外參套用於所有後續幀")
    p.add_argument("--use_gt_cameras", action="store_true", help="使用 GT 的相機內外參，並利用 GT 縮放 VGGT 深度")
    p.add_argument("--use_da3", action="store_true", help="使用 Depth Anything 3 取代 VGGT 來預測全局一致性深度")
    p.add_argument("--export_per_frame_ply", action="store_true", help="是否將每一幀的軌跡儲存為獨立的 PLY 檔以供檢查")
    p.add_argument("--segment_frames", type=int, default=50, help="每個追蹤片段的長度 (幀數)")
    p.add_argument("--voxel_size", type=float, default=0.1, help="體素最大尺寸 (預設 0.1)")
    p.add_argument("--voxel_mode", type=str, choices=["uniform", "dynamic", "surfel"], default="surfel", help="體素化模式: uniform (均勻), dynamic (動態足跡), surfel (多尺度表面基元融合 - 方案 B)")
    p.add_argument("--use_dynamic_voxel", action="store_true", default=None, help="啟用動態足跡體素化 (已整合至 --voxel_mode dynamic)")
    p.add_argument("--min_voxel_size", type=float, default=0.01, help="最小體素大小 (預設 0.01)")
    p.add_argument("--dynamic_voxel_scale", type=float, default=1.0, help="投影 Footprint 乘上的 scale 係數，用來調節動態 Voxel 大小 (預設 1.0)")
    p.add_argument("--surfel_normal_scale", type=float, default=5.0, help="Surfel 模式法向容差倍率 (預設 5.0)")
    p.add_argument("--surfel_angle_deg", type=float, default=25.0, help="Surfel 模式表面法向夾角門檻 (預設 25.0 度)")
    p.add_argument("--feature_weights_json", type=str, default=None, help="特徵權重與全域尺度 JSON 檔路徑 (預設依體素模式自動匹配)")
    p.add_argument("--w_0", type=float, default=None, help="覆寫 w0 (平滑底分)")
    p.add_argument("--w_sobel", type=float, default=None, help="覆寫 w1 (Sobel 邊界)")
    p.add_argument("--w_dct", type=float, default=None, help="覆寫 w2 (DCT 材質)")
    p.add_argument("--w_chroma", type=float, default=None, help="覆寫 w3 (Chroma 色彩)")
    p.add_argument("--export_all_frame_ply", action="store_true", help="將當前模型每幀重建結果合併成完整的4D點雲供比對")
    p.add_argument("--export_vggt_all_frame_ply", action="store_true", help="[相容舊參數] 將模型每幀重建結果合併成完整的4D點雲供比對")
    p.add_argument("--export_importance_all_frame_ply", action="store_true", help="輸出基於 RGB 特徵分數重要度採樣後的獨立 4D 點雲 PLY")
    p.add_argument("--importance_total_points", type=int, default=100000, help="全序列重要度採樣總點數預算 (預設 100,000，自動按幀數攤提)")
    p.add_argument("--importance_points_per_frame", type=int, default=None, help="每幀重要度採樣點數 (若指定則覆寫總點數預算)")
    p.add_argument("--importance_vis_style", type=str, choices=["rgb", "highlight", "feature"], default="rgb", help="重要度採樣點的顏色樣式 (rgb: 真實顏色, highlight: 金黃強調, feature: 熱力圖色)")
    p.add_argument("--use_raft_mask", action="store_true", help="使用 RAFT 光流來過濾靜態背景點")
    p.add_argument("--raft_threshold", type=float, default=1.0, help="RAFT 判定為移動的像素閾值")
    p.add_argument("--query_sort_mode", type=str, default="round_robin", help="query 排序方式: round_robin 或 kmeans")
    p.add_argument("--da3_chunk_size", type=int, default=1, help="DA3 跨時間處理的 chunk size (靜態相機視角建議設為 1)")
    p.add_argument("--da3_model", type=str, default="depth-anything/DA3-LARGE-1.1", help="DA3 模型版本 (如 depth-anything/DA3-GIANT-1.1 或 depth-anything/DA3-LARGE-1.1)")
    p.add_argument("--da3_layer_offload", action="store_true", default=None, help="強制啟用 DA3 動態分層推論 (未指定時若為 GIANT 且顯存<=16GB 則自動開啟)")
    p.add_argument("--da3_process_res", type=int, default=672, help="DA3 推論解析度 (預設 672)")
    p.add_argument("--skip_mvtracker", action="store_true", help="跳過 MVTracker 載入與追蹤，僅進行資料準備與點雲匯出")
    args = p.parse_args()

    np.random.seed(72)
    torch.manual_seed(72)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # 決定體素化模式
    if args.use_dynamic_voxel:
        voxel_mode = "dynamic"
    elif args.voxel_mode is not None:
        voxel_mode = args.voxel_mode
    else:
        voxel_mode = "surfel"
    use_dynamic = (voxel_mode in ["dynamic", "surfel"])

    t_data_start = time.time()
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

    # 儲存所有相機供 4DGS 使用
    npz_path = os.path.join(args.dir, "estimated_cameras_vggt.npz")
    np.savez(npz_path, w2c_poses=all_w2c_np, intrs=all_intrs_np)
    print(f"✅ 相機參數已儲存至 {npz_path}")

    if args.selected_cams is not None:
        SELECTED_CAMS = args.selected_cams
        print(f"🎯 使用手動指定的相機視角 Index: {SELECTED_CAMS}")
    elif args.cam_select_mode == "da3_surfel":
        SELECTED_CAMS = select_cams_by_da3_surfel(
            view_dirs=view_dirs,
            all_w2c_np=all_w2c_np,
            all_intrs_np=all_intrs_np,
            num_to_select=args.num_cams,
            voxel_size=0.2,
            exclude_indices=[0],
            da3_model=args.da3_model if args.use_da3 else "depth-anything/DA3-LARGE-1.1",
            process_res=args.da3_process_res if args.use_da3 else 672,
            device=device,
        )
    else:
        # 自動使用 FPS 在 3D 空間中挑選 (預設排除 view_0 測試視角)
        SELECTED_CAMS = select_cams_by_fps(all_w2c_np, num_to_select=args.num_cams, alpha=args.fps_alpha, exclude_indices=[0])
        print(f"🎯 自動使用 FPS (80% 位置 + 20% 角度) 挑選出最均勻的 {len(SELECTED_CAMS)} 個訓練視角 (已排除 view_0 測試視角): {SELECTED_CAMS}")

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
    # 2. 收集所有幀的 RGB 與 VGGT 深度
    # ==========================================
    print("🚀 讀取 RGB 影像並整理資料...")
    rgbs_list = []
    for cam_idx in tqdm(SELECTED_CAMS, desc="RGB 讀取進度"):
        frames = get_sorted_frame_paths(view_dirs[cam_idx])[:num_frames]
        cam_rgbs = []
        for t in range(num_frames):
            img = Image.open(frames[t]).convert('RGB')
            img_resized = img.resize((TARGET_W, TARGET_H), Image.Resampling.BILINEAR)
            cam_rgbs.append((to_tensor(img_resized) * 255).byte()) # [3, H, W] uint8
        rgbs_list.append(torch.stack(cam_rgbs)) # [T, 3, H, W]
    
    rgbs_tensor = torch.stack(rgbs_list) # [V, T, 3, H, W]
    t_data_prep = time.time() - t_data_start

    t_depth_start = time.time()
    if args.use_da3:
        # ==========================================
        # [路徑 A] 使用 Depth Anything 3 (DA3)
        # ==========================================
        print(f"🚀 提取 DA3 深度 (帶有 GT 位姿約束)...")
        with torch.no_grad():
            depths_tensor, _, intrs_model, extrs_model, *extra = _ensure_da3_aligned_cache_and_load(
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
        depths_raw_tensor = extra[0] if (len(extra) > 0 and extra[0] is not None) else None
        if depths_raw_tensor is not None:
            print(f"✅ 取得 DA3 原始推理解析度深度圖: {list(depths_raw_tensor.shape)}")

        # 對齊後，相機模型即為 GT
        intrs_model = intrs_gt.clone()
        extrs_model = extrs_gt.clone()
    else:
        # ==========================================
        # [路徑 B] 使用原始的 VGGT
        # ==========================================
        depths_raw_tensor = None
        if args.use_gt_cameras:
            seq_name = "n3d_gt_init_4d"
        elif args.use_dynamic_vggt_cameras:
            seq_name = "n3d_gt_init_aligned"
        else:
            seq_name = "n3d_gt_init_raw"
            
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
            # 透過 SL(4) Global Homography 對齊 VGGT 深度與 GT 空間
            print("🚀 執行 SL(4) Global Homography Depth Alignment...")
            depths_tensor = align_depth_sparse_to_dense(rgbs_tensor, depths_tensor, intrs_vggt, extrs_vggt, intrs_gt, extrs_gt).cpu()
            intrs_model = intrs_gt.clone()
            extrs_model = extrs_gt.clone()
        elif args.use_dynamic_vggt_cameras:
            intrs_model = intrs_vggt
            extrs_model = extrs_vggt
        else:
            # 套用第1幀內外參至所有後續幀
            intrs_model = intrs_vggt[:, 0:1].expand_as(intrs_vggt).clone()
            extrs_model = extrs_vggt[:, 0:1].expand_as(extrs_vggt).clone()

    t_depth_infer = time.time() - t_depth_start
    if torch.cuda.is_available():
        vram_after_depth = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"✅ 深度幾何推論完成！耗時: {t_depth_infer:.2f} 秒 | 峰值顯存: {vram_after_depth:.2f} GB")
    else:
        print(f"✅ 深度幾何推論完成！耗時: {t_depth_infer:.2f} 秒")

    # 決定動態模型標籤
    if args.use_da3:
        da3_m = args.da3_model.lower()
        if "giant" in da3_m:
            model_tag = "da3_giant"
        elif "large" in da3_m:
            model_tag = "da3_large"
        elif "base" in da3_m:
            model_tag = "da3_base"
        elif "small" in da3_m:
            model_tag = "da3_small"
        else:
            model_tag = "da3"
    else:
        model_tag = "vggt"

    # 載入 3DGS 特徵加權權重與全域尺度 Q_global
    feat_weights = {"w_0": 0.2409, "w_sobel": 0.2945, "w_dct": 0.2022, "w_chroma": 0.2624}
    q_global = {"s_sobel": 0.2147, "s_dct": 2.2048, "s_chroma": 0.0482}

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
        elif voxel_mode == "dynamic":
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
                feat_weights.update(cfg["optimal_weights"])
            if "q_global" in cfg:
                q_global.update(cfg["q_global"])

    if args.w_0 is not None: feat_weights["w_0"] = args.w_0
    if args.w_sobel is not None: feat_weights["w_sobel"] = args.w_sobel
    if args.w_dct is not None: feat_weights["w_dct"] = args.w_dct
    if args.w_chroma is not None: feat_weights["w_chroma"] = args.w_chroma

    w_sum = sum(feat_weights.values())
    for k in feat_weights:
        feat_weights[k] /= w_sum

    # 構造設定字串供檔名使用
    if voxel_mode == "surfel":
        config_desc = f"surfel_s{args.dynamic_voxel_scale}_min{args.min_voxel_size}_ang{int(args.surfel_angle_deg)}"
    elif voxel_mode == "dynamic":
        config_desc = f"dynamic_s{args.dynamic_voxel_scale}_min{args.min_voxel_size}"
    else:
        config_desc = f"uniform_vox{args.voxel_size}"

    if args.use_da3:
        config_desc += f"_res{args.da3_process_res}"

    # ==========================================
    # 3. 載入模型 (MVTracker, RAFT)
    # ==========================================
    if not args.skip_mvtracker:
        print("🧠 載入 MVTracker...")
        mvtracker = torch.hub.load(".", "mvtracker", source="local", pretrained=True, device=device)
        torch.set_float32_matmul_precision("high")
        amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

    raft_model = None
    raft_transforms = None
    if args.use_raft_mask:
        print("🌊 載入 RAFT 光流模型 (動靜態分離)...")
        weights = Raft_Large_Weights.DEFAULT
        raft_transforms = weights.transforms()
        raft_model = raft_large(weights=weights, progress=False).to(device)
        raft_model.eval()

    # ==========================================
    # 4. 分段追蹤 (Chunk-based Tracking)
    # ==========================================
    total_4d_export_time = 0.0
    total_tracking_time = 0.0
    last_imp_ply_path = None
    last_imp_pts_count = 0
    chunk_summaries = []

    for start_t in range(0, num_frames, args.segment_frames):
        chunk_start_time = time.time()
        end_t = min(start_t + args.segment_frames, num_frames)
        if end_t <= start_t:
            continue
            
        chunk_frames = end_t - start_t
        chunk_dir = os.path.join(args.dir, f"{start_t}-{end_t-1}frame")
        os.makedirs(chunk_dir, exist_ok=True)
        
        print(f"\n" + "="*50)
        print(f"🎬 開始處理片段: {start_t} 到 {end_t-1} 幀 (共 {chunk_frames} 幀)")
        print("="*50)

        # ------------------------------------------
        # 4.1 生成 Query Points (由 voxel_size 決定總數與密度)
        # ------------------------------------------
        t0 = start_t
        print(f"🎯 從 t={t0} 幀「全部相機」萃取追蹤點 (統一 Voxelize 融合)...")
        with torch.no_grad():
            xyz_full, rgb_full = init_pointcloud_from_rgbd(
                fmaps=rgbs_tensor[:, t0:t0+1].unsqueeze(0).float(),
                depths=depths_tensor[:, t0:t0+1].unsqueeze(0),
                intrs=intrs_model[:, t0:t0+1].unsqueeze(0),
                extrs=extrs_model[:, t0:t0+1].unsqueeze(0),
                stride=1,
                level=0,
            )
        
        pts_full = xyz_full[0]  
        colors_full = rgb_full[0] 
        
        valid_mask = depths_tensor[:, t0, 0].flatten() > 0
        motion_mask = torch.zeros_like(valid_mask)
        
        if args.use_raft_mask and raft_model is not None:
            print(f"🌊 計算 RAFT 動靜態遮罩 (Threshold: {args.raft_threshold})...")
            raft_model.to(device)
            max_flow_mag = torch.zeros(V, TARGET_H, TARGET_W, device=device)
            with torch.no_grad():
                for t in range(t0 + 5, end_t, 5):
                    # 一次送入所有視角，發揮最大並行算力
                    img1_t, img2_t = raft_transforms(rgbs_tensor[:, t0].to(device), rgbs_tensor[:, t].to(device))
                    flow = raft_model(img1_t, img2_t)[-1]
                    flow_mag = torch.norm(flow, dim=1)
                    max_flow_mag = torch.max(max_flow_mag, flow_mag)
            
            raft_model.to('cpu')
            torch.cuda.empty_cache()
            
            # ★ 新增：將第一台相機 (V=0) 的 RAFT Mask 疊加上去並存檔觀察
            cam0_rgb = rgbs_tensor[0, t0].permute(1, 2, 0).cpu().numpy().astype(np.float32) # [H, W, 3]
            cam0_mask = (max_flow_mag[0] > args.raft_threshold).cpu().numpy() # [H, W]
            
            overlay = cam0_rgb.copy()
            # 將被判定為動態的區域，染上 50% 半透明的紅色 [255, 0, 0]
            overlay[cam0_mask] = overlay[cam0_mask] * 0.5 + np.array([255, 0, 0]) * 0.5
            
            vis_path = os.path.join(chunk_dir, f"raft_mask_overlay_t{t0}.jpg")
            Image.fromarray(overlay.astype(np.uint8)).save(vis_path)
            print(f"🖼️ RAFT 動態遮罩預覽圖 (半透明紅) 已儲存至: {vis_path}")

            # 產生全域的 boolean motion_mask
            motion_mask = (max_flow_mag > args.raft_threshold).flatten().cpu()

        # 計算相機像素的 3D 足跡 (Footprints)
        if use_dynamic:
            focal_lengths = 0.5 * (intrs_model[:, t0, 0, 0] + intrs_model[:, t0, 1, 1])  # [V]
            f_map = focal_lengths[:, None, None].expand(V, TARGET_H, TARGET_W)  # [V, H, W]
            depths_t0 = depths_tensor[:, t0, 0]  # [V, H, W]
            footprint_map = depths_t0 / f_map  # [V, H, W]
            footprints_full = footprint_map.flatten()  # [V * H * W]
            valid_footprints = footprints_full[valid_mask].cpu().numpy()
        else:
            valid_footprints = None

        # 取出所有具備有效深度的點
        valid_pts = pts_full[valid_mask].cpu().numpy()
        valid_colors = colors_full[valid_mask].cpu().numpy()
        valid_is_dynamic = motion_mask[valid_mask].numpy()

        # 若為 surfel 模式，計算表面法向量
        if voxel_mode == "surfel":
            with torch.no_grad():
                if depths_raw_tensor is not None:
                    d_raw = depths_raw_tensor[:, t0, 0:1].to(device)
                    _, _, H_raw, W_raw = d_raw.shape
                    intrs_raw = intrs_model[:, t0].clone().to(device)
                    intrs_raw[:, 0, :] *= (float(W_raw) / float(TARGET_W))
                    intrs_raw[:, 1, :] *= (float(H_raw) / float(TARGET_H))
                    normals_raw, _ = depth_to_surface_normals(d_raw, intrinsics=intrs_raw, method="min_gradient")
                    normals_torch = F.interpolate(normals_raw, size=(TARGET_H, TARGET_W), mode="bilinear", align_corners=False)
                    normals_torch = F.normalize(normals_torch, dim=1, eps=1e-6)
                else:
                    normals_torch, _ = depth_to_surface_normals(depths_tensor[:, t0, 0:1].to(device), intrinsics=intrs_model[:, t0].to(device), method="min_gradient")
                norm_c = normals_torch.permute(0, 2, 3, 1).cpu().numpy()
                norm_w_list = []
                for v in range(V):
                    w2c_v = extrs_model[v, t0].cpu().numpy()
                    c2w_v = np.linalg.inv(np.vstack([w2c_v, [0, 0, 0, 1]]))
                    R_c2w = c2w_v[:3, :3]
                    n_v = norm_c[v].reshape(-1, 3)
                    n_w = n_v @ R_c2w.T
                    norm_w_list.append(n_w)
                all_norms_w = np.concatenate(norm_w_list, axis=0)
                valid_normals = all_norms_w[valid_mask.cpu().numpy()]
        else:
            valid_normals = None

        print(f"   [Voxel 融合] 準備處理總有效點數: {len(valid_pts)} (模式: {voxel_mode})")
        
        # 一次性對所有點進行 Voxelize (GPU 加速)
        v_pts, v_cols, _, v_is_dyn = voxel_downsample_unified_with_features(
            pts=valid_pts,
            colors=valid_colors,
            features=None,
            is_dynamic=valid_is_dynamic,
            normals=valid_normals,
            voxel_size=args.voxel_size,
            footprints=valid_footprints,
            min_voxel_size=args.min_voxel_size,
            scale=args.dynamic_voxel_scale,
            mode=voxel_mode,
            surfel_normal_scale=args.surfel_normal_scale,
            surfel_angle_deg=args.surfel_angle_deg,
        )
        if isinstance(v_pts, torch.Tensor):
            v_pts = v_pts.cpu().numpy()
            v_cols = v_cols.cpu().numpy()
            if v_is_dyn is not None:
                v_is_dyn = v_is_dyn.cpu().numpy()
        
        print(f"   [Voxel 融合] 融合後總點數: {len(v_pts)} (模式: {voxel_mode}, voxel_size={args.voxel_size})")

        # 根據 voxel 融合後的標籤，分流出動態與靜態點
        moving_pts_np = v_pts[v_is_dyn]
        moving_cols_np = v_cols[v_is_dyn]

        static_pts_np = v_pts[~v_is_dyn]
        static_cols_np = v_cols[~v_is_dyn]

        # ==========================================
        # ★ 新增：為動態點加回 KMeans 空間群聚排序
        # 讓 MVTracker 的 Transformer 能完美捕捉局部運動特徵
        # ==========================================
        if len(moving_pts_np) > 0:
            if args.query_sort_mode == "kmeans":
                from sklearn.cluster import MiniBatchKMeans
                chunk_size = args.track_chunk_size
                # 計算需要分成幾個群 (至少 1 群)
                num_groups = max(1, len(moving_pts_np) // chunk_size)
                
                print(f"   [空間分群] 將 {len(moving_pts_np)} 個動態點聚類為 {num_groups} 個空間區塊...")
                kmeans = MiniBatchKMeans(n_clusters=num_groups, random_state=72, n_init="auto")
                labels = kmeans.fit_predict(moving_pts_np)
                
                # 根據分群標籤進行排序 (argsort)
                # 這會確保屬於同一個 Cluster 的點在陣列中是連續的
                sort_idx = np.argsort(labels)
                moving_pts_np = moving_pts_np[sort_idx]
                moving_cols_np = moving_cols_np[sort_idx]
                print("   [空間分群] 排序完成，已為 MVTracker 最佳化 Batch 排列！")
            elif args.query_sort_mode == "round_robin":
                print(f"   [空間均勻打散] 將 {len(moving_pts_np)} 個動態點透過 3D 體素網格(Voxel Grid)進行輪詢排序...")
            
                # 取得均勻交錯的 index
                sort_idx = reorder_by_3d_grid_round_robin(moving_pts_np, max_points=args.track_chunk_size)
                
                # 套用排序
                moving_pts_np = moving_pts_np[sort_idx]
                moving_cols_np = moving_cols_np[sort_idx]
                print("   [空間均勻打散] 排序完成！現在每個 Batch 都會均勻包含全場景的特徵點！")

        # ==========================================

        # 轉回 Torch Tensor 放進 GPU
        moving_pts_sampled = torch.from_numpy(moving_pts_np).to(device)
        moving_colors_sampled = torch.from_numpy(moving_cols_np).to(device)
        
        static_pts_sampled = torch.from_numpy(static_pts_np).to(device)
        static_colors_sampled = torch.from_numpy(static_cols_np).to(device)

        # 組合所有的點 (為了保持原本程式後半段能順利執行)
        all_pts_sampled = torch.cat([moving_pts_sampled, static_pts_sampled], dim=0)
        all_colors_sampled = torch.cat([moving_colors_sampled, static_colors_sampled], dim=0)
            
        print(f"✅ 成功提取 {len(moving_pts_sampled)} 動態追蹤點, {len(static_pts_sampled)} 靜態軌跡點！ (總共 {len(all_pts_sampled)} 點)")
        # 匯出初始點雲 (.ply) 供 4DGS 初始化
        pts_np = all_pts_sampled.cpu().numpy()
        colors_np = all_colors_sampled.cpu().numpy()
        if colors_np.max() <= 1.0: 
            colors_np = (colors_np * 255).astype(np.uint8)
        times_np = np.full((len(pts_np), 1), float(t0), dtype=np.float32)

        init_ply_name = f"init_pointcloud_{model_tag}_4d.ply"
        ply_path = os.path.join(chunk_dir, init_ply_name)
        dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('time', 'f4'),
            ('t', 'f4')
        ]
        elements = np.empty(pts_np.shape[0], dtype=dtype)
        elements['x'], elements['y'], elements['z'] = pts_np[:, 0], pts_np[:, 1], pts_np[:, 2]
        elements['red'], elements['green'], elements['blue'] = colors_np[:, 0], colors_np[:, 1], colors_np[:, 2]
        elements['time'] = times_np[:, 0]
        elements['t'] = times_np[:, 0]

        init_comments = [
            f"model={model_tag}",
            f"model_id={args.da3_model if args.use_da3 else 'facebook/VGGT-1B'}",
            f"voxel_mode={voxel_mode}",
            f"voxel_size={args.voxel_size}",
            f"min_voxel_size={args.min_voxel_size}",
            f"dynamic_voxel_scale={args.dynamic_voxel_scale}",
            f"process_res={args.da3_process_res if args.use_da3 else 'native'}",
            f"t0={t0}",
            f"total_points={len(pts_np)}",
        ]
        PlyData([PlyElement.describe(elements, 'vertex')], text=False, comments=init_comments).write(ply_path)
        print(f"🎉 初始點雲已儲存至 {ply_path}")

        # 向下相容建立軟連結
        legacy_init_ply = os.path.join(chunk_dir, "init_pointcloud_vggt_4d.ply")
        if ply_path != legacy_init_ply:
            try:
                if os.path.islink(legacy_init_ply) or os.path.exists(legacy_init_ply):
                    os.remove(legacy_init_ply)
                os.symlink(init_ply_name, legacy_init_ply)
            except Exception:
                PlyData([PlyElement.describe(elements, 'vertex')], text=False, comments=init_comments).write(legacy_init_ply)

        # ==========================================
        # ★ 全幀點雲匯出：支援 Full 4D 點雲與 Importance Sampling 4D 點雲
        # ==========================================
        do_export_full = args.export_all_frame_ply or args.export_vggt_all_frame_ply
        do_export_importance = args.export_importance_all_frame_ply

        if do_export_full or do_export_importance:
            t_export_start = time.time()
            export_modes = []
            if do_export_full: export_modes.append("Full 4D")
            if do_export_importance: export_modes.append("Importance Sampling 4D")
            print(f"\n🌟 正在生成 4D 點雲 [{', '.join(export_modes)}] (Frames {start_t} to {end_t-1}, Model: {model_tag})...")

            all_t_full_pts, all_t_full_colors, all_t_full_times = [], [], []
            all_t_imp_pts, all_t_imp_colors, all_t_imp_times = [], [], []

            num_frames_in_chunk = end_t - start_t
            if args.importance_points_per_frame is not None:
                importance_k = args.importance_points_per_frame
            else:
                importance_k = max(1, args.importance_total_points // max(1, num_frames_in_chunk))

            if do_export_importance:
                print(f"🎯 重要度採樣配置: 每幀採樣 {importance_k:,} 點 (預算模式, 樣式: {args.importance_vis_style})")
                print(f"   加權係數: w0={feat_weights['w_0']:.3f}, sob={feat_weights['w_sobel']:.3f}, dct={feat_weights['w_dct']:.3f}, chr={feat_weights['w_chroma']:.3f}")

            feature_scorer = RGBTextureScorer(normalize=False)

            # 逐幀提取點雲並降採樣
            for t_idx in tqdm(range(start_t, end_t), desc="4D 點雲轉換與特徵採樣"):
                # 提取當前幀多視角 2D 特徵
                rgbs_t = rgbs_tensor[:, t_idx].float() / 255.0  # [V, 3, H, W]
                with torch.no_grad():
                    f_res = feature_scorer(rgbs_t)
                    s_sob = f_res['s_sobel'].squeeze(1).cpu()
                    s_dct = f_res['s_dct'].squeeze(1).cpu()
                    s_chr = f_res['s_chroma'].squeeze(1).cpu()

                # 應用跨視角全域尺度基準 Q_global (防禦白牆噪點放大)
                sob_scaled = torch.clamp(s_sob / max(q_global["s_sobel"], 1e-4), 0.0, 1.0).numpy()
                dct_scaled = torch.clamp(s_dct / max(q_global["s_dct"], 1e-4), 0.0, 1.0).numpy()
                chr_scaled = torch.clamp(s_chr / max(q_global["s_chroma"], 1e-4), 0.0, 1.0).numpy()

                # Surfel 模式：以 GPU LINEMOD 計算表面法向量
                if voxel_mode == "surfel":
                    with torch.no_grad():
                        if depths_raw_tensor is not None:
                            d_raw = depths_raw_tensor[:, t_idx, 0:1].to(device)
                            _, _, H_raw, W_raw = d_raw.shape
                            intrs_raw = intrs_model[:, t_idx].clone().to(device)
                            intrs_raw[:, 0, :] *= (float(W_raw) / float(TARGET_W))
                            intrs_raw[:, 1, :] *= (float(H_raw) / float(TARGET_H))
                            normals_raw, _ = depth_to_surface_normals(d_raw, intrinsics=intrs_raw, method="min_gradient")
                            normals_torch = F.interpolate(normals_raw, size=(TARGET_H, TARGET_W), mode="bilinear", align_corners=False)
                            normals_torch = F.normalize(normals_torch, dim=1, eps=1e-6)
                        else:
                            normals_torch, _ = depth_to_surface_normals(depths_tensor[:, t_idx, 0:1].to(device), intrinsics=intrs_model[:, t_idx].to(device), method="min_gradient")
                        normals_np = normals_torch.permute(0, 2, 3, 1).cpu().numpy()
                else:
                    normals_np = None

                # 多視角 3D 反投影
                pts_t, cols_t, feats_t, fps_t, norms_t = [], [], [], [], []
                for v in range(V):
                    rgb_np = rgbs_tensor[v, t_idx].permute(1, 2, 0).numpy()
                    d_np = depths_tensor[v, t_idx, 0].numpy()
                    feat_map = np.stack([sob_scaled[v], dct_scaled[v], chr_scaled[v]], axis=-1)
                    K_v = intrs_model[v, t_idx].numpy()
                    w2c_v = extrs_model[v, t_idx].numpy()
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

                # 體素下採樣與特徵 Max-Pooling
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

                # 1. 收集 Full 4D 點雲
                if do_export_full:
                    v_cols_u8 = (np.clip(v_cols, 0.0, 1.0) * 255).astype(np.uint8) if v_cols.max() <= 1.0 else v_cols.astype(np.uint8)
                    times_t = np.full((M,), float(t_idx), dtype=np.float32)
                    all_t_full_pts.append(v_pts)
                    all_t_full_colors.append(v_cols_u8)
                    all_t_full_times.append(times_t)

                # 2. 進行特徵加權重要度採樣 (Importance Sampling)
                if do_export_importance:
                    s_sobel_v = v_feats[:, 0]
                    s_dct_v = v_feats[:, 1]
                    s_chroma_v = v_feats[:, 2]

                    p_base_v = 1.0 / M
                    p_sob_v = s_sobel_v / max(np.sum(s_sobel_v), 1e-12)
                    p_dct_v = s_dct_v / max(np.sum(s_dct_v), 1e-12)
                    p_chr_v = s_chroma_v / max(np.sum(s_chroma_v), 1e-12)

                    p_k = (
                        feat_weights["w_0"] * p_base_v +
                        feat_weights["w_sobel"] * p_sob_v +
                        feat_weights["w_dct"] * p_dct_v +
                        feat_weights["w_chroma"] * p_chr_v
                    )
                    p_sum = p_k.sum()
                    if p_sum > 0:
                        p_k /= p_sum
                    else:
                        p_k = np.full(M, 1.0 / M, dtype=np.float32)

                    k_sample = min(importance_k, M)
                    p_torch = torch.from_numpy(p_k).float()
                    sampled_indices = torch.multinomial(p_torch, num_samples=k_sample, replacement=False).numpy()

                    s_pts = v_pts[sampled_indices]
                    if args.importance_vis_style == "highlight":
                        s_cols = np.tile(np.array([255, 215, 0], dtype=np.uint8), (k_sample, 1))
                    elif args.importance_vis_style == "feature":
                        s_spatial = (
                            feat_weights["w_0"] +
                            feat_weights["w_sobel"] * s_sobel_v +
                            feat_weights["w_dct"] * s_dct_v +
                            feat_weights["w_chroma"] * s_chroma_v
                        )
                        q99_score = float(np.percentile(s_spatial, 99)) if len(s_spatial) > 0 else 1.0
                        score_norm = np.clip(s_spatial / max(q99_score, 1e-4), 0.0, 1.0)
                        turbo_colors = (plt.cm.turbo(score_norm)[:, :3] * 255).astype(np.uint8)
                        s_cols = turbo_colors[sampled_indices]
                    else:  # "rgb"
                        v_c = (np.clip(v_cols[sampled_indices], 0.0, 1.0) * 255).astype(np.uint8) if v_cols.max() <= 1.0 else v_cols[sampled_indices].astype(np.uint8)
                        s_cols = v_c

                    s_times = np.full((k_sample,), float(t_idx), dtype=np.float32)
                    all_t_imp_pts.append(s_pts)
                    all_t_imp_colors.append(s_cols)
                    all_t_imp_times.append(s_times)

            # 匯出 Full 4D 點雲
            if do_export_full and len(all_t_full_pts) > 0:
                full_pts_np = np.concatenate(all_t_full_pts, axis=0)
                full_colors_np = np.concatenate(all_t_full_colors, axis=0)
                full_times_np = np.concatenate(all_t_full_times, axis=0)

                full_ply_filename = f"init_pointcloud_{model_tag}_4d_full_{config_desc}.ply"
                full_ply_path = os.path.join(chunk_dir, full_ply_filename)
                
                full_comments = [
                    f"model={model_tag}",
                    f"model_id={args.da3_model if args.use_da3 else 'facebook/VGGT-1B'}",
                    f"voxel_mode={voxel_mode}",
                    f"voxel_size={args.voxel_size}",
                    f"min_voxel_size={args.min_voxel_size}",
                    f"dynamic_voxel_scale={args.dynamic_voxel_scale}",
                    f"surfel_normal_scale={args.surfel_normal_scale}",
                    f"surfel_angle_deg={args.surfel_angle_deg}",
                    f"process_res={args.da3_process_res if args.use_da3 else 'native'}",
                    f"frames={start_t}..{end_t-1}",
                    f"total_points={len(full_pts_np)}",
                ]

                elements_full = np.empty(full_pts_np.shape[0], dtype=dtype)
                elements_full['x'], elements_full['y'], elements_full['z'] = full_pts_np[:, 0], full_pts_np[:, 1], full_pts_np[:, 2]
                elements_full['red'], elements_full['green'], elements_full['blue'] = full_colors_np[:, 0], full_colors_np[:, 1], full_colors_np[:, 2]
                elements_full['time'] = full_times_np
                elements_full['t'] = full_times_np
                PlyData([PlyElement.describe(elements_full, 'vertex')], text=False, comments=full_comments).write(full_ply_path)
                print(f"🎉 全幀 4D 完整點雲已儲存至: {full_ply_path} (總點數: {len(full_pts_np):,})")

                # 向下相容建立軟連結
                legacy_full_ply = os.path.join(chunk_dir, "init_pointcloud_vggt_4d_full.ply")
                try:
                    if os.path.islink(legacy_full_ply) or os.path.exists(legacy_full_ply):
                        os.remove(legacy_full_ply)
                    os.symlink(full_ply_filename, legacy_full_ply)
                except Exception:
                    pass

            # 匯出 Importance Sampled 4D 點雲
            if do_export_importance and len(all_t_imp_pts) > 0:
                imp_pts_np = np.concatenate(all_t_imp_pts, axis=0)
                imp_colors_np = np.concatenate(all_t_imp_colors, axis=0)
                imp_times_np = np.concatenate(all_t_imp_times, axis=0)

                pts_budget_str = f"{args.importance_total_points // 1000}k" if (args.importance_total_points % 1000 == 0 and args.importance_total_points >= 1000) else f"{args.importance_total_points}"
                style_str = f"_{args.importance_vis_style}" if args.importance_vis_style != "rgb" else ""
                imp_ply_filename = f"init_pointcloud_{model_tag}_4d_importance_{pts_budget_str}pts_{config_desc}{style_str}.ply"
                imp_ply_path = os.path.join(chunk_dir, imp_ply_filename)

                imp_comments = [
                    f"model={model_tag}",
                    f"model_id={args.da3_model if args.use_da3 else 'facebook/VGGT-1B'}",
                    f"voxel_mode={voxel_mode}",
                    f"voxel_size={args.voxel_size}",
                    f"min_voxel_size={args.min_voxel_size}",
                    f"dynamic_voxel_scale={args.dynamic_voxel_scale}",
                    f"surfel_normal_scale={args.surfel_normal_scale}",
                    f"surfel_angle_deg={args.surfel_angle_deg}",
                    f"process_res={args.da3_process_res if args.use_da3 else 'native'}",
                    f"importance_total_points={args.importance_total_points}",
                    f"importance_points_per_frame={importance_k}",
                    f"importance_vis_style={args.importance_vis_style}",
                    f"weights=w0:{feat_weights['w_0']:.4f},w_sobel:{feat_weights['w_sobel']:.4f},w_dct:{feat_weights['w_dct']:.4f},w_chroma:{feat_weights['w_chroma']:.4f}",
                    f"q_global=s_sobel:{q_global['s_sobel']:.4f},s_dct:{q_global['s_dct']:.4f},s_chroma:{q_global['s_chroma']:.4f}",
                    f"frames={start_t}..{end_t-1}",
                    f"total_points={len(imp_pts_np)}",
                ]

                elements_imp = np.empty(imp_pts_np.shape[0], dtype=dtype)
                elements_imp['x'], elements_imp['y'], elements_imp['z'] = imp_pts_np[:, 0], imp_pts_np[:, 1], imp_pts_np[:, 2]
                elements_imp['red'], elements_imp['green'], elements_imp['blue'] = imp_colors_np[:, 0], imp_colors_np[:, 1], imp_colors_np[:, 2]
                elements_imp['time'] = imp_times_np
                elements_imp['t'] = imp_times_np
                PlyData([PlyElement.describe(elements_imp, 'vertex')], text=False, comments=imp_comments).write(imp_ply_path)
                print(f"🎉 重要度採樣 4D 點雲已儲存至: {imp_ply_path} (總點數: {len(imp_pts_np):,})\n")
                last_imp_ply_path = imp_ply_path
                last_imp_pts_count = len(imp_pts_np)

                # 建立簡短名稱軟連結
                imp_short_link = os.path.join(chunk_dir, f"init_pointcloud_{model_tag}_4d_importance.ply")
                try:
                    if os.path.islink(imp_short_link) or os.path.exists(imp_short_link):
                        os.remove(imp_short_link)
                    os.symlink(imp_ply_filename, imp_short_link)
                except Exception:
                    pass

            total_4d_export_time += (time.time() - t_export_start)

        if args.skip_mvtracker:
            chunk_elapsed = time.time() - chunk_start_time
            alloc_gb = torch.cuda.memory_allocated(device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
            peak_alloc_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
            peak_res_gb = torch.cuda.max_memory_reserved(device) / (1024 ** 3) if torch.cuda.is_available() else 0.0

            chunk_info = {
                "chunk_range": f"{start_t}-{end_t-1}",
                "start_frame": start_t,
                "end_frame": end_t - 1,
                "frames": chunk_frames,
                "elapsed_seconds": round(chunk_elapsed, 2),
                "memory_allocated_gb": round(alloc_gb, 3),
                "peak_memory_allocated_gb": round(peak_alloc_gb, 3),
                "peak_memory_reserved_gb": round(peak_res_gb, 3),
                "importance_ply": last_imp_ply_path,
                "importance_points": last_imp_pts_count,
            }
            chunk_summaries.append(chunk_info)
            chunk_json_path = os.path.join(chunk_dir, "benchmark_stats.json")
            try:
                with open(chunk_json_path, "w", encoding="utf-8") as f:
                    json.dump(chunk_info, f, indent=2, ensure_ascii=False)
            except Exception:
                pass

            if torch.cuda.is_available():
                print(f"⏱️ 片段 [{start_t}..{end_t-1}] 處理完成！耗時: {chunk_elapsed:.2f} 秒 | 顯存: {alloc_gb:.2f} GB (峰值: {peak_alloc_gb:.2f} GB)")
            else:
                print(f"⏱️ 片段 [{start_t}..{end_t-1}] 處理完成！耗時: {chunk_elapsed:.2f} 秒")
            print(f"⏭️ 啟用 --skip_mvtracker，跳過該片段的 MVTracker 追蹤與後續步驟。")
            continue

        # ------------------------------------------
        # 4.2 執行 MVTracker (動態點批次處理)
        # ------------------------------------------
        # ... 下方的程式碼保持不變 ...
        ts_abs = torch.full((all_pts_sampled.shape[0], 1), float(t0), device=all_pts_sampled.device)
        query_points_abs = torch.cat([ts_abs, all_pts_sampled], dim=1).float()  
        
        ts_rel = torch.full((moving_pts_sampled.shape[0], 1), float(0), device=moving_pts_sampled.device)
        query_points_rel_moving = torch.cat([ts_rel, moving_pts_sampled], dim=1).float()
        
        all_pred_tracks = []
        all_pred_vis = []
        total_queries = query_points_rel_moving.shape[0]
        chunk_size = args.track_chunk_size
        
        print(f"🚀 開始追蹤 {total_queries} 個動態點，分批大小為 {chunk_size}...")
        start_time = time.time()
        
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=amp_dtype):
            rgbs_input = rgbs_tensor[None, :, start_t:end_t].cpu()
            depths_input = depths_tensor[None, :, start_t:end_t].cpu()
            intrs_input = intrs_model[None, :, start_t:end_t].to(device)
            extrs_input = extrs_model[None, :, start_t:end_t].to(device)
            
            for i in tqdm(range(0, total_queries, chunk_size), desc="MVTracker 追蹤進度"):
                batch_queries = query_points_rel_moving[i:i+chunk_size].to(device)
                results = mvtracker(
                    rgbs=rgbs_input,
                    depths=depths_input,
                    intrs=intrs_input,
                    extrs=extrs_input,
                    query_points_3d=batch_queries[None],
                )
                all_pred_tracks.append(results["traj_e"].cpu())
                all_pred_vis.append(results["vis_e"].cpu())
                
                del results
                torch.cuda.empty_cache()

        end_time = time.time()
        
        if len(all_pred_tracks) > 0:
            pred_tracks_moving = torch.cat(all_pred_tracks, dim=2)[0]
            pred_vis_moving = torch.cat(all_pred_vis, dim=2)[0]
        else:
            pred_tracks_moving = torch.empty((chunk_frames, 0, 3))
            pred_vis_moving = torch.empty((chunk_frames, 0))
            
        if static_pts_sampled is not None:
            N_static = static_pts_sampled.shape[0]
            pred_tracks_static = static_pts_sampled.unsqueeze(0).expand(chunk_frames, -1, -1).cpu()
            pred_vis_static = torch.ones((chunk_frames, N_static), dtype=torch.float32).cpu() * 10.0
            
            pred_tracks = torch.cat([pred_tracks_moving, pred_tracks_static], dim=1)
            pred_vis = torch.cat([pred_vis_moving, pred_vis_static], dim=1)
        else:
            pred_tracks = pred_tracks_moving
            pred_vis = pred_vis_moving

        print(f"✅ 動態追蹤與靜態軌跡合併完成！ (片段花費時間: {end_time - start_time:.2f} 秒)")

        # ------------------------------------------
        # 4.3 儲存軌跡
        # ------------------------------------------
        tracks_path = os.path.join(chunk_dir, "mvtracker_tracks.npz")
        np.savez(
            tracks_path,
            query_points=query_points_abs.cpu().numpy(),
            pred_tracks=pred_tracks.numpy(),
            pred_vis=pred_vis.numpy(),
        )
        print(f"🎉 軌跡已儲存至 {tracks_path}")

        # ------------------------------------------
        # 4.4 匯出逐幀 PLY
        # ------------------------------------------
        if args.export_per_frame_ply:
            ply_dir = os.path.join(chunk_dir, "mvtracker_plys")
            os.makedirs(ply_dir, exist_ok=True)
            print(f"📁 正在將每一幀的點雲匯出至 {ply_dir}...")
            
            pred_tracks_np = pred_tracks.numpy()
            total_output_pts = len(all_pts_sampled)  # <== 修正：確保總數是動態+靜態之和
            
            for t_rel in tqdm(range(chunk_frames), desc="匯出 PLY"):
                abs_t = start_t + t_rel
                ply_out_path = os.path.join(ply_dir, f"frame_{abs_t:04d}.ply")
                
                # <== 修正：改用 total_output_pts 初始化 elements 陣列
                elements = np.empty(total_output_pts, dtype=dtype)
                elements['x'], elements['y'], elements['z'] = pred_tracks_np[t_rel, :, 0], pred_tracks_np[t_rel, :, 1], pred_tracks_np[t_rel, :, 2]
                elements['red'], elements['green'], elements['blue'] = colors_np[:, 0], colors_np[:, 1], colors_np[:, 2]
                elements['time'] = np.full(total_output_pts, float(abs_t), dtype=np.float32)
                PlyData([PlyElement.describe(elements, 'vertex')]).write(ply_out_path)
            print("✅ 逐幀 PLY 匯出完成！")

        chunk_elapsed = time.time() - chunk_start_time
        total_tracking_time += (time.time() - start_time)
        alloc_gb = torch.cuda.memory_allocated(device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
        peak_alloc_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
        peak_res_gb = torch.cuda.max_memory_reserved(device) / (1024 ** 3) if torch.cuda.is_available() else 0.0

        chunk_info = {
            "chunk_range": f"{start_t}-{end_t-1}",
            "start_frame": start_t,
            "end_frame": end_t - 1,
            "frames": chunk_frames,
            "elapsed_seconds": round(chunk_elapsed, 2),
            "memory_allocated_gb": round(alloc_gb, 3),
            "peak_memory_allocated_gb": round(peak_alloc_gb, 3),
            "peak_memory_reserved_gb": round(peak_res_gb, 3),
            "importance_ply": last_imp_ply_path,
            "importance_points": last_imp_pts_count,
        }
        chunk_summaries.append(chunk_info)
        chunk_json_path = os.path.join(chunk_dir, "benchmark_stats.json")
        try:
            with open(chunk_json_path, "w", encoding="utf-8") as f:
                json.dump(chunk_info, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        if torch.cuda.is_available():
            print(f"⏱️ 片段 [{start_t}..{end_t-1}] 全部完成！耗時: {chunk_elapsed:.2f} 秒 | 顯存: {alloc_gb:.2f} GB (峰值: {peak_alloc_gb:.2f} GB)")
        else:
            print(f"⏱️ 片段 [{start_t}..{end_t-1}] 全部完成！耗時: {chunk_elapsed:.2f} 秒")

    # ==========================================
    # 總結統計報表 (Outside chunk loop)
    # ==========================================
    total_elapsed = time.time() - full_start_time
    curr_alloc = torch.cuda.memory_allocated(device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
    peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
    peak_res = torch.cuda.max_memory_reserved(device) / (1024 ** 3) if torch.cuda.is_available() else 0.0

    print(f"\n" + "="*70)
    print(f"📊 執行效能與 GPU 顯存統計摘要 (Performance & VRAM Summary)")
    print("="*70)
    print(f"⏱️  總執行時間: {total_elapsed:.2f} 秒 ({total_elapsed / 60:.2f} 分鐘)")
    print(f"   - 資料讀取與相機準備: {t_data_prep:.2f} 秒")
    print(f"   - 幾何深度推論與對齊 ({model_tag}): {t_depth_infer:.2f} 秒")
    if do_export_full or do_export_importance:
        print(f"   - 4D 體素特徵與點雲生成總計: {total_4d_export_time:.2f} 秒")
    if not args.skip_mvtracker:
        print(f"   - MVTracker 追蹤與軌跡匯出總計: {total_tracking_time:.2f} 秒")
    if torch.cuda.is_available():
        print(f"💾 GPU 顯存 (VRAM) 統計 ({torch.cuda.get_device_name(device)}):")
        print(f"   - 當前佔用 (Current Allocated): {curr_alloc:.2f} GB")
        print(f"   - 歷史峰值佔用 (Peak Allocated): {peak_alloc:.2f} GB")
        print(f"   - 歷史峰值保留 (Peak Reserved):  {peak_res:.2f} GB")
    else:
        print("💾 GPU 顯存: N/A (CPU 模式)")
    if last_imp_ply_path is not None:
        print(f"🎯 最新重要度採樣點雲: {last_imp_ply_path} (共 {last_imp_pts_count:,} 點)")

    # 寫入整體摘要 JSON 檔案
    summary_stats = {
        "dataset": os.path.basename(os.path.normpath(args.dir)),
        "dataset_dir": args.dir,
        "timestamp": datetime.now().isoformat(),
        "model": model_tag,
        "model_id": args.da3_model if args.use_da3 else 'facebook/VGGT-1B',
        "process_res": args.da3_process_res if args.use_da3 else 'native',
        "voxel_mode": voxel_mode,
        "total_elapsed_seconds": round(total_elapsed, 2),
        "total_elapsed_minutes": round(total_elapsed / 60, 2),
        "timing_breakdown": {
            "data_prep_seconds": round(t_data_prep, 2),
            "depth_infer_seconds": round(t_depth_infer, 2),
            "export_4d_seconds": round(total_4d_export_time, 2) if (do_export_full or do_export_importance) else 0.0,
            "tracking_seconds": round(total_tracking_time, 2) if not args.skip_mvtracker else 0.0,
        },
        "gpu_vram": {
            "device_name": torch.cuda.get_device_name(device) if torch.cuda.is_available() else "N/A",
            "current_allocated_gb": round(curr_alloc, 3),
            "peak_allocated_gb": round(peak_alloc, 3),
            "peak_reserved_gb": round(peak_res, 3),
        },
        "config": {
            "num_cams": args.num_cams,
            "segment_frames": args.segment_frames,
            "total_frames": num_frames,
            "importance_total_points": args.importance_total_points,
            "importance_vis_style": args.importance_vis_style,
            "feature_weights": feat_weights,
            "q_global": q_global,
            "skip_mvtracker": args.skip_mvtracker,
        },
        "chunks": chunk_summaries,
    }

    summary_json_path = os.path.join(args.dir, f"benchmark_stats_{model_tag}.json")
    try:
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_stats, f, indent=2, ensure_ascii=False)
        print(f"💾 效能與顯存統計已儲存至: {summary_json_path}")
    except Exception as e:
        print(f"⚠️ 寫入總結統計檔案失敗: {e}")
    print("="*70 + "\n")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", message=".*DtypeTensor constructors are no longer.*", module="pointops.query")
    warnings.filterwarnings("ignore", message=".*Plan failed with a cudnnException.*", module="torch.nn.modules.conv")
    main()
