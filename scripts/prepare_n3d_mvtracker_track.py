import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import re
import time
import glob
import gc

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
from plyfile import PlyData, PlyElement

from mvtracker.datasets.generic_scene_dataset import _ensure_vggt_aligned_cache_and_load
from mvtracker.models.core.model_utils import init_pointcloud_from_rgbd
from sklearn.cluster import MiniBatchKMeans
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

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

def main():
    full_start_time = time.time()
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=str, required=True, help="n3d 資料集路徑")
    p.add_argument("--max_frames", type=int, default=300, help="限制載入的最大幀數 (避免 MVTracker OOM)")
    p.add_argument("--selected_cams", type=int, nargs="+", default=[16, 10, 13, 1, 18], help="選擇的相機 ID 列表")
    p.add_argument("--track_chunk_size", type=int, default=10000, help="MVTracker 追蹤時的分批大小，避免 OOM")
    p.add_argument("--use_dynamic_vggt_cameras", action="store_true", help="使用動態 VGGT 預測的內外參；若不加此參數，則將第1幀的內外參套用於所有後續幀")
    p.add_argument("--export_per_frame_ply", action="store_true", help="是否將每一幀的軌跡儲存為獨立的 PLY 檔以供檢查")
    p.add_argument("--segment_frames", type=int, default=50, help="每個追蹤片段的長度 (幀數)")
    p.add_argument("--voxel_size", type=float, default=0.1, help="Voxel downsample 的體素大小")
    p.add_argument("--use_raft_mask", action="store_true", help="使用 RAFT 光流來過濾靜態背景點")
    p.add_argument("--raft_threshold", type=float, default=1.0, help="RAFT 判定為移動的像素閾值")
    args = p.parse_args()

    np.random.seed(72)
    torch.manual_seed(72)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # 1. 讀取影像與相機姿態
    # ==========================================
    view_dirs = sorted(
        glob.glob(os.path.join(args.dir, 'view_*')), 
        key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group())
    )
    sample_frames = sorted(glob.glob(os.path.join(view_dirs[0], '*.jpg')))
    num_frames = min(len(sample_frames), args.max_frames)
    sample_img = Image.open(sample_frames[0])
    W_img, H_img = sample_img.size
    print(f"📷 處理圖片: {W_img}x{H_img} | 取樣幀數: {num_frames} 幀")

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

    SELECTED_CAMS = args.selected_cams
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
        frames = sorted(glob.glob(os.path.join(view_dirs[cam_idx], '*.jpg')))[:num_frames]
        cam_rgbs = []
        for t in range(num_frames):
            img = Image.open(frames[t]).convert('RGB')
            img_resized = img.resize((TARGET_W, TARGET_H), Image.Resampling.BILINEAR)
            cam_rgbs.append((to_tensor(img_resized) * 255).byte()) # [3, H, W] uint8
        rgbs_list.append(torch.stack(cam_rgbs)) # [T, 3, H, W]
    
    rgbs_tensor = torch.stack(rgbs_list) # [V, T, 3, H, W]

    seq_name = "n3d_gt_init_aligned" if args.use_dynamic_vggt_cameras else "n3d_gt_init_raw"
    print(f"🚀 提取 VGGT 深度 (模式: {seq_name})...")
    with torch.no_grad():
        depths_tensor, _, intrs_vggt, extrs_vggt = _ensure_vggt_aligned_cache_and_load(
            rgbs=rgbs_tensor,
            seq_name=seq_name,
            dataset_root=args.dir,
            extrs_gt=extrs_gt,
            vggt_cache_subdir="vggt_cache",
            skip_if_cached=False,
            model_id="facebook/VGGT-1B",
        )

    if args.use_dynamic_vggt_cameras:
        intrs_model = intrs_vggt
        extrs_model = extrs_vggt
    else:
        # 套用第1幀內外參至所有後續幀
        intrs_model = intrs_vggt[:, 0:1].expand_as(intrs_vggt).clone()
        extrs_model = extrs_vggt[:, 0:1].expand_as(extrs_vggt).clone()

    # ==========================================
    # 3. 載入模型 (MVTracker, RAFT)
    # ==========================================
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
    for start_t in range(0, num_frames, args.segment_frames):
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

        # ==========================================
        # ★ 新增：統一的 Voxel 融合函數 (包含動態遮罩的 Max Pooling)
        # ==========================================
        def voxel_downsample_unified(pts, colors, orig_idx, is_dynamic, voxel_size):
            coords = np.round(pts / voxel_size)
            unique_coords, first_indices, inverse_indices = np.unique(coords, axis=0, return_index=True, return_inverse=True)
            
            num_voxels = len(unique_coords)
            pts_sum = np.zeros((num_voxels, 3), dtype=np.float64)
            colors_sum = np.zeros((num_voxels, 3), dtype=np.float64)
            # 建立儲存 voxel 動態標籤的陣列 (預設為 False)
            is_dynamic_max = np.zeros(num_voxels, dtype=bool)
            
            np.add.at(pts_sum, inverse_indices, pts)
            np.add.at(colors_sum, inverse_indices, colors)
            counts = np.bincount(inverse_indices)
            
            # 使用 logical_or.at 達成 Max Pooling 的效果
            # 只要 voxel 裡有任一個點是 True(動態)，整個 voxel 就是 True
            np.logical_or.at(is_dynamic_max, inverse_indices, is_dynamic)
            
            pts_voxel = (pts_sum / counts[:, None]).astype(np.float32)
            colors_voxel = (colors_sum / counts[:, None]).astype(np.float32)
            orig_idx_voxel = orig_idx[first_indices] 
            
            return pts_voxel, colors_voxel, orig_idx_voxel, is_dynamic_max

        # 取出所有具備有效深度的點
        valid_pts = pts_full[valid_mask].cpu().numpy()
        valid_colors = colors_full[valid_mask].cpu().numpy()
        valid_indices_np = torch.nonzero(valid_mask, as_tuple=True)[0].cpu().numpy()
        valid_is_dynamic = motion_mask[valid_mask].numpy()
        
        print(f"   [Voxel 融合] 準備處理總有效點數: {len(valid_pts)}")
        
        # 一次性對所有點進行 Voxelize
        v_pts, v_cols, v_idx, v_is_dyn = voxel_downsample_unified(
            valid_pts, valid_colors, valid_indices_np, valid_is_dynamic, args.voxel_size
        )
        
        print(f"   [Voxel 融合] 融合後總點數: {len(v_pts)} (voxel_size={args.voxel_size})")

        # 根據 voxel 融合後的標籤，分流出動態與靜態點
        moving_pts_np = v_pts[v_is_dyn]
        moving_cols_np = v_cols[v_is_dyn]
        moving_idx_np = v_idx[v_is_dyn]

        static_pts_np = v_pts[~v_is_dyn]
        static_cols_np = v_cols[~v_is_dyn]
        static_idx_np = v_idx[~v_is_dyn]

        # ==========================================
        # ★ 新增：為動態點加回 KMeans 空間群聚排序
        # 讓 MVTracker 的 Transformer 能完美捕捉局部運動特徵
        # ==========================================
        if len(moving_pts_np) > 0:
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
            moving_idx_np = moving_idx_np[sort_idx]
            print("   [空間分群] 排序完成，已為 MVTracker 最佳化 Batch 排列！")

        # ==========================================

        # 轉回 Torch Tensor 放進 GPU
        moving_pts_sampled = torch.from_numpy(moving_pts_np).to(device)
        moving_colors_sampled = torch.from_numpy(moving_cols_np).to(device)
        moving_original_idx = torch.from_numpy(moving_idx_np).to(device)
        
        static_pts_sampled = torch.from_numpy(static_pts_np).to(device)
        static_colors_sampled = torch.from_numpy(static_cols_np).to(device)
        static_original_idx = torch.from_numpy(static_idx_np).to(device)

        # 組合所有的點 (為了保持原本程式後半段能順利執行)
        all_pts_sampled = torch.cat([moving_pts_sampled, static_pts_sampled], dim=0)
        all_colors_sampled = torch.cat([moving_colors_sampled, static_colors_sampled], dim=0)
        all_original_idx = torch.cat([moving_original_idx, static_original_idx], dim=0)
            
        print(f"✅ 成功提取 {len(moving_pts_sampled)} 動態追蹤點, {len(static_pts_sampled)} 靜態軌跡點！ (總共 {len(all_pts_sampled)} 點)")
        # 匯出初始點雲 (.ply) 供 4DGS 初始化
        pts_np = all_pts_sampled.cpu().numpy()
        colors_np = all_colors_sampled.cpu().numpy()
        if colors_np.max() <= 1.0: 
            colors_np = (colors_np * 255).astype(np.uint8)
        times_np = np.full((len(pts_np), 1), float(t0), dtype=np.float32)

        ply_path = os.path.join(chunk_dir, "init_pointcloud_vggt_4d.ply")
        dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('time', 'f4')
        ]
        elements = np.empty(pts_np.shape[0], dtype=dtype)
        elements['x'], elements['y'], elements['z'] = pts_np[:, 0], pts_np[:, 1], pts_np[:, 2]
        elements['red'], elements['green'], elements['blue'] = colors_np[:, 0], colors_np[:, 1], colors_np[:, 2]
        elements['time'] = times_np[:, 0]
        PlyData([PlyElement.describe(elements, 'vertex')]).write(ply_path)
        print(f"🎉 初始點雲已儲存至 {ply_path}")

        # ------------------------------------------
        # 4.2 執行 MVTracker (動態點批次處理)
        # ------------------------------------------
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

        # ------------------------------------------
        # 4.5 漂移驗證 (Drift Analysis)
        # ------------------------------------------
        print("🔍 進行漂移驗證 (Drift Analysis)...")
        cam_idx = all_original_idx // (TARGET_H * TARGET_W)
        rem = all_original_idx % (TARGET_H * TARGET_W)
        y = rem // TARGET_W
        x = rem % TARGET_W

        N_pts = len(all_original_idx)
        vggt_tracks = torch.zeros((chunk_frames, N_pts, 3), device=device)
        depths_tensor_dev = depths_tensor[:, start_t:end_t].to(device)
        intrs_model_dev = intrs_model[:, start_t:end_t].to(device)
        extrs_model_dev = extrs_model[:, start_t:end_t].to(device)

        pixel_homo = torch.stack([x.float(), y.float(), torch.ones_like(x).float()], dim=-1).to(device)
        
        for t_rel in range(chunk_frames):
            d_t = depths_tensor_dev[cam_idx, t_rel, 0, y, x] 
            K_t = intrs_model_dev[cam_idx, t_rel] 
            E_t = extrs_model_dev[cam_idx, t_rel] 
            
            K_inv = torch.inverse(K_t) 
            cam_ray = torch.bmm(K_inv, pixel_homo.unsqueeze(-1)).squeeze(-1) 
            cam_xyz = cam_ray * d_t.unsqueeze(-1) 
            
            R = E_t[:, :3, :3] 
            Tr = E_t[:, :3, 3] 
            
            X_c_minus_Tr = cam_xyz - Tr
            R_inv = R.transpose(1, 2)
            world_xyz = torch.bmm(R_inv, X_c_minus_Tr.unsqueeze(-1)).squeeze(-1) 
            vggt_tracks[t_rel] = world_xyz

        pred_tracks_dev = pred_tracks.to(device)
        
        mv_drift = torch.norm(pred_tracks_dev - pred_tracks_dev[0:1], dim=-1) 
        vggt_drift = torch.norm(vggt_tracks - vggt_tracks[0:1], dim=-1) 

        mv_drift_mean = mv_drift.mean(dim=1).cpu().numpy()
        vggt_drift_mean = vggt_drift.mean(dim=1).cpu().numpy()
        
        plt.figure(figsize=(12, 6))
        plt.plot(range(start_t, end_t), mv_drift_mean, label='MVTracker Mean Drift', color='blue', linewidth=2)
        plt.plot(range(start_t, end_t), vggt_drift_mean, label='VGGT Mean Drift', color='orange', linewidth=2, linestyle='--')
        plt.xlabel('Absolute Frame')
        plt.ylabel(f'Mean Displacement from t={start_t} (World Units)')
        plt.title(f'Drift Analysis Over Time (Frames {start_t}-{end_t-1})')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(chunk_dir, "drift_analysis.png")
        plt.savefig(plot_path)
        plt.close()
        
        threshold = 0.5
        mv_bad_pct = (mv_drift[-1] > threshold).float().mean().item() * 100
        vggt_bad_pct = (vggt_drift[-1] > threshold).float().mean().item() * 100
        
        print(f"📊 漂移分析圖表已儲存至 {plot_path}")
        print(f"   - MVTracker 片段最後一幀平均漂移: {mv_drift_mean[-1]:.4f} 單位")
        print(f"   - VGGT      片段最後一幀平均漂移: {vggt_drift_mean[-1]:.4f} 單位")
        print(f"   - MVTracker > {threshold} 單位移動的點比例: {mv_bad_pct:.2f}%")
        print(f"   - VGGT      > {threshold} 單位移動的點比例: {vggt_bad_pct:.2f}%\n")
        print(f"\n🎉 全部漂移分析完成！總耗時: {full_start_time - time.time():.2f} 秒")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", message=".*DtypeTensor constructors are no longer.*", module="pointops.query")
    warnings.filterwarnings("ignore", message=".*Plan failed with a cudnnException.*", module="torch.nn.modules.conv")
    main()
