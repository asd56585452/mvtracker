import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import re
import time
import glob
import gc

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
from plyfile import PlyData, PlyElement

from mvtracker.datasets.generic_scene_dataset import _ensure_vggt_aligned_cache_and_load
from mvtracker.models.core.model_utils import init_pointcloud_from_rgbd

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
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=str, required=True, help="n3d 資料集路徑")
    p.add_argument("--max_frames", type=int, default=300, help="限制載入的最大幀數 (避免 MVTracker OOM)")
    p.add_argument("--num_queries", type=int, default=200000, help="要追蹤的點數量")
    p.add_argument("--selected_cams", type=int, nargs="+", default=[1, 6, 10, 14, 20], help="選擇的相機 ID 列表")
    p.add_argument("--track_chunk_size", type=int, default=10000, help="MVTracker 追蹤時的分批大小，避免 OOM")
    p.add_argument("--use_dynamic_vggt_cameras", action="store_true", help="使用動態 VGGT 預測的內外參；若不加此參數，則將第1幀的內外參套用於所有後續幀")
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
    # 3. 生成隨機 Query Points (於 t=0)
    # ==========================================
    t0 = 0
    print(f"🎯 從 t=0 幀「全部相機」萃取並隨機取樣 {args.num_queries} 個追蹤點...")
    with torch.no_grad():
        xyz_full, rgb_full = init_pointcloud_from_rgbd(
            fmaps=rgbs_tensor[:, t0:t0+1].unsqueeze(0).float(),
            depths=depths_tensor[:, t0:t0+1].unsqueeze(0),
            intrs=intrs_model[:, t0:t0+1].unsqueeze(0),
            extrs=extrs_model[:, t0:t0+1].unsqueeze(0),
            stride=1,
            level=0,
        )
    
    pts_full = xyz_full[t0]  # [V*H*W, 3] at t=0
    colors_full = rgb_full[t0] # [V*H*W, 3] at t=0
    
    valid_mask = depths_tensor[:, t0, 0].flatten() > 0
    pool_pts = pts_full[valid_mask]
    pool_colors = colors_full[valid_mask]
    assert pool_pts.shape[0] > 0, "沒有找到有效的深度點來產生 Queries！"
    
    if pool_pts.shape[0] > args.num_queries:
        idx = torch.randperm(pool_pts.shape[0])[:args.num_queries]
        pts_sampled = pool_pts[idx]
        colors_sampled = pool_colors[idx]
    else:
        pts_sampled = pool_pts
        colors_sampled = pool_colors
        
    print(f"✅ 成功提取 {len(pts_sampled)} 個初始點雲！")

    # 匯出初始點雲 (.ply) 供 4DGS 初始化
    pts_np = pts_sampled.cpu().numpy()
    colors_np = colors_sampled.cpu().numpy()
    if colors_np.max() <= 1.0: 
        colors_np = (colors_np * 255).astype(np.uint8)
    times_np = np.full((len(pts_np), 1), float(t0), dtype=np.float32)

    ply_path = os.path.join(args.dir, "init_pointcloud_vggt_4d.ply")
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

    # ==========================================
    # 4. 載入並執行 MVTracker (批次處理)
    # ==========================================
    ts = torch.full((pts_sampled.shape[0], 1), float(t0), device=pts_sampled.device)
    query_points = torch.cat([ts, pts_sampled], dim=1).float()  # (N,4): (t,x,y,z)

    print("🧠 載入 MVTracker...")
    mvtracker = torch.hub.load(".", "mvtracker", source="local", pretrained=True, device=device)
    torch.set_float32_matmul_precision("high")
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    
    all_pred_tracks = []
    all_pred_vis = []
    
    total_queries = query_points.shape[0]
    chunk_size = args.track_chunk_size
    
    print(f"🚀 開始追蹤 {total_queries} 個點，分批大小為 {chunk_size}...")
    start_time = time.time()
    
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=amp_dtype):
        # 預先將環境變數移至 GPU / CPU 節省轉移時間
        rgbs_input = rgbs_tensor[None].cpu()
        depths_input = depths_tensor[None].cpu()
        intrs_input = intrs_model[None].to(device)
        extrs_input = extrs_model[None].to(device)
        
        for i in tqdm(range(0, total_queries, chunk_size), desc="MVTracker 追蹤進度"):
            batch_queries = query_points[i:i+chunk_size].to(device)
            results = mvtracker(
                rgbs=rgbs_input,
                depths=depths_input,
                intrs=intrs_input,
                extrs=extrs_input,
                query_points_3d=batch_queries[None],
            )
            all_pred_tracks.append(results["traj_e"].cpu())  # [1, T, N_chunk, 3]
            all_pred_vis.append(results["vis_e"].cpu())      # [1, T, N_chunk]
            
            # 清理快取
            del results
            torch.cuda.empty_cache()

    end_time = time.time()
    
    # 組合結果，shape 為 [T, N, 3] 與 [T, N]
    pred_tracks = torch.cat(all_pred_tracks, dim=2)[0]
    pred_vis = torch.cat(all_pred_vis, dim=2)[0]
    
    print(f"✅ 追蹤完成！ (花費時間: {end_time - start_time:.2f} 秒)")

    # ==========================================
    # 5. 儲存軌跡
    # ==========================================
    tracks_path = os.path.join(args.dir, "mvtracker_tracks.npz")
    np.savez(
        tracks_path,
        query_points=query_points.cpu().numpy(),
        pred_tracks=pred_tracks.numpy(),
        pred_vis=pred_vis.numpy(),
    )
    print(f"🎉 軌跡已儲存至 {tracks_path}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", message=".*DtypeTensor constructors are no longer.*", module="pointops.query")
    warnings.filterwarnings("ignore", message=".*Plan failed with a cudnnException.*", module="torch.nn.modules.conv")
    main()
