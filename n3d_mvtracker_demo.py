import argparse
import os
import re
import time
import warnings
import glob

import numpy as np
import rerun as rr
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from mvtracker.datasets.generic_scene_dataset import _ensure_vggt_aligned_cache_and_load,_ensure_vggt_raw_cache_and_load
from mvtracker.models.core.model_utils import init_pointcloud_from_rgbd
from mvtracker.utils.visualizer_rerun import log_pointclouds_to_rerun, log_tracks_to_rerun


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
    p.add_argument("--num_queries", type=int, default=512, help="要追蹤的點數量")
    p.add_argument("--use_dynamic_vggt_cameras", action="store_true", help="使用動態 VGGT 預測的內外參；若不加此參數，則將第1幀的內外參套用於所有後續幀")
    p.add_argument("--mask_img", type=str, default="mask.jpg", help="提供白色像素標註追蹤點的遮罩圖片路徑 (例如: mask.png)")
    p.add_argument("--mode", type=str, choices=["viz", "full", "both", "prove"], default="viz", help="運行模式: viz(視覺化), full(抽20萬點測速), both(測速並視覺化), prove(證明點間干擾)")
    p.add_argument("--full_num_queries", type=int, default=200000, help="正式測試模式要追蹤的總點數量")
    p.add_argument("--rerun", choices=["save", "spawn", "stream"], default="save")
    p.add_argument("--lightweight", action="store_true", help="使用輕量級視覺化 (適合網頁版 Viewer)")
    p.add_argument("--rrd", default="n3d_mvtracker_demo.rrd", help="輸出的 Rerun 檔名")
    p.add_argument("--start_frame", type=int, default=0, help="開始追蹤的幀數 (設定 t0)")
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

    SELECTED_CAMS = [16, 10, 13, 1, 18]
    V = len(SELECTED_CAMS)
    
    # 建構 Extrinsics 與 Intrinsics，維度擴展為 [V, T, ...]
    TARGET_W, TARGET_H = 768, 576
    scale_x = TARGET_W / W_img
    scale_y = TARGET_H / H_img
    extrs_list, intrs_list = [], []
    for cam_idx in SELECTED_CAMS:
        extrs_list.append(torch.from_numpy(all_w2c_np[cam_idx]).float())
        K_original = all_intrs_np[cam_idx].copy()
        K_original[0, :] *= scale_x  # 縮放 fx, cx
        K_original[1, :] *= scale_y  # 縮放 fy, cy
        intrs_list.append(torch.from_numpy(K_original).float())
        
    # extrs_gt shape: [V, T, 3, 4]
    extrs_gt = torch.stack(extrs_list).unsqueeze(1).repeat(1, num_frames, 1, 1)
    # intrs_gt shape: [V, T, 3, 3]
    intrs_gt = torch.stack(intrs_list).unsqueeze(1).repeat(1, num_frames, 1, 1)

    # ==========================================
    # 2. 收集所有幀的 RGB 與 VGGT 深度
    # ==========================================
    print("🚀 讀取 RGB 影像並整理資料...")
    rgbs_list = []

    for cam_idx in tqdm(SELECTED_CAMS, desc="RGBD 讀取進度"):
        frames = sorted(glob.glob(os.path.join(view_dirs[cam_idx], '*.jpg')))[:num_frames]
        cam_rgbs = []
        for t in range(num_frames):
            img = Image.open(frames[t]).convert('RGB')
            img_resized = img.resize((TARGET_W, TARGET_H), Image.Resampling.BILINEAR)
            cam_rgbs.append((to_tensor(img_resized) * 255).byte()) # [3, H, W] uint8
        rgbs_list.append(torch.stack(cam_rgbs)) # [T, 3, H, W]
    
    rgbs_tensor = torch.stack(rgbs_list) # [V, T, 3, H, W]

    if args.use_dynamic_vggt_cameras:
        print("🚀 提取 VGGT 深度並進行全部外參對齊...")
        with torch.no_grad():
            depths_tensor, _, intrs_model, extrs_model = _ensure_vggt_aligned_cache_and_load(
                rgbs=rgbs_tensor,
                seq_name="n3d_gt_init_aligned",
                dataset_root=args.dir,
                extrs_gt=extrs_gt,
                vggt_cache_subdir="vggt_cache",
                skip_if_cached=False,
                model_id="facebook/VGGT-1B",
            )
    else:
        print("🚀 提取 VGGT Raw 深度並將第1幀內外參套用至所有後續幀...")
        with torch.no_grad():
            depths_tensor, _, intrs_vggt, extrs_vggt = _ensure_vggt_aligned_cache_and_load(
                rgbs=rgbs_tensor,
                seq_name="n3d_gt_init_raw",
                dataset_root=args.dir,
                extrs_gt=extrs_gt,
                vggt_cache_subdir="vggt_cache",
                skip_if_cached=False,
                model_id="facebook/VGGT-1B",
            )

        intrs_model = intrs_vggt[:, 0:1].expand_as(intrs_vggt).clone()
        extrs_model = extrs_vggt[:, 0:1].expand_as(extrs_vggt).clone()

    # ==========================================
    # 3. 生成隨機 Query Points (於 t0)
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
        pts_viz = xyz_viz[0]  # [H*W, 3] at t=t0
        
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
        pts_full = xyz_full[0]  # [V*H*W, 3] at t=t0
        valid_mask = depths_tensor[:, t0, 0].flatten() > 0
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
        pool = pts_full[valid_mask]
        assert pool.shape[0] > 0, "沒有找到有效的深度點來產生 Queries！"
        
        if pool.shape[0] > args.full_num_queries:
            rand_idx = torch.randperm(pool.shape[0])[:args.full_num_queries]
            full_pts_sampled = pool[rand_idx]
            chosen_idx = valid_indices[rand_idx]
        else:
            full_pts_sampled = pool
            chosen_idx = valid_indices
            
        full_pixel_coords = torch.stack([
            chosen_idx // (TARGET_H * TARGET_W), 
            (chosen_idx % (TARGET_H * TARGET_W)) // TARGET_W, 
            chosen_idx % TARGET_W 
        ], dim=1)
        
        print(f"✅ 成功提取 {len(full_pts_sampled)} 個全景點雲！")

    if args.mode == "viz":
        pts_sampled = viz_pts_sampled
        pixel_coords = viz_pixel_coords
    elif args.mode in ["full", "prove"]:
        pts_sampled = full_pts_sampled
        pixel_coords = full_pixel_coords
    else:
        # both: 將 viz (mask) 和 full 合併追蹤
        len_viz = len(viz_pts_sampled)
        len_full = len(full_pts_sampled)
        pts_sampled = torch.cat([viz_pts_sampled, full_pts_sampled], dim=0)
        pixel_coords = torch.cat([viz_pixel_coords, full_pixel_coords], dim=0)
    
    ts = torch.full((pts_sampled.shape[0], 1), float(t0), device=pts_sampled.device)
    query_points = torch.cat([ts, pts_sampled], dim=1).float()  # (N,4): (t,x,y,z)

    # ==========================================
    # 4. 載入並執行 MVTracker
    # ==========================================
    print("🧠 載入 MVTracker 並開始追蹤...")
    mvtracker = torch.hub.load(".", "mvtracker", source="local", pretrained=True, device=device)
    
    torch.set_float32_matmul_precision("high")
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    
    if args.mode == "prove":
        print("\n" + "="*50)
        print("🎯 [PROVE 模式] 證明點間干擾 (Interference Proof)")
        print("="*50)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=amp_dtype):
            print("🚀 [1/2] 單獨追蹤第 0 個目標點...")
            res_single = mvtracker(
                rgbs=rgbs_tensor[None].cpu(),
                depths=depths_tensor[None].cpu(),
                intrs=intrs_model[None].to(device),
                extrs=extrs_model[None].to(device),
                query_points_3d=query_points[None, 0:1].to(device), # Only point 0
            )
            print(f"🚀 [2/2] 與其他 {args.full_num_queries-1} 個點一起追蹤...")
            res_group = mvtracker(
                rgbs=rgbs_tensor[None].cpu(),
                depths=depths_tensor[None].cpu(),
                intrs=intrs_model[None].to(device),
                extrs=extrs_model[None].to(device),
                query_points_3d=query_points[None].to(device), # All points
            )
            
        track_single = res_single["traj_e"].cpu()
        if track_single.dim() == 4:
            track_single = track_single[0] # [T, N, 3]
        track_single = track_single[:, 0, :] # [T, 3]
        
        track_group_target = res_group["traj_e"].cpu()
        if track_group_target.dim() == 4:
            track_group_target = track_group_target[0] # [T, N, 3]
        track_group_target = track_group_target[:, 0, :] # [T, 3]
        
        diff = torch.norm(track_single - track_group_target, dim=-1) # [T]
        print("\n📊 結論: 同一個目標點在「單獨追蹤」vs「群體追蹤」的位移差異 (Euclidean Error):")
        for t in range(diff.shape[0]):
            print(f"Frame {t}: 相差 {diff[t].item():.6f} 單位")
            
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(diff.numpy(), label='Deviation caused by other queries', color='red', marker='o')
        plt.xlabel('Frame')
        plt.ylabel('Deviation Error (World Units)')
        plt.title('Target Point Trajectory Deviation (Single vs Group)')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(args.dir, "interference_proof.png")
        plt.savefig(plot_path)
        print(f"✅ 干擾證明圖表已儲存至 {plot_path}")
        return

    start_time = time.time()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=amp_dtype):
        # 這裡需要加上 [None] batch 維度，並把 RGB 縮放到 0~1
        results = mvtracker(
            rgbs=rgbs_tensor[None].cpu(), # 直接傳遞 uint8 給 predictor 內部，省下大量記憶體
            depths=depths_tensor[None].cpu(),
            intrs=intrs_model[None].to(device),
            extrs=extrs_model[None].to(device),
            query_points_3d=query_points[None].to(device),
        )
    end_time = time.time()
    pred_tracks = results["traj_e"].cpu()  # [T,N,3]
    pred_vis = results["vis_e"].cpu()      # [T,N]
    
    print(f"✅ 追蹤完成！ (花費時間: {end_time - start_time:.2f} 秒)")

    # ==========================================
    # 5. 視覺化 (匯出至 Rerun)
    # ==========================================
    if args.mode == "full":
        print("⏭️  [FULL 模式] 略過視覺化，測試結束。")
        return

    if args.mode == "both":
        print(f"🎲 [BOTH 模式] 從測速的 {len_full} 個點中隨機抽樣 {args.num_queries} 個點，再加上 {len_viz} 個標註點進行視覺化...")
        
        # 標註點的 indices (前 len_viz 個)
        idx_viz = torch.arange(len_viz)
        
        # 從 full random 取樣的 points 中隨機挑 args.num_queries 個
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
    if args.rerun == "stream":
        rr.connect_tcp()
    elif args.rerun == "spawn":
        rr.spawn()
        
    gt_centers_all = -extrs_model[:, 0, :3, :3].transpose(-1, -2) @ extrs_model[:, 0, :3, 3].unsqueeze(-1)
    scene_center = gt_centers_all.mean(dim=0).flatten().cpu().numpy()

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
        sphere_radius_crop=None,  # 可視需求調整視野裁切
        sphere_center_crop=scene_center,
        log_rgb_image=False,
        log_depthmap_as_image_v1=False,
        log_depthmap_as_image_v2=False,
        log_camera_frustrum=True,
        log_rgb_pointcloud=True,
    )
    
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
        rr.save(args.rrd)
        print(f"🎉 成功儲存 Rerun 檔案至: {os.path.abspath(args.rrd)}")

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
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(mv_drift_mean, label='MVTracker Mean Drift', color='blue', linewidth=2)
    plt.plot(vggt_drift_mean, label='VGGT Mean Drift', color='orange', linewidth=2, linestyle='--')
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
    print(f"   - VGGT      最後一幀平均漂移: {vggt_drift_mean[-1]:.4f} 單位")
    print(f"   - MVTracker > {threshold} 單位移動的點比例: {mv_bad_pct:.2f}%")
    print(f"   - VGGT      > {threshold} 單位移動的點比例: {vggt_bad_pct:.2f}%")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*DtypeTensor constructors are no longer.*", module="pointops.query")
    warnings.filterwarnings("ignore", message=".*Plan failed with a cudnnException.*", module="torch.nn.modules.conv")
    main()