import argparse
import os
import cv2
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=str, required=True, help="bullpen 資料集路徑")
    p.add_argument("--max_frames", type=int, default=300, help="限制載入的最大幀數 (避免 MVTracker OOM)")
    p.add_argument("--num_queries", type=int, default=512, help="要追蹤的點數量")
    p.add_argument("--use_dynamic_vggt_cameras", action="store_true", help="使用動態 VGGT 預測的內外參；若不加此參數，則將第1幀的內外參套用於所有後續幀")
    p.add_argument("--mask_img", type=str, default="mask.jpg", help="提供白色像素標註追蹤點的遮罩圖片路徑 (例如: mask.png)")
    p.add_argument("--rerun", choices=["save", "spawn", "stream"], default="save")
    p.add_argument("--lightweight", action="store_true", help="使用輕量級視覺化 (適合網頁版 Viewer)")
    p.add_argument("--rrd", default="bullpen_mvtracker_demo.rrd", help="輸出的 Rerun 檔名")
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

    intri_path = os.path.join(args.dir, "intri.yml")
    extri_path = os.path.join(args.dir, "extri.yml")
    
    intri_fs = cv2.FileStorage(intri_path, cv2.FILE_STORAGE_READ)
    extri_fs = cv2.FileStorage(extri_path, cv2.FILE_STORAGE_READ)
    
    SELECTED_CAMS = [0, 1, 2, 3] # view_0 to view_3 correspond to cam1 to cam4
    V = len(SELECTED_CAMS)
    cam_names = [f"cam{i+1}" for i in SELECTED_CAMS]
    
    TARGET_W, TARGET_H = 768, 576
    
    extrs_list, intrs_list = [], []
    for cam in cam_names:
        intr = intri_fs.getNode(f"K_{cam}").mat().astype(np.float32)
        R = extri_fs.getNode(f"Rot_{cam}").mat().astype(np.float32)
        T = extri_fs.getNode(f"T_{cam}").mat().astype(np.float32).reshape(3)
        w2c = np.concatenate([R, T[:, None]], axis=1)
        
        orig_W = intr[0, 2] * 2
        orig_H = intr[1, 2] * 2
        
        scale_x = TARGET_W / orig_W
        scale_y = TARGET_H / orig_H
        
        K_original = intr.copy()
        K_original[0, :] *= scale_x
        K_original[1, :] *= scale_y
        
        extrs_list.append(torch.from_numpy(w2c).float())
        intrs_list.append(torch.from_numpy(K_original).float())
        
    intri_fs.release()
    extri_fs.release()
        
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
                seq_name="bullpen_gt_init_aligned",
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
                seq_name="bullpen_gt_init_raw",
                dataset_root=args.dir,
                extrs_gt=extrs_gt,
                vggt_cache_subdir="vggt_cache",
                skip_if_cached=False,
                model_id="facebook/VGGT-1B",
            )

        intrs_model = intrs_vggt[:, 0:1].expand_as(intrs_vggt).clone()
        extrs_model = extrs_vggt[:, 0:1].expand_as(extrs_vggt).clone()

    # ==========================================
    # 3. 生成隨機 Query Points (於 t=0)
    # ==========================================
    print(f"🎯 從 t=0 幀深度圖取樣 {args.num_queries} 個追蹤點...")
    t0 = 0
    xyz, _ = init_pointcloud_from_rgbd(
        fmaps=rgbs_tensor[0:1, t0:t0+1].unsqueeze(0).float(),
        depths=depths_tensor[0:1, t0:t0+1].unsqueeze(0),
        intrs=intrs_model[0:1, t0:t0+1].unsqueeze(0),
        extrs=extrs_model[0:1, t0:t0+1].unsqueeze(0),
        stride=1,
        level=0,
    )
    pts = xyz[t0]  # [H*W, 3] at t=0
    
    if args.mask_img and os.path.exists(args.mask_img):
        print(f"🖼️ 讀取選點遮罩: {args.mask_img}")
        mask_pil = Image.open(args.mask_img).convert('L')
        # 縮放到與追蹤時相同的解析度
        mask_resized = mask_pil.resize((TARGET_W, TARGET_H), Image.Resampling.BILINEAR)
        mask_np = np.array(mask_resized)
        
        # 找到圖中帶有白色標註的地方 (亮度大於 200)
        ys, xs = np.where(mask_np > 250)
        
        indices = []
        for y, x in zip(ys, xs):
            indices.append(y * TARGET_W + x)
            
        if len(indices) == 0:
            print("⚠️ 遮罩中未找到明顯的白色像素標註，將改回隨機取樣。")
            valid_mask = depths_tensor[0:1, t0, 0].flatten() > 0
            pool = pts[valid_mask]
            pts_sampled = pool[torch.randperm(pool.shape[0])[:args.num_queries]]
        else:
            # 去除重複點並直接取樣，也可以視情況做抽樣控制點數
            indices = list(set(indices))
            pool = pts[indices]
            
            # 若標註的數量超過設定的查詢數量上限，做隨機降採樣
            if pool.shape[0] > args.num_queries:
                idx = torch.randperm(pool.shape[0])[:args.num_queries]
                pts_sampled = pool[idx]
            else:
                pts_sampled = pool
                
            print(f"✅ 成功從圖片中讀取 {len(pts_sampled)} 個手動標註點！")
    else:
        valid_mask = depths_tensor[0:1, t0, 0].flatten() > 0
        pool = pts[valid_mask]
        assert pool.shape[0] > 0, "沒有找到有效的深度點來產生 Queries！"
        idx = torch.randperm(pool.shape[0])[:args.num_queries]
        pts_sampled = pool[idx]
    
    ts = torch.full((pts_sampled.shape[0], 1), float(t0), device=pts_sampled.device)
    query_points = torch.cat([ts, pts_sampled], dim=1).float()  # (N,4): (t,x,y,z)

    # ==========================================
    # 4. 載入並執行 MVTracker
    # ==========================================
    print("🧠 載入 MVTracker 並開始追蹤...")
    mvtracker = torch.hub.load(".", "mvtracker", source="local", pretrained=True, device=device)
    
    torch.set_float32_matmul_precision("high")
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=amp_dtype):
        # 這裡需要加上 [None] batch 維度，並把 RGB 縮放到 0~1
        results = mvtracker(
            rgbs=rgbs_tensor[None].cpu(), # 直接傳遞 uint8 給 predictor 內部，省下大量記憶體
            depths=depths_tensor[None].cpu(),
            intrs=intrs_model[None].to(device),
            extrs=extrs_model[None].to(device),
            query_points_3d=query_points[None].to(device),
        )
    pred_tracks = results["traj_e"].cpu()  # [T,N,3]
    pred_vis = results["vis_e"].cpu()      # [T,N]
    print("✅ 追蹤完成！")

    # ==========================================
    # 5. 視覺化 (匯出至 Rerun)
    # ==========================================
    print("🎨 正在打包 Rerun 視覺化檔案...")
    rr.init("bullpen_tracking", recording_id="v0.16")
    if args.rerun == "stream":
        rr.connect_tcp()
    elif args.rerun == "spawn":
        rr.spawn()
        
    gt_centers_all = -extrs_model[:, 0, :3, :3].transpose(-1, -2) @ extrs_model[:, 0, :3, 3].unsqueeze(-1)
    scene_center = gt_centers_all.mean(dim=0).flatten().cpu().numpy()

    log_pointclouds_to_rerun(
        dataset_name="bullpen",
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
        dataset_name="bullpen",
        datapoint_idx=0,
        predictor_name="MVTracker",
        gt_trajectories_3d_worldspace=None,
        gt_visibilities_any_view=None,
        query_points_3d=query_points[None],
        pred_trajectories=pred_tracks,
        pred_visibilities=pred_vis,
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

if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*DtypeTensor constructors are no longer.*", module="pointops.query")
    warnings.filterwarnings("ignore", message=".*Plan failed with a cudnnException.*", module="torch.nn.modules.conv")
    main()