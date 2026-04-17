import sys
import os
import re
import time
import gc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import torch
import numpy as np
from plyfile import PlyData, PlyElement
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from mvtracker.datasets.generic_scene_dataset import _ensure_vggt_raw_cache_and_load
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

# 👇 新增了 frame_step 參數
def main(dataset_dir, target_size=392, frames_chunk_size=1, frame_step=2):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    view_dirs = sorted(
        glob.glob(os.path.join(dataset_dir, 'view_*')), 
        key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group())
    )
    
    sample_frames = sorted(glob.glob(os.path.join(view_dirs[0], '*.jpg')))
    num_frames = len(sample_frames)
    sample_img = Image.open(sample_frames[0])
    W_img, H_img = sample_img.size
    print(f"📷 實際圖片: {W_img}x{H_img} | 總幀數: {num_frames} | 跳幀步長: {frame_step}")

    # ==========================================
    # 1. 讀取 poses_bounds 並匯出相機 NPZ
    # ==========================================
    poses_path = os.path.join(dataset_dir, 'poses_bounds.npy')
    assert os.path.exists(poses_path), f"找不到 {poses_path}"
    poses_bounds = np.load(poses_path)
    all_poses_llff = poses_bounds[:, :-2].reshape([-1, 3, 5])
    
    all_w2c, all_intrs = [], []
    for i in range(len(all_poses_llff)):
        w2c, K = llff_to_opencv_w2c(all_poses_llff[i], W_img, H_img)
        all_w2c.append(w2c)
        all_intrs.append(K)
        
    all_w2c_np = np.stack(all_w2c)
    all_intrs_np = np.stack(all_intrs)
    
    npz_path = os.path.join(dataset_dir, "estimated_cameras_vggt.npz")
    np.savez(npz_path, w2c_poses=all_w2c_np, intrs=all_intrs_np)

    # ==========================================
    # 2. 手動篩選相機
    # ==========================================
    SELECTED_CAMS = [1,6,10,14,20]
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
        
    extrs_gt = torch.stack(extrs_list).unsqueeze(1).to(device) 
    intrs_gt = torch.stack(intrs_list).unsqueeze(1).to(device) 

    def get_centers(w2c_tensor):
        homo = torch.tensor([[[0,0,0,1]]]).expand(w2c_tensor.shape[0], -1, -1).to(w2c_tensor.device)
        c2w = torch.inverse(torch.cat([w2c_tensor, homo], dim=1))
        return c2w[:, :3, 3]
    gt_centers = get_centers(extrs_gt[:, 0])

    all_pts, all_colors, all_times = [], [], []

    # ==========================================
    # 3. 逐幀提取 (加入跳幀邏輯)
    # ==========================================
    print(f"🚀 開始提取 4D 初始點雲 (每 {frame_step} 幀取樣一次)...")
    
    # 👇 修改這裡的迴圈，加入 frame_step 作為 step 參數
    for t in tqdm(range(0, num_frames, frame_step), desc="4D 點雲生成進度"):
        t0 = time.time()
        
        rgbs_list = []
        for cam_idx in SELECTED_CAMS:
            frames = sorted(glob.glob(os.path.join(view_dirs[cam_idx], '*.jpg')))
            img = Image.open(frames[t]).convert('RGB')
            img_resized = img.resize((TARGET_W, TARGET_H), Image.Resampling.BILINEAR)
            rgbs_list.append((to_tensor(img_resized) * 255).byte()) # [3, H, W] uint8
        
        rgbs = torch.stack(rgbs_list).unsqueeze(1) 
        V, T_vggt, C, H, W = rgbs.shape
        t1 = time.time()
        
        with torch.no_grad():
            depths_raw, _, _, extrs_vggt = _ensure_vggt_raw_cache_and_load(
                rgbs=rgbs, 
                seq_name=f"n3d_gt_init_f{t:04d}", 
                dataset_root=dataset_dir,
                vggt_cache_subdir="vggt_cache", 
                skip_if_cached=False, 
                model_id="facebook/VGGT-1B", 
                target=target_size, 
                frames_chunk_size=frames_chunk_size,
            )
        t2 = time.time()

        vggt_centers = get_centers(extrs_vggt[:, 0].to(device))
        scale_factor = (torch.pdist(gt_centers).mean() / torch.pdist(vggt_centers).mean()).item()
        depths_metric = depths_raw.to(device) * scale_factor

        xyz, rgb = init_pointcloud_from_rgbd(
            fmaps=rgbs.to(device).unsqueeze(0).float(), 
            depths=depths_metric.unsqueeze(0),
            intrs=intrs_gt.unsqueeze(0),
            extrs=extrs_gt.unsqueeze(0),
            stride=1, 
            level=0
        )
        t3 = time.time()
        
        pts, colors = xyz[0].cpu().numpy(), rgb[0].cpu().numpy()
        depths_flat = depths_metric[:, 0, 0].flatten().cpu().numpy()
        
        valid_mask = depths_flat > 0
        pts_clean, colors_clean = pts[valid_mask], colors[valid_mask]
        
        if colors_clean.max() <= 1.0: 
            colors_clean = (colors_clean * 255).astype(np.uint8)

        # ==========================================
        # 💡 救命優化 1：單幀立刻降採樣 (不要等最後才做)
        # ==========================================
        # 計算這一幀應該要貢獻多少點 (總共 20 萬點 / 總處理幀數)
        total_processed_frames = len(range(0, num_frames, frame_step))
        target_pts_per_frame = 200000 // total_processed_frames
        
        if len(pts_clean) > target_pts_per_frame:
            indices = np.random.choice(len(pts_clean), target_pts_per_frame, replace=False)
            pts_clean = pts_clean[indices]
            colors_clean = colors_clean[indices]

        # 現在塞進陣列的點數已經非常少了，RAM 絕對不會爆
        all_pts.append(pts_clean)
        all_colors.append(colors_clean)
        all_times.append(np.full((len(pts_clean), 1), t, dtype=np.float32))

        t4 = time.time()
        print(f"\n[Frame {t}] 總耗時:{t4-t0:.2f}s (VGGT:{t2-t1:.2f}s)")

        # ==========================================
        # 💡 救命優化 2：殘酷無情的強制記憶體回收
        # ==========================================
        # 刪除這回合佔用空間的巨大 Tensor 與變數
        del rgbs, depths_raw, extrs_vggt, depths_metric, xyz, rgb, pts, colors, valid_mask
        
        # 強制 PyTorch 清空 VRAM 快取
        torch.cuda.empty_cache()
        
        # 強制 Python 清空 RAM 垃圾
        gc.collect()

    # ==========================================
    # 4. 全域降採樣與匯出
    # ==========================================
    print("整合所有時間步的點雲...")
    final_pts = np.concatenate(all_pts, axis=0)
    final_colors = np.concatenate(all_colors, axis=0)
    final_times = np.concatenate(all_times, axis=0)
    
    total_points = len(final_pts)
    print(f"✅ 成功萃取總點數: {total_points} 點")

    max_points = 200000
    if total_points > max_points:
        print(f"正在全域降採樣至 {max_points} 點...")
        indices = np.random.choice(total_points, max_points, replace=False)
        final_pts = final_pts[indices]
        final_colors = final_colors[indices]
        final_times = final_times[indices]

    ply_path = os.path.join(dataset_dir, "init_pointcloud_vggt_4d.ply")
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('time', 'f4')
    ]
    
    elements = np.empty(final_pts.shape[0], dtype=dtype)
    elements['x'], elements['y'], elements['z'] = final_pts[:, 0], final_pts[:, 1], final_pts[:, 2]
    elements['red'], elements['green'], elements['blue'] = final_colors[:, 0], final_colors[:, 1], final_colors[:, 2]
    elements['time'] = final_times[:, 0]
    
    PlyData([PlyElement.describe(elements, 'vertex')]).write(ply_path)
    print(f"🎉 完美的 4D 初始點雲已儲存至 {ply_path} (共 {len(final_pts)} 個點)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="資料集路徑")
    parser.add_argument("--vggt_target_size", type=int, default=392, help="VGGT 解析度")
    parser.add_argument("--vggt_chunk_size", type=int, default=1, help="解碼並發數")
    
    # 👇 將參數暴露給終端機
    parser.add_argument("--frame_step", type=int, default=2, help="跳幀步長 (1=每幀算, 2=每隔1幀, 3=每隔2幀...)")
    
    args = parser.parse_args()
    main(args.dir, args.vggt_target_size, args.vggt_chunk_size, args.frame_step)