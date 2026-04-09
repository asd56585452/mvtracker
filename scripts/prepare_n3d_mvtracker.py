import sys
import os
import re # 新增正則表達式，解決排序問題
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import torch
import numpy as np
from plyfile import PlyData, PlyElement
from PIL import Image
from torchvision.transforms.functional import to_tensor

from mvtracker.datasets.generic_scene_dataset import _ensure_vggt_raw_cache_and_load
from mvtracker.models.core.model_utils import init_pointcloud_from_rgbd

def llff_to_opencv_w2c(pose_llff, actual_W, actual_H):
    """
    將 LLFF 矩陣轉換為 OpenCV w2c，並【自動根據實際圖片大小縮放內參】
    """
    H_bound, W_bound, f_bound = pose_llff[:, 4]
    
    # 計算解析度縮放比例 (這步超級關鍵！)
    scale_x = actual_W / W_bound
    scale_y = actual_H / H_bound
    
    K = np.array([
        [f_bound * scale_x, 0, actual_W / 2.0],
        [0, f_bound * scale_y, actual_H / 2.0],
        [0, 0, 1]
    ], dtype=np.float32)

    R_llff = pose_llff[:, :3]
    t_llff = pose_llff[:, 3]

    # LLFF [Down, Right, Backwards] -> OpenCV [Right, Down, Forward]
    R_cv = np.zeros_like(R_llff)
    R_cv[:, 0] = R_llff[:, 1]
    R_cv[:, 1] = R_llff[:, 0]
    R_cv[:, 2] = -R_llff[:, 2]

    c2w_cv = np.eye(4, dtype=np.float32)
    c2w_cv[:3, :3] = R_cv
    c2w_cv[:3, 3] = t_llff

    w2c_cv = np.linalg.inv(c2w_cv)[:3, :4]
    return w2c_cv, K

def main(dataset_dir, target_size=392, frames_chunk_size=1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 💡 修正 1：確保 view_dirs 是按照數字大小排序，而不是字母排序
    view_dirs = sorted(
        glob.glob(os.path.join(dataset_dir, 'view_*')), 
        key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group())
    )
    
    # 💡 修正 2：先讀取第一張圖片，取得真實解析度
    sample_img_path = glob.glob(os.path.join(view_dirs[0], '*.jpg'))[0]
    sample_img = Image.open(sample_img_path)
    W_img, H_img = sample_img.size
    print(f"📷 偵測到實際圖片解析度為: {W_img}x{H_img}")

    # ==========================================
    # 1. 讀取 poses_bounds.npy 並匯出「所有」相機的 NPZ
    # ==========================================
    poses_path = os.path.join(dataset_dir, 'poses_bounds.npy')
    assert os.path.exists(poses_path), f"找不到 {poses_path}"
    
    poses_bounds = np.load(poses_path)
    all_poses_llff = poses_bounds[:, :-2].reshape([-1, 3, 5])
    num_all_cams = len(all_poses_llff)
    
    all_w2c, all_intrs = [], []
    for i in range(num_all_cams):
        # 傳入實際寬高，動態修正焦距
        w2c, K = llff_to_opencv_w2c(all_poses_llff[i], W_img, H_img)
        all_w2c.append(w2c)
        all_intrs.append(K)
        
    all_w2c_np = np.stack(all_w2c)    # [V, 3, 4]
    all_intrs_np = np.stack(all_intrs) # [V, 3, 3]
    
    npz_path = os.path.join(dataset_dir, "estimated_cameras_vggt.npz")
    np.savez(npz_path, w2c_poses=all_w2c_np, intrs=all_intrs_np)

    # ==========================================
    # 2. 篩選相機並準備資料
    # ==========================================
    SELECTED_CAMS = [0, 3, 6, 9, 12, 14, 17, 20]
    
    rgbs_list, extrs_list, intrs_list = [], [], []
    for cam_idx in SELECTED_CAMS:
        frames = sorted(glob.glob(os.path.join(view_dirs[cam_idx], '*.jpg')))
        img = Image.open(frames[0]).convert('RGB')
        
        rgbs_list.append((to_tensor(img) * 255).byte())
        extrs_list.append(torch.from_numpy(all_w2c_np[cam_idx]).float())
        intrs_list.append(torch.from_numpy(all_intrs_np[cam_idx]).float())

    rgbs = torch.stack(rgbs_list).unsqueeze(1)    # [V, 1, 3, H, W]
    extrs_gt = torch.stack(extrs_list).unsqueeze(1) # [V, 1, 3, 4]
    intrs_gt = torch.stack(intrs_list).unsqueeze(1) # [V, 1, 3, 3]

    V, T, C, H, W = rgbs.shape

    # ==========================================
    # 3. 執行 VGGT (提取深度) 並縮放至真實尺度
    # ==========================================
    print("啟動 VGGT 提取深度圖...")
    with torch.no_grad():
        depths_raw, _, intrs_vggt, extrs_vggt = _ensure_vggt_raw_cache_and_load(
            rgbs=rgbs, seq_name="n3d_gt_init", dataset_root=dataset_dir,
            vggt_cache_subdir="vggt_cache", skip_if_cached=False,
            model_id="facebook/VGGT-1B", target=target_size, frames_chunk_size=frames_chunk_size,
        )

    print("正在將 VGGT 深度尺度對齊至 N3D 物理尺度...")
    def get_centers(w2c_tensor):
        homo = torch.tensor([[[0,0,0,1]]]).expand(V, -1, -1).to(w2c_tensor.device)
        c2w = torch.inverse(torch.cat([w2c_tensor, homo], dim=1))
        return c2w[:, :3, 3]

    gt_centers = get_centers(extrs_gt[:, 0])
    vggt_centers = get_centers(extrs_vggt[:, 0])

    gt_dist_mean = torch.pdist(gt_centers).mean()
    vggt_dist_mean = torch.pdist(vggt_centers).mean()
    scale_factor = (gt_dist_mean / vggt_dist_mean).item()
    depths_metric = depths_raw * scale_factor

    # ==========================================
    # 4. 生成 3D 點雲
    # ==========================================
    print("使用真實相機參數與尺度化深度圖生成 3D 點雲...")
    xyz, rgb = init_pointcloud_from_rgbd(
        fmaps=rgbs.unsqueeze(0).float(), 
        depths=depths_metric.unsqueeze(0),
        intrs=intrs_gt.unsqueeze(0),
        extrs=extrs_gt.unsqueeze(0),
        stride=1, level=0
    )
    
    pts, colors = xyz[0].cpu().numpy(), rgb[0].cpu().numpy()
    depths_flat = depths_metric[:, 0, 0].flatten().cpu().numpy()
    
    pts_clean, colors_clean = pts[depths_flat > 0], colors[depths_flat > 0]
    if colors_clean.max() <= 1.0: colors_clean = (colors_clean * 255).astype(np.uint8)

    max_points = 200000
    if len(pts_clean) > max_points:
        indices = np.random.choice(len(pts_clean), max_points, replace=False)
        pts_clean, colors_clean = pts_clean[indices], colors_clean[indices]

    ply_path = os.path.join(dataset_dir, "init_pointcloud_vggt.ply")
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    elements = np.empty(pts_clean.shape[0], dtype=dtype)
    elements['x'], elements['y'], elements['z'] = pts_clean[:, 0], pts_clean[:, 1], pts_clean[:, 2]
    elements['red'], elements['green'], elements['blue'] = colors_clean[:, 0], colors_clean[:, 1], colors_clean[:, 2]
    PlyData([PlyElement.describe(elements, 'vertex')]).write(ply_path)
    print(f"✅ 初始 3D 點雲已儲存至 {ply_path} (共 {len(pts_clean)} 個點)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="資料集路徑")
    parser.add_argument("--vggt_target_size", type=int, default=392, help="VGGT 解析度")
    parser.add_argument("--vggt_chunk_size", type=int, default=1, help="解碼並發數")
    args = parser.parse_args()
    main(args.dir, args.vggt_target_size, args.vggt_chunk_size)