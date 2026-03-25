import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import torch
import numpy as np
from plyfile import PlyData, PlyElement
from PIL import Image
from torchvision.transforms.functional import to_tensor

# 匯入 MVTracker 內建的 VGGT 處理函式與 3D 轉換工具
from mvtracker.datasets.generic_scene_dataset import _ensure_vggt_raw_cache_and_load
from mvtracker.models.core.model_utils import init_pointcloud_from_rgbd

def main(dataset_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. 抓取每個視角的第一幀 (00001.jpg)
    view_dirs = sorted(glob.glob(os.path.join(dataset_dir, 'view_*')))
    assert len(view_dirs) > 0, "找不到視角資料夾！"

    rgbs_list = []
    for v_dir in view_dirs:
        frames = sorted(glob.glob(os.path.join(v_dir, '*.jpg')))
        img = Image.open(frames[0]).convert('RGB')
        # 轉換為 [3, H, W] 的 uint8 張量，符合 VGGT 預期格式
        tensor = (to_tensor(img) * 255).byte()
        rgbs_list.append(tensor)

    # 封裝維度成 [V, T=1, 3, H, W]
    rgbs = torch.stack(rgbs_list).unsqueeze(1)
    V, T, C, H, W = rgbs.shape
    print(f"成功載入 {V} 個視角，影像解析度為 {W}x{H}。")

    # 2. 執行 VGGT 獲取深度與相機參數
    print("啟動 VGGT 進行全局推論 (初次執行會下載模型權重)...")
    depths_raw, confs, intrs_raw, extrs_raw = _ensure_vggt_raw_cache_and_load(
        rgbs=rgbs,
        seq_name="frame0_init", # 作為快取資料夾名稱
        dataset_root=dataset_dir,
        vggt_cache_subdir="vggt_cache",
        skip_if_cached=False,   # 強制重新執行提取
        model_id="facebook/VGGT-1B"
    )

    # 3. 生成 3D 點雲
    print("將深度與參數轉換為 3D 點雲...")
    fmaps_input = rgbs.unsqueeze(0).float() # [1, V, T, 3, H, W]
    
    xyz, rgb = init_pointcloud_from_rgbd(
        fmaps=fmaps_input, 
        depths=depths_raw.unsqueeze(0),
        intrs=intrs_raw.unsqueeze(0),
        extrs=extrs_raw.unsqueeze(0),
        stride=1,
        level=0
    )
    
    # 提取 t=0 的點座標與顏色
    pts = xyz[0].cpu().numpy()
    colors = rgb[0].cpu().numpy()
    
    # 4. 過濾掉無效深度的點
    depths_flat = depths_raw[:, 0, 0].flatten().cpu().numpy()
    valid_mask = depths_flat > 0
    
    pts_clean = pts[valid_mask]
    colors_clean = colors[valid_mask]
    if colors_clean.max() <= 1.0:
        colors_clean = (colors_clean * 255).astype(np.uint8)

    max_points = 200000  # 設定 4DGS 完美的初始點數：20 萬點
    if len(pts_clean) > max_points:
        print(f"點數過多 ({len(pts_clean)})，正在隨機降採樣至 {max_points} 點...")
        # 產生不重複的隨機索引
        indices = np.random.choice(len(pts_clean), max_points, replace=False)
        pts_clean = pts_clean[indices]
        colors_clean = colors_clean[indices]

    # 5. 匯出 4DGS 支援的 PLY 檔案
    ply_path = os.path.join(dataset_dir, "init_pointcloud_vggt.ply")
    
    # 定義 plyfile 需要的資料結構 (3個 float32 座標, 3個 uint8 顏色)
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    # 建立結構化 Numpy 陣列
    elements = np.empty(pts_clean.shape[0], dtype=dtype)
    elements['x'] = pts_clean[:, 0]
    elements['y'] = pts_clean[:, 1]
    elements['z'] = pts_clean[:, 2]
    elements['red'] = colors_clean[:, 0]
    elements['green'] = colors_clean[:, 1]
    elements['blue'] = colors_clean[:, 2]
    
    # 寫入檔案
    vertex_element = PlyElement.describe(elements, 'vertex')
    PlyData([vertex_element]).write(ply_path)
    
    print(f"初始 3D 點雲已儲存至 {ply_path} (共 {len(pts_clean)} 個點)")

    # 6. 儲存相機參數供 4DGS 轉換使用
    npz_path = os.path.join(dataset_dir, "estimated_cameras_vggt.npz")
    # VGGT 輸出的 extrs_raw 是 World-to-Camera (w2c) 矩陣
    np.savez(
        npz_path, 
        w2c_poses=extrs_raw[:, 0].cpu().numpy(), 
        intrs=intrs_raw[:, 0].cpu().numpy()
    )
    print(f"相機參數 (w2c & intrs) 已儲存至 {npz_path}")
    # =========================================================================
    # 7. 匯出 COLMAP TXT 格式，供 GUI 視覺化檢查內外參與點雲座標系
    # =========================================================================
    print("正在輸出 COLMAP TXT 格式...")
    colmap_dir = os.path.join(dataset_dir, "colmap_text", "sparse", "0")
    os.makedirs(colmap_dir, exist_ok=True)

    # 旋轉矩陣轉四元數 (w, x, y, z) 函數 (完全依照 COLMAP 官方定義)
    def rotmat2qvec(R):
        Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
        K = np.array([
            [Rxx - Ryy - Rzz, 0, 0, 0],
            [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
            [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
            [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
        eigvals, eigvecs = np.linalg.eigh(K)
        qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
        if qvec[0] < 0:
            qvec *= -1
        return qvec

    # 7.1 寫入 cameras.txt
    with open(os.path.join(colmap_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for vid in range(V):
            cam_id = vid + 1
            K_mat = intrs_raw[vid, 0].cpu().numpy()
            fx, fy = K_mat[0, 0], K_mat[1, 1]
            cx, cy = K_mat[0, 2], K_mat[1, 2]
            f.write(f"{cam_id} PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n")

    # 7.2 寫入 images.txt
    with open(os.path.join(colmap_dir, "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for vid in range(V):
            img_id = vid + 1
            cam_id = vid + 1
            # VGGT 輸出的是 W2C 矩陣
            w2c = extrs_raw[vid, 0].cpu().numpy()
            R_mat = w2c[:3, :3]
            T_vec = w2c[:3, 3]
            qvec = rotmat2qvec(R_mat)
            img_name = f"view_{vid}_f0000.jpg" 
            
            f.write(f"{img_id} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {T_vec[0]} {T_vec[1]} {T_vec[2]} {cam_id} {img_name}\n")
            f.write("\n") # 特徵點留空，我們只看相機位置

    # 7.3 寫入 points3D.txt
    with open(os.path.join(colmap_dir, "points3D.txt"), "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for pid, (pt, color) in enumerate(zip(pts_clean, colors_clean)):
            f.write(f"{pid+1} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {color[0]} {color[1]} {color[2]} 0.0\n")

    print(f"COLMAP TXT 格式已輸出至 {colmap_dir}，可使用 COLMAP GUI 匯入檢查。")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="資料集路徑 (如 datasets/bullpen)")
    args = parser.parse_args()
    main(args.dir)
# python scripts/prepare_4dgs_vggt.py --dir datasets/bullpen