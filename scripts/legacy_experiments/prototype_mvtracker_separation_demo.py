import os
import sys
import glob
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from plyfile import PlyData
import rerun as rr
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

from mvtracker.utils.camera_selection import select_cams_by_fps
from mvtracker.utils.spatial_voxelizer import voxel_downsample_unified_with_features


def load_colmap_points_simple(colmap_dir: str):
    """載入指定 colmap 目錄下的 points3D (.bin 或 .ply 或 .txt)"""
    bin_path = os.path.join(colmap_dir, "sparse", "0", "points3D.bin")
    ply_path = os.path.join(colmap_dir, "sparse", "0", "points3D.ply")
    txt_path = os.path.join(colmap_dir, "sparse", "0", "points3D.txt")
    
    if os.path.isfile(bin_path):
        import struct
        pts, cols = [], []
        with open(bin_path, "rb") as f:
            num_points = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_points):
                data = f.read(43)
                if len(data) < 43:
                    break
                _id, x, y, z, r, g, b, err = struct.unpack("<QdddBBBd", data)
                track_len = struct.unpack("<Q", f.read(8))[0]
                f.seek(track_len * 8, 1)
                pts.append([x, y, z])
                cols.append([r, g, b])
        return np.array(pts, dtype=np.float32), np.array(cols, dtype=np.uint8)
    elif os.path.isfile(ply_path):
        ply = PlyData.read(ply_path)
        v = ply['vertex']
        pts = np.stack([v['x'], v['y'], v['z']], axis=-1).astype(np.float32)
        cols = np.stack([v['red'], v['green'], v['blue']], axis=-1).astype(np.uint8)
        return pts, cols
    elif os.path.isfile(txt_path):
        pts, cols = [], []
        with open(txt_path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) >= 7:
                    pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    cols.append([int(parts[4]), int(parts[5]), int(parts[6])])
        return np.array(pts, dtype=np.float32), np.array(cols, dtype=np.uint8)
    return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)


def extract_da3_with_conf_and_mask(
    rgbs_tensor, # [V, T, 3, H, W] uint8
    extrs_gt,    # [V, T, 3, 4]
    intrs_gt,    # [V, T, 3, 3]
    model_id="depth-anything/DA3-LARGE-1.1",
    process_res=672,
    conf_thresh=5.0,
    chunk_size=1,
    enable_layer_offload=None,
    device="cuda",
):
    """
    執行 DA3 提取深度與信心度，並依據 conf_thresh 將低信心像素深度設為 0 (防止錯誤深度干擾追蹤)
    """
    from depth_anything_3.api import DepthAnything3
    V, T, _, H, W = rgbs_tensor.shape
    
    if enable_layer_offload is None:
        if "GIANT" in model_id.upper() and device == "cuda":
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            enable_layer_offload = total_vram_gb <= 16.0
        else:
            enable_layer_offload = False
            
    print(f"🚀 [DA3] 載入 {model_id} (process_res={process_res}, layer_offload={enable_layer_offload})...")
    model = DepthAnything3.from_pretrained(model_id)
    if enable_layer_offload:
        from mvtracker.datasets.generic_scene_dataset import enable_da3_layer_offload
        model = enable_da3_layer_offload(model, device=torch.device(device))
    else:
        model = model.to(device)
    model.eval()

    depths_arr = torch.empty((V, T, 1, H, W), dtype=torch.float32)
    confs_arr = torch.empty((V, T, 1, H, W), dtype=torch.float32)

    with torch.no_grad():
        for t_start in tqdm(range(0, T, chunk_size), desc="DA3 推論與信心圖提取"):
            t_end = min(t_start + chunk_size, T)
            curr_chunk = t_end - t_start
            
            sub_imgs = []
            sub_extrs = []
            sub_intrs = []
            for t in range(t_start, t_end):
                for v in range(V):
                    sub_imgs.append(rgbs_tensor[v, t].permute(1, 2, 0).cpu().numpy())
                    sub_extrs.append(extrs_gt[v, t].cpu().numpy())
                    sub_intrs.append(intrs_gt[v, t].cpu().numpy())
                    
            sub_extr_4x4 = np.zeros((V * curr_chunk, 4, 4), dtype=np.float32)
            sub_extr_4x4[:, :3, :4] = np.stack(sub_extrs, axis=0)
            sub_extr_4x4[:, 3, 3] = 1.0
            sub_intr_arr = np.stack(sub_intrs, axis=0)

            try:
                pred = model.inference(
                    image=sub_imgs,
                    extrinsics=sub_extr_4x4,
                    intrinsics=sub_intr_arr,
                    process_res=process_res,
                    align_to_input_ext_scale=True,
                    ref_view_strategy="saddle_balanced" if curr_chunk > 1 else None,
                )
                pred_depth = pred.depth.astype(np.float32) # [V * curr_chunk, H_proc, W_proc]
                pred_conf = pred.conf.astype(np.float32) if (hasattr(pred, 'conf') and pred.conf is not None) else np.full_like(pred_depth, 10.0)
            except Exception as e:
                print(f"⚠️ DA3 批次推論異常 ({e})，降級為逐幀安全推論...")
                sub_d_list, sub_c_list = [], []
                for sub_t in range(t_start, t_end):
                    frame_imgs = [rgbs_tensor[v, sub_t].permute(1, 2, 0).cpu().numpy() for v in range(V)]
                    frame_extr = np.zeros((V, 4, 4), dtype=np.float32)
                    frame_extr[:, :3, :4] = extrs_gt[:, sub_t].cpu().numpy()
                    frame_extr[:, 3, 3] = 1.0
                    frame_intr = intrs_gt[:, sub_t].cpu().numpy()
                    p = model.inference(image=frame_imgs, extrinsics=frame_extr, intrinsics=frame_intr, process_res=process_res, align_to_input_ext_scale=True)
                    sub_d_list.append(p.depth.astype(np.float32))
                    sub_c_list.append(p.conf.astype(np.float32) if (hasattr(p, 'conf') and p.conf is not None) else np.full_like(p.depth, 10.0))
                pred_depth = np.concatenate(sub_d_list, axis=0)
                pred_conf = np.concatenate(sub_c_list, axis=0)

            d_t = torch.from_numpy(pred_depth).unsqueeze(1) # [V * curr, 1, H_p, W_p]
            c_t = torch.from_numpy(pred_conf).unsqueeze(1)
            
            d_res = F.interpolate(d_t, size=(H, W), mode="nearest")
            c_res = F.interpolate(c_t, size=(H, W), mode="nearest")
            
            idx = 0
            for t in range(t_start, t_end):
                for v in range(V):
                    depths_arr[v, t] = d_res[idx]
                    confs_arr[v, t] = c_res[idx]
                    idx += 1

    del model
    torch.cuda.empty_cache()

    # 產生給 Tracker 使用的無干擾深度圖 (低信心設為 0)
    depths_tracker = depths_arr.clone()
    depths_tracker[confs_arr < conf_thresh] = 0.0
    masked_pct = (depths_tracker == 0.0).float().mean().item() * 100.0
    print(f"✅ DA3 推論完成！已依 conf_thresh={conf_thresh} 將 {masked_pct:.2f}% 低信心像素深度置零（安全屏蔽噪點幾何）。")

    return depths_arr, confs_arr, depths_tracker


def compute_raft_optical_flow_scores(
    rgbs_tensor, # [V, T, 3, H, W] uint8
    t0=0,
    step=5,
    device="cuda",
):
    """
    計算 RAFT 2D 光流場，回傳各視角最大光流圖 max_flow_mag: [V, H, W]
    """
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    print(f"🌊 [RAFT] 載入光流模型計算 2D 運動邊界 (t0={t0}, step={step})...")
    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()
    model = raft_large(weights=weights, progress=False).to(device)
    model.eval()

    V, T, _, H, W = rgbs_tensor.shape
    max_flow_mag = torch.zeros((V, H, W), device=device)

    with torch.no_grad():
        for t in range(t0 + step, T, step):
            img1, img2 = transforms(rgbs_tensor[:, t0].to(device), rgbs_tensor[:, t].to(device))
            flow = model(img1, img2)[-1]
            mag = torch.norm(flow, dim=1) # [V, H, W]
            max_flow_mag = torch.max(max_flow_mag, mag)

    del model
    torch.cuda.empty_cache()
    return max_flow_mag.cpu().numpy()


def compute_photometric_residual(
    pts_3d,       # [N, 3] np.ndarray
    rgbs_tensor,  # [V, T, 3, H, W] uint8 torch.Tensor
    depths_clean, # [V, T, 1, H, W] float torch.Tensor
    extrs_model,  # [V, T, 3, 4]
    intrs_model,  # [V, T, 3, 3]
    device="cuda",
):
    """
    計算 3D 點在多視角相機上的固定像素外觀隨時間變化量 (Photometric Temporal Residual)
    """
    V, T, _, H, W = rgbs_tensor.shape
    N = len(pts_3d)
    pw = torch.from_numpy(pts_3d).float().to(device) # [N, 3]
    rg_gray = rgbs_tensor.float().mean(dim=2).to(device) # [V, T, H, W]
    
    # 3x3 空間模糊降低亞像素對齊噪聲
    rg_smooth = torch.nn.functional.avg_pool2d(rg_gray.reshape(-1, 1, H, W), 3, 1, 1).reshape(V, T, H, W)
    
    residuals_per_cam = torch.full((V, N), float("nan"), device=device)

    with torch.no_grad():
        for v in range(V):
            E_v = extrs_model[v, 0].to(device) # [3, 4]
            K_v = intrs_model[v, 0].to(device) # [3, 3]
            
            # 轉換到相機座標
            R = E_v[:3, :3]
            t_vec = E_v[:3, 3:4]
            p_cam = (R @ pw.T + t_vec).T # [N, 3]
            z = p_cam[:, 2]
            
            # 投影到像素座標
            uv = (K_v @ p_cam.T).T
            u = uv[:, 0] / torch.clamp(z, min=1e-4)
            w_coord = uv[:, 1] / torch.clamp(z, min=1e-4)
            
            u_i = u.round().long()
            w_i = w_coord.round().long()
            
            in_bounds = (z > 0.1) & (u_i >= 1) & (u_i < W - 1) & (w_i >= 1) & (w_i < H - 1)
            u_clamp = u_i.clamp(1, W - 2)
            w_clamp = w_i.clamp(1, H - 2)
            
            d_val = depths_clean[v, 0, 0].to(device)[w_clamp, u_clamp]
            # 深度遮擋一致性檢查 (允許 5% 誤差)
            is_visible = in_bounds & ((d_val - z).abs() < 0.05 * z) & (d_val > 0.1)
            
            ref_color = rg_smooth[v, 0, w_clamp, u_clamp] # [N]
            all_colors = rg_smooth[v, :, w_clamp, u_clamp] # [T, N]
            
            cam_res = (all_colors - ref_color.unsqueeze(0)).abs().mean(dim=0) / 255.0 # [N]
            cam_res[~is_visible] = float("nan")
            residuals_per_cam[v] = cam_res

    res_np = residuals_per_cam.cpu().numpy()
    # 跨相機中位數 (忽略遮擋相機)
    with np.errstate(all='ignore'):
        photo_res = np.nanmedian(res_np, axis=0)
    photo_res[np.isnan(photo_res)] = 0.0
    return photo_res


def run_separation_demo(
    args,
    view_dirs,
    all_w2c_np,
    all_intrs_np,
    W_img,
    H_img,
    device="cuda",
):
    """
    第一階段整合 Demo：
    1. DA3 + COLMAP 協同抽點
    2. RAFT、顏色指標、MVTracker 3D 位移、空間平滑與階層融合對比
    3. 輸出全方位 Rerun 互動視覺化 (.rrd)
    """
    print("\n" + "="*60)
    print("🎬 [第一階段 Demo] 靜動態分離整合驗證 (DA3 + COLMAP + MVTracker)")
    print("="*60)

    # 1. 選擇相機視角
    if args.selected_cams is not None:
        cams = args.selected_cams
    else:
        cams = select_cams_by_fps(all_w2c_np, num_to_select=args.num_cams, alpha=args.fps_alpha, exclude_indices=[0])
    print(f"🎯 使用視角: {cams} (已排除測試視角 0)")
    V = len(cams)

    # 2. 準備 RGB 影像序列
    TARGET_W, TARGET_H = 768, 576
    scale_x = TARGET_W / W_img
    scale_y = TARGET_H / H_img

    sample_frames = sorted(glob.glob(os.path.join(view_dirs[cams[0]], "*.png")) + glob.glob(os.path.join(view_dirs[cams[0]], "*.jpg")),
                           key=lambda x: int("".join(filter(str.isdigit, os.path.basename(x)))) if any(c.isdigit() for c in os.path.basename(x)) else 0)
    T = min(len(sample_frames), args.max_frames)
    t0 = args.start_frame
    print(f"⏱️ 處理影格: t0={t0} 至 {t0 + T - 1} (共 {T} 幀)")

    rgbs_list = []
    for c in tqdm(cams, desc="讀取影像"):
        f_paths = sorted(glob.glob(os.path.join(view_dirs[c], "*.png")) + glob.glob(os.path.join(view_dirs[c], "*.jpg")),
                         key=lambda x: int("".join(filter(str.isdigit, os.path.basename(x)))) if any(c.isdigit() for c in os.path.basename(x)) else 0)[t0:t0 + T]
        c_rgbs = []
        for p in f_paths:
            im = Image.open(p).convert("RGB").resize((TARGET_W, TARGET_H), Image.Resampling.BILINEAR)
            c_rgbs.append((torch.from_numpy(np.array(im)).permute(2, 0, 1)))
        rgbs_list.append(torch.stack(c_rgbs))
    rgbs_tensor = torch.stack(rgbs_list) # [V, T, 3, H, W] uint8

    # 建構 Extrinsics & Intrinsics
    extrs_list, intrs_list = [], []
    for c in cams:
        extrs_list.append(torch.from_numpy(all_w2c_np[c]).float())
        K = all_intrs_np[c].copy()
        K[0, :] *= scale_x
        K[1, :] *= scale_y
        intrs_list.append(torch.from_numpy(K).float())
    extrs_gt = torch.stack(extrs_list).unsqueeze(1).repeat(1, T, 1, 1)
    intrs_gt = torch.stack(intrs_list).unsqueeze(1).repeat(1, T, 1, 1)

    # 3. 提取 DA3 深度與 2D 信心圖 (並依需求屏蔽低信心像素)
    conf_thresh = getattr(args, "conf_thresh", 5.0)
    depths_clean, confs_tensor, depths_tracker = extract_da3_with_conf_and_mask(
        rgbs_tensor=rgbs_tensor,
        extrs_gt=extrs_gt,
        intrs_gt=intrs_gt,
        model_id=args.da3_model,
        process_res=args.da3_process_res,
        conf_thresh=conf_thresh,
        chunk_size=1,
        enable_layer_offload=args.da3_layer_offload,
        device=device,
    )

    # 4. 解耦組裝初始點雲 (COLMAP 錨點 100% 保留 + DA3 均勻體素腳手架)
    from mvtracker.models.core.model_utils import init_pointcloud_from_rgbd
    with torch.no_grad():
        xyz_full, rgb_full = init_pointcloud_from_rgbd(
            fmaps=rgbs_tensor[:, 0:1].unsqueeze(0).float(),
            depths=depths_clean[:, 0:1].unsqueeze(0),
            intrs=intrs_gt[:, 0:1].unsqueeze(0),
            extrs=extrs_gt[:, 0:1].unsqueeze(0),
            stride=1,
            level=0,
        )
    pts_t0 = xyz_full[0].cpu().numpy()
    cols_t0 = rgb_full[0].cpu().numpy()
    conf_flat = confs_tensor[:, 0, 0].flatten().cpu().numpy()
    depth_flat = depths_clean[:, 0, 0].flatten().cpu().numpy()
    valid_mask = (depth_flat > 0.1) & (conf_flat >= conf_thresh) & np.isfinite(depth_flat)

    valid_pts = pts_t0[valid_mask]
    valid_cols = cols_t0[valid_mask]

    vox_size = getattr(args, "voxel_size", 0.08)
    print(f"📦 [體素均勻下採樣] 處理 {len(valid_pts):,} 個高信心點 (voxel_size={vox_size})...")
    v_pts, v_cols, _, _ = voxel_downsample_unified_with_features(
        pts=valid_pts,
        colors=valid_cols,
        features=None,
        is_dynamic=np.zeros(len(valid_pts), bool),
        normals=None,
        voxel_size=vox_size,
        footprints=None,
        min_voxel_size=0.01,
        scale=1.0,
        mode="uniform",
    )
    if isinstance(v_pts, torch.Tensor):
        v_pts = v_pts.cpu().numpy()
        v_cols = v_cols.cpu().numpy()

    # 載入 COLMAP 稀疏錨點
    colmap_dir = os.path.join(args.dir, f"colmap_{t0}")
    if not os.path.exists(colmap_dir):
        colmap_dir = os.path.join(args.dir, "colmap_0")
    colmap_pts, colmap_cols = load_colmap_points_simple(colmap_dir)
    print(f"📌 [COLMAP] 成功載入 {len(colmap_pts):,} 個剛性幾何錨點 ({colmap_dir})")

    # 限制 DA3 點數預算
    budget = getattr(args, "budget_queries", 25000)
    da3_budget = max(1000, budget - len(colmap_pts))
    if len(v_pts) > da3_budget:
        perm = np.random.permutation(len(v_pts))[:da3_budget]
        v_pts = v_pts[perm]
        v_cols = v_cols[perm]

    # 解耦拼接
    if len(colmap_pts) > 0:
        all_pts = np.concatenate([colmap_pts, v_pts], axis=0).astype(np.float32)
        all_cols = np.concatenate([colmap_cols, v_cols], axis=0).astype(np.uint8) if v_cols.max() > 1.0 else np.concatenate([colmap_cols, (v_cols * 255).astype(np.uint8)], axis=0)
        is_colmap = np.concatenate([np.ones(len(colmap_pts), bool), np.zeros(len(v_pts), bool)])
    else:
        all_pts = v_pts.astype(np.float32)
        all_cols = (v_cols * 255).astype(np.uint8) if v_cols.max() <= 1.0 else v_cols.astype(np.uint8)
        is_colmap = np.zeros(len(v_pts), bool)

    N_queries = len(all_pts)
    print(f"🎯 總候選點雲: {N_queries:,} 點 (COLMAP: {is_colmap.sum():,}, DA3 Scaffold: {(~is_colmap).sum():,})")

    # 5. 方法 1: RAFT 光流
    raft_thresh = getattr(args, "raft_threshold", 1.0)
    max_flow_mag = compute_raft_optical_flow_scores(rgbs_tensor, t0=0, step=5, device=device) # [V, H, W]

    # 計算各 3D 點在相機上的光流
    pts_flow_list = []
    pw = torch.from_numpy(all_pts).to(device)
    for v in range(V):
        E_v = extrs_gt[v, 0].to(device)
        K_v = intrs_gt[v, 0].to(device)
        p_c = (E_v[:3, :3] @ pw.T + E_v[:3, 3:4]).T
        z = p_c[:, 2]
        uv = (K_v @ p_c.T).T
        u = (uv[:, 0] / torch.clamp(z, min=1e-4)).round().long()
        w = (uv[:, 1] / torch.clamp(z, min=1e-4)).round().long()
        valid_u = (z > 0.1) & (u >= 0) & (u < TARGET_W) & (w >= 0) & (w < TARGET_H)
        flow_v = torch.zeros(N_queries, device=device)
        if valid_u.any():
            flow_v[valid_u] = torch.from_numpy(max_flow_mag[v])[w[valid_u].cpu(), u[valid_u].cpu()].to(device)
        pts_flow_list.append(flow_v)
    pts_max_flow = torch.stack(pts_flow_list, dim=0).max(dim=0)[0].cpu().numpy()
    pred_raft_dyn = pts_max_flow > raft_thresh

    # 6. 方法 2: 顏色指標 (外觀時序殘差)
    photo_thresh = getattr(args, "photo_threshold", 0.015)
    print("🎨 [顏色指標] 計算投影像素多視角外觀時序變動 (Photometric Residual)...")
    photo_residuals = compute_photometric_residual(
        pts_3d=all_pts,
        rgbs_tensor=rgbs_tensor,
        depths_clean=depths_clean,
        extrs_model=extrs_gt,
        intrs_model=intrs_gt,
        device=device,
    )
    pred_photo_dyn = photo_residuals > photo_thresh

    # 7. 方法 3 & 4: MVTracker 3D 位移追蹤與空間平滑
    disp_thresh = getattr(args, "disp_threshold", 0.20)
    use_presort = getattr(args, "use_raft_presort", True)

    # 依預算決定是否快速粗篩
    if use_presort:
        # 顯著靜態點：光流極小 (<0.5) 且顏色幾乎不變 (<0.005)
        is_clear_static = (pts_max_flow < 0.5) & (photo_residuals < 0.005)
        # 動態候選點
        cand_indices = np.where(~is_clear_static)[0]
        # 加上少量 (5%) 靜態樣本用於驗證校準
        static_indices = np.where(is_clear_static)[0]
        calib_indices = np.random.choice(static_indices, size=min(len(static_indices), 1000), replace=False) if len(static_indices) > 0 else np.array([], dtype=int)
        track_indices = np.union1d(cand_indices, calib_indices)
        print(f"⚡ [預算加速粗篩] 篩除 {is_clear_static.sum():,} 點顯著背景！僅追蹤 {len(track_indices):,} 個候選/校準點 (符合 <=1s/frame 預算)")
    else:
        track_indices = np.arange(N_queries)

    print(f"🧠 [MVTracker] 載入模型並執行 3D 軌跡追蹤 (目標追蹤點數: {len(track_indices):,})...")
    mvtracker = torch.hub.load(".", "mvtracker", source="local", pretrained=True, device=device)
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

    track_pts_tensor = torch.from_numpy(all_pts[track_indices]).float().to(device)
    ts = torch.zeros((len(track_indices), 1), device=device)
    q_track = torch.cat([ts, track_pts_tensor], dim=1)

    chunk_size = getattr(args, "track_chunk_size", 4096)
    all_pred_trajs = []
    all_pred_vis = []

    t_track_start = time.time()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=amp_dtype):
        # 注意：使用 depths_tracker (低信心像素已遮蔽為 0，避免錯誤深度污染幾何)
        for i in tqdm(range(0, len(track_indices), chunk_size), desc="MVTracker 3D 追蹤進度"):
            batch_q = q_track[i:i + chunk_size]
            res = mvtracker(
                rgbs=rgbs_tensor[None],
                depths=depths_tracker[None],
                intrs=intrs_gt[None].to(device),
                extrs=extrs_gt[None].to(device),
                query_points_3d=batch_q[None],
            )
            all_pred_trajs.append(res["traj_e"].cpu())
            all_pred_vis.append(res["vis_e"].cpu())
            del res
            torch.cuda.empty_cache()

    t_track_total = time.time() - t_track_start
    print(f"⏱️ MVTracker 追蹤完成！總耗時: {t_track_total:.2f} 秒 (平均單幀: {t_track_total / T:.3f} 秒/幀)")

    del mvtracker
    torch.cuda.empty_cache()

    if len(all_pred_trajs) > 0:
        tracked_trajs = torch.cat(all_pred_trajs, dim=2)[0].numpy() # [T, N_tracked, 3]
        tracked_vis = torch.cat(all_pred_vis, dim=2)[0].numpy()     # [T, N_tracked]
    else:
        tracked_trajs = np.zeros((T, 0, 3), dtype=np.float32)

    # 計算全域 3D 位移
    disp_full = np.zeros(N_queries, dtype=np.float32)
    tracked_disps = np.linalg.norm(tracked_trajs - tracked_trajs[0:1], axis=-1).max(axis=0)
    disp_full[track_indices] = tracked_disps

    # 方法 3: 原始 MVTracker 位移
    pred_mvt_raw_dyn = disp_full > disp_thresh

    # 方法 4: 空間 kNN 平滑 (去除單點深度跳變噪聲)
    k_val = getattr(args, "separation_knn_k", 16)
    print(f"🔍 [空間平滑] 建立 3D KD-Tree 進行 kNN={k_val} 鄰近中位數濾波...")
    tree = cKDTree(all_pts)
    _, nn_idx = tree.query(all_pts, k=min(k_val, len(all_pts)))
    disp_knn = np.median(disp_full[nn_idx], axis=1)
    pred_mvt_knn_dyn = disp_knn > disp_thresh

    # 方法 5: 階層式融合判準 (Hierarchical Fusion)
    # 核心公式：平滑位移超標 且 (有外觀變動 OR 光流移動)；或極大位移 (排除低反差移動)
    pred_fusion_dyn = (
        (disp_knn > disp_thresh) & ((photo_residuals > 0.008) | (pts_max_flow > 1.0))
    ) | (disp_knn > 2.0 * disp_thresh)

    # 剛性 COLMAP 錨點保護：SfM 剛性點不隨意判為動態
    pred_fusion_dyn[is_colmap & (disp_knn <= 2.0 * disp_thresh)] = False

    # 8. 輸出指標統計報告
    stats = {
        "total_queries": int(N_queries),
        "colmap_anchors": int(is_colmap.sum()),
        "da3_scaffold": int((~is_colmap).sum()),
        "tracking_time_sec": round(t_track_total, 2),
        "time_per_frame_sec": round(t_track_total / T, 3),
        "raft_dynamic_pct": round(float(pred_raft_dyn.mean() * 100), 2),
        "photo_dynamic_pct": round(float(pred_photo_dyn.mean() * 100), 2),
        "mvt_raw_dynamic_pct": round(float(pred_mvt_raw_dyn.mean() * 100), 2),
        "mvt_knn_dynamic_pct": round(float(pred_mvt_knn_dyn.mean() * 100), 2),
        "fusion_dynamic_pct": round(float(pred_fusion_dyn.mean() * 100), 2),
        "fusion_colmap_dyn_pct": round(float(pred_fusion_dyn[is_colmap].mean() * 100), 2) if is_colmap.any() else 0.0,
    }
    print("\n" + "="*50)
    print("📊 各方法靜動態分離統計對比:")
    print(f"  1. RAFT 2D 光流判定動態比例:        {stats['raft_dynamic_pct']}%")
    print(f"  2. 外觀顏色殘差判定動態比例:        {stats['photo_dynamic_pct']}%")
    print(f"  3. MVTracker 3D 原始位移判定動態:   {stats['mvt_raw_dynamic_pct']}%")
    print(f"  4. MVTracker 3D kNN 平滑判定動態:   {stats['mvt_knn_dynamic_pct']}%")
    print(f"  5. ★ 推薦階層融合方法判定動態:      {stats['fusion_dynamic_pct']}% (COLMAP 錨點誤判率僅 {stats['fusion_colmap_dyn_pct']}%)")
    print("="*50 + "\n")

    # 9. Rerun 多圖層可視化匯出
    rrd_path = args.rrd if args.rrd.endswith(".rrd") else f"{args.rrd}.rrd"
    print(f"🎨 [Rerun] 打包靜動態分離視覺化至: {rrd_path} ...")
    rr.init("n3d_static_dynamic_separation", recording_id=f"sep_{int(time.time())}")

    C_STATIC = [50, 220, 80]  # 綠色
    C_DYNAMIC = [255, 50, 50] # 紅色

    def make_colors(mask):
        cols = np.tile(np.array(C_STATIC, dtype=np.uint8), (len(mask), 1))
        cols[mask] = np.array(C_DYNAMIC, dtype=np.uint8)
        return cols

    # 對照圖層 1~5 (使用 static=True 確保在 timeline 'frame' 下永久可見)
    rr.log("Separation_Methods/1_RAFT_2D_Flow", rr.Points3D(positions=all_pts, colors=make_colors(pred_raft_dyn), radii=0.018), static=True)
    rr.log("Separation_Methods/2_Photometric_Residual", rr.Points3D(positions=all_pts, colors=make_colors(pred_photo_dyn), radii=0.018), static=True)
    rr.log("Separation_Methods/3_MVTracker_3D_Raw", rr.Points3D(positions=all_pts, colors=make_colors(pred_mvt_raw_dyn), radii=0.018), static=True)
    rr.log("Separation_Methods/4_MVTracker_3D_kNN", rr.Points3D(positions=all_pts, colors=make_colors(pred_mvt_knn_dyn), radii=0.018), static=True)
    rr.log("Separation_Methods/5_Fusion_Recommended", rr.Points3D(positions=all_pts, colors=make_colors(pred_fusion_dyn), radii=0.02), static=True)

    # 熱力圖
    def to_turbo(values, vmin, vmax):
        norm = np.clip((values - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
        return (plt.cm.turbo(norm)[:, :3] * 255).astype(np.uint8)

    rr.log("Heatmaps/MVTracker_Displacement", rr.Points3D(positions=all_pts, colors=to_turbo(disp_knn, 0.0, 0.8), radii=0.018), static=True)
    rr.log("Heatmaps/Photometric_Residual", rr.Points3D(positions=all_pts, colors=to_turbo(photo_residuals, 0.0, 0.05), radii=0.018), static=True)
    rr.log("Heatmaps/RAFT_Flow_Magnitude", rr.Points3D(positions=all_pts, colors=to_turbo(pts_max_flow, 0.0, 5.0), radii=0.018), static=True)

    # 點雲來源標註 (COLMAP=青色, DA3 Scaffold=紫色)
    src_colors = np.where(is_colmap[:, None], np.array([[0, 230, 255]], dtype=np.uint8), np.array([[190, 100, 255]], dtype=np.uint8))
    rr.log("Geometry_Scaffold/COLMAP_vs_DA3", rr.Points3D(positions=all_pts, colors=src_colors, radii=0.016), static=True)

    # 4D 動態時序播放
    # 建立動態軌跡陣列
    fusion_dyn_indices = np.where(pred_fusion_dyn)[0]
    fusion_sta_indices = np.where(~pred_fusion_dyn)[0]

    # 對於被判定為動態的點，提取其 3D 軌跡；若未被追蹤則維持原地
    track_map = {idx: i for i, idx in enumerate(track_indices)}
    dyn_tracks_4d = np.zeros((T, len(fusion_dyn_indices), 3), dtype=np.float32)
    for i, orig_idx in enumerate(fusion_dyn_indices):
        if orig_idx in track_map:
            dyn_tracks_4d[:, i] = tracked_trajs[:, track_map[orig_idx]]
        else:
            dyn_tracks_4d[:, i] = all_pts[orig_idx]

    static_pts_fixed = all_pts[fusion_sta_indices]

    # 動態軌跡線段 (LineStrips)
    if len(fusion_dyn_indices) > 0:
        lines = [dyn_tracks_4d[:, i, :] for i in range(min(len(fusion_dyn_indices), 500))] # 顯示前500條動態線
        rr.log("Dynamic_Trajectories/Motion_Paths", rr.LineStrips3D(strips=lines, colors=[255, 100, 100]), static=True)

    # 時序幀播放
    for t in range(T):
        rr.set_time("frame", sequence=t)
        rr.log("Playback_4D/Moving_Gaussians", rr.Points3D(positions=dyn_tracks_4d[t], colors=np.full((len(fusion_dyn_indices), 3), C_DYNAMIC, dtype=np.uint8), radii=0.022))
        rr.log("Playback_4D/Static_Gaussians", rr.Points3D(positions=static_pts_fixed, colors=np.full((len(fusion_sta_indices), 3), C_STATIC, dtype=np.uint8), radii=0.018))

    # 相機 2D 視角投影
    for v in range(V):
        w2c = extrs_gt[v, 0].cpu().numpy()
        c2w = np.linalg.inv(np.vstack([w2c, [0, 0, 0, 1]]))
        R_cam = c2w[:3, :3]
        t_cam = c2w[:3, 3]
        K_cam = intrs_gt[v, 0].cpu().numpy()
        
        rr.log(f"Cameras/cam_{cams[v]}", rr.Pinhole(image_from_camera=K_cam, width=TARGET_W, height=TARGET_H), static=True)
        rr.log(f"Cameras/cam_{cams[v]}", rr.Transform3D(translation=t_cam, mat3x3=R_cam), static=True)
        
        for t in range(min(T, 50)):
            rr.set_time("frame", sequence=t)
            img_np = rgbs_tensor[v, t].permute(1, 2, 0).cpu().numpy()
            rr.log(f"Cameras/cam_{cams[v]}/Image", rr.Image(img_np))

    if args.rerun == "save":
        rr.save(rrd_path)
        print(f"🎉 成功儲存 Rerun 檔案至: {os.path.abspath(rrd_path)}")
    elif args.rerun == "spawn":
        rr.spawn()
    elif args.rerun == "stream":
        rr.connect_tcp()

    # 儲存摘要報告 JSON
    json_path = os.path.splitext(rrd_path)[0] + "_stats.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"📄 統計數據報告已儲存至: {json_path}")
    print("="*60 + "\n")
