import os
import re
import glob
import time
import gc
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from mvtracker.utils.feature_scores import depth_to_surface_normals


def select_cams_by_fps(
    all_w2c_np: np.ndarray,
    num_to_select: int,
    alpha: float = 0.8,
    exclude_indices: List[int] = [0],
) -> List[int]:
    """
    使用幾何外參最遠點採樣 (Farthest Point Sampling, FPS) 挑選相機。
    - alpha: 距離權重 (0.8 代表 80% 相機中心 3D 距離 + 20% 視線方向角度差)
    - exclude_indices: 排除的相機 (預設排除 index 0 測試視角)
    """
    num_total = len(all_w2c_np)
    valid_candidates = [i for i in range(num_total) if i not in exclude_indices]

    if num_to_select >= len(valid_candidates):
        return sorted(valid_candidates)

    centers, dirs = [], []
    for w2c in all_w2c_np:
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        c = -R.T @ t                  # 世界座標相機中心
        d = R.T @ np.array([0, 0, 1])  # 相機光軸方向
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


def select_cams_by_da3_surfel(
    view_dirs: List[str],
    all_w2c_np: np.ndarray,
    all_intrs_np: np.ndarray,
    num_to_select: int = 5,
    voxel_size: float = 0.2,
    exclude_indices: List[int] = [0],
    da3_model: str = "depth-anything/DA3-GIANT-1.1",
    process_res: int = 728,
    device: str = "cuda",
) -> List[int]:
    """
    基於第 0 幀 Depth Anything 3 (DA3) 預測的多視角深度與法向，
    透過「3D 表面基元 (Surfel) 次模貪婪最大覆蓋」挑選出視角覆蓋率最高的相機組合。
    
    優勢：
    1. 不受限於純幾何相機外參距離，真切依據目標物體可見表面做覆蓋決策。
    2. 利用 Surfel (空間座標 + 法向量角度量化) 能有效區分正反面與不同朝向表面，
       自然挑出 360 度環繞主體表面之相機群。
    3. 僅推論第 0 幀 (約 20 張)，在現代 GPU 上耗時僅約 3 秒。
    """
    num_total = len(all_w2c_np)
    candidates = [i for i in range(num_total) if i not in exclude_indices]

    if num_to_select >= len(candidates):
        print(f"⚠️ 要求的相機數 ({num_to_select}) 大於等於可用候選視角數 ({len(candidates)})，回傳所有視角。")
        return sorted(candidates)

    print(f"\n🔍 [相機自動選擇] 啟動基於第 0 幀 DA3 Surfel 最大覆蓋度演算法...")
    t0 = time.time()

    # 1. 讀取所有候選相機的第 0 幀影像
    images_f0 = []
    for vid in candidates:
        v_dir = view_dirs[vid]
        # 尋找 0.png 或 000000.png / jpg
        f_list = sorted(
            glob.glob(os.path.join(v_dir, "*.png")) + glob.glob(os.path.join(v_dir, "*.jpg")),
            key=lambda x: int(re.search(r"\d+", os.path.basename(x)).group()) if re.search(r"\d+", os.path.basename(x)) else 0
        )
        if not f_list:
            raise FileNotFoundError(f"在相機資料夾 {v_dir} 中找不到任何影像！")
        img_f0 = Image.open(f_list[0]).convert("RGB")
        images_f0.append(np.array(img_f0))

    H_orig, W_orig = images_f0[0].shape[:2]

    # 2. 呼叫 DA3 推論第 0 幀深度
    try:
        from depth_anything_3.api import DepthAnything3
        from mvtracker.datasets.generic_scene_dataset import enable_da3_layer_offload
    except ImportError:
        print("⚠️ 未安裝 depth_anything_3，自動降級為 FPS 幾何外參選擇。")
        return select_cams_by_fps(all_w2c_np, num_to_select, exclude_indices=exclude_indices)

    # 相機覆蓋率挑選僅需宏觀幾何與法向，使用 DA3-LARGE (res<=504) 僅耗時 ~4 秒且杜絕 20 視角 VRAM 溢出
    selection_model = "depth-anything/DA3-LARGE-1.1" if "GIANT" in da3_model.upper() else da3_model
    selection_res = min(process_res, 504)

    print(f"🤖 正在使用 {selection_model} (res={selection_res}) 推論 {len(candidates)} 個候選視角在第 0 幀之幾何深度...")
    t_infer_start = time.time()
    
    extr_4x4 = np.zeros((len(candidates), 4, 4), dtype=np.float32)
    extr_4x4[:, :3, :4] = all_w2c_np[candidates]
    extr_4x4[:, 3, 3] = 1.0
    intr_batch = all_intrs_np[candidates]

    dev = torch.device(device)
    model = DepthAnything3.from_pretrained(selection_model).to(dev)
    model.eval()

    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    with torch.no_grad(), torch.autocast(device, dtype=amp_dtype):
        pred = model.inference(
            images_f0,
            extrinsics=extr_4x4,
            intrinsics=intr_batch,
            process_res=selection_res,
            align_to_input_ext_scale=True,
            ref_view_strategy="saddle_balanced"
        )
        depths_np = pred.depth  # [N_cands, H_proc, W_proc]

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    t_infer = time.time() - t_infer_start
    print(f"⚡ DA3 第 0 幀深度推論完成！耗時: {t_infer:.2f} 秒")

    # 3. 反投影為 3D Surfel (空間座標 + 表面法向量量化)
    cam_surfel_sets = {}
    total_surfels_seen = set()

    for idx, vid in enumerate(candidates):
        d = depths_np[idx]
        d_torch = torch.from_numpy(d).unsqueeze(0).unsqueeze(0).to(device)
        norm_res = depth_to_surface_normals(d_torch)
        norm_c = (norm_res[0] if isinstance(norm_res, tuple) else norm_res).squeeze(0).permute(1, 2, 0).cpu().numpy()

        # 4x 空間步進下採樣 (加速計算，幾千點即可代表表面)
        d_sub = d[::4, ::4]
        norm_sub = norm_c[::4, ::4]
        valid = (d_sub >= 0.5) & (d_sub <= 50.0)

        ys, xs = np.where(valid)
        if len(ys) == 0:
            cam_surfel_sets[vid] = set()
            continue

        d_vals = d_sub[ys, xs]
        n_vals = norm_sub[ys, xs]

        K = intr_batch[idx].copy()
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        H_d, W_d = d.shape
        sx, sy = W_d / W_orig, H_d / H_orig
        fx, fy = fx * sx, fy * sy
        cx, cy = cx * sx, cy * sy

        xs_full = xs * 4
        ys_full = ys * 4

        x_c = (xs_full - cx) * d_vals / fx
        y_c = (ys_full - cy) * d_vals / fy
        z_c = d_vals
        pts_c = np.stack([x_c, y_c, z_c, np.ones_like(z_c)], axis=0)

        c2w = np.linalg.inv(extr_4x4[idx])
        pts_w = (c2w @ pts_c)[:3].T
        norms_w = (c2w[:3, :3] @ n_vals.T).T

        # 量化空間座標 (voxel_size) 與法向方向 (1.5 倍量化至 ~8-14 個方向槽)
        v_xyz = np.floor(pts_w / voxel_size).astype(np.int32)
        v_norm = np.round(norms_w * 1.5).astype(np.int16)

        surfel_keys = set(zip(v_xyz[:, 0], v_xyz[:, 1], v_xyz[:, 2], v_norm[:, 0], v_norm[:, 1], v_norm[:, 2]))
        cam_surfel_sets[vid] = surfel_keys
        total_surfels_seen.update(surfel_keys)

    print(f"🌐 候選視角在第 0 幀共觀測到 {len(total_surfels_seen):,} 個獨立 3D 表面微基元 (Surfels)")

    # 4. 次模貪婪最大覆蓋選擇 (Greedy Max Coverage)
    selected = []
    covered_surfels = set()

    for step in range(num_to_select):
        best_cam = None
        best_gain = -1
        for vid in candidates:
            if vid in selected:
                continue
            new_surfels = len(cam_surfel_sets[vid] - covered_surfels)
            if new_surfels > best_gain:
                best_gain = new_surfels
                best_cam = vid

        if best_cam is None:
            break

        selected.append(best_cam)
        covered_surfels.update(cam_surfel_sets[best_cam])
        cov_pct = (len(covered_surfels) / max(len(total_surfels_seen), 1)) * 100
        print(f"   👉 [步驟 {step+1}/{num_to_select}] 選擇相機 cam{best_cam:02d} | 增加 {best_gain:,} 個新表面基元 | 累計表面覆蓋率: {cov_pct:.1f}%")

    total_time = time.time() - t0
    final_selected = sorted(selected)
    print(f"🎉 [相機挑選完成] 最佳 Surfel 覆蓋相機組合: {final_selected} (總挑選耗時: {total_time:.2f} 秒)\n")
    return final_selected
