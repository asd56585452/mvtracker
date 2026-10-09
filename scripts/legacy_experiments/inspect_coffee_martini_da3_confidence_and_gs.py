#!/usr/bin/env python3
"""
Inspect DA3-GIANT confidence score maps and direct 3DGS prediction on coffee_martini.
Analyzes whether DA3 confidence drops significantly on the window/outdoor area,
and evaluates if the directly predicted 3DGS correctly places outdoor geometry.
"""

import os
import sys
import glob
import json
import re
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.export.gs import save_gaussian_ply


def llff_to_opencv_w2c(pose_llff, actual_W, actual_H):
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


def select_cams_by_fps(all_w2c_np, num_to_select, alpha=0.8, exclude_indices=[0]):
    num_total = len(all_w2c_np)
    valid_candidates = [i for i in range(num_total) if i not in exclude_indices]

    if num_to_select is None or num_to_select >= len(valid_candidates):
        return sorted(valid_candidates)

    centers, dirs = [], []
    for w2c in all_w2c_np:
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        c = -R.T @ t
        d = R.T @ np.array([0, 0, 1])
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


def enable_da3_layer_offload(model, device: torch.device):
    blocks = model.model.backbone.pretrained.blocks
    model.to(device)
    for blk in blocks:
        blk.to("cpu")
    torch.cuda.empty_cache()

    for blk in blocks:
        def make_pre():
            def pre_hook(m, args, kwargs):
                m.to(device)
                return args, kwargs
            return pre_hook

        def make_post():
            def post_hook(m, args, out):
                m.to("cpu")
                return out
            return post_hook

        blk.register_forward_pre_hook(make_pre(), with_kwargs=True)
        blk.register_forward_hook(make_post())

    print("⚡ DA3-GIANT 動態分層推論已掛載 (常駐顯存 ~840 MB)")
    return model


def main():
    scene_name = "coffee_martini"
    frame_idx = 25
    process_res = 728
    artifact_dir = os.path.join(PROJECT_ROOT, "mvtracker_results", "da3_confidence_inspect")
    os.makedirs(artifact_dir, exist_ok=True)

    scene_dir = os.path.join(PROJECT_ROOT, "datasets", scene_name)
    poses_path = os.path.join(scene_dir, "poses_bounds.npy")
    poses_bounds = np.load(poses_path)
    all_poses_llff = poses_bounds[:, :-2].reshape([-1, 3, 5])

    view_dirs = sorted([
        os.path.join(scene_dir, d) for d in os.listdir(scene_dir)
        if os.path.isdir(os.path.join(scene_dir, d)) and d.startswith("cam")
    ], key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))

    first_frame_path = os.path.join(view_dirs[0], f"{frame_idx}.png")
    if not os.path.exists(first_frame_path):
        first_frame_path = glob.glob(os.path.join(view_dirs[0], "*.png"))[0]
    with Image.open(first_frame_path) as im:
        W_orig, H_orig = im.size

    all_w2c, all_intrs = [], []
    for i in range(len(all_poses_llff)):
        w2c, K = llff_to_opencv_w2c(all_poses_llff[i], W_orig, H_orig)
        all_w2c.append(w2c)
        all_intrs.append(K)
    all_w2c_np = np.stack(all_w2c)
    all_intrs_np = np.stack(all_intrs)

    # 1. FPS 挑選 5 台相機 (排除 cam00 測試視角)
    selected_indices = select_cams_by_fps(all_w2c_np, num_to_select=5, alpha=0.8, exclude_indices=[0])
    selected_cams = [view_dirs[i] for i in selected_indices]
    selected_names = [os.path.basename(d) for d in selected_cams]
    print(f"🎯 FPS 選取的 5 台相機: {selected_names} (索引: {selected_indices})")

    # 讀取這 5 台相機的影像
    images_list = []
    cam_imgs_np = []
    for d in selected_cams:
        img_path = os.path.join(d, f"{frame_idx}.png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"找不到檔案: {img_path}")
        im = Image.open(img_path).convert("RGB")
        images_list.append(im)
        cam_imgs_np.append(np.array(im))

    # 準備外參與內參
    extr_4x4 = np.zeros((5, 4, 4), dtype=np.float32)
    extr_4x4[:, :3, :4] = all_w2c_np[selected_indices]
    extr_4x4[:, 3, 3] = 1.0
    intr_3x3 = all_intrs_np[selected_indices]

    # 2. 載入 DA3-GIANT 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 載入 DepthAnything3 DA3-GIANT-1.1 on {device}...")
    model = DepthAnything3.from_pretrained("depth-anything/DA3-GIANT-1.1")
    model = enable_da3_layer_offload(model, device)
    model.eval()

    # 3. 執行推論 (infer_gs=True)
    print(f"🚀 開始執行 5 視角 DA3-GIANT 推論 (process_res={process_res}, infer_gs=True)...")
    torch.cuda.empty_cache()
    t0 = time.time()
    with torch.no_grad():
        pred = model.inference(
            images_list,
            extrinsics=extr_4x4,
            intrinsics=intr_3x3,
            process_res=process_res,
            infer_gs=True,
            align_to_input_ext_scale=True,
            ref_view_strategy="saddle_balanced",
        )
    infer_time = time.time() - t0
    print(f"✅ 推論完成！耗時: {infer_time:.2f} 秒")

    # 4. 分析深度與信心分數
    depths = pred.depth  # (5, H_p, W_p)
    confs = pred.conf    # (5, H_p, W_p)
    print(f"📊 Depth shape: {depths.shape}, range: [{depths.min():.2f}, {depths.max():.2f}]")
    if confs is not None:
        print(f"📊 Conf shape: {confs.shape}, range: [{confs.min():.2f}, {confs.max():.2f}], mean: {confs.mean():.2f}")
    else:
        print("⚠️ 未取得 pred.conf！")

    # 5. 匯出 DA3Giant 直接預測的 3DGS PLY 檔
    gs_export_dir = os.path.join(scene_dir, f"colmap_{frame_idx}")
    os.makedirs(gs_export_dir, exist_ok=True)
    out_ply_path = os.path.join(gs_export_dir, "da3_giant_direct_3dgs_f25.ply")
    artifact_ply_path = os.path.join(artifact_dir, "da3_giant_direct_3dgs_f25.ply")

    print(f"💾 匯出 3DGS PLY 檔案至: {out_ply_path}...")
    gs_world = pred.gaussians
    pred_depth_tensor = torch.from_numpy(pred.depth).unsqueeze(-1).to(gs_world.means)
    save_gaussian_ply(
        gaussians=gs_world,
        save_path=out_ply_path,
        ctx_depth=pred_depth_tensor,
        shift_and_scale=False,
        save_sh_dc_only=True,
        gs_views_interval=1,
        inv_opacity=True,
        prune_by_depth_percent=0.9,
        prune_border_gs=True,
        match_3dgs_mcmc_dev=False,
    )
    # 同步複製一份至 artifact 資料夾
    import shutil
    shutil.copyfile(out_ply_path, artifact_ply_path)
    ply_size_mb = os.path.getsize(out_ply_path) / (1024 ** 2)
    print(f"✅ 3DGS PLY 已成功匯出！大小: {ply_size_mb:.2f} MB ({out_ply_path})")

    # 分析 3DGS 點雲座標分佈
    means = gs_world.means.detach().cpu().numpy().reshape(-1, 3)
    opacities = gs_world.opacities.detach().cpu().numpy().reshape(-1)
    scales = gs_world.scales.detach().cpu().numpy().reshape(-1, 3)
    print(f"\n🌐 [3DGS 幾何座標統計] 總高斯數: {len(means):,}")
    print(f"   X 軸範圍: [{means[:,0].min():.2f}, {means[:,0].max():.2f}], 均值: {means[:,0].mean():.2f}")
    print(f"   Y 軸範圍: [{means[:,1].min():.2f}, {means[:,1].max():.2f}], 均值: {means[:,1].mean():.2f}")
    print(f"   Z 軸範圍: [{means[:,2].min():.2f}, {means[:,2].max():.2f}], 均值: {means[:,2].mean():.2f}")
    print(f"   Opacity 範圍: [{opacities.min():.4f}, {opacities.max():.4f}], 均值: {opacities.mean():.4f}")

    # 6. 分析窗戶區域 vs 室內前景的信心分數
    # 觀察 coffee_martini 影像：窗戶位於視野上方/右上方的室外背景區域 (大約 Y: [0, 0.45 * H], X: [0.35 * W, 0.95 * W])
    H_p, W_p = confs.shape[1], confs.shape[2]
    # 定義窗戶 ROI 與 前景雞尾酒/吧台 ROI
    # 窗戶 ROI (大略比例)
    win_y0, win_y1 = int(0.05 * H_p), int(0.40 * H_p)
    win_x0, win_x1 = int(0.40 * W_p), int(0.85 * W_p)

    # 前景調酒/吧台 ROI
    fg_y0, fg_y1 = int(0.50 * H_p), int(0.85 * H_p)
    fg_x0, fg_x1 = int(0.30 * W_p), int(0.70 * W_p)

    # 室內牆面/酒架 ROI
    wall_y0, wall_y1 = int(0.10 * H_p), int(0.45 * H_p)
    wall_x0, wall_x1 = int(0.05 * W_p), int(0.35 * W_p)

    roi_stats = []
    print("\n🔍 [各視角信心分數 ROI 深入分析]")
    for v in range(5):
        c_v = confs[v]
        d_v = depths[v]
        win_conf = c_v[win_y0:win_y1, win_x0:win_x1].mean()
        fg_conf = c_v[fg_y0:fg_y1, fg_x0:fg_x1].mean()
        wall_conf = c_v[wall_y0:wall_y1, wall_x0:wall_x1].mean()

        win_depth = d_v[win_y0:win_y1, win_x0:win_x1].mean()
        fg_depth = d_v[fg_y0:fg_y1, fg_x0:fg_x1].mean()
        wall_depth = d_v[wall_y0:wall_y1, wall_x0:wall_x1].mean()

        roi_stats.append({
            "cam": selected_names[v],
            "win_conf": float(win_conf),
            "fg_conf": float(fg_conf),
            "wall_conf": float(wall_conf),
            "win_depth": float(win_depth),
            "fg_depth": float(fg_depth),
            "wall_depth": float(wall_depth),
        })
        print(f"📷 {selected_names[v]}:")
        print(f"   - 窗戶/室外區域: 信心 = {win_conf:.3f} | 預測深度 = {win_depth:.3f}m")
        print(f"   - 前景調酒吧台: 信心 = {fg_conf:.3f} | 預測深度 = {fg_depth:.3f}m")
        print(f"   - 室內酒架牆面: 信心 = {wall_conf:.3f} | 預測深度 = {wall_depth:.3f}m")

    # 7. 繪製視覺化圖表
    # 圖 1: 5 台相機的信心分數圖與 RGB 對照圖
    fig, axes = plt.subplots(3, 5, figsize=(20, 11), dpi=200)
    plt.subplots_adjust(wspace=0.08, hspace=0.15)

    # 統一信心分數顏色範圍以便比較
    conf_vmin = 1.0
    conf_vmax = max(confs.max(), 8.0)

    for v in range(5):
        # Row 0: 原圖 RGB
        ax_rgb = axes[0, v]
        im_resized = images_list[v].resize((W_p, H_p), Image.Resampling.BILINEAR)
        ax_rgb.imshow(im_resized)
        ax_rgb.set_title(f"Cam: {selected_names[v]}\nRGB Input", fontsize=12, fontweight='bold', pad=6)
        # 標出窗戶與前景檢測框
        rect_win = plt.Rectangle((win_x0, win_y0), win_x1 - win_x0, win_y1 - win_y0,
                                 fill=False, edgecolor='cyan', linewidth=1.8, linestyle='--')
        rect_fg = plt.Rectangle((fg_x0, fg_y0), fg_x1 - fg_x0, fg_y1 - fg_y0,
                                fill=False, edgecolor='yellow', linewidth=1.8, linestyle=':')
        ax_rgb.add_patch(rect_win)
        ax_rgb.add_patch(rect_fg)
        ax_rgb.axis("off")

        # Row 1: DA3 深度圖
        ax_d = axes[1, v]
        d_map = ax_d.imshow(depths[v], cmap="plasma")
        ax_d.set_title(f"DA3 Depth (m)\nMean: {depths[v].mean():.2f}m", fontsize=11, pad=6)
        ax_d.axis("off")

        # Row 2: 信心分數圖
        ax_c = axes[2, v]
        c_map = ax_c.imshow(confs[v], cmap="turbo", vmin=conf_vmin, vmax=conf_vmax)
        ax_c.set_title(f"Confidence Score\nWin: {roi_stats[v]['win_conf']:.2f} | FG: {roi_stats[v]['fg_conf']:.2f}",
                       fontsize=11, fontweight='bold', pad=6)
        ax_c.axis("off")

    # 添加 Colorbar
    cbar_ax_d = fig.add_axes([0.92, 0.38, 0.012, 0.24])
    cbar_d = fig.colorbar(d_map, cax=cbar_ax_d)
    cbar_d.set_label("Depth (meters)", fontsize=10)

    cbar_ax_c = fig.add_axes([0.92, 0.08, 0.012, 0.24])
    cbar_c = fig.colorbar(c_map, cax=cbar_ax_c)
    cbar_c.set_label("Confidence (expp1: exp(x)+1)", fontsize=10)

    fig.suptitle(f"Coffee Martini (Frame {frame_idx}) - DA3-GIANT 5 FPS Cameras Confidence & Depth Analysis\nCyan dashed: Window/Outdoor ROI | Yellow dotted: Foreground Table ROI",
                 fontsize=15, fontweight='bold', y=0.98)

    viz_out_path = os.path.join(artifact_dir, "coffee_martini_fps5_confidence_maps.png")
    plt.savefig(viz_out_path, bbox_inches="tight")
    plt.close()
    print(f"📊 5 視角信心分數圖已儲存至: {viz_out_path}")

    # 儲存統計 JSON
    stats_json_path = os.path.join(artifact_dir, "coffee_martini_da3_confidence_stats.json")
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "scene": scene_name,
            "frame": frame_idx,
            "selected_cams": selected_names,
            "roi_stats": roi_stats,
            "ply_path": out_ply_path,
            "artifact_ply_path": artifact_ply_path,
            "num_gaussians": len(means),
            "means_bounds": {
                "x": [float(means[:,0].min()), float(means[:,0].max())],
                "y": [float(means[:,1].min()), float(means[:,1].max())],
                "z": [float(means[:,2].min()), float(means[:,2].max())],
            }
        }, f, indent=2, ensure_ascii=False)
    print(f"📝 統計摘要已儲存至: {stats_json_path}")


if __name__ == "__main__":
    main()
