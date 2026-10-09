#!/usr/bin/env python3
"""
Comprehensive DA3-GIANT Evaluation on Coffee Martini (Frame 25):
1. 5 FPS Cameras: RGB, Depth, Confidence, and 3DGS Projected Depth.
2. All 17 Training Cameras (excluding cam00): RGB, Depth, Confidence, and 3DGS Projected Depth.
3. In-depth analysis of Left Window, Center Blinds, and Right Outdoor Window.
"""

import os
import sys
import glob
import re
import json
import time
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from depth_anything_3.api import DepthAnything3
from depth_anything_3.model.utils.gs_renderer import render_3dgs
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

    print("⚡ DA3-GIANT Layer Offload Mounted (~840 MB base VRAM)")
    return model


def run_experiment(model, device, scene_dir, cam_dirs, frame_idx, process_res, exp_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cam_names = [os.path.basename(d) for d in cam_dirs]
    num_cams = len(cam_dirs)
    print(f"\n=======================================================")
    print(f"🚀 Running {exp_name}: {num_cams} cameras, process_res={process_res}")
    print(f"   Cameras: {cam_names}")
    print(f"=======================================================")

    # 1. 讀取相機參數
    poses_bounds = np.load(os.path.join(scene_dir, "poses_bounds.npy"))
    all_poses_llff = poses_bounds[:, :-2].reshape([-1, 3, 5])

    all_view_dirs = sorted([
        os.path.join(scene_dir, d) for d in os.listdir(scene_dir)
        if os.path.isdir(os.path.join(scene_dir, d)) and d.startswith("cam")
    ], key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))

    sample_img = Image.open(os.path.join(cam_dirs[0], f"{frame_idx}.png"))
    W_orig, H_orig = sample_img.size

    all_w2c, all_intrs = [], []
    for i in range(len(all_poses_llff)):
        w2c, K = llff_to_opencv_w2c(all_poses_llff[i], W_orig, H_orig)
        all_w2c.append(w2c)
        all_intrs.append(K)
    all_w2c_np = np.stack(all_w2c)
    all_intrs_np = np.stack(all_intrs)

    cam_indices = [all_view_dirs.index(d) for d in cam_dirs]
    extr_4x4 = np.zeros((num_cams, 4, 4), dtype=np.float32)
    extr_4x4[:, :3, :4] = all_w2c_np[cam_indices]
    extr_4x4[:, 3, 3] = 1.0
    intr_3x3 = all_intrs_np[cam_indices]

    # 2. 讀取影像
    imgs = [Image.open(os.path.join(d, f"{frame_idx}.png")).convert("RGB") for d in cam_dirs]

    # 3. 執行推論
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        pred = model.inference(
            imgs,
            extrinsics=extr_4x4,
            intrinsics=intr_3x3,
            process_res=process_res,
            infer_gs=True,
            align_to_input_ext_scale=True,
            ref_view_strategy="saddle_balanced",
        )
    infer_time = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"✅ 推論完成！耗時: {infer_time:.2f}s | 顯存峰值: {peak_vram:.1f} MB")

    depths = pred.depth  # (num_cams, H_p, W_p)
    confs = pred.conf    # (num_cams, H_p, W_p)
    H_p, W_p = depths.shape[1], depths.shape[2]

    # 4. 渲染 3DGS 深度圖 (將 XYZ 反投影回各視角)
    print(f"🎯 正在透過 gsplat 將 3DGS 幾何投影回 {num_cams} 台相機生成深度圖...")
    gs = pred.gaussians
    intr_normed = torch.from_numpy(intr_3x3).clone().to(device)
    intr_normed[:, 0, :] /= W_orig
    intr_normed[:, 1, :] /= H_orig
    extr_tensor = torch.from_numpy(extr_4x4).clone().to(device)

    with torch.no_grad():
        rgb_rend, depth_rend = render_3dgs(
            extrinsics=extr_tensor,
            intrinsics=intr_normed,
            image_shape=(H_p, W_p),
            gaussian=gs,
            num_view=num_cams,
            color_mode="RGB+D"
        )
    gs_depths_np = depth_rend.detach().cpu().numpy()  # (num_cams, H_p, W_p)
    print(f"✅ 3DGS 反投影深度完成！範圍: [{gs_depths_np.min():.2f}, {gs_depths_np.max():.2f}] m")

    # 5. 匯出 3DGS PLY (若是 5 台相機或特定實驗)
    ply_path = os.path.join(out_dir, f"{exp_name}_3dgs.ply")
    pred_depth_tensor = torch.from_numpy(depths).unsqueeze(-1).to(gs.means)
    save_gaussian_ply(
        gaussians=gs,
        save_path=ply_path,
        ctx_depth=pred_depth_tensor,
        shift_and_scale=False,
        save_sh_dc_only=True,
        gs_views_interval=1,
        inv_opacity=True,
        prune_by_depth_percent=0.9,
        prune_border_gs=True,
        match_3dgs_mcmc_dev=False,
    )
    print(f"💾 3DGS PLY 已儲存: {ply_path} ({os.path.getsize(ply_path)/(1024**2):.1f} MB)")

    # 6. 儲存原始數值矩陣
    npz_path = os.path.join(out_dir, f"{exp_name}_data.npz")
    np.savez_compressed(
        npz_path,
        cam_names=cam_names,
        depths=depths,
        confs=confs,
        gs_depths=gs_depths_np,
        extr_4x4=extr_4x4,
        intr_3x3=intr_3x3,
    )

    return {
        "exp_name": exp_name,
        "cam_names": cam_names,
        "imgs": imgs,
        "depths": depths,
        "confs": confs,
        "gs_depths": gs_depths_np,
        "ply_path": ply_path,
        "npz_path": npz_path,
        "infer_time": infer_time,
        "peak_vram": peak_vram,
        "H_p": H_p,
        "W_p": W_p,
    }


def plot_5cams_results(res5, out_png):
    cam_names = res5["cam_names"]
    imgs = res5["imgs"]
    depths = res5["depths"]
    confs = res5["confs"]
    gs_depths = res5["gs_depths"]
    H_p, W_p = res5["H_p"], res5["W_p"]

    fig, axes = plt.subplots(4, 5, figsize=(22, 14), dpi=200)
    plt.subplots_adjust(wspace=0.08, hspace=0.18)

    # 統一顏色刻度
    d_vmax = 24.0
    c_vmin = 1.0
    c_vmax = max(confs.max(), 12.0)

    for v in range(5):
        # 1. RGB
        ax = axes[0, v]
        im_res = imgs[v].resize((W_p, H_p), Image.Resampling.BILINEAR)
        ax.imshow(im_res)
        ax.set_title(f"{cam_names[v]}\nRGB Input", fontsize=11, fontweight="bold", pad=4)
        ax.axis("off")

        # 2. DA3 Depth
        ax = axes[1, v]
        d_im = ax.imshow(depths[v], cmap="plasma", vmin=0, vmax=d_vmax)
        ax.set_title(f"DA3 Depth\nMean: {depths[v].mean():.1f}m (Max: {depths[v].max():.1f}m)", fontsize=10, pad=4)
        ax.axis("off")

        # 3. DA3 Confidence
        ax = axes[2, v]
        c_im = ax.imshow(confs[v], cmap="turbo", vmin=c_vmin, vmax=c_vmax)
        ax.set_title(f"DA3 Confidence\nMean: {confs[v].mean():.1f} (Min: {confs[v].min():.1f})", fontsize=10, pad=4)
        ax.axis("off")

        # 4. Projected 3DGS Depth
        ax = axes[3, v]
        gs_d_im = ax.imshow(gs_depths[v], cmap="plasma", vmin=0, vmax=d_vmax)
        ax.set_title(f"3DGS Projected Depth\nMean: {gs_depths[v].mean():.1f}m", fontsize=10, pad=4)
        ax.axis("off")

    # Colorbars
    cb_d_ax = fig.add_axes([0.92, 0.54, 0.012, 0.16])
    fig.colorbar(d_im, cax=cb_d_ax).set_label("Depth (m)", fontsize=9)

    cb_c_ax = fig.add_axes([0.92, 0.30, 0.012, 0.16])
    fig.colorbar(c_im, cax=cb_c_ax).set_label("Confidence", fontsize=9)

    cb_gs_ax = fig.add_axes([0.92, 0.07, 0.012, 0.16])
    fig.colorbar(gs_d_im, cax=cb_gs_ax).set_label("3DGS Depth (m)", fontsize=9)

    fig.suptitle("Coffee Martini (Frame 25) - DA3-GIANT 5 FPS Cameras Evaluation\nComparing RGB, Base Depth, Confidence Map, and Direct 3DGS Projected Depth",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    print(f"📊 5 台相機完整對照圖已儲存: {out_png}")


def plot_17cams_full_matrix(res17, out_png):
    cam_names = res17["cam_names"]
    imgs = res17["imgs"]
    depths = res17["depths"]
    confs = res17["confs"]
    gs_depths = res17["gs_depths"]
    H_p, W_p = res17["H_p"], res17["W_p"]
    num_cams = len(cam_names)

    # 17 rows, 4 columns: [RGB | DA3 Depth | Confidence | 3DGS Projected Depth]
    fig, axes = plt.subplots(num_cams, 4, figsize=(16, 42), dpi=150)
    plt.subplots_adjust(wspace=0.06, hspace=0.12)

    d_vmax = 24.0
    c_vmin = 1.0
    c_vmax = 14.0

    for i in range(num_cams):
        # Col 0: RGB
        ax = axes[i, 0]
        im_res = imgs[i].resize((W_p, H_p), Image.Resampling.BILINEAR)
        ax.imshow(im_res)
        ax.set_ylabel(cam_names[i], fontsize=11, fontweight="bold", labelpad=8)
        if i == 0:
            ax.set_title("RGB Input", fontsize=12, fontweight="bold", pad=6)
        ax.set_xticks([])
        ax.set_yticks([])

        # Col 1: DA3 Depth
        ax = axes[i, 1]
        ax.imshow(depths[i], cmap="plasma", vmin=0, vmax=d_vmax)
        if i == 0:
            ax.set_title("DA3 Base Depth (m)", fontsize=12, fontweight="bold", pad=6)
        ax.set_xticks([])
        ax.set_yticks([])

        # Col 2: Confidence
        ax = axes[i, 2]
        ax.imshow(confs[i], cmap="turbo", vmin=c_vmin, vmax=c_vmax)
        if i == 0:
            ax.set_title("DA3 Confidence Map", fontsize=12, fontweight="bold", pad=6)
        ax.set_xticks([])
        ax.set_yticks([])

        # Col 3: Projected 3DGS Depth
        ax = axes[i, 3]
        ax.imshow(gs_depths[i], cmap="plasma", vmin=0, vmax=d_vmax)
        if i == 0:
            ax.set_title("3DGS Projected Depth (m)", fontsize=12, fontweight="bold", pad=6)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Coffee Martini (Frame 25) - ALL 17 Training Cameras Full Evaluation (cam01 ~ cam20, Excl. cam00)\nColumns: RGB | DA3 Depth | DA3 Confidence | 3DGS Projected Depth",
                 fontsize=14, fontweight="bold", y=0.995)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    print(f"📊 17 台相機完整矩陣圖已儲存: {out_png}")


def plot_window_anomaly_spotlight(res17, out_png):
    """精準聚焦分析左側窗戶 (Left Window) 與右側室外大窗 (Right Window) 在多相機下的信心與深度異常"""
    cam_names = res17["cam_names"]
    imgs = res17["imgs"]
    depths = res17["depths"]
    confs = res17["confs"]
    gs_depths = res17["gs_depths"]
    H_p, W_p = res17["H_p"], res17["W_p"]

    # 挑選具備代表性視角的 6 台相機：
    # cam01, cam02: 左側窗戶最明顯
    # cam06, cam10: 中央與右側窗戶轉折
    # cam16, cam20: 右側室外窗戶最大視角
    spotlight_names = ["cam01", "cam02", "cam06", "cam10", "cam16", "cam20"]
    spotlight_indices = [cam_names.index(c) for c in spotlight_names]

    fig, axes = plt.subplots(4, 6, figsize=(24, 14), dpi=200)
    plt.subplots_adjust(wspace=0.08, hspace=0.18)

    d_vmax = 24.0
    c_vmin = 1.0
    c_vmax = 14.0

    for col, idx in enumerate(spotlight_indices):
        c_name = cam_names[idx]
        # Row 0: RGB
        ax = axes[0, col]
        im_res = imgs[idx].resize((W_p, H_p), Image.Resampling.BILINEAR)
        ax.imshow(im_res)
        ax.set_title(f"{c_name}\nRGB Input", fontsize=11, fontweight="bold", pad=4)
        ax.axis("off")

        # Row 1: DA3 Base Depth
        ax = axes[1, col]
        ax.imshow(depths[idx], cmap="plasma", vmin=0, vmax=d_vmax)
        ax.set_title("DA3 Base Depth", fontsize=10, pad=4)
        ax.axis("off")

        # Row 2: Confidence Map
        ax = axes[2, col]
        c_im = ax.imshow(confs[idx], cmap="turbo", vmin=c_vmin, vmax=c_vmax)
        ax.set_title("DA3 Confidence Map", fontsize=10, pad=4)
        ax.axis("off")

        # Row 3: 3DGS Projected Depth
        ax = axes[3, col]
        d_im = ax.imshow(gs_depths[idx], cmap="plasma", vmin=0, vmax=d_vmax)
        ax.set_title("3DGS Projected Depth", fontsize=10, pad=4)
        ax.axis("off")

    cb_c_ax = fig.add_axes([0.92, 0.31, 0.012, 0.16])
    fig.colorbar(c_im, cax=cb_c_ax).set_label("Confidence Score", fontsize=9)

    cb_d_ax = fig.add_axes([0.92, 0.08, 0.012, 0.16])
    fig.colorbar(d_im, cax=cb_d_ax).set_label("Depth (meters)", fontsize=9)

    fig.suptitle("Coffee Martini - Multi-Window Spotlight Analysis (Left Window vs Center Blinds vs Right Window)\nObserving Left Window Confidence Drop (Deep Blue < 3.0) and Right Window Squashed Depth (~20m)",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    print(f"📊 窗戶焦點深入對照圖已儲存: {out_png}")


def main():
    scene_name = "coffee_martini"
    frame_idx = 25
    scene_dir = os.path.join(PROJECT_ROOT, "datasets", scene_name)
    out_dir = os.path.join(PROJECT_ROOT, "mvtracker_results", "da3_confidence_inspect")
    os.makedirs(out_dir, exist_ok=True)

    # 搜尋所有訓練相機 (排除 cam00)
    all_view_dirs = sorted([
        os.path.join(scene_dir, d) for d in os.listdir(scene_dir)
        if os.path.isdir(os.path.join(scene_dir, d)) and d.startswith("cam")
    ], key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))

    train_view_dirs = [d for d in all_view_dirs if os.path.basename(d) != "cam00"]
    print(f"📷 總共相機數: {len(all_view_dirs)} | 排除 cam00，訓練相機共: {len(train_view_dirs)} 台")

    # FPS 選取的 5 台相機
    fps_names = ["cam01", "cam06", "cam09", "cam14", "cam20"]
    fps_view_dirs = [d for d in all_view_dirs if os.path.basename(d) in fps_names]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 載入 DA3-GIANT-1.1 模型 on {device}...")
    model = DepthAnything3.from_pretrained("depth-anything/DA3-GIANT-1.1")
    model = enable_da3_layer_offload(model, device)
    model.eval()

    # 1. 執行 5 台相機測試
    res5 = run_experiment(
        model=model,
        device=device,
        scene_dir=scene_dir,
        cam_dirs=fps_view_dirs,
        frame_idx=frame_idx,
        process_res=728,
        exp_name="fps_5cams",
        out_dir=out_dir
    )
    png5_path = os.path.join(out_dir, "coffee_martini_fps5_full_analysis.png")
    plot_5cams_results(res5, png5_path)

    # 2. 執行所有 17 台相機測試
    res17 = run_experiment(
        model=model,
        device=device,
        scene_dir=scene_dir,
        cam_dirs=train_view_dirs,
        frame_idx=frame_idx,
        process_res=504,
        exp_name="all_17cams",
        out_dir=out_dir
    )
    png17_path = os.path.join(out_dir, "coffee_martini_all17cams_full_matrix.png")
    plot_17cams_full_matrix(res17, png17_path)

    spotlight_png_path = os.path.join(out_dir, "coffee_martini_window_spotlight.png")
    plot_window_anomaly_spotlight(res17, spotlight_png_path)

    # 3. 輸出詳細數值分析與統計 JSON
    summary_stats = {
        "scene": scene_name,
        "frame": frame_idx,
        "fps_5cams": {
            "cams": res5["cam_names"],
            "depth_means": [float(d.mean()) for d in res5["depths"]],
            "conf_means": [float(c.mean()) for c in res5["confs"]],
            "conf_mins": [float(c.min()) for c in res5["confs"]],
            "gs_depth_means": [float(g.mean()) for g in res5["gs_depths"]],
            "ply": res5["ply_path"]
        },
        "all_17cams": {
            "cams": res17["cam_names"],
            "depth_means": [float(d.mean()) for d in res17["depths"]],
            "conf_means": [float(c.mean()) for c in res17["confs"]],
            "conf_mins": [float(c.min()) for c in res17["confs"]],
            "gs_depth_means": [float(g.mean()) for g in res17["gs_depths"]],
            "ply": res17["ply_path"]
        }
    }
    stats_json_path = os.path.join(out_dir, "coffee_martini_full_evaluation_stats.json")
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2, ensure_ascii=False)
    print(f"📝 完整統計摘要 JSON 已儲存: {stats_json_path}")
    print("🎉 全部測試順利完成！")


if __name__ == "__main__":
    main()
