#!/usr/bin/env python3
"""
測試 DA3-GIANT 的 3D Gaussian Splatting (3DGS) 生成功能。
測試條件與 4DGS 對齊：
1. 視角：除 cam00 (測試視角) 外的所有相機視角 (cam01 ~ cam20, 共 20 個視角)。
2. 幀數：單幀推論 (第 0 幀)。
3. 解析度：優先嘗試原圖一半解析度 (1352x1014, 即 process_res=1352)；
   若 12GB 顯存無法負擔 20 視角的高解析度 Cross-Attention，則自動平滑降級至 560 進行測試。
"""

import argparse
import glob
import os
import re
import sys
import time
import numpy as np
from PIL import Image
import torch

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.export.gs import export_to_gs_ply


def parse_cam_idx(path):
    m = re.search(r'\d+', os.path.basename(path))
    return int(m.group()) if m else 0


def llff_to_opencv_w2c(pose_llff, actual_W, actual_H):
    """將 LLFF 相機位姿轉為 OpenCV 座標系 (X-right, Y-down, Z-forward)"""
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
    """
    啟用 DA3 ViT Block 動態分層推論 (Sequential Block CPU Offload)。
    先將全模型移入 device，隨即將 40 個 ViT blocks 搬回 CPU，並清空顯存快取。
    使常駐顯存降至 ~840 MB。
    """
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


def get_sorted_frame_paths(view_dir):
    frames = sorted(glob.glob(os.path.join(view_dir, '*.png')) + glob.glob(os.path.join(view_dir, '*.jpg')))
    if not frames:
        frames = sorted(glob.glob(os.path.join(view_dir, 'images', '*.png')) + glob.glob(os.path.join(view_dir, 'images', '*.jpg')))
    return frames


def run_gs_inference(model, images_list, extr_4x4, intr_3x3, process_res, output_dir):
    """執行 GS 推論並匯出 PLY"""
    print(f"\n🚀 正在以 process_res={process_res} 執行 DA3-GIANT 3DGS 推論 (共 {len(images_list)} 視角)...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_t = time.time()

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

    infer_time = time.time() - start_t
    peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_resv = torch.cuda.max_memory_reserved() / (1024 ** 2)

    print(f"✅ 推論完成！耗時: {infer_time:.2f} 秒")
    print(f"📊 顯存峰值: Allocated: {peak_alloc:.1f} MB | Reserved: {peak_resv:.1f} MB")

    # 匯出 3D Gaussian Splatting PLY
    os.makedirs(output_dir, exist_ok=True)
    gs_export_dir = os.path.join(output_dir, f"res_{process_res}")
    os.makedirs(gs_export_dir, exist_ok=True)
    
    print(f"💾 正在匯出 3D Gaussian (.ply) 至 {gs_export_dir} ...")
    export_to_gs_ply(pred, gs_export_dir, gs_views_interval=1)

    ply_path = os.path.join(gs_export_dir, "gs_ply/0000.ply")
    if os.path.isfile(ply_path):
        ply_size_mb = os.path.getsize(ply_path) / (1024 ** 2)
        print(f"🎉 3D Gaussian 檔案生成成功: {ply_path} (大小: {ply_size_mb:.2f} MB)")
    else:
        print(f"⚠️ PLY 匯出路徑檢查: {gs_export_dir}/gs_ply")

    return True


def try_inference(model, images_list, extr_4x4, intr_3x3, process_res, output_dir):
    try:
        run_gs_inference(model, images_list, extr_4x4, intr_3x3, process_res, output_dir)
        return True
    except torch.cuda.OutOfMemoryError as e:
        import traceback, gc
        print(f"\n⚠️ 提示: {len(images_list)} 視角在 process_res={process_res} 下超出 12GB 顯存上限 (觸發 OOM)")
        traceback.clear_frames(e.__traceback__)
        del e
        # 確保所有 ViT blocks 歸位回 CPU
        for blk in model.model.backbone.pretrained.blocks:
            blk.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()
        current_vram = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"🧹 已清空暫存顯存，目前常駐顯存: {current_vram:.1f} MB")
        return False


def fps_sampling(points, n_samples):
    """使用最遠點取樣 (Farthest Point Sampling) 挑選幾何基線最大、覆蓋率最佳的相機光心"""
    sampled_indices = [0]
    distances = np.linalg.norm(points - points[0], axis=1)
    for _ in range(1, n_samples):
        farthest_idx = int(np.argmax(distances))
        sampled_indices.append(farthest_idx)
        distances = np.minimum(distances, np.linalg.norm(points - points[farthest_idx], axis=1))
    return sampled_indices


def main():
    parser = argparse.ArgumentParser(description="測試 DA3-GIANT 的 3D Gaussian Splatting 生成功能")
    parser.add_argument("--dir", type=str, default="datasets/sear_steak", help="資料集路徑 (預設: datasets/sear_steak)")
    parser.add_argument("--frame_idx", type=int, default=0, help="要測試的幀索引 (預設: 0)")
    parser.add_argument("--output_dir", type=str, default="output_da3_gs", help="3DGS 輸出目錄")
    parser.add_argument("--model_id", type=str, default="depth-anything/DA3-GIANT-1.1", help="DA3 模型 ID")
    parser.add_argument("--force_res", type=int, default=None, help="強制指定推論解析度 (例如 560, 1008 或 1232)，不進行自動降級測試")
    parser.add_argument("--num_cams", type=int, default=None, help="使用 FPS 最遠點取樣選取的相機數量 (例如 4 或 6，預設 None 為全部 20 視角)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 運算裝置: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # 1. 搜尋所有相機視角
    view_dirs = [f for f in glob.glob(os.path.join(args.dir, 'cam*')) if os.path.isdir(f)]
    if not view_dirs:
        view_dirs = [f for f in glob.glob(os.path.join(args.dir, 'view_*')) if os.path.isdir(f)]
    if not view_dirs:
        raise FileNotFoundError(f"在 {args.dir} 找不到任何相機視角資料夾！")

    view_dirs.sort(key=parse_cam_idx)

    # 2. 依照 4DGS 條件：排除 cam00 (測試相機)，保留 cam01 ~ cam20 (訓練相機)
    all_training_dirs = [d for d in view_dirs if parse_cam_idx(d) != 0]
    all_training_indices = [parse_cam_idx(d) for d in all_training_dirs]

    # 讀取相機參數以備 FPS 採樣與後續推論
    poses_path = os.path.join(args.dir, 'poses_bounds.npy')
    if not os.path.isfile(poses_path):
        raise FileNotFoundError(f"找不到相機參數檔: {poses_path}")
    poses_bounds = np.load(poses_path)
    all_poses_llff = poses_bounds[:, :-2].reshape([-1, 3, 5])

    # 若指定 --num_cams，則使用 FPS 最遠點採樣選取幾何分佈最廣的相機光心
    if args.num_cams is not None and args.num_cams < len(all_training_dirs):
        centers = np.array([all_poses_llff[idx, :, 3] for idx in all_training_indices])
        sampled_sub_indices = fps_sampling(centers, args.num_cams)
        selected_cams_dirs = [all_training_dirs[i] for i in sampled_sub_indices]
        selected_cam_indices = [all_training_indices[i] for i in sampled_sub_indices]
        print(f"📷 總共相機數: {len(view_dirs)} | 排除 cam00 後有 {len(all_training_dirs)} 視角")
        print(f"🎯 啟用 FPS 最遠點取樣: 精選出 {len(selected_cams_dirs)} 個最大基線覆蓋視角: {[os.path.basename(d) for d in selected_cams_dirs]}")
    else:
        selected_cams_dirs = all_training_dirs
        selected_cam_indices = all_training_indices
        print(f"📷 總共相機數: {len(view_dirs)} | 已排除 cam00，選中訓練相機: {len(selected_cams_dirs)} 個視角")
        print(f"   選取相機清單: {[os.path.basename(d) for d in selected_cams_dirs]}")

    # 3. 讀取第 0 幀影像
    images_list = []
    for d in selected_cams_dirs:
        frames = get_sorted_frame_paths(d)
        if len(frames) <= args.frame_idx:
            raise IndexError(f"視角 {d} 的幀數不足 ({len(frames)} <= {args.frame_idx})")
        img_path = frames[args.frame_idx]
        img = Image.open(img_path).convert("RGB")
        images_list.append(img)

    W_orig, H_orig = images_list[0].size
    print(f"🖼️ 原圖解析度: {W_orig}x{H_orig} (長邊一半約為 {W_orig // 2})")

    # 4. 構建 OpenCV 相機外參與內參矩陣
    extr_list, intr_list = [], []
    for cam_idx in selected_cam_indices:
        w2c, K = llff_to_opencv_w2c(all_poses_llff[cam_idx], W_orig, H_orig)
        extr_list.append(w2c)
        intr_list.append(K)

    num_views = len(images_list)
    extr_4x4 = np.zeros((num_views, 4, 4), dtype=np.float32)
    extr_4x4[:, :3, :4] = np.stack(extr_list, axis=0)
    extr_4x4[:, 3, 3] = 1.0
    intr_3x3 = np.stack(intr_list, axis=0)

    # 5. 載入模型並啟用分層推論
    print(f"\n🧠 載入 {args.model_id} ...")
    model = DepthAnything3.from_pretrained(args.model_id)
    # 在 12GB 顯存的 5070 上自動啟用動態分層推論
    if device.type == "cuda":
        total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        if total_vram_gb <= 16.0:
            model = enable_da3_layer_offload(model, device=device)
        else:
            model = model.to(device)
    else:
        model = model.to(device)
    model.eval()

    # 6. 解析度策略：如果指定 force_res 則直接跑；否則先嘗試 1352，若 OOM 則切換至對應視角數的最佳安全解析度
    if args.force_res is not None:
        run_gs_inference(model, images_list, extr_4x4, intr_3x3, args.force_res, args.output_dir)
        return

    # 依相機數量自適應決定安全最佳解析度
    if num_views <= 3:
        default_safe_res = 1232
    elif num_views <= 4:
        default_safe_res = 1008
    elif num_views <= 6:
        default_safe_res = 728
    elif num_views <= 8:
        default_safe_res = 672
    else:
        default_safe_res = 560

    # 先嘗試原生一半解析度 (1352)
    half_native_res = W_orig // 2 # 2704 // 2 = 1352
    print(f"\n========================================================")
    print(f"▶️ [階段一] 嘗試理想條件: 原生一半解析度 ({half_native_res})")
    print(f"========================================================")

    success = try_inference(model, images_list, extr_4x4, intr_3x3, half_native_res, args.output_dir)
    if success:
        print(f"\n🎉 狂賀！RTX 5070 成功在原生一半解析度 ({half_native_res}) 下完成 {num_views} 視角 DA3 3DGS 生成！")
    else:
        print(f"\n========================================================")
        print(f"▶️ [階段二] 自動切換至 {num_views} 視角之最佳安全高解析度 ({default_safe_res}) 執行 DA3 3DGS 生成")
        print(f"========================================================")
        success_safe = try_inference(model, images_list, extr_4x4, intr_3x3, default_safe_res, args.output_dir)
        if success_safe:
            print(f"\n🎉 {default_safe_res} 解析度下 {num_views} 視角 3DGS PLY 生成成功！")
        else:
            print(f"\n❌ {default_safe_res} 解析度推論失敗")


if __name__ == "__main__":
    main()
