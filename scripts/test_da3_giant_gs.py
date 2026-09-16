#!/usr/bin/env python3
"""
測試 DA3-GIANT 的 3D Gaussian Splatting (3DGS) 序列生成功能。
測試條件與 4DGS 對齊：
1. 視角：除 cam00 (測試視角) 外的所有相機視角 (cam01 ~ cam20, 共 20 個視角)，或使用 FPS 選取指定數量視角。
2. 幀數：批次推論序列 (預設前 300 幀，從第 0 幀開始)。
3. 解析度：優先嘗試原圖一半解析度 (1352x1014, 即 process_res=1352)；
   若 12GB 顯存無法負擔當前視角數的高解析度，則自動平滑切換至對應視角數之最佳安全解析度。
"""

import argparse
import datetime
import glob
import json
import os
import re
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.export.gs import save_gaussian_ply


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
    frames = sorted(glob.glob(os.path.join(view_dir, '*.png')) + glob.glob(os.path.join(view_dir, '*.jpg')),
                    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]) if os.path.splitext(os.path.basename(x))[0].isdigit() else os.path.basename(x))
    if not frames:
        frames = sorted(glob.glob(os.path.join(view_dir, 'images', '*.png')) + glob.glob(os.path.join(view_dir, 'images', '*.jpg')),
                        key=lambda x: int(os.path.splitext(os.path.basename(x))[0]) if os.path.splitext(os.path.basename(x))[0].isdigit() else os.path.basename(x))
    return frames


def export_gs_frame(prediction, save_path, gs_views_interval=1):
    """匯出單幀 3D Gaussian Splatting (.ply)"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    gs_world = prediction.gaussians
    pred_depth = torch.from_numpy(prediction.depth).unsqueeze(-1).to(gs_world.means)
    save_gaussian_ply(
        gaussians=gs_world,
        save_path=save_path,
        ctx_depth=pred_depth,
        shift_and_scale=False,
        save_sh_dc_only=True,
        gs_views_interval=gs_views_interval,
        inv_opacity=True,
        prune_by_depth_percent=0.9,
        prune_border_gs=True,
        match_3dgs_mcmc_dev=False,
    )


def run_gs_inference(model, images_list, extr_4x4, intr_3x3, process_res, output_dir, frame_idx=0, verbose=True):
    """執行單幀 GS 推論並匯出 PLY"""
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

    gs_export_dir = os.path.join(output_dir, f"res_{process_res}", "gs_ply")
    ply_path = os.path.join(gs_export_dir, f"{frame_idx:04d}.ply")

    export_gs_frame(pred, ply_path, gs_views_interval=1)
    ply_size_mb = os.path.getsize(ply_path) / (1024 ** 2) if os.path.isfile(ply_path) else 0

    if verbose:
        print(f"✅ 幀 {frame_idx:04d} 推論完成！耗時: {infer_time:.2f}s | 顯存 Allocated: {peak_alloc:.1f} MB (Reserved: {peak_resv:.1f} MB) | PLY: {ply_size_mb:.1f} MB")

    return infer_time, peak_alloc, peak_resv, ply_size_mb


def try_inference(model, images_list, extr_4x4, intr_3x3, process_res, output_dir, frame_idx=0):
    """安全推論函式，若遇到 OOM 則自動清理暫存並恢復模型權重於 CPU"""
    try:
        res = run_gs_inference(model, images_list, extr_4x4, intr_3x3, process_res, output_dir, frame_idx=frame_idx, verbose=True)
        return True, res
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
        return False, None


def select_cams_by_fps(all_w2c_np, num_to_select, alpha=0.8, exclude_indices=[0]):
    """
    使用 FPS (最遠點採樣) 根據 3D 位置與視線方向篩選訓練視角 (與 prepare_n3d_mvtracker_track.py 保持 100% 一致)
    - exclude_indices: 排除的視角 (預設排除 index 0，即 cam00 / view_0 測試視角)
    - alpha: 評分權重 (0.8 代表 80% 位置距離 + 20% 角度差距)
    """
    num_total = len(all_w2c_np)
    valid_candidates = [i for i in range(num_total) if i not in exclude_indices]

    if num_to_select is None or num_to_select >= len(valid_candidates):
        return sorted(valid_candidates)

    centers, dirs = [], []
    for w2c in all_w2c_np:
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        c = -R.T @ t                 # 世界座標下的相機中心
        d = R.T @ np.array([0, 0, 1]) # 相機 Look-at 方向
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


class DualLogger:
    """雙向日誌器：同時輸出至終端機與檔案，並帶有即時刷新"""
    def __init__(self, log_filepath):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
        self.log_file = open(log_filepath, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


def main():
    parser = argparse.ArgumentParser(description="DA3-GIANT 3D Gaussian Splatting 序列生成")
    parser.add_argument("--dir", type=str, default="datasets/sear_steak", help="資料集路徑 (預設: datasets/sear_steak)")
    parser.add_argument("--start_frame", type=int, default=0, help="開始生成的幀索引 (預設: 0)")
    parser.add_argument("--max_frames", type=int, default=300, help="總共要生成的最大幀數 (預設: 300)")
    parser.add_argument("--output_dir", type=str, default="mvtracker_results/output_da3_gs", help="3DGS 輸出目錄 (預設: mvtracker_results/output_da3_gs)")
    parser.add_argument("--model_id", type=str, default="depth-anything/DA3-GIANT-1.1", help="DA3 模型 ID")
    parser.add_argument("--force_res", type=int, default=None, help="強制指定推論解析度 (例如 560, 1008 或 1232)，不進行自動降級探測")
    parser.add_argument("--num_cams", type=int, default=None, help="使用 FPS 最遠點取樣選取的相機數量 (例如 4 或 6，預設 None 為全部 20 視角)")
    parser.add_argument("--fps_alpha", type=float, default=0.8, help="FPS 篩選權重: 3D 位置比例 (預設 0.8，即 80% 位置 + 20% 視線方向)")
    parser.add_argument("--skip_existing", action="store_true", default=True, help="若目標 PLY 檔案已存在則跳過生成 (預設: True)")
    parser.add_argument("--no_skip_existing", dest="skip_existing", action="store_false", help="強制重新生成已存在的 PLY 檔案")
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

    # 讀取相機參數
    poses_path = os.path.join(args.dir, 'poses_bounds.npy')
    if not os.path.isfile(poses_path):
        raise FileNotFoundError(f"找不到相機參數檔: {poses_path}")
    poses_bounds = np.load(poses_path)
    all_poses_llff = poses_bounds[:, :-2].reshape([-1, 3, 5])

    if len(view_dirs) != len(all_poses_llff):
        raise ValueError(f"相機視角資料夾數量 ({len(view_dirs)}) 與 poses_bounds.npy 數量 ({len(all_poses_llff)}) 不相符！")

    # 檢視第 0 個相機首幀以確認原圖尺寸
    sample_frames = get_sorted_frame_paths(view_dirs[0])
    if not sample_frames:
        raise FileNotFoundError(f"在 {view_dirs[0]} 找不到影像檔案")
    test_img = Image.open(sample_frames[0])
    W_orig, H_orig = test_img.size
    print(f"🖼️ 原圖解析度: {W_orig}x{H_orig} (長邊一半為 {W_orig // 2})")

    # 預先計算所有相機的 OpenCV 外參 (W2C) 與內參 (K)
    all_w2c, all_intrs = [], []
    for i in range(len(all_poses_llff)):
        w2c, K = llff_to_opencv_w2c(all_poses_llff[i], W_orig, H_orig)
        all_w2c.append(w2c)
        all_intrs.append(K)
    all_w2c_np = np.stack(all_w2c)
    all_intrs_np = np.stack(all_intrs)

    # 2. 依照 4DGS 與 prepare_n3d_mvtracker_track.py 條件：
    # 排除 cam00 (測試視角，即 index 0)，並使用 FPS (80% 位置 + 20% 視角方向) 挑選訓練相機
    if args.num_cams is not None:
        selected_cam_indices = select_cams_by_fps(all_w2c_np, num_to_select=args.num_cams, alpha=args.fps_alpha, exclude_indices=[0])
        selected_cams_dirs = [view_dirs[i] for i in selected_cam_indices]
        print(f"📷 總共相機數: {len(view_dirs)} | 排除 cam00 (view_0) 測試視角")
        print(f"🎯 自動使用 FPS ({round(args.fps_alpha*100)}% 位置 + {round((1-args.fps_alpha)*100)}% 角度) 挑選出 {len(selected_cams_dirs)} 個訓練視角: {[os.path.basename(d) for d in selected_cams_dirs]}")
    else:
        selected_cam_indices = [i for i in range(len(view_dirs)) if parse_cam_idx(view_dirs[i]) != 0]
        selected_cams_dirs = [view_dirs[i] for i in selected_cam_indices]
        print(f"📷 總共相機數: {len(view_dirs)} | 已排除 cam00，選中全部訓練相機: {len(selected_cams_dirs)} 個視角")
        print(f"   選取相機清單: {[os.path.basename(d) for d in selected_cams_dirs]}")

    # 3. 讀取全部選定視角的幀路徑列表
    cam_frame_paths = {}
    for d in selected_cams_dirs:
        frames = get_sorted_frame_paths(d)
        if not frames:
            raise FileNotFoundError(f"在 {d} 找不到影像檔案")
        cam_frame_paths[d] = frames

    total_available_frames = min(len(frames) for frames in cam_frame_paths.values())
    start_frame = args.start_frame
    end_frame = min(start_frame + args.max_frames, total_available_frames)
    num_frames_to_run = max(0, end_frame - start_frame)
    if num_frames_to_run == 0:
        raise ValueError(f"無有效幀可處理 (start_frame={start_frame}, total_available_frames={total_available_frames})")

    print(f"🎬 預計生成幀區間: 第 {start_frame} 幀 至 第 {end_frame - 1} 幀 (共 {num_frames_to_run} 幀)")

    # 4. 構建選中視角的 OpenCV 外參與內參矩陣 (靜態相機陣列只需計算一次)
    num_views = len(selected_cams_dirs)
    extr_4x4 = np.zeros((num_views, 4, 4), dtype=np.float32)
    extr_4x4[:, :3, :4] = all_w2c_np[selected_cam_indices]
    extr_4x4[:, 3, 3] = 1.0
    intr_3x3 = all_intrs_np[selected_cam_indices]

    # 5. 載入模型並啟用分層推論
    print(f"\n🧠 載入 {args.model_id} ...")
    model = DepthAnything3.from_pretrained(args.model_id)
    if device.type == "cuda":
        total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        if total_vram_gb <= 16.0:
            model = enable_da3_layer_offload(model, device=device)
        else:
            model = model.to(device)
    else:
        model = model.to(device)
    model.eval()

    # 6. 解析度策略探測
    if args.force_res is not None:
        target_res = args.force_res
        print(f"🎯 使用指定推論解析度: process_res={target_res}")
    else:
        # 根據相機數量決定推薦的安全高解析度
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

        half_native_res = W_orig // 2 # 1352
        print(f"\n========================================================")
        print(f"▶️ [解析度探測] 測試首幀 ({start_frame}) 原生一半解析度 ({half_native_res})...")
        print(f"========================================================")
        first_frame_imgs = [Image.open(cam_frame_paths[d][start_frame]).convert("RGB") for d in selected_cams_dirs]
        success, _ = try_inference(model, first_frame_imgs, extr_4x4, intr_3x3, half_native_res, args.output_dir, frame_idx=start_frame)
        if success:
            target_res = half_native_res
            print(f"🎉 狂賀！RTX 5070 成功支援原生一半解析度 ({half_native_res})，全序列將以此解析度生成！")
        else:
            target_res = default_safe_res
            print(f"🔄 原生一半解析度顯存超出上限，全序列自動採用最優安全解析度: process_res={target_res}")

    # 7. 批次生成 300 幀 GS PLY
    dataset_name = os.path.basename(os.path.normpath(args.dir))
    if os.path.basename(os.path.normpath(args.output_dir)) == dataset_name:
        dataset_out_dir = args.output_dir
    else:
        dataset_out_dir = os.path.join(args.output_dir, dataset_name)

    dataset_res_dir = os.path.join(dataset_out_dir, f"res_{target_res}")
    gs_export_dir = os.path.join(dataset_res_dir, "gs_ply")
    os.makedirs(gs_export_dir, exist_ok=True)

    # 掛載雙向日誌器 (同時輸出到終端機與 generation.log 檔案)
    log_file_path = os.path.join(dataset_res_dir, "generation.log")
    sys.stdout = DualLogger(log_file_path)

    print(f"\n========================================================")
    print(f"🚀 開始批次生成 {num_frames_to_run} 幀 3DGS (.ply)")
    print(f"📂 輸出資料夾: {gs_export_dir}")
    print(f"📝 日誌檔案: {log_file_path}")
    print(f"⚙️ 執行配置: 資料集={dataset_name} | 相機數={num_views} 視角 | 解析度 process_res={target_res}")
    print(f"========================================================\n")

    total_start_t = time.time()
    processed_count = 0

    for idx, f_idx in enumerate(range(start_frame, end_frame)):
        ply_path = os.path.join(gs_export_dir, f"{f_idx:04d}.ply")
        if args.skip_existing and os.path.isfile(ply_path) and os.path.getsize(ply_path) > 0:
            print(f"⏩ [{idx+1:03d}/{num_frames_to_run:03d}] 幀 {f_idx:04d} 已存在 ({os.path.getsize(ply_path)/(1024**2):.1f} MB)，跳過生成")
            continue

        frame_imgs = [Image.open(cam_frame_paths[d][f_idx]).convert("RGB") for d in selected_cams_dirs]
        infer_time, peak_alloc, peak_resv, ply_size = run_gs_inference(
            model, frame_imgs, extr_4x4, intr_3x3, target_res, dataset_out_dir, frame_idx=f_idx, verbose=False
        )
        processed_count += 1

        elapsed = time.time() - total_start_t
        avg_time = elapsed / (idx + 1)
        remaining = avg_time * (num_frames_to_run - (idx + 1))
        rem_m, rem_s = divmod(int(remaining), 60)
        rem_h, rem_m = divmod(rem_m, 60)

        print(f"✅ [{idx+1:03d}/{num_frames_to_run:03d}] 幀 {f_idx:04d} 完成 | 耗時: {infer_time:.2f}s | 顯存: {peak_alloc:.1f}MB | PLY: {ply_size:.1f}MB | 預估剩餘: {rem_h:02d}h {rem_m:02d}m {rem_s:02d}s")
        torch.cuda.empty_cache()

    total_time = time.time() - total_start_t
    print(f"\n🎉 批次生成完成！共處理 {num_frames_to_run} 幀 (新生成 {processed_count} 幀)，總耗時: {total_time/60:.2f} 分鐘")
    print(f"📁 產出 3DGS PLY 檔案位於: {gs_export_dir}")
    print(f"📝 完整日誌已記錄至: {log_file_path}")

    # 輸出結構化統計摘要 JSON
    summary = {
        "dataset": dataset_name,
        "model_id": args.model_id,
        "target_res": target_res,
        "num_views": num_views,
        "selected_cams": [os.path.basename(d) for d in selected_cams_dirs],
        "start_frame": start_frame,
        "end_frame": end_frame,
        "total_frames_processed": num_frames_to_run,
        "new_frames_generated": processed_count,
        "total_time_seconds": round(total_time, 2),
        "total_time_minutes": round(total_time / 60, 2),
        "avg_time_per_frame_seconds": round(total_time / max(processed_count, 1), 2) if processed_count > 0 else 0,
        "output_ply_dir": gs_export_dir,
        "log_path": log_file_path,
        "finished_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    summary_path = os.path.join(dataset_res_dir, "generation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"📊 統計摘要 JSON 已儲存至: {summary_path}")


if __name__ == "__main__":
    main()
