#!/usr/bin/env python3
"""
統計與評估 6 Scenes x 6 Keyframes 5 組設定之 3DGS 成果：
1. COLMAP (~7k) [reset=3000, refine=300]
2. Group B (7k Importance) [reset=3000, refine=500]
3. Group C (14k Uniform) [reset=3000, refine=500]
4. 50k Importance [reset=3000, refine=500]
5. 50k Importance [reset=1500, refine=500]
"""

import os
import json
import numpy as np

RESULTS_BASE = "/home/tsaichenghan/GSCodec_Studio/results/144exp"

SCENES = [
    "coffee_martini",
    "cook_spinach",
    "cut_roasted_beef",
    "flame_salmon_1",
    "flame_steak",
    "sear_steak",
]

FRAMES = [25, 75, 125, 175, 225, 275]

METHODS = [
    {
        "key": "colmap_reset3000",
        "name": "純 COLMAP (r3000, ref300)",
        "dir_pattern": "f{frame}_3dgs_colmap_reset3000",
    },
    {
        "key": "colmap_refine500_reset3000",
        "name": "純 COLMAP (r3000, ref500)",
        "dir_pattern": "f{frame}_3dgs_colmap_refine500_reset3000",
    },
    {
        "key": "group_b_14k_imp_refine500",
        "name": "組別 B (14k Imp, r3000, ref500)",
        "dir_pattern": "f{frame}_3dgs_giant_colmap_14k_imp_refine500_reset3000",
    },
    {
        "key": "group_b_7k_imp_refine500",
        "name": "組別 B (7k Imp, r3000, ref500)",
        "dir_pattern": "f{frame}_3dgs_giant_colmap_7k_imp_refine500_reset3000",
    },
    {
        "key": "group_c_14k_uni_refine500",
        "name": "組別 C (14k Uni, r3000, ref500)",
        "dir_pattern": "f{frame}_3dgs_giant_colmap_14k_uni_refine500_reset3000",
    },
    {
        "key": "50k_imp_refine500_reset3000",
        "name": "50k Imp (r3000, ref500)",
        "dir_pattern": "f{frame}_3dgs_giant_colmap_50k_imp_refine500_reset3000",
    },
    {
        "key": "50k_uni_refine500_reset3000",
        "name": "50k Uni (r3000, ref500)",
        "dir_pattern": "f{frame}_3dgs_giant_colmap_50k_uni_refine500_reset3000",
    },
    {
        "key": "50k_imp_refine500_reset1500",
        "name": "50k Imp (r1500, ref500)",
        "dir_pattern": "f{frame}_3dgs_giant_colmap_50k_imp_refine500_reset1500",
    },
]


def load_stats(scene, frame, dir_pattern):
    folder_name = dir_pattern.format(frame=frame)
    stats_file = os.path.join(RESULTS_BASE, scene, folder_name, "stats", "val_step4999.json")
    if os.path.isfile(stats_file):
        try:
            with open(stats_file, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def main():
    print("=" * 100)
    print("📊 6 Scenes x 6 Keyframes 3DGS 消融評估總覽 (5000 Steps)")
    print("=" * 100)

    method_scene_psnr = {m["key"]: {s: [] for s in SCENES} for m in METHODS}
    method_global_psnr = {m["key"]: [] for m in METHODS}
    method_global_ssim = {m["key"]: [] for m in METHODS}
    method_global_lpips = {m["key"]: [] for m in METHODS}
    method_global_gs = {m["key"]: [] for m in METHODS}
    completed_counts = {m["key"]: 0 for m in METHODS}

    total_runs = len(SCENES) * len(FRAMES)

    for scene in SCENES:
        print(f"\n🎬 場景: {scene}")
        header = f"{'Frame':<8}"
        for m in METHODS:
            header += f" | {m['name'][:22]:<22}"
        print("-" * len(header))
        print(header)
        print("-" * len(header))

        for frame in FRAMES:
            row = f"f{frame:<6}"
            for m in METHODS:
                stats = load_stats(scene, frame, m["dir_pattern"])
                if stats:
                    psnr = stats.get("psnr", 0.0)
                    ssim = stats.get("ssim", 0.0)
                    lpips = stats.get("lpips", 0.0)
                    num_gs = stats.get("num_GS", 0)
                    method_scene_psnr[m["key"]][scene].append(psnr)
                    method_global_psnr[m["key"]].append(psnr)
                    method_global_ssim[m["key"]].append(ssim)
                    method_global_lpips[m["key"]].append(lpips)
                    method_global_gs[m["key"]].append(num_gs)
                    completed_counts[m["key"]] += 1
                    row += f" | {psnr:6.3f} dB ({num_gs//1000}k)  "
                else:
                    row += f" | {'⏳ 尚未完成':<22}"
            print(row)

        # 場景平均
        avg_row = f"{'均值':<7}"
        for m in METHODS:
            vals = method_scene_psnr[m["key"]][scene]
            if vals:
                avg_val = np.mean(vals)
                avg_row += f" | {avg_val:6.3f} dB ({len(vals)}/{len(FRAMES)})   "
            else:
                avg_row += f" | {'-':<22}"
        print("-" * len(header))
        print(avg_row)

    print("\n" + "=" * 100)
    print("🏆 全局 6 Scene 總結成績比較 (Overall Summary)")
    print("=" * 100)
    summary_header = f"{'Method':<32} | {'進度':<8} | {'平均 PSNR':<11} | {'平均 SSIM':<11} | {'平均 LPIPS':<11} | {'平均 GS數':<10}"
    print(summary_header)
    print("-" * len(summary_header))

    for m in METHODS:
        cnt = completed_counts[m["key"]]
        psnrs = method_global_psnr[m["key"]]
        ssims = method_global_ssim[m["key"]]
        lpipss = method_global_lpips[m["key"]]
        gss = method_global_gs[m["key"]]
        if psnrs:
            m_psnr = f"{np.mean(psnrs):6.3f} dB"
            m_ssim = f"{np.mean(ssims):.4f}"
            m_lpips = f"{np.mean(lpipss):.4f}"
            m_gs = f"{int(np.mean(gss)):,}"
        else:
            m_psnr, m_ssim, m_lpips, m_gs = "-", "-", "-", "-"
        prog = f"{cnt}/{total_runs}"
        print(f"{m['name']:<32} | {prog:<8} | {m_psnr:<11} | {m_ssim:<11} | {m_lpips:<11} | {m_gs:<10}")
    print("=" * 100)


if __name__ == "__main__":
    main()
