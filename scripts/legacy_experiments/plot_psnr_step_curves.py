#!/usr/bin/env python3
"""
繪製 3DGS 訓練步數與驗證 PSNR 折線圖 (PSNR vs. Training Step):
1. Subgroup 1: Coffee Martini + Flame Salmon 1 (2 場景, 12 runs/方法)
2. Subgroup 2: 其餘 4 場景 (Cook Spinach, Cut Roasted Beef, Flame Steak, Sear Steak, 24 runs/方法)

比較方法:
- COLMAP (no reset)
- 舊版 DA3 50k Importance
- DA3 14k Uniform + COLMAP
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 設定字體
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(FONT_PATH):
    fm.fontManager.addfont(FONT_PATH)
    prop = fm.FontProperties(fname=FONT_PATH)
    plt.rcParams["font.family"] = prop.get_name()
else:
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = "/media/tsaichenghan/2TBSSD/results/144exp"
ARTIFACT_DIR = "/home/tsaichenghan/.gemini/antigravity-ide/brain/cd1c9aec-32ca-4d5b-8491-fa8bfec9b350"
OUT_SSD_DIR = "/media/tsaichenghan/2TBSSD/results"

SCENES_SUB1 = ["coffee_martini", "flame_salmon_1"]
SCENES_SUB2 = ["cook_spinach", "cut_roasted_beef", "flame_steak", "sear_steak"]
FRAMES = [25, 75, 125, 175, 225, 275]
EVAL_STEPS = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

METHODS = [
    {
        "id": "colmap_no_reset",
        "name": "COLMAP (no reset)",
        "label": "COLMAP (No Reset, ~7k)",
        "pat": "f{frame}_3dgs_colmap_no_reset",
        "color": "#1f77b4",  # Steel Blue
        "marker": "o",
        "linestyle": "--",
    },
    {
        "id": "da3_50k_old",
        "name": "舊版 DA3 50k Importance",
        "label": "舊版 DA3 50k (無 COLMAP)",
        "pat": "f{frame}_3dgs_da3_50k",
        "color": "#2ca02c",  # Forest Green
        "marker": "^",
        "linestyle": "-.",
    },
    {
        "id": "da3_14k_uni_colmap",
        "name": "DA3 14k Uniform + COLMAP",
        "label": "DA3 14k Uniform + COLMAP (Ours)",
        "pat": "f{frame}_3dgs_giant_colmap_14k_uni_refine500_reset3000",
        "color": "#d62728",  # Crimson Red
        "marker": "s",
        "linestyle": "-",
    },
]


def collect_data(scenes):
    results = {}
    for m in METHODS:
        step_data = {step: [] for step in EVAL_STEPS}
        for s in scenes:
            for f in FRAMES:
                path = os.path.join(BASE_DIR, s, m["pat"].format(frame=f), "stats")
                for step in EVAL_STEPS:
                    fn = os.path.join(path, f"val_step{step-1:04d}.json")
                    with open(fn, "r") as fp:
                        data = json.load(fp)
                        step_data[step].append(data["psnr"])
        means = np.array([np.mean(step_data[st]) for st in EVAL_STEPS])
        stds = np.array([np.std(step_data[st]) for st in EVAL_STEPS])
        sems = stds / np.sqrt(len(scenes) * len(FRAMES))
        results[m["id"]] = {
            "means": means,
            "stds": stds,
            "sems": sems,
            "raw": step_data,
        }
    return results


def plot_single(data, title, subtitle, out_filenames, ylim=None):
    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=300)

    for m in METHODS:
        res = data[m["id"]]
        means = res["means"]
        sems = res["sems"]

        ax.plot(
            EVAL_STEPS,
            means,
            label=f"{m['label']} (Final: {means[-1]:.2f} dB)",
            color=m["color"],
            marker=m["marker"],
            markersize=7,
            linewidth=2.2,
            linestyle=m["linestyle"],
            markeredgewidth=1.2,
            markeredgecolor="white",
            zorder=4 if "Ours" in m["label"] else 3,
        )

        # 半透明誤差帶 (SEM)
        ax.fill_between(
            EVAL_STEPS,
            means - sems,
            means + sems,
            color=m["color"],
            alpha=0.15,
            zorder=2,
        )

    ax.set_title(f"{title}\n{subtitle}", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("訓練步數 (Training Step)", fontsize=12, fontweight="medium", labelpad=8)
    ax.set_ylabel("平均驗證 PSNR (dB) ↑", fontsize=12, fontweight="medium", labelpad=8)
    ax.set_xticks(EVAL_STEPS)
    ax.set_xticklabels([str(s) for s in EVAL_STEPS], fontsize=10)
    if ylim:
        ax.set_ylim(ylim)
    ax.tick_params(axis="both", which="major", labelsize=10.5)

    ax.grid(True, linestyle=":", alpha=0.6, color="#888888")
    ax.legend(fontsize=10.5, loc="lower right", framealpha=0.92, edgecolor="#cccccc")

    plt.tight_layout()
    for fn in out_filenames:
        fig.savefig(fn, dpi=300)
        print(f"Saved: {fn}")
    plt.close(fig)


def plot_combined(data_sub1, data_sub2, out_filenames):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 6.2), dpi=300)

    # Subgroup 1
    for m in METHODS:
        res = data_sub1[m["id"]]
        means = res["means"]
        sems = res["sems"]
        ax1.plot(
            EVAL_STEPS,
            means,
            label=f"{m['label']}\n(Final: {means[-1]:.2f} dB)",
            color=m["color"],
            marker=m["marker"],
            markersize=6.5,
            linewidth=2.0,
            linestyle=m["linestyle"],
            markeredgewidth=1.2,
            markeredgecolor="white",
            zorder=4 if "Ours" in m["label"] else 3,
        )
        ax1.fill_between(
            EVAL_STEPS,
            means - sems,
            means + sems,
            color=m["color"],
            alpha=0.15,
            zorder=2,
        )

    ax1.set_title(
        "子群 1：Coffee Martini + Flame Salmon 1 (2 場景平均)\n[高反光 / 透明杯皿挑戰場景]",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax1.set_xlabel("訓練步數 (Training Step)", fontsize=11, fontweight="medium")
    ax1.set_ylabel("平均驗證 PSNR (dB) ↑", fontsize=11, fontweight="medium")
    ax1.set_xticks(EVAL_STEPS)
    ax1.tick_params(axis="both", which="major", labelsize=9.5)
    ax1.grid(True, linestyle=":", alpha=0.6, color="#888888")
    ax1.legend(fontsize=9.5, loc="lower right", framealpha=0.92, edgecolor="#cccccc")
    ax1.set_ylim(21.5, 28.5)

    # Subgroup 2
    for m in METHODS:
        res = data_sub2[m["id"]]
        means = res["means"]
        sems = res["sems"]
        ax2.plot(
            EVAL_STEPS,
            means,
            label=f"{m['label']}\n(Final: {means[-1]:.2f} dB)",
            color=m["color"],
            marker=m["marker"],
            markersize=6.5,
            linewidth=2.0,
            linestyle=m["linestyle"],
            markeredgewidth=1.2,
            markeredgecolor="white",
            zorder=4 if "Ours" in m["label"] else 3,
        )
        ax2.fill_between(
            EVAL_STEPS,
            means - sems,
            means + sems,
            color=m["color"],
            alpha=0.15,
            zorder=2,
        )

    ax2.set_title(
        "子群 2：其餘 4 場景平均 (Cook Spinach, Roasted Beef, Flame Steak, Sear Steak)\n[實體食材 / 紋理豐富場景]",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax2.set_xlabel("訓練步數 (Training Step)", fontsize=11, fontweight="medium")
    ax2.set_ylabel("平均驗證 PSNR (dB) ↑", fontsize=11, fontweight="medium")
    ax2.set_xticks(EVAL_STEPS)
    ax2.tick_params(axis="both", which="major", labelsize=9.5)
    ax2.grid(True, linestyle=":", alpha=0.6, color="#888888")
    ax2.legend(fontsize=9.5, loc="lower right", framealpha=0.92, edgecolor="#cccccc")
    ax2.set_ylim(25.5, 34.5)

    fig.suptitle(
        "3DGS 收斂動態對比：PSNR vs. Training Steps 消融實驗曲線",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()

    for fn in out_filenames:
        fig.savefig(fn, dpi=300)
        print(f"Saved: {fn}")
    plt.close(fig)


def main():
    print("正在收集 Subgroup 1 數據 (Coffee Martini + Flame Salmon 1)...")
    data_sub1 = collect_data(SCENES_SUB1)

    print("正在收集 Subgroup 2 數據 (其餘 4 場景)...")
    data_sub2 = collect_data(SCENES_SUB2)

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs(OUT_SSD_DIR, exist_ok=True)

    # 1. 輸出 Subgroup 1 單獨圖檔
    plot_single(
        data_sub1,
        "Coffee Martini + Flame Salmon 1 平均 PSNR 收斂曲線",
        "(2 場景 × 6 關鍵幀，共 12 Runs 平均；高反光/透明杯皿挑戰場景)",
        [
            os.path.join(ARTIFACT_DIR, "subgroup1_psnr_step_curve.png"),
            os.path.join(OUT_SSD_DIR, "subgroup1_psnr_step_curve.png"),
        ],
        ylim=(21.5, 28.5),
    )

    # 2. 輸出 Subgroup 2 單獨圖檔
    plot_single(
        data_sub2,
        "其餘 4 場景平均 PSNR 收斂曲線",
        "(Cook Spinach, Cut Roasted Beef, Flame Steak, Sear Steak，共 24 Runs 平均)",
        [
            os.path.join(ARTIFACT_DIR, "subgroup2_psnr_step_curve.png"),
            os.path.join(OUT_SSD_DIR, "subgroup2_psnr_step_curve.png"),
        ],
        ylim=(25.5, 34.5),
    )

    # 3. 輸出 Combined 雙欄圖檔
    plot_combined(
        data_sub1,
        data_sub2,
        [
            os.path.join(ARTIFACT_DIR, "psnr_vs_step_comparison_2panel.png"),
            os.path.join(OUT_SSD_DIR, "psnr_vs_step_comparison_2panel.png"),
        ],
    )

    # 4. 輸出數值 JSON 供後續分析
    summary_data = {
        "steps": EVAL_STEPS,
        "subgroup1_2scenes": {
            m["id"]: {
                "means": data_sub1[m["id"]]["means"].tolist(),
                "stds": data_sub1[m["id"]]["stds"].tolist(),
                "sems": data_sub1[m["id"]]["sems"].tolist(),
            }
            for m in METHODS
        },
        "subgroup2_4scenes": {
            m["id"]: {
                "means": data_sub2[m["id"]]["means"].tolist(),
                "stds": data_sub2[m["id"]]["stds"].tolist(),
                "sems": data_sub2[m["id"]]["sems"].tolist(),
            }
            for m in METHODS
        },
    }
    with open(os.path.join(ARTIFACT_DIR, "psnr_step_curves_data.json"), "w") as fp:
        json.dump(summary_data, fp, indent=2, ensure_ascii=False)
    print("JSON 數據已成功寫入！")


if __name__ == "__main__":
    main()
