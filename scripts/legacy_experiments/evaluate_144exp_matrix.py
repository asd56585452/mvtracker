#!/usr/bin/env python3
"""
彙整與評估 144 組 3DGS 大矩陣實驗 (6 Scenes x 6 Frames x 4 Methods)
輸出完整跨場景、跨時序對照報告與統計圖表。
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_BASE = "/home/tsaichenghan/GSCodec_Studio/results/144exp"
ARTIFACTS_DIR = "/home/tsaichenghan/.gemini/antigravity-ide/brain/cd1c9aec-32ca-4d5b-8491-fa8bfec9b350"

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
    ("3dgs_colmap", "3DGS (COLMAP 7.3k)", "SfM Baseline (with Refine)"),
    ("3dgs_da3_7k", "3DGS (DA3 7.3k)", "DA3 Equal Points (with Refine)"),
    ("3dgs_da3_50k", "3DGS (DA3 50k)", "DA3 Golden Capacity (with Refine)"),
    ("3dgs_da3_250k_no_refine", "3DGS (DA3 250k No-Refine)", "DA3 Dense Surfel (NO Refine/Prune)"),
]

EVAL_STEPS = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

def parse_all_runs():
    records = []
    
    for scene in SCENES:
        for frame in FRAMES:
            for m_id, m_name, m_desc in METHODS:
                run_dir = os.path.join(RESULTS_BASE, scene, f"f{frame}_{m_id}")
                final_stats_path = os.path.join(run_dir, "stats", "val_step4999.json")
                
                status = "Finished" if os.path.exists(final_stats_path) else "Pending"
                
                row = {
                    "scene": scene,
                    "frame": frame,
                    "method_id": m_id,
                    "method_name": m_name,
                    "method_desc": m_desc,
                    "status": status,
                    "psnr": None,
                    "ssim": None,
                    "lpips": None,
                    "num_GS": None,
                    "eval_curve": {},
                }
                
                if status == "Finished":
                    with open(final_stats_path, "r") as f:
                        d = json.load(f)
                    row["psnr"] = d.get("psnr", None)
                    row["ssim"] = d.get("ssim", None)
                    row["lpips"] = d.get("lpips", None)
                    row["num_GS"] = d.get("num_GS", None)
                    
                    # 讀取中間 10 個步驟的 eval 曲線
                    for step in EVAL_STEPS:
                        s_str = f"{step-1:04d}"
                        s_path = os.path.join(run_dir, "stats", f"val_step{s_str}.json")
                        if os.path.exists(s_path):
                            with open(s_path, "r") as sf:
                                s_data = json.load(sf)
                            row["eval_curve"][step] = s_data.get("psnr", None)
                            
                records.append(row)
                
    return pd.DataFrame(records)

def generate_report(df: pd.DataFrame):
    total = len(df)
    finished = len(df[df["status"] == "Finished"])
    
    lines = []
    lines.append("# N3D 144 組 3DGS 大矩陣實驗綜合評估報告\n")
    lines.append(f"> **執行進度**: 已完成 {finished} / {total} 組實驗 ({finished/total*100:.1f}%)\n")
    lines.append(f"> **測試視角**: Novel Test View `cam00` (held-out)\n")
    lines.append(f"> **評估時序**: 6 均勻幀 (f25, f75, f125, f175, f225, f275)\n\n")
    
    if finished == 0:
        lines.append("⚠️ 目前尚未有已完成的實驗數據。請在啟動 `run_matrix_144exp_da3.sh` 後再次執行本評估腳本。\n")
        return "\n".join(lines)
        
    # 1. 四大方法全場景平均表現
    lines.append("## 一、 四大方法全場景平均指標對比\n")
    lines.append("| 評估方法 | 初始點數 | 分裂/剪枝機制 | 平均 PSNR (dB) | 平均 SSIM | 平均 LPIPS | 平均高斯數 (Num GS) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for m_id, m_name, m_desc in METHODS:
        m_df = df[(df["method_id"] == m_id) & (df["status"] == "Finished")]
        if len(m_df) > 0:
            p_mean = m_df["psnr"].mean()
            s_mean = m_df["ssim"].mean()
            l_mean = m_df["lpips"].mean()
            g_mean = m_df["num_GS"].mean()
            
            init_str = "COLMAP (~7.3k)" if "colmap" in m_id else ("7,330" if "7k" in m_id else ("50,000" if "50k" in m_id else "250,000"))
            refine_str = "禁止 (純固定拓撲優化)" if "no_refine" in m_id else "標準 (300~3000步)"
            
            lines.append(f"| **{m_name}** | {init_str} | {refine_str} | **{p_mean:.2f}** | {s_mean:.4f} | {l_mean:.4f} | {int(g_mean):,} |")
            
    lines.append("\n---\n")
    
    # 2. 6 個場景分別之表現摘要
    lines.append("## 二、 6 個場景跨方法平均 PSNR (dB)\n")
    lines.append("| 場景名稱 (Scene) | 3DGS (COLMAP) | 3DGS (DA3 7.3k) | 3DGS (DA3 50k) | 3DGS (DA3 250k No-Refine) | DA3 50k 提升 (vs COLMAP) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    
    for scene in SCENES:
        row_str = f"| **{scene}** |"
        c_psnr = None
        d50_psnr = None
        for m_id, _, _ in METHODS:
            sub = df[(df["scene"] == scene) & (df["method_id"] == m_id) & (df["status"] == "Finished")]
            if len(sub) > 0:
                val = sub["psnr"].mean()
                row_str += f" {val:.2f} |"
                if m_id == "3dgs_colmap": c_psnr = val
                if m_id == "3dgs_da3_50k": d50_psnr = val
            else:
                row_str += " - |"
        if c_psnr is not None and d50_psnr is not None:
            diff = d50_psnr - c_psnr
            sign = "+" if diff >= 0 else ""
            row_str += f" **{sign}{diff:.2f} dB** |"
        else:
            row_str += " - |"
        lines.append(row_str)
        
    lines.append("\n---\n")
    
    # 3. 關鍵假說驗證解析
    lines.append("## 三、 核心假說驗證解析\n")
    lines.append("1. **同點數絕對公平對決 (COLMAP 7.3k vs. DA3 7.3k)**：\n")
    lines.append("   - 驗證在相同的幾何點數預算下，DA3 重要度採樣是否在多視角未見測試集上顯著勝過傳統 SfM 點雲。\n")
    lines.append("2. **黃金容量擴展 (DA3 50k)**：\n")
    lines.append("   - 驗證在 6 個全動態場景中，提供充足背景幾何種子能否徹底消除背景模糊與浮動雜訊。\n")
    lines.append("3. **終極假說：DA3 250k 不分裂剪枝 (No-Refine) 是否成立？**：\n")
    lines.append("   - 若 250k 固定拓撲模型的 PSNR 接近或超越 COLMAP，將宣告：**Foundation Model 幾何先驗可直接取代傳統 3DGS 3,000 步繁複且不穩定的分裂與剪枝啟發式機制！**\n")
    
    return "\n".join(lines)

def main():
    df = parse_all_runs()
    report = generate_report(df)
    
    report_path = os.path.join(ARTIFACTS_DIR, "matrix_144exp_evaluation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
        
    print(f"📊 評估報告已更新至: {report_path}")
    
    # 印出目前進度
    finished = len(df[df["status"] == "Finished"])
    print(f"   目前進度: {finished} / {len(df)} runs 已完成")

if __name__ == "__main__":
    main()
