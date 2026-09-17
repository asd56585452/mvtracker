#!/usr/bin/env python3
"""
2D 影像特徵空間評分 (S_2D = w_tex * S_tex + w_depth * S_depth) 視覺化工具

功能：
1. 支援任意解析度與縮放 (--max_dim, --scale, --resize)。
2. 計算並視覺化三大 RGB 紋理特徵 (S_Sobel, S_DCT_MidHigh, S_ColorVar / ChromaGrad)。
3. 支援 DA3 幾何深度特徵 (對齊 OpenCV LINEMOD, PCL, COLMAP 業界標準)：
   - 對數深度一階斷差梯度 (S_depth_grad)
   - 空間表面單位法向量 (Surface Normals: X-右, Y-下, Z-前)
   - PCL 深度跳變門檻保護之局部表面曲率 (S_curv)
   - COLMAP / 2DGS 掠射角防拉扯濾波 (w_grazing)
   - 幾何斷差互斥閘門 (Edge Gating)
   - DA3 信心度 (c_DA3) 硬門檻過濾與調製
4. 支援線上直接調用 Depth-Anything-3 零配置推論 (--use_da3) 或讀取快取深度檔 (--depth)。
5. 輸出 3x3 多面板全景對比圖、四大抗噪防護圖、各項高解析度熱力圖。
"""

import os
import sys
import argparse
import time
from typing import Tuple, Dict, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import scipy.ndimage
import scipy.optimize
import cv2

# 加入專案根目錄至 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mvtracker.utils.feature_scores import (
    RGBTextureScorer,
    DepthGeometryScorer,
    Spatial2DScorer,
    rgb_to_ycbcr,
    gaussian_blur2d,
    sobel_gradient,
    block_dct8x8_midhigh,
    chroma_gradient,
    noise_coring,
    robust_quantile_clamp,
    log_depth_gradient,
    depth_to_surface_normals,
    depth_gated_surface_curvature,
    grazing_angle_filter
)


def load_and_preprocess_image(
    image_path: str,
    max_dim: int = 0,
    scale: float = 1.0,
    resize: tuple = None
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int]]:
    """
    載入影像並根據參數調整解析度。
    Returns:
        tensor: [1, 3, H, W] float32 in [0, 1]
        pil_img: 縮放後的 PIL.Image
        orig_size: (orig_W, orig_H)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到輸入影像: {image_path}")

    img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img.size

    target_w, target_h = orig_w, orig_h

    if resize is not None and len(resize) == 2 and resize[0] > 0 and resize[1] > 0:
        target_w, target_h = int(resize[0]), int(resize[1])
    elif max_dim > 0:
        longest = max(orig_w, orig_h)
        if longest > max_dim:
            ratio = float(max_dim) / float(longest)
            target_w = int(round(orig_w * ratio))
            target_h = int(round(orig_h * ratio))
    elif scale != 1.0 and scale > 0:
        target_w = int(round(orig_w * scale))
        target_h = int(round(orig_h * scale))

    if (target_w, target_h) != (orig_w, orig_h):
        img_resized = img.resize((target_w, target_h), Image.Resampling.BILINEAR)
    else:
        img_resized = img

    np_img = np.array(img_resized, dtype=np.float32) / 255.0 # [H, W, 3]
    tensor = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0) # [1, 3, H, W]

    return tensor, img_resized, (orig_w, orig_h)


def load_depth_from_file(depth_path: str, target_h: int, target_w: int) -> torch.Tensor:
    """從檔案載入既有深度圖並縮放至目標解析度"""
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"找不到深度檔案: {depth_path}")

    if depth_path.endswith('.npy'):
        arr = np.load(depth_path).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[0]
    else:
        depth_img = Image.open(depth_path)
        arr = np.array(depth_img, dtype=np.float32)
        if arr.max() > 255.0:
            arr = arr / 1000.0  # 假定 16-bit 毫米為單位

    tensor = torch.from_numpy(arr).float()
    while tensor.ndim < 4:
        tensor = tensor.unsqueeze(0)
    
    if tensor.shape[-2:] != (target_h, target_w):
        tensor = F.interpolate(tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)

    return tensor


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """轉換單通道 [1, 1, H, W] 或 [H, W] 為 numpy 2D array"""
    if t.ndim == 4:
        t = t[0, 0]
    elif t.ndim == 3:
        t = t[0]
    return t.detach().cpu().numpy()


def save_heatmap(data: np.ndarray, save_path: str, colormap: str = 'turbo', vmin: float = 0.0, vmax: float = 1.0):
    """將單通道數據以指定 colormap 存為高解析度圖片"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 8), dpi=150)
    plt.imshow(data, cmap=colormap, vmin=vmin, vmax=vmax)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
    plt.close()


def print_stats_table(stats_dict: dict):
    """美化輸出特徵統計表"""
    print("\n" + "=" * 84)
    print(f"{'特徵名稱':<24} | {'最小值':<8} | {'最大值':<8} | {'平均值':<8} | {'99%分位數':<10} | {'非零比例 (>0)':<12}")
    print("-" * 84)
    for name, s in stats_dict.items():
        arr = s.flatten()
        min_v = float(np.min(arr))
        max_v = float(np.max(arr))
        mean_v = float(np.mean(arr))
        q99_v = float(np.percentile(arr, 99))
        nonzero_ratio = float(np.mean(arr > 1e-4) * 100.0)
        print(f"{name:<24} | {min_v:<8.4f} | {max_v:<8.4f} | {mean_v:<8.4f} | {q99_v:<10.4f} | {nonzero_ratio:<11.1f}%")
    print("=" * 84 + "\n")


def load_and_process_gs_density(
    density_path: str,
    target_h: int,
    target_w: int,
    kde_sigma: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    載入 4DGS 高斯中心投影計數圖 (.npy)，並計算連續 KDE 機率密度分佈 (P_4DGS)。
    
    Returns:
        raw_counts: [H, W] float32 原始離散高斯中心數量
        p_4dgs: [H, W] float32 連續平滑且總和為 1.0 的機率密度函數 (PDF)
    """
    if not os.path.exists(density_path):
        raise FileNotFoundError(f"找不到 4DGS 密度檔案: {density_path}")

    raw = np.load(density_path).astype(np.float32)
    while raw.ndim > 2:
        raw = raw[0]

    orig_h, orig_w = raw.shape
    if (orig_h, orig_w) != (target_h, target_w):
        # 尺寸不同時使用 INTER_AREA 進行保持能量總和的縮放
        scale_factor = (target_h * target_w) / float(orig_h * orig_w)
        raw_counts = cv2.resize(raw, (target_w, target_h), interpolation=cv2.INTER_AREA) * scale_factor
    else:
        raw_counts = raw.copy()

    raw_counts = np.clip(raw_counts, 0.0, None)

    # 連續高斯平滑 (KDE 估計 2D Gaussian splatter footprint)
    if kde_sigma > 0:
        kde = scipy.ndimage.gaussian_filter(raw_counts, sigma=kde_sigma)
    else:
        kde = raw_counts.copy()

    kde = np.clip(kde, 0.0, None)
    total_kde = np.sum(kde)
    if total_kde > 0:
        p_4dgs = kde / total_kde
    else:
        p_4dgs = np.ones_like(kde, dtype=np.float32) / kde.size

    return raw_counts, p_4dgs


def evaluate_distribution_alignment(
    scores_dict: Dict[str, np.ndarray],
    p_4dgs: np.ndarray,
    raw_counts: np.ndarray,
    scales: List[int] = [1, 2, 4, 8],
    optimize_w0: bool = True
) -> Tuple[Dict[str, dict], Optional[dict]]:
    """
    在多解析度金字塔 (1x, 1/2x, 1/4x, 1/8x) 下評估各項預測特徵分佈與 4DGS 實際高斯分佈的對齊指標。
    支援：
    1. 獨立特徵：求解平滑背景基準分最佳混合比 alpha^* in [0, 1]，使 (1-alpha) P_k + alpha * u 的 KL 散度最小化。
    2. 綜合特徵：透過 NNLS 與直接 KL 散度凸優化求解最佳權重向量 [w0, w1, w2, w3]。
    """
    results = {}
    total_gaussians = float(np.sum(raw_counts))
    H, W = p_4dgs.shape
    b_flat = p_4dgs.flatten()
    p_u = np.ones_like(p_4dgs, dtype=np.float32) / p_4dgs.size
    p_u_flat = p_u.flatten()

    def calc_metrics_for_pdf(p_pdf):
        kl_by_scale = {}
        corr_by_scale = {}
        for sc in scales:
            if sc == 1:
                p_p_sc = p_pdf
                p_g_sc = p_4dgs
            else:
                th, tw = max(1, H // sc), max(1, W // sc)
                p_p_sc = cv2.resize(p_pdf, (tw, th), interpolation=cv2.INTER_AREA)
                p_g_sc = cv2.resize(p_4dgs, (tw, th), interpolation=cv2.INTER_AREA)
                p_p_sc = p_p_sc / (np.sum(p_p_sc) + 1e-12)
                p_g_sc = p_g_sc / (np.sum(p_g_sc) + 1e-12)

            kl = float(np.sum(p_g_sc * np.log((p_g_sc + 1e-12) / (p_p_sc + 1e-12))))
            flat_g = p_g_sc.flatten()
            flat_p = p_p_sc.flatten()
            std_g = np.std(flat_g)
            std_p = np.std(flat_p)
            corr = float(np.corrcoef(flat_g, flat_p)[0, 1]) if (std_g > 1e-12 and std_p > 1e-12) else 0.0

            kl_by_scale[sc] = kl
            corr_by_scale[sc] = corr
        return kl_by_scale, corr_by_scale

    single_pdfs = {}

    for name, s in scores_dict.items():
        eps = 1e-7
        s_pos = np.clip(s, 0.0, None) + eps
        p_raw = s_pos / np.sum(s_pos)
        single_pdfs[name] = p_raw

        kl_raw, corr_raw = calc_metrics_for_pdf(p_raw)

        # 最佳化單一特徵之平滑基準分 alpha
        alpha_opt = 0.0
        p_opt = p_raw
        kl_opt = kl_raw
        corr_opt = corr_raw

        if optimize_w0:
            p_feat_flat = p_raw.flatten()
            def kl_1d(alpha):
                p_mix = (1.0 - alpha) * p_feat_flat + alpha * p_u_flat
                return np.sum(b_flat * np.log((b_flat + 1e-12) / (p_mix + 1e-12)))
            
            res_1d = scipy.optimize.minimize_scalar(kl_1d, bounds=(0.0, 1.0), method='bounded')
            if res_1d.success:
                alpha_opt = float(res_1d.x)
                p_opt = (1.0 - alpha_opt) * p_raw + alpha_opt * p_u
                kl_opt, corr_opt = calc_metrics_for_pdf(p_opt)

        top_k_recalls = {}
        if total_gaussians > 0:
            for k in [10, 20, 30, 50]:
                cutoff = np.percentile(s, 100 - k)
                mask = (s >= cutoff)
                captured = np.sum(raw_counts[mask])
                top_k_recalls[k] = float(captured / total_gaussians * 100.0)

        results[name] = {
            "p_pred": p_raw,
            "p_pred_opt": p_opt,
            "alpha_opt": alpha_opt,
            "kl_by_scale": kl_raw,
            "kl_opt_by_scale": kl_opt,
            "corr_by_scale": corr_raw,
            "corr_opt_by_scale": corr_opt,
            "top_k_recall": top_k_recalls
        }

    # 綜合最佳化 [w0 (uniform), w1, w2, w3...]
    comb_info = None
    if optimize_w0 and len(scores_dict) > 1:
        # 排除包含 Total 或 Combined 的彙總名稱，保留子特徵
        feat_keys = [k for k in scores_dict.keys() if "Total" not in k and "Texture" not in k and "Depth" not in k]
        if not feat_keys:
            feat_keys = list(scores_dict.keys())

        feat_matrix = [p_u_flat] + [single_pdfs[k].flatten() for k in feat_keys]
        A = np.stack(feat_matrix, axis=1) # [N, 1 + K]
        b = b_flat

        # 1. NNLS 快速非負最小平方法求解
        w_nnls, _ = scipy.optimize.nnls(A, b)
        w_nnls = w_nnls / (np.sum(w_nnls) + 1e-12)

        # 2. 直接 KL 散度梯度下降微調 (SLSQP)
        def kl_multi(w):
            p_mix = A @ w
            return np.sum(b * np.log((b + 1e-12) / (p_mix + 1e-12)))

        bounds = [(0.0, 1.0) for _ in range(A.shape[1])]
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        res_opt = scipy.optimize.minimize(kl_multi, w_nnls, bounds=bounds, constraints=constraints, method='SLSQP')
        w_opt = res_opt.x if res_opt.success else w_nnls
        w_opt = np.clip(w_opt, 0.0, None)
        w_opt = w_opt / np.sum(w_opt)

        p_opt_comb = (A @ w_opt).reshape(H, W)
        kl_comb, corr_comb = calc_metrics_for_pdf(p_opt_comb)

        comb_name = "S_tex (+opt w0)" if "S_tex (Texture)" in scores_dict else "S_2D (+opt w0)"
        results[comb_name] = {
            "p_pred": p_opt_comb,
            "p_pred_opt": p_opt_comb,
            "alpha_opt": float(w_opt[0]),
            "kl_by_scale": kl_comb,
            "kl_opt_by_scale": kl_comb,
            "corr_by_scale": corr_comb,
            "corr_opt_by_scale": corr_comb,
            "top_k_recall": results.get("S_tex (Texture)", list(results.values())[0])["top_k_recall"]
        }

        comb_info = {
            "feat_keys": feat_keys,
            "w_nnls": w_nnls,
            "w_opt": w_opt,
            "w_0_opt": float(w_opt[0]),
            "p_opt_comb": p_opt_comb
        }

    return results, comb_info


def print_multiscale_alignment_table(eval_results: Dict[str, dict], comb_info: Optional[dict] = None):
    """美化輸出 4DGS 分佈對齊評估表 (含平滑區域基準分 w0 優化)"""
    print("\n" + "=" * 110)
    print("📊 4DGS 實際高斯分佈 vs 空間特徵評分 多尺度對齊評估表 (含平滑背景基準分 w0 凸優化)")
    print("=" * 110)
    print(f"{'特徵名稱':<24} | {'最優 w0 比例':<12} | {'原始 KL(1x)':<12} | {'優化 KL(1x)':<12} | {'優化 KL(1/4x)':<14} | {'相關係數 (1x)':<13} | {'Top-20% Recall':<14}")
    print("-" * 110)
    for name, res in eval_results.items():
        w0_str = f"{res['alpha_opt']*100:.1f}%" if res['alpha_opt'] > 0 else "-"
        kl_raw = res["kl_by_scale"].get(1, 0.0)
        kl_opt = res["kl_opt_by_scale"].get(1, 0.0)
        kl_opt_4x = res["kl_opt_by_scale"].get(4, 0.0)
        corr = res["corr_opt_by_scale"].get(1, 0.0)
        rec20 = res["top_k_recall"].get(20, 0.0)
        print(f"{name:<24} | {w0_str:<12} | {kl_raw:<12.4f} | {kl_opt:<12.4f} | {kl_opt_4x:<14.4f} | {corr:<13.4f} | {rec20:<13.1f}%")
    print("=" * 110)

    if comb_info is not None:
        print("\n🎛️ 綜合多特徵最佳權重組合 [w0* (平滑背景) + w_k* (高頻特徵)]：")
        print(f"   • 平滑背景基準分 (w0*): {comb_info['w_0_opt']*100:.2f}%")
        for k, w_val in zip(comb_info['feat_keys'], comb_info['w_opt'][1:]):
            print(f"   • {k} 最佳權重: {w_val*100:.2f}%")
        print("=" * 110 + "\n")


def compute_dominant_feature_map(
    subfeatures_dict: Dict[str, np.ndarray],
    weights_dict: Dict[str, float],
    w0_weight: float = 0.40,
    s_comb: Optional[np.ndarray] = None,
    raw_counts: Optional[np.ndarray] = None,
    agg_sigma: float = 2.0,
    use_prob_space: bool = True,
    bg_thresh: float = 0.05
) -> dict:
    """
    計算每個像素之「優勢特徵 (Winner-take-all Dominant Feature)」：
    - 方案 A (機率空間重要度採樣比大小，use_prob_space=True):
      直接比較各候選機率項:
        c0 = w0 * u (平滑底分均勻分佈)
        c_k = w_k * P_k(u, v) (各高頻特徵的機率分量)
      經適度高斯空間平滑 (agg_sigma=2.0) 後，由 argmax 決定勝出者。
      無需人工設定背景門檻 (bg_thresh)，當所有特徵 c_k < w0*u 時底分自然勝出！
    - 方案 B (特徵顯著性空間，use_prob_space=False):
      以 Q99.5 歸一化後乘上特徵權重，並以 bg_thresh 截斷背景。
    """
    CATEGORY_DEFS = [
        {"key": "bg", "name": f"Flat Floor (w0*u: {w0_weight*100:.1f}%)" if use_prob_space else "Flat / Low Energy", "color": np.array([45, 48, 55], dtype=np.uint8)},
        {"key": "s_sobel", "name": "Sobel Edge (Boundary)", "color": np.array([230, 25, 75], dtype=np.uint8)},
        {"key": "s_dct", "name": "DCT MidHigh (Texture)", "color": np.array([60, 180, 75], dtype=np.uint8)},
        {"key": "s_chroma", "name": "ChromaGrad (Color Var)", "color": np.array([255, 205, 25], dtype=np.uint8)},
        {"key": "s_depth_grad", "name": "Depth Silhouette (Discontinuity)", "color": np.array([0, 130, 200], dtype=np.uint8)},
        {"key": "s_curv", "name": "Surface Curvature (PCL-Gated)", "color": np.array([150, 40, 190], dtype=np.uint8)},
    ]

    active_cats = [CATEGORY_DEFS[0]]
    H, W = next(iter(subfeatures_dict.values())).shape
    N = float(H * W)
    u = 1.0 / N

    if use_prob_space:
        # 候選 0: 平滑底色均勻機率 (高斯平滑 uniform 仍為 uniform)
        c0 = np.full((H, W), w0_weight * u, dtype=np.float32)
        candidate_arrays = [c0]

        for cdef in CATEGORY_DEFS[1:]:
            k = cdef["key"]
            if k in subfeatures_dict:
                active_cats.append(cdef)
                raw_s = subfeatures_dict[k]
                sum_s = float(np.sum(raw_s))
                p_k = (raw_s / sum_s) if sum_s > 0 else np.zeros_like(raw_s)
                w_k = weights_dict.get(k, 0.0)
                weighted_p = w_k * p_k
                
                if agg_sigma > 0.0:
                    c_k = scipy.ndimage.gaussian_filter(weighted_p, sigma=agg_sigma)
                else:
                    c_k = weighted_p
                candidate_arrays.append(c_k)

        stack = np.stack(candidate_arrays, axis=0) # [1 + K, H, W]
        dominant_idx = np.argmax(stack, axis=0)     # 0..K (0 代表 Flat Floor)
        
        # 總機率密度
        p_total = np.sum(stack, axis=0)
        p_norm = np.clip(p_total / (np.percentile(p_total, 99.0) + 1e-12), 0.0, 1.0)[..., None]
        intensity_scale = np.where(
            dominant_idx[..., None] == 0,
            0.40,
            0.35 + 0.65 * p_norm
        )
    else:
        candidate_arrays = []
        for cdef in CATEGORY_DEFS[1:]:
            k = cdef["key"]
            if k in subfeatures_dict:
                active_cats.append(cdef)
                raw_s = subfeatures_dict[k]
                q99 = np.percentile(raw_s, 99.5)
                s_norm = np.clip(raw_s / (q99 + 1e-8), 0.0, 1.0)
                w = weights_dict.get(k, 1.0)
                weighted_s = w * s_norm
                
                if agg_sigma > 0.0:
                    smoothed_s = scipy.ndimage.gaussian_filter(weighted_s, sigma=agg_sigma)
                else:
                    smoothed_s = weighted_s
                candidate_arrays.append(smoothed_s)

        stack = np.stack(candidate_arrays, axis=0) # [K, H, W]
        max_vals = np.max(stack, axis=0)           # [H, W]
        argmax_indices = np.argmax(stack, axis=0) + 1 # 1..K
        dominant_idx = np.where(max_vals >= bg_thresh, argmax_indices, 0).astype(np.int32)
        
        if s_comb is not None:
            if agg_sigma > 0.0:
                s_comb_sm = scipy.ndimage.gaussian_filter(s_comb, sigma=max(2.0, agg_sigma * 0.67))
            else:
                s_comb_sm = s_comb
            s_norm = np.clip(s_comb_sm / (np.percentile(s_comb_sm, 99.0) + 1e-8), 0.0, 1.0)[..., None]
            intensity_scale = np.where(dominant_idx[..., None] == 0, 0.45, 0.35 + 0.65 * s_norm)
        else:
            intensity_scale = 1.0

    total_pixels = float(H * W)
    total_gaussians = float(np.sum(raw_counts)) if raw_counts is not None else 0.0

    base_rgb = np.zeros((H, W, 3), dtype=np.float32)
    category_stats = []

    for cat_idx, cat in enumerate(active_cats):
        mask = (dominant_idx == cat_idx)
        base_rgb[mask] = cat["color"].astype(np.float32)
        pixel_ratio = float(np.sum(mask) / total_pixels * 100.0)
        
        gs_ratio = 0.0
        if raw_counts is not None and total_gaussians > 0:
            gs_ratio = float(np.sum(raw_counts[mask]) / total_gaussians * 100.0)

        category_stats.append({
            "idx": cat_idx,
            "name": cat["name"],
            "color": cat["color"],
            "pixel_ratio": pixel_ratio,
            "gs_ratio": gs_ratio
        })

    modulated_rgb = np.clip(base_rgb * intensity_scale, 0, 255).astype(np.uint8)

    return {
        "dominant_idx": dominant_idx,
        "modulated_rgb": modulated_rgb,
        "base_rgb": base_rgb.astype(np.uint8),
        "category_stats": category_stats,
        "active_cats": active_cats,
        "use_prob_space": use_prob_space
    }


def plot_dominant_feature_canvas(
    pil_img: Image.Image,
    dominant_res: dict,
    raw_counts: Optional[np.ndarray],
    save_path: str,
    base_name: str,
    agg_sigma: float = 2.0
):
    """
    繪製優勢特徵 (Winner-take-all) 分割圖與 4DGS 實際高斯中心落點對比
    """
    fig, axes = plt.subplots(1, 3, figsize=(23, 7), dpi=150, gridspec_kw={'width_ratios': [1.3, 1.3, 1.0]})

    mod_rgb = dominant_res["modulated_rgb"]
    cat_stats = dominant_res["category_stats"]
    is_prob = dominant_res.get("use_prob_space", False)

    # 1. 優勢特徵分類圖
    axes[0].imshow(mod_rgb)
    smooth_txt = f"Spatial Smoothing (sigma={agg_sigma})" if agg_sigma > 0 else "1-Pixel Raw"
    method_txt = "Scheme A: Prob Space [w_k*P_k vs w0*u]" if is_prob else "Scheme B: Saliency Space"
    axes[0].set_title(f"Dominant Feature Classification ({method_txt})\n[{smooth_txt} | Intensity Modulated by Sampling Density]", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    legend_patches = []
    for cat in cat_stats:
        c_norm = cat["color"] / 255.0
        label = f"{cat['name']} ({cat['pixel_ratio']:.1f}%)"
        legend_patches.append(mpatches.Patch(color=c_norm, label=label))
    axes[0].legend(handles=legend_patches, loc='lower left', fontsize=9, framealpha=0.85, facecolor='black', labelcolor='white')

    # 2. 優勢特徵圖 + 4DGS 高斯中心散點疊加
    axes[1].imshow(mod_rgb)
    if raw_counts is not None and np.sum(raw_counts) > 0:
        ys, xs = np.where(raw_counts > 0)
        axes[1].scatter(xs, ys, s=1.2, c='white', alpha=0.35, linewidths=0, label=f"4DGS Centers ({len(xs)} pts)")
        axes[1].legend(loc='lower left', fontsize=10, framealpha=0.85, facecolor='black', labelcolor='white')
    axes[1].set_title("Dominant Feature + 4DGS Centers Overlay\n[White dots: Actual 4DGS Gaussians]", fontsize=13, fontweight='bold')
    axes[1].axis('off')

    # 3. 佔比對比長條圖
    names = [c["name"].split('(')[0].strip() for c in cat_stats]
    pixel_ratios = [c["pixel_ratio"] for c in cat_stats]
    gs_ratios = [c["gs_ratio"] for c in cat_stats]
    colors = [c["color"] / 255.0 for c in cat_stats]

    y_pos = np.arange(len(names))
    height = 0.35

    axes[2].barh(y_pos - height/2, pixel_ratios, height, label='Image Area %', color='dimgray', alpha=0.8)
    axes[2].barh(y_pos + height/2, gs_ratios, height, label='4DGS Captured %', color=colors, alpha=0.95, edgecolor='black')

    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(names, fontsize=10, fontweight='bold')
    axes[2].set_xlabel("Percentage (%)", fontsize=11, fontweight='bold')
    axes[2].set_title("Category Breakdown:\nImage Area vs. 4DGS Allocation", fontsize=13, fontweight='bold')
    axes[2].grid(axis='x', linestyle='--', alpha=0.5)
    axes[2].legend(loc='lower right', fontsize=10)
    axes[2].invert_yaxis()

    plt.suptitle(f"Winner-take-all Feature Dominance & 4DGS Allocation Analysis - [{base_name}]", fontsize=16, fontweight='heavy', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def plot_4dgs_comparison_canvas(
    pil_img: Image.Image,
    p_4dgs: np.ndarray,
    raw_counts: np.ndarray,
    scores_dict: Dict[str, np.ndarray],
    eval_results: Dict[str, dict],
    save_path: str,
    base_name: str,
    comb_info: Optional[dict] = None,
    is_rgb_only: bool = True
):
    """
    繪製特徵預測與 4DGS 實際高斯分佈之全方位對比面板 (3x3 畫布)
    """
    fig, axes = plt.subplots(3, 3, figsize=(20, 18), dpi=150)
    W, H = pil_img.size

    # (0, 0) 原始 RGB 影像
    axes[0, 0].imshow(pil_img)
    axes[0, 0].set_title(f"Original RGB ({W}x{H})", fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')

    # (0, 1) 4DGS 連續機率密度 (P_4DGS)
    total_gs = int(np.sum(raw_counts))
    vmax_gs = np.percentile(p_4dgs, 99.8)
    im_gs = axes[0, 1].imshow(p_4dgs, cmap='turbo', vmin=0.0, vmax=vmax_gs)
    axes[0, 1].set_title(f"4DGS Actual Density (P_4DGS, KDE)\nSum=1.0, Total Gaussians: {total_gs:,}", fontsize=13, fontweight='bold')
    axes[0, 1].axis('off')
    fig.colorbar(im_gs, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # 決定主要比較分佈 (若有最優組合則用最優組合，否則用總分)
    best_key = "S_tex (+opt w0)" if "S_tex (+opt w0)" in eval_results else ("S_2D (+opt w0)" if "S_2D (+opt w0)" in eval_results else ("S_tex (Texture)" if "S_tex (Texture)" in eval_results else list(eval_results.keys())[0]))
    p_best = eval_results[best_key]["p_pred_opt"]
    vmax_best = np.percentile(p_best, 99.8)

    # (0, 2) 最優預測空間分佈
    im_best = axes[0, 2].imshow(p_best, cmap='turbo', vmin=0.0, vmax=vmax_best)
    opt_w0_txt = f" (Opt w0={eval_results[best_key]['alpha_opt']*100:.1f}%)" if eval_results[best_key]['alpha_opt'] > 0 else ""
    axes[0, 2].set_title(f"Predicted Sampling Density\n{best_key}{opt_w0_txt}", fontsize=13, fontweight='bold', color='crimson')
    axes[0, 2].axis('off')
    fig.colorbar(im_best, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # 輔助函式：繪製對稱發散殘差圖
    def plot_residual(ax, p_pred, title):
        res = p_pred - p_4dgs
        bound = np.percentile(np.abs(res), 99.5)
        im = ax.imshow(res, cmap='coolwarm', vmin=-bound, vmax=bound)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Blue: Under (<0) | Red: Over (>0)", fontsize=8)

    # (1, 0) 總體殘差 (P_best - P_4DGS)
    kl_best = eval_results[best_key]["kl_opt_by_scale"][1]
    corr_best = eval_results[best_key]["corr_opt_by_scale"][1]
    plot_residual(axes[1, 0], p_best, f"Total Residual ({best_key})\nKL(1x): {kl_best:.4f}, Corr: {corr_best:.4f}")

    if is_rgb_only:
        # (1, 1) Sobel 殘差
        p_sob = eval_results["S_sobel (Edge)"]["p_pred_opt"]
        kl_sob = eval_results["S_sobel (Edge)"]["kl_opt_by_scale"][1]
        corr_sob = eval_results["S_sobel (Edge)"]["corr_opt_by_scale"][1]
        w0_sob = eval_results["S_sobel (Edge)"]["alpha_opt"]
        plot_residual(axes[1, 1], p_sob, f"Sobel Edge (+w0={w0_sob*100:.1f}%)\nKL(1x): {kl_sob:.4f}, Corr: {corr_sob:.4f}")

        # (1, 2) DCT 殘差
        p_dct = eval_results["S_dct (MidHigh)"]["p_pred_opt"]
        kl_dct = eval_results["S_dct (MidHigh)"]["kl_opt_by_scale"][1]
        corr_dct = eval_results["S_dct (MidHigh)"]["corr_opt_by_scale"][1]
        w0_dct = eval_results["S_dct (MidHigh)"]["alpha_opt"]
        plot_residual(axes[1, 2], p_dct, f"DCT Texture (+w0={w0_dct*100:.1f}%)\nKL(1x): {kl_dct:.4f}, Corr: {corr_dct:.4f}")
    else:
        p_tex = eval_results["S_tex (Texture)"]["p_pred_opt"]
        kl_tex = eval_results["S_tex (Texture)"]["kl_opt_by_scale"][1]
        corr_tex = eval_results["S_tex (Texture)"]["corr_opt_by_scale"][1]
        plot_residual(axes[1, 1], p_tex, f"Residual (S_tex)\nKL: {kl_tex:.4f}")

        p_dep = eval_results["S_depth (Depth)"]["p_pred_opt"]
        kl_dep = eval_results["S_depth (Depth)"]["kl_opt_by_scale"][1]
        corr_dep = eval_results["S_depth (Depth)"]["corr_opt_by_scale"][1]
        plot_residual(axes[1, 2], p_dep, f"Residual (S_depth)\nKL: {kl_dep:.4f}")

    # (2, 0) 多解析度 KL 散度曲線 (1x, 1/2x, 1/4x, 1/8x)
    scale_labels = ['1x', '1/2x', '1/4x', '1/8x']
    scales = [1, 2, 4, 8]
    for name, res in eval_results.items():
        if name in [best_key, "S_tex (Texture)", "S_sobel (Edge)", "S_dct (MidHigh)", "S_chroma (Color)"]:
            y_vals = [res["kl_opt_by_scale"][sc] for sc in scales]
            axes[2, 0].plot(scale_labels, y_vals, marker='o', linewidth=2, label=name.split('(')[0].strip())
    axes[2, 0].set_title("Multi-Scale KL Divergence (with w0)\n[Lower = Better Fit to 4DGS]", fontsize=13, fontweight='bold')
    axes[2, 0].set_xlabel("Resolution Scale", fontsize=11, fontweight='bold')
    axes[2, 0].set_ylabel("D_KL(P_4DGS || P_pred)", fontsize=11, fontweight='bold')
    axes[2, 0].grid(True, linestyle='--', alpha=0.6)
    axes[2, 0].legend(loc='upper right', fontsize=9)

    # (2, 1) 權重分配或相關係數柱狀圖
    if comb_info is not None:
        labels = ["w0 (Smooth Floor)"] + [k.split('(')[0].strip() for k in comb_info["feat_keys"]]
        weights_vals = [comb_info["w_0_opt"] * 100.0] + [w * 100.0 for w in comb_info["w_opt"][1:]]
        x_idx = np.arange(len(labels))
        bar_colors = ['dimgray', 'crimson', 'forestgreen', 'goldenrod']
        axes[2, 1].bar(x_idx, weights_vals, color=bar_colors[:len(labels)], alpha=0.85, edgecolor='black')
        axes[2, 1].set_xticks(x_idx)
        axes[2, 1].set_xticklabels(labels, rotation=20, ha='right', fontsize=9, fontweight='bold')
        axes[2, 1].set_ylabel("Optimal Weight (%)", fontsize=11, fontweight='bold')
        axes[2, 1].set_title(f"Optimal Mixture Weights (KL-Minimizing)\n[w0={comb_info['w_0_opt']*100:.1f}%, KL={kl_best:.4f}]", fontsize=13, fontweight='bold')
        axes[2, 1].grid(axis='y', linestyle='--', alpha=0.6)
        for i, v in enumerate(weights_vals):
            axes[2, 1].text(i, v + 1.0, f"{v:.1f}%", ha='center', fontsize=9, fontweight='bold')
    else:
        # 顯示相關係數
        names_to_show = [k for k in ["S_sobel (Edge)", "S_dct (MidHigh)", "S_chroma (Color)"] if k in eval_results]
        x_idx = np.arange(len(names_to_show))
        corrs = [eval_results[k]["corr_opt_by_scale"][1] for k in names_to_show]
        axes[2, 1].bar(x_idx, corrs, color='steelblue', alpha=0.85)
        axes[2, 1].set_xticks(x_idx)
        axes[2, 1].set_xticklabels([k.split('(')[0].strip() for k in names_to_show], rotation=20, ha='right', fontsize=9)
        axes[2, 1].set_title("Pearson Correlation (1x)", fontsize=13, fontweight='bold')

    # (2, 2) Top-K% 空間重要度採樣之高斯捕捉召回率曲線
    k_percs = [10, 20, 30, 50]
    axes[2, 2].plot(k_percs, k_percs, 'k--', label='Uniform Random (y=x)', alpha=0.6)
    for name, res in eval_results.items():
        if name in [best_key, "S_tex (Texture)", "S_sobel (Edge)", "S_dct (MidHigh)", "S_chroma (Color)"]:
            y_rec = [res["top_k_recall"][k] for k in k_percs]
            axes[2, 2].plot(k_percs, y_rec, marker='s', linewidth=2, label=name.split('(')[0].strip())
    axes[2, 2].set_title("Gaussian Capture Recall vs Area %\n[Higher Above Dashed = More Efficient]", fontsize=13, fontweight='bold')
    axes[2, 2].set_xlabel("Top-K% Pixel Area Sampled", fontsize=11, fontweight='bold')
    axes[2, 2].set_ylabel("4DGS Points Captured (%)", fontsize=11, fontweight='bold')
    axes[2, 2].grid(True, linestyle='--', alpha=0.6)
    axes[2, 2].legend(loc='lower right', fontsize=9)

    plt.suptitle(f"4DGS Ground Truth Alignment & Sampling Efficiency Analysis - [{base_name}]", fontsize=16, fontweight='heavy', y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="2D 空間綜合評分器 (RGB 紋理 S_tex + DA3 幾何深度 S_depth) 視覺化工具")
    # 輸入與輸出
    parser.add_argument("--image", type=str, required=True, help="輸入影像路徑 (如 cam08.png)")
    parser.add_argument("--output_dir", type=str, default="visualizations/rgb_scores", help="輸出視覺化圖表目錄")
    
    # 解析度調整選項
    parser.add_argument("--max_dim", type=int, default=0, help="限制長邊最大尺寸並等比縮放 (例如 1280)")
    parser.add_argument("--scale", type=float, default=1.0, help="等比例縮放倍率 (例如 0.5)")
    parser.add_argument("--resize", type=int, nargs=2, default=None, metavar=('W', 'H'), help="強制指定寬高 (例如 1280 720)")
    
    # RGB 紋理特徵超參數
    parser.add_argument("--sigma", type=float, default=0.8, help="高斯預平滑 sigma (預設 0.8)")
    parser.add_argument("--ksize", type=int, default=3, help="高斯核大小 (預設 3)")
    parser.add_argument("--coring_ratio", type=float, default=0.08, help="Noise Coring 門檻係數 (預設 0.08)")
    parser.add_argument("--quantile", type=float, default=0.99, help="極值截斷分位數 (預設 0.99)")
    parser.add_argument("--dct_stride", type=int, default=1, choices=[1, 8], help="8x8 DCT 步長 (1: 滑動窗; 8: Block)")
    parser.add_argument("--num_high_to_zero", type=int, default=12, help="DCT 遮蔽最高頻係數個數 (預設 12)")
    parser.add_argument("--border_margin", type=int, default=0, help="畫面邊界遮蔽邊緣寬度 (像素)")
    parser.add_argument("--w_grad", type=float, default=1.0, help="Sobel 邊界權重 (預設 1.0)")
    parser.add_argument("--w_band", type=float, default=1.0, help="DCT MidHigh 紋理權重 (預設 1.0)")
    parser.add_argument("--w_color", type=float, default=1.0, help="ChromaGrad 色度權重 (預設 1.0)")
    
    # DA3 幾何深度超參數
    parser.add_argument("--depth", type=str, default=None, help="輸入深度檔案路徑 (.npy 或影像)")
    parser.add_argument("--conf", type=str, default=None, help="輸入信心度檔案路徑 (.npy 或影像)")
    parser.add_argument("--use_da3", action="store_true", help="啟用 Depth-Anything-3 線上即時推論")
    parser.add_argument("--da3_model", type=str, default="depth-anything/DA3-LARGE-1.1", help="DA3 預訓練模型名稱")
    parser.add_argument("--da3_process_res", type=int, default=504, help="DA3 推論內部處理解析度 (預設 504)")
    parser.add_argument("--w_d_edge", type=float, default=1.0, help="對數深度一階斷差梯度權重 (預設 1.0)")
    parser.add_argument("--w_curv", type=float, default=1.0, help="表面法向局部曲率權重 (預設 1.0)")
    parser.add_argument("--curv_ksize", type=int, default=5, help="曲率池化窗口大小 (預設 5)")
    parser.add_argument("--max_depth_change_factor", type=float, default=0.05, help="PCL 深度跳變門檻容許比率 (預設 0.05)")
    parser.add_argument("--conf_thresh", type=float, default=0.3, help="DA3 信心度硬過濾門檻 (預設 0.3)")
    
    # 2D 空間整合權重
    parser.add_argument("--w_0", type=float, default=0.0, help="空間基礎常數權重 (預設 0.0)")
    parser.add_argument("--w_tex", type=float, default=1.0, help="RGB 紋理分數權重 (預設 1.0)")
    parser.add_argument("--w_depth", type=float, default=1.0, help="幾何深度分數權重 (預設 1.0)")

    # 4DGS 密度比較與分佈對齊
    parser.add_argument("--gs_density", type=str, default=None, help="4DGS 實際投影高斯中心密度圖檔案路徑 (.npy)")
    parser.add_argument("--kde_sigma", type=float, default=2.0, help="4DGS 連續機率密度估計之高斯平滑核大小 (預設 2.0)")
    parser.add_argument("--dominant_bg_thresh", type=float, default=0.05, help="優勢特徵分類之平坦背景門檻 (預設 0.05)")
    parser.add_argument("--agg_sigma", type=float, default=2.0, help="優勢特徵空間高斯能量平滑尺度 (預設 2.0，設為 0 則不平滑)")
    parser.add_argument("--rgb_only", action="store_true", help="強制純 RGB 紋理色彩評估模式，暫停深度幾何分數")
    parser.add_argument("--optimize_w0", action="store_true", default=True, help="自動最佳化求解平滑背景基準分 w0 與特徵權重")

    # 視覺化風格與儲存
    parser.add_argument("--colormap", type=str, default="turbo", choices=["turbo", "viridis", "inferno", "magma", "plasma", "jet"], help="熱力圖色盤 (預設 turbo)")
    parser.add_argument("--show_defenses", action="store_true", help="額外輸出抗噪與防護防線的中間步驟圖")
    parser.add_argument("--save_individual", action="store_true", help="額外儲存各分數的高解析度獨立圖檔")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="運算裝置 (cuda 或 cpu)")

    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("🎨 2D 空間特徵評分 (RGB 紋理 + DA3 幾何深度) 視覺化分析")
    print("=" * 60)
    print(f"📁 輸入圖片: {args.image}")
    print(f"💾 輸出目錄: {args.output_dir}")
    print(f"⚙️ 運算裝置: {device}")

    # 1. 載入並調整 RGB 解析度
    tensor_rgb, pil_img, (orig_w, orig_h) = load_and_preprocess_image(
        args.image,
        max_dim=args.max_dim,
        scale=args.scale,
        resize=args.resize
    )
    H, W = tensor_rgb.shape[-2:]
    print(f"📐 原始尺寸: {orig_w}x{orig_h} ──► 處理解析度: {W}x{H} (長寬比: {W/H:.2f})")

    tensor_rgb = tensor_rgb.to(device)

    # 2. 獲取深度資訊 (若有指定 --depth 或 --use_da3，且未開啟 --rgb_only)
    tensor_depth = None
    tensor_conf = None
    tensor_intrs = None

    if not args.rgb_only and args.use_da3:
        print(f"🤖 啟用 Depth-Anything-3 即時推論 ({args.da3_model})...")
        from depth_anything_3.api import DepthAnything3
        t_da3_start = time.time()
        da3_model = DepthAnything3.from_pretrained(args.da3_model).to(device)
        da3_model.eval()

        with torch.no_grad():
            pred = da3_model.inference([pil_img], process_res=args.da3_process_res)
        
        cost_da3 = (time.time() - t_da3_start) * 1000.0
        print(f"⚡ DA3 推論完成！耗時: {cost_da3:.2f} ms")

        # 保留原生解析度，所有幾何微分計算 (斷差/法向/曲率) 在原生網格執行以杜絕插值摺痕
        depth_raw = torch.from_numpy(pred.depth).float().unsqueeze(1).to(device) # [1, 1, H_proc, W_proc]
        tensor_depth = depth_raw

        if hasattr(pred, 'conf') and pred.conf is not None:
            tensor_conf = torch.from_numpy(pred.conf).float().unsqueeze(1).to(device)

        if hasattr(pred, 'intrinsics') and pred.intrinsics is not None:
            tensor_intrs = torch.from_numpy(pred.intrinsics[0]).float().to(device) # 原生相機內參

        del da3_model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    elif not args.rgb_only and args.depth is not None:
        print(f"📂 載入深度快取檔案: {args.depth}")
        tensor_depth = load_depth_from_file(args.depth, H, W).to(device)
        if args.conf is not None:
            print(f"📂 載入信心度快取檔案: {args.conf}")
            tensor_conf = load_depth_from_file(args.conf, H, W).to(device)
    elif args.rgb_only:
        print("🎯 模式設定: 純 RGB 紋理色彩分析 (已暫停深度幾何分數)")

    # 3. 建立綜合評分器
    spatial_scorer = Spatial2DScorer(
        w_0=args.w_0,
        w_tex=args.w_tex,
        w_depth=args.w_depth,
        w_sobel=args.w_grad,
        w_dct=args.w_band,
        w_chroma=args.w_color,
        w_d_edge=args.w_d_edge,
        w_curv=args.w_curv,
        curv_ksize=args.curv_ksize,
        max_depth_change_factor=args.max_depth_change_factor,
        conf_thresh=args.conf_thresh
    ).to(device)

    # 4. 前向特徵計算
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_start = time.time()

    with torch.no_grad():
        results = spatial_scorer(
            rgb=tensor_rgb,
            depth=tensor_depth,
            conf=tensor_conf,
            intrinsics=tensor_intrs
        )

    if device.type == 'cuda':
        torch.cuda.synchronize()
    cost_ms = (time.time() - t_start) * 1000.0
    print(f"⚡ 特徵計算完成！耗時: {cost_ms:.2f} ms")

    # 5. 提取特徵 Numpy 陣列
    s_sobel_np = tensor_to_numpy(results['s_sobel'])
    s_dct_np = tensor_to_numpy(results['s_dct'])
    s_chroma_np = tensor_to_numpy(results['s_chroma'])
    s_tex_np = tensor_to_numpy(results['s_tex'])
    s_2d_np = tensor_to_numpy(results['s_2d'])

    has_depth = (tensor_depth is not None)
    if has_depth:
        depth_disp = F.interpolate(tensor_depth, size=(H, W), mode='bilinear', align_corners=False)
        depth_np = tensor_to_numpy(depth_disp)
        s_depth_grad_np = tensor_to_numpy(results['s_depth_grad'])
        s_curv_np = tensor_to_numpy(results['s_curv'])
        s_curv_gated_np = tensor_to_numpy(results['s_curv_gated'])
        w_grazing_np = tensor_to_numpy(results['w_grazing'])
        s_depth_np = tensor_to_numpy(results['s_depth'])
        normals_np = results['normals'][0].permute(1, 2, 0).detach().cpu().numpy()  # [H, W, 3]
        # 法向量視覺化 RGB 映射: [-1, 1] ──► [0, 1]
        normal_rgb = np.clip((normals_np + 1.0) / 2.0, 0.0, 1.0)

    # 6. 印出統計資訊表
    stats = {
        "Sobel (Boundary)": s_sobel_np,
        "DCT MidHigh (Texture)": s_dct_np,
        "ChromaGrad (Color)": s_chroma_np,
        "S_tex (Texture Sum)": s_tex_np
    }
    if has_depth:
        stats.update({
            "Depth (Meters)": depth_np,
            "S_depth_grad (Silhouette)": s_depth_grad_np,
            "S_curv (Normal Curvature)": s_curv_np,
            "w_grazing (Anti-Streaking)": w_grazing_np,
            "S_depth (Depth Sum)": s_depth_np,
            "S_2D (Total Spatial)": s_2d_np
        })
    print_stats_table(stats)

    base_name = os.path.splitext(os.path.basename(args.image))[0]

    # 7. 生成綜合對比面板圖
    consolidated_path = os.path.join(args.output_dir, f"{base_name}_scores_comparison.png")

    if has_depth:
        # 3x3 九宮格綜合畫布
        fig, axes = plt.subplots(3, 3, figsize=(19, 17), dpi=150)

        # (0, 0) 原始 RGB 影像
        axes[0, 0].imshow(pil_img)
        axes[0, 0].set_title(f"Original RGB ({W}x{H})", fontsize=13, fontweight='bold')
        axes[0, 0].axis('off')

        # (0, 1) 對數深度圖 (Log-Depth)
        im_d = axes[0, 1].imshow(np.log(np.clip(depth_np, 1e-4, None)), cmap='magma')
        axes[0, 1].set_title(f"Log-Depth Map (ln D)\nMin: {depth_np.min():.2f}m, Max: {depth_np.max():.2f}m", fontsize=13, fontweight='bold')
        axes[0, 1].axis('off')
        fig.colorbar(im_d, ax=axes[0, 1], fraction=0.046, pad=0.04)

        # (0, 2) 表面法向量視覺化圖 (Surface Normal RGB: (n+1)/2)
        axes[0, 2].imshow(normal_rgb)
        axes[0, 2].set_title("Surface Normals (LINEMOD)\nX-Right (R), Y-Down (G), Z-Facing (B)", fontsize=13, fontweight='bold')
        axes[0, 2].axis('off')

        # (1, 0) 幾何輪廓斷差梯度 S_depth_grad
        im_dg = axes[1, 0].imshow(s_depth_grad_np, cmap=args.colormap, vmin=0.0, vmax=1.0)
        axes[1, 0].set_title(f"Silhouette Discontinuity (S_depth_grad)\nw_d_edge={args.w_d_edge}", fontsize=13, fontweight='bold')
        axes[1, 0].axis('off')
        fig.colorbar(im_dg, ax=axes[1, 0], fraction=0.046, pad=0.04)

        # (1, 1) 表面法向曲率 S_curv (含 PCL 深度門檻保護)
        im_cv = axes[1, 1].imshow(s_curv_gated_np, cmap=args.colormap, vmin=0.0, vmax=max(0.1, float(s_curv_gated_np.max())))
        axes[1, 1].set_title(f"Surface Curvature (S_curv)\nPCL Depth-Gated, ksize={args.curv_ksize}", fontsize=13, fontweight='bold')
        axes[1, 1].axis('off')
        fig.colorbar(im_cv, ax=axes[1, 1], fraction=0.046, pad=0.04)

        # (1, 2) 掠射角防拉扯濾波 w_grazing
        im_gr = axes[1, 2].imshow(w_grazing_np, cmap='cividis', vmin=0.0, vmax=1.0)
        axes[1, 2].set_title("Anti-Streaking Filter (w_grazing)\nZero at Grazing Angles (>80°)", fontsize=13, fontweight='bold')
        axes[1, 2].axis('off')
        fig.colorbar(im_gr, ax=axes[1, 2], fraction=0.046, pad=0.04)

        # (2, 0) RGB 紋理加權總分 S_tex
        im_tx = axes[2, 0].imshow(s_tex_np, cmap=args.colormap, vmin=0.0, vmax=max(1.0, float(s_tex_np.max())))
        axes[2, 0].set_title(f"Texture & Color Score (S_tex)\nw_sobel={args.w_grad}, w_dct={args.w_band}, w_color={args.w_color}", fontsize=13, fontweight='bold', color='darkblue')
        axes[2, 0].axis('off')
        fig.colorbar(im_tx, ax=axes[2, 0], fraction=0.046, pad=0.04)

        # (2, 1) 幾何深度加權總分 S_depth
        im_dp = axes[2, 1].imshow(s_depth_np, cmap=args.colormap, vmin=0.0, vmax=max(1.0, float(s_depth_np.max())))
        axes[2, 1].set_title(f"Geometry Depth Score (S_depth)\nw_edge={args.w_d_edge}, w_curv={args.w_curv}", fontsize=13, fontweight='bold', color='darkgreen')
        axes[2, 1].axis('off')
        fig.colorbar(im_dp, ax=axes[2, 1], fraction=0.046, pad=0.04)

        # (2, 2) 最終空間總分 S_2D 疊加圖 (Overlay)
        rgb_arr = np.array(pil_img, dtype=np.float32) / 255.0
        cmap_obj = plt.get_cmap(args.colormap)
        norm_score = s_2d_np / (np.max(s_2d_np) + 1e-8)
        heatmap_rgb = cmap_obj(norm_score)[..., :3]
        overlay = 0.55 * rgb_arr + 0.45 * heatmap_rgb
        overlay = np.clip(overlay, 0.0, 1.0)
        axes[2, 2].imshow(overlay)
        axes[2, 2].set_title(f"Final S_2D Sampling Overlay\nw_tex={args.w_tex}, w_depth={args.w_depth}", fontsize=13, fontweight='bold', color='crimson')
        axes[2, 2].axis('off')

        plt.suptitle(f"Unified 2D Spatial Scoring (S_2D = S_tex + S_depth) - [{base_name}]", fontsize=17, fontweight='heavy', y=0.99)

    else:
        # 2x3 RGB-Only 畫布
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=150)
        axes[0, 0].imshow(pil_img)
        axes[0, 0].set_title(f"Original RGB ({W}x{H})", fontsize=13, fontweight='bold')
        axes[0, 0].axis('off')

        im1 = axes[0, 1].imshow(s_sobel_np, cmap=args.colormap, vmin=0.0, vmax=1.0)
        axes[0, 1].set_title(f"1. Precision Boundary (S_Sobel)\nw_grad={args.w_grad}", fontsize=13, fontweight='bold')
        axes[0, 1].axis('off')
        fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        im2 = axes[0, 2].imshow(s_dct_np, cmap=args.colormap, vmin=0.0, vmax=1.0)
        axes[0, 2].set_title(f"2. Band-pass Texture (S_DCT_MidHigh)\nw_band={args.w_band}", fontsize=13, fontweight='bold')
        axes[0, 2].axis('off')
        fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

        im3 = axes[1, 0].imshow(s_chroma_np, cmap=args.colormap, vmin=0.0, vmax=1.0)
        axes[1, 0].set_title(f"3. Chroma Boundary (S_ColorVar)\nw_color={args.w_color}", fontsize=13, fontweight='bold')
        axes[1, 0].axis('off')
        fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

        im4 = axes[1, 1].imshow(s_tex_np, cmap=args.colormap, vmin=0.0, vmax=max(1.0, float(s_tex_np.max())))
        axes[1, 1].set_title(f"Combined Texture Score (S_tex)\nw_grad={args.w_grad}, w_band={args.w_band}, w_color={args.w_color}", fontsize=13, fontweight='bold', color='darkblue')
        axes[1, 1].axis('off')
        fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

        rgb_arr = np.array(pil_img, dtype=np.float32) / 255.0
        cmap_obj = plt.get_cmap(args.colormap)
        norm_tex = s_tex_np / (np.max(s_tex_np) + 1e-8)
        heatmap_rgb = cmap_obj(norm_tex)[..., :3]
        overlay = 0.55 * rgb_arr + 0.45 * heatmap_rgb
        overlay = np.clip(overlay, 0.0, 1.0)
        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title("Gaussian Sampling Overlay on RGB\n(High Response = Denser Gaussians)", fontsize=13, fontweight='bold', color='crimson')
        axes[1, 2].axis('off')

        plt.suptitle(f"RGB Texture & Color Entropy Scoring (S_tex) - [{base_name}]", fontsize=16, fontweight='heavy', y=0.98)

    plt.tight_layout()
    plt.savefig(consolidated_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"🖼️ 綜合對比面板圖已儲存至: {consolidated_path}")

    # 8. 儲存個別特徵圖檔 (--save_individual)
    if args.save_individual:
        ind_dir = os.path.join(args.output_dir, f"{base_name}_individual")
        os.makedirs(ind_dir, exist_ok=True)
        pil_img.save(os.path.join(ind_dir, "00_original_rgb.png"))
        save_heatmap(s_sobel_np, os.path.join(ind_dir, "01_s_sobel.png"), colormap=args.colormap, vmin=0.0, vmax=1.0)
        save_heatmap(s_dct_np, os.path.join(ind_dir, "02_s_dct_midhigh.png"), colormap=args.colormap, vmin=0.0, vmax=1.0)
        save_heatmap(s_chroma_np, os.path.join(ind_dir, "03_s_chroma.png"), colormap=args.colormap, vmin=0.0, vmax=1.0)
        save_heatmap(s_tex_np, os.path.join(ind_dir, "04_s_tex_combined.png"), colormap=args.colormap, vmin=0.0, vmax=float(s_tex_np.max()))

        if has_depth:
            save_heatmap(depth_np, os.path.join(ind_dir, "05_depth_metric.png"), colormap='magma', vmin=float(depth_np.min()), vmax=float(depth_np.max()))
            Image.fromarray((normal_rgb * 255).astype(np.uint8)).save(os.path.join(ind_dir, "06_surface_normals_rgb.png"))
            save_heatmap(s_depth_grad_np, os.path.join(ind_dir, "07_s_depth_grad.png"), colormap=args.colormap, vmin=0.0, vmax=1.0)
            save_heatmap(s_curv_gated_np, os.path.join(ind_dir, "08_s_curv_gated.png"), colormap=args.colormap, vmin=0.0, vmax=max(0.1, float(s_curv_gated_np.max())))
            save_heatmap(w_grazing_np, os.path.join(ind_dir, "09_w_grazing.png"), colormap='cividis', vmin=0.0, vmax=1.0)
            save_heatmap(s_depth_np, os.path.join(ind_dir, "10_s_depth_combined.png"), colormap=args.colormap, vmin=0.0, vmax=float(s_depth_np.max()))
            save_heatmap(s_2d_np, os.path.join(ind_dir, "11_s_2d_final.png"), colormap=args.colormap, vmin=0.0, vmax=float(s_2d_np.max()))
            Image.fromarray((overlay * 255).astype(np.uint8)).save(os.path.join(ind_dir, "12_overlay_s_2d.png"))
        else:
            Image.fromarray((overlay * 255).astype(np.uint8)).save(os.path.join(ind_dir, "05_overlay_s_tex.png"))

        print(f"📁 各特徵獨立高解析度熱圖已儲存至: {ind_dir}/")

    # 9. 4DGS 密度比較與分佈對齊分析 (--gs_density)
    if args.gs_density is not None:
        print(f"\n" + "=" * 60)
        print("🎯 執行 4DGS 實際高斯分佈對齊與重要度採樣評估...")
        print("=" * 60)
        raw_counts, p_4dgs = load_and_process_gs_density(
            args.gs_density,
            target_h=H,
            target_w=W,
            kde_sigma=args.kde_sigma
        )
        total_gaussians = int(np.sum(raw_counts))
        print(f"📊 4DGS 高斯總數: {total_gaussians:,} 顆，非零像素: {int(np.sum(raw_counts > 0)):,} ({np.mean(raw_counts > 0)*100:.2f}%)")

        # 彙整待評估特徵分佈
        eval_scores = {}
        if not args.rgb_only and has_depth:
            eval_scores["S_2D (Total)"] = s_2d_np
            eval_scores["S_depth (Depth)"] = s_depth_np
            eval_scores["S_depth_grad (Silhouette)"] = s_depth_grad_np
            eval_scores["S_curv (Curvature)"] = s_curv_gated_np

        eval_scores.update({
            "S_tex (Texture)": s_tex_np,
            "S_sobel (Edge)": s_sobel_np,
            "S_dct (MidHigh)": s_dct_np,
            "S_chroma (Color)": s_chroma_np,
        })

        # 多解析度分佈對齊評估 (KL, Pearson, Top-K Recall, w0 最佳化)
        eval_results, comb_info = evaluate_distribution_alignment(
            eval_scores,
            p_4dgs=p_4dgs,
            raw_counts=raw_counts,
            scales=[1, 2, 4, 8],
            optimize_w0=args.optimize_w0
        )
        print_multiscale_alignment_table(eval_results, comb_info=comb_info)

        # 輸出 4DGS 對比面板圖
        comp_name = "cam08_rgb_4dgs_comparison.png" if args.rgb_only else f"{base_name}_4dgs_comparison.png"
        comp_4dgs_path = os.path.join(args.output_dir, comp_name)
        plot_4dgs_comparison_canvas(
            pil_img=pil_img,
            p_4dgs=p_4dgs,
            raw_counts=raw_counts,
            scores_dict=eval_scores,
            eval_results=eval_results,
            save_path=comp_4dgs_path,
            base_name=base_name,
            comb_info=comb_info,
            is_rgb_only=args.rgb_only
        )
        print(f"🖼️ 4DGS 分佈對齊與採樣效率面板已儲存至: {comp_4dgs_path}")

        # 10. 像素優勢特徵分類地圖 (Winner-take-all Dominant Feature Map - 方案 A: 機率空間比大小)
        subfeatures = {
            "s_sobel": s_sobel_np,
            "s_dct": s_dct_np,
            "s_chroma": s_chroma_np,
        }
        if not args.rgb_only and has_depth:
            subfeatures.update({
                "s_depth_grad": s_depth_grad_np,
                "s_curv": s_curv_gated_np,
            })

        # 從 comb_info 獲取分佈對齊求解出的最優權重組合 [w0*, w1*, w2*, w3*...]
        subweights = {}
        w0_opt_weight = 0.40
        if comb_info is not None:
            w0_opt_weight = float(comb_info["w_0_opt"])
            for fkey, wval in zip(comb_info["feat_keys"], comb_info["w_opt"][1:]):
                fl = fkey.lower()
                if "sobel" in fl:
                    subweights["s_sobel"] = float(wval)
                elif "dct" in fl:
                    subweights["s_dct"] = float(wval)
                elif "chroma" in fl:
                    subweights["s_chroma"] = float(wval)
                elif "depth" in fl:
                    subweights["s_depth_grad"] = float(wval)
                elif "curv" in fl:
                    subweights["s_curv"] = float(wval)
        else:
            subweights = {
                "s_sobel": args.w_grad,
                "s_dct": args.w_band,
                "s_chroma": args.w_color,
            }
            if not args.rgb_only and has_depth:
                subweights["s_depth_grad"] = args.w_d_edge
                subweights["s_curv"] = args.w_curv
            tot_w = sum(subweights.values())
            subweights = {k: v / (tot_w + 1e-8) for k, v in subweights.items()}

        dominant_res = compute_dominant_feature_map(
            subfeatures_dict=subfeatures,
            weights_dict=subweights,
            w0_weight=w0_opt_weight,
            s_comb=s_tex_np if args.rgb_only else s_2d_np,
            raw_counts=raw_counts,
            agg_sigma=args.agg_sigma,
            use_prob_space=True,
            bg_thresh=args.dominant_bg_thresh
        )

        dom_name = "cam08_rgb_dominant_feature_map.png" if args.rgb_only else f"{base_name}_dominant_feature_map.png"
        dominant_path = os.path.join(args.output_dir, dom_name)
        plot_dominant_feature_canvas(
            pil_img=pil_img,
            dominant_res=dominant_res,
            raw_counts=raw_counts,
            save_path=dominant_path,
            base_name=base_name,
            agg_sigma=args.agg_sigma
        )
        print(f"🖼️ 優勢特徵分類與高斯落點分析圖已儲存至: {dominant_path}")

        # 印出優勢特徵統計摘要
        print("\n" + "-" * 72)
        print(f"{'優勢特徵類別':<32} | {'像素面積佔比':<14} | {'4DGS 高斯捕捉率':<16}")
        print("-" * 72)
        for cat in dominant_res["category_stats"]:
            print(f"{cat['name']:<32} | {cat['pixel_ratio']:<13.1f}% | {cat['gs_ratio']:<15.1f}%")
        print("-" * 72 + "\n")

        # 若啟用 --save_individual，額外輸出高解析度 4DGS 相關獨立圖檔
        if args.save_individual:
            ind_dir = os.path.join(args.output_dir, f"{base_name}_individual")
            save_heatmap(p_4dgs, os.path.join(ind_dir, "13_4dgs_density_kde.png"), colormap='turbo', vmin=0.0, vmax=float(np.percentile(p_4dgs, 99.8)))
            
            # 殘差圖
            best_key = "S_tex (+opt w0)" if "S_tex (+opt w0)" in eval_results else ("S_2D (+opt w0)" if "S_2D (+opt w0)" in eval_results else list(eval_results.keys())[0])
            p_best = eval_results[best_key]["p_pred_opt"]
            res_best = p_best - p_4dgs
            b_best = float(np.percentile(np.abs(res_best), 99.5))
            save_heatmap(res_best, os.path.join(ind_dir, "14_residual_opt_vs_4dgs.png"), colormap='coolwarm', vmin=-b_best, vmax=b_best)

            p_sob = eval_results["S_sobel (Edge)"]["p_pred_opt"]
            res_sob = p_sob - p_4dgs
            b_sob = float(np.percentile(np.abs(res_sob), 99.5))
            save_heatmap(res_sob, os.path.join(ind_dir, "15_residual_ssobel_opt_vs_4dgs.png"), colormap='coolwarm', vmin=-b_sob, vmax=b_sob)

            p_dct = eval_results["S_dct (MidHigh)"]["p_pred_opt"]
            res_dct = p_dct - p_4dgs
            b_dct = float(np.percentile(np.abs(res_dct), 99.5))
            save_heatmap(res_dct, os.path.join(ind_dir, "16_residual_sdct_opt_vs_4dgs.png"), colormap='coolwarm', vmin=-b_dct, vmax=b_dct)

            if not args.rgb_only and has_depth:
                p_dep = eval_results["S_depth (Depth)"]["p_pred_opt"]
                res_dep = p_dep - p_4dgs
                b_dep = float(np.percentile(np.abs(res_dep), 99.5))
                save_heatmap(res_dep, os.path.join(ind_dir, "17_residual_sdepth_vs_4dgs.png"), colormap='coolwarm', vmin=-b_dep, vmax=b_dep)

            Image.fromarray(dominant_res["modulated_rgb"]).save(os.path.join(ind_dir, "18_dominant_features_modulated.png"))
            Image.fromarray(dominant_res["base_rgb"]).save(os.path.join(ind_dir, "19_dominant_features_labels.png"))

    print(f"\n🎉 視覺化分析圓滿完成！\n")


if __name__ == "__main__":
    main()
