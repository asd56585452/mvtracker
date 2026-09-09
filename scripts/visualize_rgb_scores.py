#!/usr/bin/env python3
"""
RGB 影像紋理與色彩熵分數 (S_tex) 獨立視覺化工具

功能：
1. 支援任意解析度與縮放 (--max_dim, --scale, --resize)。
2. 計算並視覺化三大子項：
   - 像素級精準邊界 (S_Sobel)
   - 去除噪聲的中高頻材質紋理 (S_DCT_MidHigh)
   - 跨通道色彩對比 (S_ColorVar / ChromaGrad)
3. 支援四大抗噪防禦中間步驟對比 (--show_defenses)。
4. 儲存多面板合成對比圖與個別高解析度熱力圖 (--save_individual)。
5. 輸出各分數通道之統計數值 (Mean, Max, Quantile 99%, 零值抑制比例)。
"""

import os
import sys
import argparse
import time
from typing import Tuple, Dict, Optional, List

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# 加入專案根目錄至 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mvtracker.utils.feature_scores import (
    RGBTextureScorer,
    rgb_to_ycbcr,
    gaussian_blur2d,
    sobel_gradient,
    block_dct8x8_midhigh,
    chroma_gradient,
    noise_coring,
    robust_quantile_clamp
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
    print("\n" + "=" * 80)
    print(f"{'特徵名稱':<22} | {'最小值':<8} | {'最大值':<8} | {'平均值':<8} | {'99%分位數':<10} | {'非零比例 (>0)':<12}")
    print("-" * 80)
    for name, s in stats_dict.items():
        arr = s.flatten()
        min_v = float(np.min(arr))
        max_v = float(np.max(arr))
        mean_v = float(np.mean(arr))
        q99_v = float(np.percentile(arr, 99))
        nonzero_ratio = float(np.mean(arr > 1e-4) * 100.0)
        print(f"{name:<22} | {min_v:<8.4f} | {max_v:<8.4f} | {mean_v:<8.4f} | {q99_v:<10.4f} | {nonzero_ratio:<11.1f}%")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="RGB 影像紋理與色彩熵分數 (S_tex) 獨立視覺化工具")
    # 輸入與輸出
    parser.add_argument("--image", type=str, required=True, help="輸入影像路徑 (如 mask.jpg)")
    parser.add_argument("--output_dir", type=str, default="visualizations/rgb_scores", help="輸出視覺化圖表目錄")
    
    # 解析度調整選項
    parser.add_argument("--max_dim", type=int, default=0, help="限制長邊最大尺寸並等比縮放 (例如 1280)")
    parser.add_argument("--scale", type=float, default=1.0, help="等比例縮放倍率 (例如 0.5)")
    parser.add_argument("--resize", type=int, nargs=2, default=None, metavar=('W', 'H'), help="強制指定寬高 (例如 1280 720)")
    
    # 特徵與抗噪超參數
    parser.add_argument("--sigma", type=float, default=0.8, help="高斯預平滑 sigma (預設 0.8)")
    parser.add_argument("--ksize", type=int, default=3, help="高斯核大小 (預設 3)")
    parser.add_argument("--coring_ratio", type=float, default=0.08, help="Noise Coring 門檻係數 (預設 0.08，即均值的 8%)")
    parser.add_argument("--quantile", type=float, default=0.99, help="極值截斷分位數 (預設 0.99)")
    parser.add_argument("--dct_stride", type=int, default=1, choices=[1, 8], help="8x8 DCT 卷積步長 (1: 滑動窗全解析度無瑕疵; 8: 傳統 Block 切塊模式)")
    parser.add_argument("--num_high_to_zero", type=int, default=12, help="DCT 遮蔽最高頻係數個數 (預設 12)")
    parser.add_argument("--border_margin", type=int, default=0, help="畫面最外圈邊界遮蔽邊緣寬度 (像素)")
    
    # 融合權重
    parser.add_argument("--w_grad", type=float, default=1.0, help="Sobel 邊界權重 (預設 1.0)")
    parser.add_argument("--w_band", type=float, default=1.0, help="DCT MidHigh 紋理權重 (預設 1.0)")
    parser.add_argument("--w_color", type=float, default=1.0, help="ChromaGrad 色度權重 (預設 1.0)")
    
    # 視覺化風格與儲存
    parser.add_argument("--colormap", type=str, default="turbo", choices=["turbo", "viridis", "inferno", "magma", "plasma", "jet"], help="熱力圖色盤 (預設 turbo)")
    parser.add_argument("--show_defenses", action="store_true", help="額外輸出四大抗噪防線的中間步驟對比圖")
    parser.add_argument("--save_individual", action="store_true", help="額外儲存各分數的高解析度獨立圖檔")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="運算裝置 (cuda 或 cpu)")

    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 50)
    print("🎨 RGB 影像紋理與色彩熵分數視覺化分析")
    print("=" * 50)
    print(f"📁 輸入圖片: {args.image}")
    print(f"💾 輸出目錄: {args.output_dir}")
    print(f"⚙️ 運算裝置: {device}")

    # 1. 載入並調整解析度
    tensor_rgb, pil_img, (orig_w, orig_h) = load_and_preprocess_image(
        args.image,
        max_dim=args.max_dim,
        scale=args.scale,
        resize=args.resize
    )
    H, W = tensor_rgb.shape[-2:]
    print(f"📐 原始尺寸: {orig_w}x{orig_h} ──► 處理解析度: {W}x{H} (長寬比: {W/H:.2f})")

    tensor_rgb = tensor_rgb.to(device)

    # 2. 建立評分器
    scorer = RGBTextureScorer(
        gaussian_sigma=args.sigma,
        gaussian_ksize=args.ksize,
        dct_stride=args.dct_stride,
        num_high_to_zero=args.num_high_to_zero,
        coring_ratio=args.coring_ratio,
        quantile=args.quantile,
        border_margin=args.border_margin,
        default_weights=(args.w_grad, args.w_band, args.w_color)
    ).to(device)

    # 3. 前向計算特徵與耗時統計
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_start = time.time()

    with torch.no_grad():
        results = scorer(tensor_rgb)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    cost_ms = (time.time() - t_start) * 1000.0
    print(f"⚡ 特徵計算完成！耗時: {cost_ms:.2f} ms (dct_stride={args.dct_stride})")

    # 提取 numpy 陣列
    s_sobel_np = tensor_to_numpy(results['s_sobel'])
    s_dct_np = tensor_to_numpy(results['s_dct'])
    s_chroma_np = tensor_to_numpy(results['s_chroma'])
    s_tex_np = tensor_to_numpy(results['s_tex'])
    
    s_sobel_raw_np = tensor_to_numpy(results['s_sobel_raw'])
    s_dct_raw_np = tensor_to_numpy(results['s_dct_raw'])
    s_chroma_raw_np = tensor_to_numpy(results['s_chroma_raw'])

    ycbcr_np = results['ycbcr'][0].detach().cpu().numpy()
    y_channel = ycbcr_np[0]
    cb_channel = ycbcr_np[1]
    cr_channel = ycbcr_np[2]
    y_smooth_np = tensor_to_numpy(results['y_smooth'])

    # 4. 印出數值統計分析
    stats = {
        "Sobel (Raw)": s_sobel_raw_np,
        "Sobel (Clean/Norm)": s_sobel_np,
        "DCT MidHigh (Raw)": s_dct_raw_np,
        "DCT MidHigh (Clean)": s_dct_np,
        "ChromaGrad (Raw)": s_chroma_raw_np,
        "ChromaGrad (Clean)": s_chroma_np,
        "S_tex (Combined)": s_tex_np
    }
    print_stats_table(stats)

    # 5. 生成主要對比面板 (Consolidated Comparison Figure)
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    consolidated_path = os.path.join(args.output_dir, f"{base_name}_scores_comparison.png")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=150)
    
    # (0, 0) 原始 RGB 影像
    axes[0, 0].imshow(pil_img)
    axes[0, 0].set_title(f"Original RGB ({W}x{H})", fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')

    # (0, 1) 像素級精準邊界 S_Sobel
    im1 = axes[0, 1].imshow(s_sobel_np, cmap=args.colormap, vmin=0.0, vmax=1.0)
    axes[0, 1].set_title(f"1. Precision Boundary (S_Sobel)\nw_grad={args.w_grad}", fontsize=13, fontweight='bold')
    axes[0, 1].axis('off')
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # (0, 2) 去除噪聲的中高頻材質紋理 S_DCT_MidHigh
    im2 = axes[0, 2].imshow(s_dct_np, cmap=args.colormap, vmin=0.0, vmax=1.0)
    axes[0, 2].set_title(f"2. Band-pass Texture (S_DCT_MidHigh)\nw_band={args.w_band}, stride={args.dct_stride}", fontsize=13, fontweight='bold')
    axes[0, 2].axis('off')
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # (1, 0) 跨通道色彩對比 S_ColorVar (ChromaGrad)
    im3 = axes[1, 0].imshow(s_chroma_np, cmap=args.colormap, vmin=0.0, vmax=1.0)
    axes[1, 0].set_title(f"3. Chroma Boundary (S_ColorVar)\nw_color={args.w_color}", fontsize=13, fontweight='bold')
    axes[1, 0].axis('off')
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # (1, 1) 最終加權融合總分 S_tex
    im4 = axes[1, 1].imshow(s_tex_np, cmap=args.colormap, vmin=0.0, vmax=max(1.0, float(s_tex_np.max())))
    axes[1, 1].set_title(f"Combined Texture & Color Score (S_tex)\nw_grad={args.w_grad}, w_band={args.w_band}, w_color={args.w_color}", fontsize=13, fontweight='bold', color='darkblue')
    axes[1, 1].axis('off')
    fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # (1, 2) S_tex 疊加至原圖上的採樣熱力圖 (Overlay)
    rgb_arr = np.array(pil_img, dtype=np.float32) / 255.0
    cmap_obj = plt.get_cmap(args.colormap)
    norm_tex = s_tex_np / (np.max(s_tex_np) + 1e-8)
    heatmap_rgb = cmap_obj(norm_tex)[..., :3]
    # 半透明疊加: 60% 原圖 + 40% 熱力圖
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

    # 6. 生成四大抗噪防禦中間步驟對比圖 (--show_defenses)
    if args.show_defenses:
        defenses_path = os.path.join(args.output_dir, f"{base_name}_noise_defenses_breakdown.png")
        fig_def, axs = plt.subplots(3, 3, figsize=(18, 15), dpi=150)

        # 第 1 橫列: 空間平滑防禦
        axs[0, 0].imshow(y_channel, cmap='gray')
        axs[0, 0].set_title("Raw Luminance Y", fontsize=12, fontweight='bold')
        axs[0, 0].axis('off')

        axs[0, 1].imshow(y_smooth_np, cmap='gray')
        axs[0, 1].set_title(f"1. Gaussian Pre-filtered Y (sigma={args.sigma})", fontsize=12, fontweight='bold')
        axs[0, 1].axis('off')

        diff_smooth = np.abs(y_channel - y_smooth_np)
        axs[0, 2].imshow(diff_smooth, cmap='hot')
        axs[0, 2].set_title("Filtered High-freq Sensor Noise", fontsize=12, fontweight='bold')
        axs[0, 2].axis('off')

        # 第 2 橫列: Sobel & Chroma 門檻抑制 (Coring)
        axs[1, 0].imshow(s_sobel_raw_np, cmap='viridis')
        axs[1, 0].set_title("Raw Sobel Gradient (Before Coring)", fontsize=12, fontweight='bold')
        axs[1, 0].axis('off')

        axs[1, 1].imshow(s_sobel_np, cmap='viridis', vmin=0.0, vmax=1.0)
        axs[1, 1].set_title(f"3. Sobel After Coring ({args.coring_ratio*100:.0f}%) & 99% Clamp", fontsize=12, fontweight='bold')
        axs[1, 1].axis('off')

        sobel_suppressed = (s_sobel_raw_np > 0) & (s_sobel_np == 0)
        axs[1, 2].imshow(sobel_suppressed, cmap='Reds')
        axs[1, 2].set_title("Suppressed Flat Noise Pixels (Strictly 0)", fontsize=12, fontweight='bold', color='crimson')
        axs[1, 2].axis('off')

        # 第 3 橫列: 頻域帶通與極值截斷
        axs[2, 0].imshow(s_dct_raw_np, cmap='plasma')
        axs[2, 0].set_title("2. Band-pass DCT Energy (DC & Extreme High Zeroed)", fontsize=12, fontweight='bold')
        axs[2, 0].axis('off')

        axs[2, 1].imshow(s_dct_np, cmap='plasma', vmin=0.0, vmax=1.0)
        axs[2, 1].set_title("4. DCT After 99% Quantile Clamping & Norm", fontsize=12, fontweight='bold')
        axs[2, 1].axis('off')

        axs[2, 2].imshow(s_chroma_np, cmap='turbo', vmin=0.0, vmax=1.0)
        axs[2, 2].set_title("Chroma Gradient (Iso-luminant Color Boundaries)", fontsize=12, fontweight='bold')
        axs[2, 2].axis('off')

        plt.suptitle(f"4-Layer Noise Defense Verification - [{base_name}]", fontsize=16, fontweight='heavy', y=0.98)
        plt.tight_layout()
        plt.savefig(defenses_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"🛡️ 四大抗噪防禦分解圖已儲存至: {defenses_path}")

    # 7. 個別高解析度圖檔匯出 (--save_individual)
    if args.save_individual:
        ind_dir = os.path.join(args.output_dir, f"{base_name}_individual")
        os.makedirs(ind_dir, exist_ok=True)
        save_heatmap(s_sobel_np, os.path.join(ind_dir, "01_s_sobel.png"), colormap=args.colormap, vmin=0.0, vmax=1.0)
        save_heatmap(s_dct_np, os.path.join(ind_dir, "02_s_dct_midhigh.png"), colormap=args.colormap, vmin=0.0, vmax=1.0)
        save_heatmap(s_chroma_np, os.path.join(ind_dir, "03_s_chroma.png"), colormap=args.colormap, vmin=0.0, vmax=1.0)
        save_heatmap(s_tex_np, os.path.join(ind_dir, "04_s_tex_combined.png"), colormap=args.colormap, vmin=0.0, vmax=float(s_tex_np.max()))
        # 原圖與疊加圖
        pil_img.save(os.path.join(ind_dir, "00_original_rgb.png"))
        Image.fromarray((overlay * 255).astype(np.uint8)).save(os.path.join(ind_dir, "05_overlay.png"))
        print(f"📁 各分數獨立高畫質熱圖已儲存至: {ind_dir}/")

    print(f"\n🎉 視覺化處理圓滿完成！\n")


if __name__ == "__main__":
    main()
