"""
RGB 影像紋理與色彩熵分數 (S_tex) 計算模組
包含三大子項特徵與四大抗噪防禦機制：
- 空間域像素級邊界: S_Sobel (亮度一階微分)
- 頻域帶通材質紋理: S_DCT_MidHigh (8x8 2D-DCT 帶通濾波)
- 色度空間跳變: S_ColorVar / ChromaGrad (Cb/Cr 跨通道色彩對比)
- 四大抗噪機制: 高斯預平滑、DCT 帶通截斷、雜訊門檻抑制 (Noise Coring)、全域 99% 分位數截斷
- 影像邊緣處理: 邊界複製/反射填充 (Replicate/Reflect Padding)，無假邊緣高梯度
"""

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# 8x8 DCT Zigzag 掃描索引表 (0 ~ 63)
ZIGZAG_8x8 = [
    [ 0,  1,  5,  6, 14, 15, 27, 28],
    [ 2,  4,  7, 13, 16, 26, 29, 42],
    [ 3,  8, 12, 17, 25, 30, 41, 43],
    [ 9, 11, 18, 24, 31, 40, 44, 53],
    [10, 19, 23, 32, 39, 45, 52, 54],
    [20, 22, 33, 38, 46, 51, 55, 60],
    [21, 34, 37, 47, 50, 56, 59, 61],
    [35, 36, 48, 49, 57, 58, 62, 63]
]


def rgb_to_ycbcr(rgb: torch.Tensor) -> torch.Tensor:
    """
    可微分 RGB 轉 YCbCr 色彩空間轉換 (符合 ITU-R BT.601 標準)。
    Args:
        rgb: [..., 3, H, W] 張量，數值範圍建議在 [0, 1] 或 [0, 255] (若 > 1 則自動正規化至 [0, 1])。
    Returns:
        ycbcr: [..., 3, H, W] 張量，通道 0 為亮度 Y，通道 1 為 Cb，通道 2 為 Cr。
    """
    orig_shape = rgb.shape
    if rgb.shape[-3] != 3:
        raise ValueError(f"預期輸入張量倒數第三維為 3 (RGB)，但收到 shape: {orig_shape}")

    # 若輸入為 uint8 或數值 > 1.0，自動轉為 float32 並縮放至 [0, 1]
    if rgb.dtype == torch.uint8 or rgb.max() > 1.0 + 1e-4:
        rgb = rgb.float() / 255.0
    else:
        rgb = rgb.float()

    r = rgb[..., 0:1, :, :]
    g = rgb[..., 1:2, :, :]
    b = rgb[..., 2:3, :, :]

    # ITU-R BT.601 轉換公式
    y  =  0.299000 * r + 0.587000 * g + 0.114000 * b
    cb = -0.168736 * r - 0.331264 * g + 0.500000 * b + 0.5
    cr =  0.500000 * r - 0.418688 * g - 0.081312 * b + 0.5

    return torch.cat([y, cb, cr], dim=-3)


def get_gaussian_kernel2d(kernel_size: int = 3, sigma: float = 0.8, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    生成 2D 歸一化高斯卷積核 [1, 1, K, K]。
    """
    radius = kernel_size // 2
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    gauss1d = torch.exp(- (x ** 2) / (2.0 * sigma * sigma))
    gauss2d = torch.outer(gauss1d, gauss1d)
    gauss2d = gauss2d / gauss2d.sum()
    return gauss2d.view(1, 1, kernel_size, kernel_size)


def gaussian_blur2d(x: torch.Tensor, kernel_size: int = 3, sigma: float = 0.8, padding_mode: str = 'replicate') -> torch.Tensor:
    """
    空間域輕量高斯預平滑 (抗噪防禦第 1 層)。
    採用邊界複製/反射填充，徹底消除影像邊框因零填充造成的假邊界高梯度。
    Args:
        x: [B, C, H, W] 或 [C, H, W]
        kernel_size: 核大小 (預設 3x3)
        sigma: 高斯標準差 (預設 0.8)
        padding_mode: 'replicate' 或 'reflect'
    Returns:
        平滑後的張量，尺寸與輸入相同。
    """
    is_3d = (x.ndim == 3)
    if is_3d:
        x = x.unsqueeze(0)
    
    B, C, H, W = x.shape
    kernel = get_gaussian_kernel2d(kernel_size, sigma, device=x.device, dtype=x.dtype)
    weight = kernel.repeat(C, 1, 1, 1)

    pad = kernel_size // 2
    x_padded = F.pad(x, (pad, pad, pad, pad), mode=padding_mode)
    out = F.conv2d(x_padded, weight, groups=C)

    return out.squeeze(0) if is_3d else out


def sobel_gradient(y: torch.Tensor, padding_mode: str = 'replicate', normalize_kernel: bool = True) -> torch.Tensor:
    """
    像素級精準邊界 (S_Sobel) - 空間域一階微分。
    在亮度通道 Y 上利用 3x3 Sobel 算子卷積計算水平與垂直梯度幅值。
    使用邊界複製填充，防止邊緣高斯擴散與邊框假特徵。
    Args:
        y: [B, 1, H, W] 亮度通道 (值域 [0, 1])
        padding_mode: 邊界填充方式 (預設 'replicate')
        normalize_kernel: 是否對 Sobel 核做 1/4 正規化 (預設 True)
    Returns:
        S_Sobel: [B, 1, H, W] 梯度幅值
    """
    is_3d = (y.ndim == 3)
    if is_3d:
        y = y.unsqueeze(0)
    
    B, C, H, W = y.shape
    assert C == 1, f"Sobel 預期單通道輸入，但收到通道數: {C}"

    scale = 0.25 if normalize_kernel else 1.0
    kx = torch.tensor([[-1.0, 0.0, 1.0],
                       [-2.0, 0.0, 2.0],
                       [-1.0, 0.0, 1.0]], device=y.device, dtype=y.dtype).view(1, 1, 3, 3) * scale
    ky = torch.tensor([[-1.0, -2.0, -1.0],
                       [ 0.0,  0.0,  0.0],
                       [ 1.0,  2.0,  1.0]], device=y.device, dtype=y.dtype).view(1, 1, 3, 3) * scale

    y_padded = F.pad(y, (1, 1, 1, 1), mode=padding_mode)
    grad_x = F.conv2d(y_padded, kx)
    grad_y = F.conv2d(y_padded, ky)

    # 加上微小 epsilon = 1e-8 確保 sqrt 在 0 處數值穩定可微分
    s_sobel = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

    return s_sobel.squeeze(0) if is_3d else s_sobel


def get_dct_midhigh_filters(
    exclude_dc: bool = True,
    num_high_to_zero: int = 12,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成 8x8 2D-DCT 帶通濾波器組 (一步到位完成頻域帶通截斷)。
    直接跳過 DC (0,0) 與右下角最高頻的 num_high_to_zero 個係數，
    卷積核只包含中高頻帶通基底，無須分開計算全頻譜與遮罩。
    Args:
        exclude_dc: 是否排除直流分量 (0,0)
        num_high_to_zero: 排除最高頻係數個數 (預設 12，範圍 10~15)
        device: 運算裝置
        dtype: 數據型態
    Returns:
        filters: [K, 1, 8, 8] 中高頻帶通 DCT 濾波器組
        selected_zigzag_indices: [K] 被選中的 Zigzag 頻率索引清單
    """
    N = 8
    max_high_idx = 64 - max(0, num_high_to_zero)
    min_idx = 1 if exclude_dc else 0

    # 建立全 64 頻率基底並依 Zigzag 挑選有效頻率
    selected_filters = []
    selected_indices = []

    for u in range(N):
        for v in range(N):
            zz_idx = ZIGZAG_8x8[u][v]
            if min_idx <= zz_idx < max_high_idx:
                alpha_u = 1.0 / math.sqrt(N) if u == 0 else math.sqrt(2.0 / N)
                alpha_v = 1.0 / math.sqrt(N) if v == 0 else math.sqrt(2.0 / N)
                
                # 8x8 空間網格
                x_coord = torch.arange(N, device=device, dtype=dtype).view(N, 1)
                y_coord = torch.arange(N, device=device, dtype=dtype).view(1, N)

                basis_u = torch.cos((2.0 * x_coord + 1.0) * (u * math.pi / (2.0 * N)))
                basis_v = torch.cos((2.0 * y_coord + 1.0) * (v * math.pi / (2.0 * N)))
                basis_2d = alpha_u * alpha_v * (basis_u * basis_v)
                
                selected_filters.append(basis_2d.unsqueeze(0).unsqueeze(0))
                selected_indices.append(zz_idx)

    filters = torch.cat(selected_filters, dim=0) # [K, 1, 8, 8]
    indices = torch.tensor(selected_indices, device=device, dtype=torch.long)
    return filters, indices


def block_dct8x8_midhigh(
    y: torch.Tensor,
    dct_stride: int = 1,
    exclude_dc: bool = True,
    num_high_to_zero: int = 12,
    freq_weights: Optional[torch.Tensor] = None,
    padding_mode: str = 'replicate'
) -> torch.Tensor:
    """
    去除噪聲的中高頻材質紋理 (S_DCT_MidHigh) - 頻域帶通 AC 能量總和。
    透過剪枝後的中高頻 8x8 濾波器組直接卷積完成帶通濾波。
    支援：
    - dct_stride=1 (滑動窗模式，全解析度 HxW 無馬賽克階梯感，細緻度高)
    - dct_stride=8 (傳統 Block 模式，運算極快 < 0.5ms，Nearest-Neighbor 上採樣)
    Args:
        y: [B, 1, H, W] 亮度通道
        dct_stride: 1 或 8
        exclude_dc: 是否排除直流 (0,0)
        num_high_to_zero: 砍掉右下角最高頻個數 (預設 12)
        freq_weights: 可選的個別頻率權重 [K]，若為 None 則均勻平方和相加
        padding_mode: 邊界填充方式 (預設 'replicate')
    Returns:
        S_DCT_MidHigh: [B, 1, H, W] 帶通能量圖 (非負)
    """
    is_3d = (y.ndim == 3)
    if is_3d:
        y = y.unsqueeze(0)

    B, C, H, W = y.shape
    assert C == 1, f"DCT 預期單通道輸入，但收到通道數: {C}"

    filters, _ = get_dct_midhigh_filters(
        exclude_dc=exclude_dc,
        num_high_to_zero=num_high_to_zero,
        device=y.device,
        dtype=y.dtype
    )
    K = filters.shape[0]

    if dct_stride == 1:
        # 滑動窗卷積模式:
        # 為了使 8x8 卷積後長寬保持完全相同的 (H, W)，填充 left=3, right=4, top=3, bottom=4 (3+4=7=8-1)
        y_padded = F.pad(y, (3, 4, 3, 4), mode=padding_mode)
        coeffs = F.conv2d(y_padded, filters, stride=1) # [B, K, H, W]
    elif dct_stride == 8:
        # 傳統不重疊 Block 模式:
        # 補齊長寬至 8 的整數倍
        pad_w = (8 - (W % 8)) % 8
        pad_h = (8 - (H % 8)) % 8
        y_padded = F.pad(y, (0, pad_w, 0, pad_h), mode=padding_mode)
        coeffs = F.conv2d(y_padded, filters, stride=8) # [B, K, H_b, W_b]
    else:
        raise ValueError(f"dct_stride 建議為 1 (滑動窗) 或 8 (Block)，收到: {dct_stride}")

    # 計算帶通能量 (平方和)
    if freq_weights is not None:
        assert freq_weights.shape[0] == K, f"freq_weights 長度預期為 {K}，但收到: {freq_weights.shape[0]}"
        w = freq_weights.view(1, K, 1, 1).to(device=coeffs.device, dtype=coeffs.dtype)
        energy = torch.sum(w * (coeffs ** 2), dim=1, keepdim=True)
    else:
        energy = torch.sum(coeffs ** 2, dim=1, keepdim=True)

    if dct_stride == 8:
        # 將 Block 能量擴展回原始全解析度，並裁切回 [H, W]
        energy_up = F.interpolate(energy, size=(H + pad_h, W + pad_w), mode='nearest')
        energy = energy_up[:, :, :H, :W]

    return energy.squeeze(0) if is_3d else energy


def chroma_gradient(cb: torch.Tensor, cr: torch.Tensor, padding_mode: str = 'replicate', normalize_kernel: bool = True) -> torch.Tensor:
    """
    跨通道色彩對比 (S_ColorVar / ChromaGrad) - 色度空間梯度。
    衡量純色相 (Hue) 與飽和度 (Saturation) 的變化速率，解決等亮度色相跳變盲點。
    S_ColorVar = sqrt(||∇Cb||^2 + ||∇Cr||^2)
    Args:
        cb: [B, 1, H, W]
        cr: [B, 1, H, W]
        padding_mode: 邊界填充方式 (預設 'replicate')
        normalize_kernel: 是否對 Sobel 算子做 1/4 縮放
    Returns:
        S_ColorVar: [B, 1, H, W]
    """
    is_3d = (cb.ndim == 3)
    if is_3d:
        cb = cb.unsqueeze(0)
        cr = cr.unsqueeze(0)

    scale = 0.25 if normalize_kernel else 1.0
    kx = torch.tensor([[-1.0, 0.0, 1.0],
                       [-2.0, 0.0, 2.0],
                       [-1.0, 0.0, 1.0]], device=cb.device, dtype=cb.dtype).view(1, 1, 3, 3) * scale
    ky = torch.tensor([[-1.0, -2.0, -1.0],
                       [ 0.0,  0.0,  0.0],
                       [ 1.0,  2.0,  1.0]], device=cb.device, dtype=cb.dtype).view(1, 1, 3, 3) * scale

    cb_pad = F.pad(cb, (1, 1, 1, 1), mode=padding_mode)
    cr_pad = F.pad(cr, (1, 1, 1, 1), mode=padding_mode)

    grad_cb_x = F.conv2d(cb_pad, kx)
    grad_cb_y = F.conv2d(cb_pad, ky)
    grad_cr_x = F.conv2d(cr_pad, kx)
    grad_cr_y = F.conv2d(cr_pad, ky)

    chroma_sq = (grad_cb_x ** 2) + (grad_cb_y ** 2) + (grad_cr_x ** 2) + (grad_cr_y ** 2)
    s_chroma = torch.sqrt(chroma_sq + 1e-8)

    return s_chroma.squeeze(0) if is_3d else s_chroma


def noise_coring(x: torch.Tensor, coring_ratio: float = 0.08, min_noise_floor: float = 0.005) -> torch.Tensor:
    """
    雜訊門檻抑制 (Noise Coring / Soft Thresholding) - 抗噪防禦第 3 層。
    低於門檻 tau 的微小雜訊直接歸零，保證平坦純色區域嚴格為 0。
    tau = max(coring_ratio * mean(x), min_noise_floor)
    formula: max(0, x - tau)
    """
    if coring_ratio <= 0.0 and min_noise_floor <= 0.0:
        return x
    
    # 逐圖或全域計算平均值
    if x.ndim >= 3:
        mean_val = torch.mean(x, dim=(-2, -1), keepdim=True)
    else:
        mean_val = torch.mean(x)
        
    tau = coring_ratio * mean_val
    if min_noise_floor > 0.0:
        tau = torch.clamp(tau, min=min_noise_floor)
    return F.relu(x - tau)


def robust_quantile_clamp(x: torch.Tensor, quantile: float = 0.99, normalize: bool = True, min_scale: float = 0.02) -> torch.Tensor:
    """
    全域極值截斷 (Robust 99% Quantile Clamping) - 抗噪防禦第 4 層。
    使用 99% 分位數截斷，防止鏡面高光/過曝反光點主導採樣分佈。
    Args:
        x: [B, 1, H, W] 或任意維度張量
        quantile: 截斷分位數 (預設 0.99)
        normalize: 是否將截斷後的張量正規化至 [0, 1]
        min_scale: 最小尺度門檻，防止純平坦無訊號通道將微弱噪訊放大為 1.0
    Returns:
        截斷且可選歸一化的特徵圖
    """
    if quantile >= 1.0 or quantile <= 0.0:
        if normalize:
            m = torch.clamp(x.max(), min=min_scale)
            return x / m
        return x

    # 針對大影像抽樣或扁平化計算分位數
    flat_x = x.detach().flatten()
    if flat_x.numel() > 1_000_000:
        step = flat_x.numel() // 500_000
        q_val = torch.quantile(flat_x[::step], quantile)
    else:
        q_val = torch.quantile(flat_x, quantile)

    q_val = torch.clamp(q_val, min=min_scale)
    clamped = torch.clamp(x, min=0.0, max=float(q_val))

    if normalize:
        clamped = clamped / q_val

    return clamped


class RGBTextureScorer(nn.Module):
    """
    RGB 影像紋理與色彩熵分數 (S_tex) 計算器。
    
    輸出：
        - s_sobel: 像素級精準邊界 [B, 1, H, W]
        - s_dct: 去除噪聲的中高頻材質能量 [B, 1, H, W]
        - s_chroma: 跨通道色彩對比 [B, 1, H, W]
        - s_tex: 三者線性加權組合總分 [B, 1, H, W]
    """
    def __init__(
        self,
        gaussian_sigma: float = 0.8,
        gaussian_ksize: int = 3,
        dct_stride: int = 1,
        num_high_to_zero: int = 12,
        coring_ratio: float = 0.08,
        min_noise_floor: float = 0.005,
        quantile: float = 0.99,
        min_scale: float = 0.02,
        border_margin: int = 0,
        default_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ):
        super().__init__()
        self.gaussian_sigma = gaussian_sigma
        self.gaussian_ksize = gaussian_ksize
        self.dct_stride = dct_stride
        self.num_high_to_zero = num_high_to_zero
        self.coring_ratio = coring_ratio
        self.min_noise_floor = min_noise_floor
        self.quantile = quantile
        self.min_scale = min_scale
        self.border_margin = border_margin
        self.default_weights = default_weights

    def forward(
        self,
        rgb: torch.Tensor,
        w_grad: Optional[float] = None,
        w_band: Optional[float] = None,
        w_color: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向傳導計算各特徵與融合分數。
        Args:
            rgb: [B, 3, H, W] 或 [3, H, W]，值域 [0, 1] 或 [0, 255]
            w_grad, w_band, w_color: 融合權重 (若為 None 則使用 default_weights)
        Returns:
            dict 包含:
                's_sobel': 像素級邊界分數 [B, 1, H, W] (經抗噪與正規化)
                's_dct': 中高頻材質紋理分數 [B, 1, H, W]
                's_chroma': 色度空間跳變分數 [B, 1, H, W]
                's_tex': 線性加權總分 [B, 1, H, W]
                'ycbcr': 轉換後的 YCbCr 影像 [B, 3, H, W]
                'y_smooth': 預平滑後的亮度圖 [B, 1, H, W]
        """
        is_3d = (rgb.ndim == 3)
        if is_3d:
            rgb = rgb.unsqueeze(0)

        wg = self.default_weights[0] if w_grad is None else w_grad
        wb = self.default_weights[1] if w_band is None else w_band
        wc = self.default_weights[2] if w_color is None else w_color

        # 1. 轉為 YCbCr 色彩空間
        ycbcr = rgb_to_ycbcr(rgb)
        y = ycbcr[:, 0:1]
        cb = ycbcr[:, 1:2]
        cr = ycbcr[:, 2:3]

        # 2. 抗噪第 1 層：空間域輕量預平滑
        y_smooth = gaussian_blur2d(y, kernel_size=self.gaussian_ksize, sigma=self.gaussian_sigma, padding_mode='replicate')
        cb_smooth = gaussian_blur2d(cb, kernel_size=self.gaussian_ksize, sigma=self.gaussian_sigma, padding_mode='replicate')
        cr_smooth = gaussian_blur2d(cr, kernel_size=self.gaussian_ksize, sigma=self.gaussian_sigma, padding_mode='replicate')

        # 3. 三大子項特徵計算
        # 子項 1: 空間域 Sobel (作用於平滑後的 Y)
        s_sobel_raw = sobel_gradient(y_smooth, padding_mode='replicate')

        # 子項 2: 頻域 8x8 帶通 DCT (直接透過剪枝濾波器組融合完成，作用於 Y)
        s_dct_raw = block_dct8x8_midhigh(
            y,
            dct_stride=self.dct_stride,
            exclude_dc=True,
            num_high_to_zero=self.num_high_to_zero,
            padding_mode='replicate'
        )

        # 子項 3: 色度空間梯度 (作用於平滑後的 Cb/Cr)
        s_chroma_raw = chroma_gradient(cb_smooth, cr_smooth, padding_mode='replicate')

        # 4. 抗噪第 3 層：雜訊門檻抑制 (Noise Coring)
        s_sobel_cored = noise_coring(s_sobel_raw, coring_ratio=self.coring_ratio, min_noise_floor=self.min_noise_floor)
        s_dct_cored = noise_coring(s_dct_raw, coring_ratio=self.coring_ratio, min_noise_floor=self.min_noise_floor)
        s_chroma_cored = noise_coring(s_chroma_raw, coring_ratio=self.coring_ratio, min_noise_floor=self.min_noise_floor)

        # 5. 抗噪第 4 層：全域 99% 極值截斷與正規化至 [0, 1]
        s_sobel_clean = robust_quantile_clamp(s_sobel_cored, quantile=self.quantile, normalize=True, min_scale=self.min_scale)
        s_dct_clean = robust_quantile_clamp(s_dct_cored, quantile=self.quantile, normalize=True, min_scale=self.min_scale)
        s_chroma_clean = robust_quantile_clamp(s_chroma_cored, quantile=self.quantile, normalize=True, min_scale=self.min_scale)

        # 可選邊界抑制 (Border Margin Suppression)
        if self.border_margin > 0:
            m = self.border_margin
            for s in [s_sobel_clean, s_dct_clean, s_chroma_clean]:
                s[:, :, :m, :] = 0.0
                s[:, :, -m:, :] = 0.0
                s[:, :, :, :m] = 0.0
                s[:, :, :, -m:] = 0.0

        # 6. 線性加權融合
        s_tex = wg * s_sobel_clean + wb * s_dct_clean + wc * s_chroma_clean

        if is_3d:
            return {
                's_sobel': s_sobel_clean.squeeze(0),
                's_dct': s_dct_clean.squeeze(0),
                's_chroma': s_chroma_clean.squeeze(0),
                's_tex': s_tex.squeeze(0),
                'ycbcr': ycbcr.squeeze(0),
                'y_smooth': y_smooth.squeeze(0),
                # 保留未過濾的原生圖供視覺化對比
                's_sobel_raw': s_sobel_raw.squeeze(0),
                's_dct_raw': s_dct_raw.squeeze(0),
                's_chroma_raw': s_chroma_raw.squeeze(0)
            }

        return {
            's_sobel': s_sobel_clean,
            's_dct': s_dct_clean,
            's_chroma': s_chroma_clean,
            's_tex': s_tex,
            'ycbcr': ycbcr,
            'y_smooth': y_smooth,
            's_sobel_raw': s_sobel_raw,
            's_dct_raw': s_dct_raw,
            's_chroma_raw': s_chroma_raw
        }


# ==============================================================================
# 第 3 節：DA3 幾何深度與曲率分數 (S_depth) 模組
# 全面對齊業界標準函式庫：
# - 法向量：OpenCV LINEMOD (Stefan Holzer et al., BMVC 2012) 單側最小差分 (Min-Gradient)
# - 曲率：PCL (Point Cloud Library) max_depth_change_factor 深度斷差門檻過濾
# - 掠射角濾波：COLMAP / 2DGS (SIGGRAPH 2024) 80度餘弦平滑截斷 (Anti-Streaking)
# ==============================================================================

def log_depth_gradient(
    depth: torch.Tensor,
    quantile: float = 0.98,
    normalize: bool = True,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    計算對數深度的一階空間梯度幅值 S_depth_grad。
    原理: ∇(ln D) = ∇D / D，具備尺度不變性 (Scale-Invariant)，專職捕捉物體幾何輪廓與遮擋斷差 (Silhouettes)。
    邊緣處理: 採用 Replicate Padding，消除影像邊界假跳變。
    極值防護: 採用動態分位數 (預設 98%) 截斷，排除飛點與極值噪聲。
    
    Args:
        depth: [..., 1, H, W] 或 [..., H, W] 深度圖 (公尺單位，需 > 0)。
        quantile: 動態分位數截斷上限 (0.0 ~ 1.0)。
        normalize: 是否歸一化至 [0, 1]。
        eps: 數值穩定 epsilon。
    Returns:
        s_depth_grad: [..., 1, H, W] 梯度幅值圖。
    """
    orig_ndim = depth.ndim
    if orig_ndim == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)
    elif orig_ndim == 3:
        if depth.shape[0] == 1:
            depth = depth.unsqueeze(0)
        else:
            depth = depth.unsqueeze(1)

    device = depth.device
    dtype = depth.dtype

    # 對數深度轉換
    depth_clean = torch.clamp(depth, min=eps)
    log_d = torch.log(depth_clean)

    # 3x3 Sobel 卷積核 (歸一化 1/8)
    sobel_x = torch.tensor([
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0]
    ], device=device, dtype=dtype).view(1, 1, 3, 3) / 8.0

    sobel_y = torch.tensor([
        [-1.0, -2.0, -1.0],
        [ 0.0,  0.0,  0.0],
        [ 1.0,  2.0,  1.0]
    ], device=device, dtype=dtype).view(1, 1, 3, 3) / 8.0

    # 邊界複製填充
    log_d_padded = F.pad(log_d, (1, 1, 1, 1), mode='replicate')
    gx = F.conv2d(log_d_padded, sobel_x)
    gy = F.conv2d(log_d_padded, sobel_y)

    grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)

    # 98% 動態分位數截斷
    if quantile is not None and 0.0 < quantile < 1.0:
        flat = grad_mag.detach().flatten()
        if flat.numel() > 100:
            q_val = torch.quantile(flat, quantile)
            if q_val > 1e-6:
                grad_mag = torch.clamp(grad_mag, max=q_val)
                if normalize:
                    grad_mag = grad_mag / q_val

    if normalize and (quantile is None or quantile <= 0.0):
        max_v = grad_mag.max()
        if max_v > 1e-6:
            grad_mag = grad_mag / max_v

    return grad_mag


def depth_to_surface_normals(
    depth: torch.Tensor,
    intrinsics: Optional[torch.Tensor] = None,
    method: str = "min_gradient",
    eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    從深度圖反投影求得三維表面單位法向量 n(u, v) 與 3D 空間點 P(u, v)。
    支援 OpenCV LINEMOD 標準之單側最小差分法 (Min-Gradient):
    在每個像素點分別計算左右差分 (ΔL, ΔR) 與上下差分 (ΔT, ΔB)，選取深度變化小的一側計算切線，
    從源頭杜絕跨越前後景遮擋邊界的連線，徹底避免生成掠射角假懸崖 (Phantom Wall)。

    Args:
        depth: [..., 1, H, W] 或 [..., H, W] 深度圖 (公尺)。
        intrinsics: 可選相機內參矩陣 [3, 3] 或 [..., 3, 3] (fx, fy, cx, cy)。
                    若為 None 則以標準針孔視角自適應推算 (fx=fy=max(H,W), cx=W/2, cy=H/2)。
        method: 'min_gradient' (OpenCV LINEMOD 標準，強烈推薦) 或 'central' (經典中心差分)。
        eps: 數值穩定 epsilon。
    Returns:
        normals: [..., 3, H, W] 單位法向量，統一朝向相機側 (標準相機座標: X-右, Y-下, Z-前)。
        points_3d: [..., 3, H, W] 反投影 3D 空間點。
    """
    orig_ndim = depth.ndim
    if orig_ndim == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)
    elif orig_ndim == 3:
        if depth.shape[0] == 1:
            depth = depth.unsqueeze(0)
        else:
            depth = depth.unsqueeze(1)

    *batch_dims, _, H, W = depth.shape
    device = depth.device
    dtype = depth.dtype

    # 建立像素座標網格 (u: 0~W-1, v: 0~H-1)
    v_grid, u_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )
    u_grid = u_grid.expand(*batch_dims, 1, H, W)
    v_grid = v_grid.expand(*batch_dims, 1, H, W)

    # 內參解析
    if intrinsics is not None:
        if intrinsics.ndim == 2:
            fx = intrinsics[0, 0]
            fy = intrinsics[1, 1]
            cx = intrinsics[0, 2]
            cy = intrinsics[1, 2]
        else:
            fx = intrinsics[..., 0:1, 0:1].unsqueeze(-1)
            fy = intrinsics[..., 1:2, 1:2].unsqueeze(-1)
            cx = intrinsics[..., 0:1, 2:3].unsqueeze(-1)
            cy = intrinsics[..., 1:2, 2:3].unsqueeze(-1)
    else:
        focal = float(max(H, W))
        fx = fy = focal
        cx = W / 2.0
        cy = H / 2.0

    depth_clean = torch.clamp(depth, min=eps)
    x_3d = (u_grid - cx) * depth_clean / fx
    y_3d = (v_grid - cy) * depth_clean / fy
    z_3d = depth_clean

    # 空間 3D 點張量 [..., 3, H, W]
    points_3d = torch.cat([x_3d, y_3d, z_3d], dim=-3)

    if method == "min_gradient":
        # ======================================================================
        # OpenCV LINEMOD 標準實做: 單側最小差分 (Min-Gradient)
        # ======================================================================
        p_pad = F.pad(points_3d, (1, 1, 1, 1), mode='replicate')

        # 水平切向量候選: 左差分 (中心 - 左) vs 右差分 (右 - 中心)
        delta_l = points_3d - p_pad[..., 1:-1, :-2]
        delta_r = p_pad[..., 1:-1, 2:] - points_3d

        dz_l = torch.abs(delta_l[..., 2:3, :, :])
        dz_r = torch.abs(delta_r[..., 2:3, :, :])

        use_left = (dz_l < dz_r).clone()
        # 邊界保全: 左邊界 col 0 無左鄰居，強制用右差分；右邊界 col W-1 無右鄰居，強制用左差分
        use_left[..., :, :, 0:1] = False
        use_left[..., :, :, -1:] = True
        dx_3d = torch.where(use_left.expand_as(delta_l), delta_l, delta_r)

        # 垂直切向量候選: 上差分 (中心 - 上) vs 下差分 (下 - 中心)
        delta_t = points_3d - p_pad[..., :-2, 1:-1]
        delta_b = p_pad[..., 2:, 1:-1] - points_3d

        dz_t = torch.abs(delta_t[..., 2:3, :, :])
        dz_b = torch.abs(delta_b[..., 2:3, :, :])

        use_top = (dz_t < dz_b).clone()
        # 邊界保全: 上邊界 row 0 無上鄰居，強制用下差分；下邊界 row H-1 無下鄰居，強制用上差分
        use_top[..., :, 0:1, :] = False
        use_top[..., :, -1:, :] = True
        dy_3d = torch.where(use_top.expand_as(delta_t), delta_t, delta_b)

    else:
        # 經典中心差分 (Central Difference)
        p_pad = F.pad(points_3d, (1, 1, 1, 1), mode='replicate')
        dx_3d = (p_pad[..., 1:-1, 2:] - p_pad[..., 1:-1, :-2]) / 2.0
        dy_3d = (p_pad[..., 2:, 1:-1] - p_pad[..., :-2, 1:-1]) / 2.0

    # 切向量外積求法向量: n = dx × dy
    # 相機座標系: X-右, Y-下, Z-前
    # dx × dy = (dx_y*dy_z - dx_z*dy_y, dx_z*dy_x - dx_x*dy_z, dx_x*dy_y - dx_y*dy_x)
    nx = dx_3d[..., 1:2, :, :] * dy_3d[..., 2:3, :, :] - dx_3d[..., 2:3, :, :] * dy_3d[..., 1:2, :, :]
    ny = dx_3d[..., 2:3, :, :] * dy_3d[..., 0:1, :, :] - dx_3d[..., 0:1, :, :] * dy_3d[..., 2:3, :, :]
    nz = dx_3d[..., 0:1, :, :] * dy_3d[..., 1:2, :, :] - dx_3d[..., 1:2, :, :] * dy_3d[..., 0:1, :, :]

    normal_raw = torch.cat([nx, ny, nz], dim=-3)

    # 統一法向量朝向觀察者側 (在相機座標系中，面對相機的平面其 nz 應為負值，即朝向原點；
    # 為了讓 RGB 視覺化呈標準淺藍色 [0.5, 0.5, 1.0]，我們將朝向相機方向規定為 nz > 0)
    flip_mask = (normal_raw[..., 2:3, :, :] < 0.0)
    normal_oriented = torch.where(flip_mask.expand_as(normal_raw), -normal_raw, normal_raw)

    # 單位長度歸一化
    norm_len = torch.sqrt(torch.sum(normal_oriented ** 2, dim=-3, keepdim=True) + 1e-12)
    normals = normal_oriented / norm_len

    return normals, points_3d


def depth_gated_surface_curvature(
    normals: torch.Tensor,
    depth: torch.Tensor,
    kernel_size: int = 5,
    max_depth_change_factor: float = 0.05,
    soft_gating: bool = True,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    對齊 PCL (Point Cloud Library) max_depth_change_factor 標準之表面法向曲率計算。
    原理:
        在 k x k 滑動窗口內計算平均法向量 n_avg。
        若相鄰像素與中心像素的相對深度跳變 |D_neighbor - D_center| / D_center > max_depth_change_factor，
        判定為跨物體表面斷差，權重予以剔除或高斯衰減。
        局部曲率離散度: S_curv = 1.0 - ||n_avg||
        - 完全平坦表面: S_curv = 0.0 (即便在物體邊緣，背景點被剔除後，窗口內同表面法向一致，仍為 0，杜絕假光暈)
        - 連續彎曲表面 (如球面、人臉): S_curv > 0.0

    Args:
        normals: [..., 3, H, W] 單位法向量。
        depth: [..., 1, H, W] 深度圖 (公尺)。
        kernel_size: 滑動窗口大小 (奇數，預設 5)。
        max_depth_change_factor: PCL 深度跳變容許比率 (預設 0.05 即 5%)。
        soft_gating: 是否採用可微分的軟性雙邊深度加權 (預設 True)。
        eps: 數值穩定 epsilon。
    Returns:
        s_curv: [..., 1, H, W] 曲率分數圖 [0, 1]。
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2

    *batch_dims, C, H, W = normals.shape
    device = normals.device
    dtype = normals.dtype

    # 將 depth 調整為 [B, 1, H, W] 進行 Unfold 操作
    b_size = 1
    for d in batch_dims:
        b_size *= d
    
    depth_flat = depth.view(b_size, 1, H, W)
    depth_clean = torch.clamp(depth_flat, min=eps)
    normals_flat = normals.view(b_size, 3, H, W)

    # 利用 F.unfold 提取局部 k x k 窗口
    depth_pad = F.pad(depth_clean, (pad, pad, pad, pad), mode='replicate')
    # patches: [B, k*k, H*W]
    d_patches = F.unfold(depth_pad, kernel_size=(kernel_size, kernel_size))
    
    # 中心點深度 [B, 1, H*W]
    d_center = depth_clean.view(b_size, 1, H * W)

    # 計算局部相對深度跳變 |D_neighbor - D_center| / D_center
    rel_depth_diff = torch.abs(d_patches - d_center) / (d_center + 1e-7)

    if soft_gating:
        # 可微分雙邊深度權重: exp(- (diff / factor)^2 / 2)
        sigma = max_depth_change_factor / 2.0
        weights = torch.exp(- (rel_depth_diff ** 2) / (2.0 * sigma * sigma)) # [B, k*k, H*W]
    else:
        # 嚴格硬門檻 (Hard Mask)
        weights = (rel_depth_diff < max_depth_change_factor).to(dtype)

    # 空間反向展開法向量 patches
    normals_pad = F.pad(normals_flat, (pad, pad, pad, pad), mode='replicate')
    n_patches = F.unfold(normals_pad, kernel_size=(kernel_size, kernel_size)) # [B, 3 * k*k, H*W]
    n_patches = n_patches.view(b_size, 3, kernel_size * kernel_size, H * W)

    # 深度加權法向量平均: n_avg = Σ(w * n) / Σ(w)
    weights_expanded = weights.unsqueeze(1) # [B, 1, k*k, H*W]
    sum_w = torch.sum(weights_expanded, dim=2) + 1e-8 # [B, 1, H*W]
    sum_wn = torch.sum(weights_expanded * n_patches, dim=2) # [B, 3, H*W]
    n_avg = sum_wn / sum_w

    # 計算平均法向向量長度
    n_avg_len = torch.sqrt(torch.sum(n_avg ** 2, dim=1, keepdim=True) + 1e-12) # [B, 1, H*W]
    s_curv = torch.clamp(1.0 - n_avg_len, min=0.0, max=1.0)
    s_curv = s_curv.view(*batch_dims, 1, H, W)

    return s_curv


def grazing_angle_filter(
    normals: torch.Tensor,
    points_3d: torch.Tensor,
    max_angle_deg: float = 80.0,
    min_angle_deg: float = 70.0,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    對齊 COLMAP 與 3DGS (SIGGRAPH 2024 2DGS) 之掠射角防拉扯濾波器 (Anti-Streaking Filter)。
    原理:
        計算空間三維點視線方向 v = P / ||P|| 與表面法向量 n 的夾角餘弦 cos(θ) = |n · v|。
        當夾角接近 90 度 (掠射角 / 視線邊緣切向) 時，反投影極易在空間中拉扯出長條虛擬網格或空中浮點 (Floaters)。
        本濾波器在 max_angle_deg (預設 80°) 處將權重平滑衰減至 0。

    Args:
        normals: [..., 3, H, W] 單位法向量。
        points_3d: [..., 3, H, W] 空間三維點座標。
        max_angle_deg: 權重完全歸零之最大夾角 (預設 80.0 度)。
        min_angle_deg: 開始進行衰減之起始夾角 (預設 70.0 度)。
    Returns:
        w_grazing: [..., 1, H, W] 掠射角濾波權重 [0.0, 1.0]。
    """
    # 視線方向向量 (由相機原點指向 3D 空間點)
    ray_len = torch.sqrt(torch.sum(points_3d ** 2, dim=-3, keepdim=True) + 1e-12)
    view_ray = points_3d / ray_len

    # 夾角餘弦值 cos(θ) = |n · v|
    cos_theta = torch.abs(torch.sum(normals * view_ray, dim=-3, keepdim=True))

    cos_max = math.cos(math.radians(max_angle_deg)) # 80° ≈ 0.1736
    cos_min = math.cos(math.radians(min_angle_deg)) # 70° ≈ 0.3420

    # 平滑過渡權重: 在 cos_max 以下為 0，在 cos_min 以上為 1
    w_grazing = torch.clamp((cos_theta - cos_max) / (cos_min - cos_max + 1e-7), min=0.0, max=1.0)

    return w_grazing


class DepthGeometryScorer(nn.Module):
    """
    DA3 幾何深度與曲率評分模組 (S_depth)。
    整合:
    1. 對數深度一階斷差梯度 S_depth_grad (外輪廓邊界鎖定)
    2. PCL 深度門檻保護之局部表面曲率 S_curv (連續曲面感知，無邊緣光暈)
    3. 掠射角防拉扯濾波 w_grazing (保護曲面，不干涉真實外輪廓)
    4. 幾何斷差互斥閘門 (Edge Gating): 在強斷差處曲率自動讓位
    5. DA3 信心度 (c_DA3) 硬門檻剔除與軟加權調製
    """
    def __init__(
        self,
        w_d_edge: float = 1.0,
        w_curv: float = 1.0,
        curv_ksize: int = 5,
        max_depth_change_factor: float = 0.05,
        conf_thresh: float = 0.3,
        grazing_max_angle: float = 80.0,
        edge_gate_thresh: float = 0.4
    ):
        super().__init__()
        self.w_d_edge = w_d_edge
        self.w_curv = w_curv
        self.curv_ksize = curv_ksize
        self.max_depth_change_factor = max_depth_change_factor
        self.conf_thresh = conf_thresh
        self.grazing_max_angle = grazing_max_angle
        self.edge_gate_thresh = edge_gate_thresh

    def forward(
        self,
        depth: torch.Tensor,
        conf: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        計算幾何深度分數。
        Args:
            depth: [..., 1, H, W] 深度圖 (公尺)。
            conf: 可選 DA3 信心度圖 [..., 1, H, W]。
            intrinsics: 可選相機內參 [3, 3] 或 [..., 3, 3]。
        Returns:
            Dict 包含:
                's_depth': 最終幾何深度綜合分數
                's_depth_grad': 一階對數深度斷差梯度
                's_curv': 原始 PCL 門檻曲率
                's_curv_gated': 互斥與掠射角防護後之曲率
                'normals': 表面法向量圖 [..., 3, H, W]
                'w_grazing': 掠射角權重圖
                'conf_mask': 信心度遮罩
        """
        # 1. 幾何輪廓斷差梯度
        s_depth_grad = log_depth_gradient(depth, quantile=0.98, normalize=True)

        # 2. OpenCV LINEMOD 標準法向量
        normals, points_3d = depth_to_surface_normals(
            depth, intrinsics=intrinsics, method="min_gradient"
        )

        # 3. PCL 深度門檻曲率
        s_curv = depth_gated_surface_curvature(
            normals=normals,
            depth=depth,
            kernel_size=self.curv_ksize,
            max_depth_change_factor=self.max_depth_change_factor
        )

        # 4. 掠射角防拉扯濾波
        w_grazing = grazing_angle_filter(
            normals=normals,
            points_3d=points_3d,
            max_angle_deg=self.grazing_max_angle
        )

        # 5. 斷差互斥閘門 (Edge Gating): 當深度梯度極大時，曲率自動退場
        edge_mask = torch.sigmoid(20.0 * (s_depth_grad - self.edge_gate_thresh))
        s_curv_gated = s_curv * w_grazing * (1.0 - edge_mask)

        # 6. 綜合加權幾何分數
        geom_raw = self.w_d_edge * s_depth_grad + self.w_curv * s_curv_gated

        # 7. DA3 信心度過濾與調製
        if conf is not None:
            # 確保 conf 維度一致
            if conf.ndim == 2:
                conf = conf.unsqueeze(0).unsqueeze(0)
            elif conf.ndim == 3 and conf.shape[0] != 1:
                conf = conf.unsqueeze(1)

            # 硬門檻遮罩 (剔除無效天空、過曝或窗外飄浮噪點)
            conf_mask = (conf > self.conf_thresh).to(depth.dtype)

            # 軟加權歸一化調製 (以 95% 信心度為飽和上限)
            c_max = torch.quantile(conf.detach().flatten(), 0.95)
            if c_max > 1e-4:
                conf_soft = torch.clamp(conf / c_max, max=1.0)
            else:
                conf_soft = conf

            s_depth = geom_raw * conf_mask * conf_soft
        else:
            conf_mask = torch.ones_like(s_depth_grad)
            s_depth = geom_raw

        return {
            's_depth': s_depth,
            's_depth_grad': s_depth_grad,
            's_curv': s_curv,
            's_curv_gated': s_curv_gated,
            'normals': normals,
            'w_grazing': w_grazing,
            'conf_mask': conf_mask,
            'points_3d': points_3d
        }


class Spatial2DScorer(nn.Module):
    """
    一站式 2D 空間綜合評分器 (S_2D)。
    整合 RGB 紋理分數 (S_tex) 與 DA3 幾何深度分數 (S_depth)：
        S_2D = w_0 + w_tex * S_tex + w_depth * S_depth
    """
    def __init__(
        self,
        w_0: float = 0.0,
        w_tex: float = 1.0,
        w_depth: float = 1.0,
        w_sobel: float = 1.0,
        w_dct: float = 1.0,
        w_chroma: float = 0.5,
        w_d_edge: float = 1.0,
        w_curv: float = 1.0,
        curv_ksize: int = 5,
        max_depth_change_factor: float = 0.05,
        conf_thresh: float = 0.3
    ):
        super().__init__()
        self.w_0 = w_0
        self.w_tex = w_tex
        self.w_depth = w_depth

        self.tex_scorer = RGBTextureScorer(
            default_weights=(w_sobel, w_dct, w_chroma)
        )
        self.geom_scorer = DepthGeometryScorer(
            w_d_edge=w_d_edge,
            w_curv=w_curv,
            curv_ksize=curv_ksize,
            max_depth_change_factor=max_depth_change_factor,
            conf_thresh=conf_thresh
        )

    def forward(
        self,
        rgb: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
        conf: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        計算綜合空間分數。
        """
        tex_res = self.tex_scorer(rgb)
        s_tex = tex_res['s_tex']

        res = {
            's_tex': s_tex,
            's_sobel': tex_res['s_sobel'],
            's_dct': tex_res['s_dct'],
            's_chroma': tex_res['s_chroma']
        }

        if depth is not None:
            # 幾何評分在深度圖的原生解析度下進行計算，避免雙線性插值對深度微分造成摺痕與偽曲率
            geom_res = self.geom_scorer(depth, conf=conf, intrinsics=intrinsics)
            s_depth = geom_res['s_depth']

            target_hw = rgb.shape[-2:]
            # 若深度圖尺寸與 RGB 尺寸不同，平滑插值幾何特徵至 RGB 尺寸
            if s_depth.shape[-2:] != target_hw:
                s_depth = F.interpolate(s_depth, size=target_hw, mode='bilinear', align_corners=False)
                for k in ['s_depth_grad', 's_curv', 's_curv_gated', 'w_grazing', 'conf_mask']:
                    if k in geom_res and geom_res[k] is not None:
                        geom_res[k] = F.interpolate(geom_res[k], size=target_hw, mode='bilinear', align_corners=False)
                if 'normals' in geom_res and geom_res['normals'] is not None:
                    n_up = F.interpolate(geom_res['normals'], size=target_hw, mode='bilinear', align_corners=False)
                    n_len = torch.sqrt(torch.sum(n_up ** 2, dim=-3, keepdim=True) + 1e-12)
                    geom_res['normals'] = n_up / n_len

            s_2d = self.w_0 + self.w_tex * s_tex + self.w_depth * s_depth

            res.update({
                's_depth': s_depth,
                's_depth_grad': geom_res['s_depth_grad'],
                's_curv': geom_res['s_curv'],
                's_curv_gated': geom_res['s_curv_gated'],
                'normals': geom_res['normals'],
                'w_grazing': geom_res['w_grazing'],
                'conf_mask': geom_res['conf_mask'],
                's_2d': s_2d
            })
        else:
            res['s_2d'] = self.w_0 + self.w_tex * s_tex

        return res


class DistributionMatcherLoss(nn.Module):
    """
    可微分分佈對齊與權重超參數優化器 (第 5 節)。
    待優化參數向量:
        theta = Softplus(phi) >= 0
    預測採樣機率:
        P_pred(k; theta) = W_k(theta) / sum(W)
    損失函數:
        L(theta) = D_KL(P_ideal || P_pred(theta)) = - sum(P_ideal * ln(P_pred + eps)) + lambda * ||theta||^2
    """
    def __init__(self, num_features: int = 7, init_val: float = 1.0, l2_reg: float = 1e-4):
        super().__init__()
        # 以非受限變數 phi 參數化，透過 Softplus 保證各特徵權重 strictly >= 0
        self.phi = nn.Parameter(torch.full((num_features,), init_val, dtype=torch.float32))
        self.l2_reg = l2_reg

    @property
    def theta(self) -> torch.Tensor:
        """非負權重參數 theta = Softplus(phi)"""
        return F.softplus(self.phi)

    def forward(
        self,
        features: torch.Tensor,
        p_ideal: torch.Tensor,
        motion_features: Optional[torch.Tensor] = None,
        eps: float = 1e-12
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        計算 KL 散度損失。
        Args:
            features: [N, D] 各體素的特徵向量 [1, S_grad, S_band, S_chroma, S_d_edge, S_curv]
            p_ideal: [N] 理想目標分佈 (需滿足 sum(p_ideal) == 1)
            motion_features: [N] 可選運動特徵 S_motion
            eps: 數值穩定 epsilon
        Returns:
            loss: 標量 Loss
            p_pred: [N] 預測機率分佈
        """
        th = self.theta
        # 空間線性加權
        w_spatial = torch.matmul(features, th[:features.shape[-1]])

        # 4D 時空乘積
        if motion_features is not None:
            w_motion = th[-1]
            w_total = w_spatial * (1.0 + w_motion * motion_features)
        else:
            w_total = w_spatial

        w_total = torch.clamp(w_total, min=eps)
        p_pred = w_total / (torch.sum(w_total) + eps)

        # KL 散度 cross entropy
        kl_loss = - torch.sum(p_ideal * torch.log(p_pred + eps))
        reg_loss = self.l2_reg * torch.sum(th ** 2)
        total_loss = kl_loss + reg_loss

        return total_loss, p_pred

