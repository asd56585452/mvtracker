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
