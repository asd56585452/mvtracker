"""
3D 空間特徵體素化融合模組 (Spatial Feature Voxelizer)

提供與 prepare_n3d_mvtracker_track.py 100% 對齊的體素聚合演算法：
- 支援 Uniform Voxelization (均勻體素) 與 Dynamic Footprint Voxelization (動態足跡體素)
- 支援解析度優先級過濾 (Resolution Prioritization)
- 幾何與色彩採用 Mean 平均，特徵分數採用 np.maximum.at 向量化 Max-Pooling (保留最銳利視角特徵)
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F


def _voxel_downsample_gpu(
    pts: torch.Tensor,
    colors: torch.Tensor,
    features: Optional[torch.Tensor] = None,
    is_dynamic: Optional[torch.Tensor] = None,
    voxel_size: float = 0.1,
    footprints: Optional[torch.Tensor] = None,
    min_voxel_size: float = 0.01,
    scale: float = 1.0,
    confidences: Optional[torch.Tensor] = None,
    conf_agg: str = "min",
    device: torch.device = torch.device("cuda"),
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """PyTorch GPU 向量化體素下採樣與特徵 Max-Pooling (支援均勻與動態足跡體素化，及信心度 min/mean 融合)。"""
    dev = device
    pts = pts.to(dev).float()
    colors = colors.to(dev).float()
    has_features = features is not None
    if has_features:
        features = features.to(dev).float()
    has_dynamic = is_dynamic is not None
    if has_dynamic:
        is_dynamic = is_dynamic.to(dev)
    has_footprints = footprints is not None
    if has_footprints:
        footprints = footprints.to(dev).float()
    has_conf = confidences is not None
    if has_conf:
        confidences = confidences.to(dev).float().squeeze()

    if not has_footprints:
        # =========================================================================
        # 模式 A：GPU 均勻體素化 (Uniform Voxelization)
        # =========================================================================
        coords = torch.round(pts / voxel_size).long()
        c_min = coords.min(dim=0).values
        coords = coords - c_min
        c_dim = coords.max(dim=0).values + 1
        flat_c = coords[:, 0] + coords[:, 1] * c_dim[0] + coords[:, 2] * (c_dim[0] * c_dim[1])

        u_c, inv = torch.unique(flat_c, return_inverse=True)
        M = len(u_c)

        pts_sum = torch.zeros(M, 3, device=dev).scatter_add_(0, inv[:, None].expand(-1, 3), pts)
        colors_sum = torch.zeros(M, 3, device=dev).scatter_add_(0, inv[:, None].expand(-1, 3), colors)
        counts = torch.bincount(inv, minlength=M).unsqueeze(1).float()

        pts_v = pts_sum / counts
        colors_v = colors_sum / counts

        if has_features:
            feat_dim = features.shape[1]
            feats_v = torch.zeros(M, feat_dim, device=dev).scatter_reduce_(
                0, inv[:, None].expand(-1, feat_dim), features, reduce="amax", include_self=False
            )
        else:
            feats_v = None

        if has_dynamic:
            dyn_v = torch.zeros(M, dtype=torch.long, device=dev).scatter_reduce_(
                0, inv, is_dynamic.long(), reduce="amax", include_self=False
            ).bool()
        else:
            dyn_v = None

        if has_conf:
            if conf_agg == "min":
                conf_v = torch.full((M,), float("inf"), device=dev).scatter_reduce_(
                    0, inv, confidences, reduce="amin", include_self=False
                )
            elif conf_agg == "mean":
                conf_v = torch.zeros(M, device=dev).scatter_add_(0, inv, confidences) / counts.squeeze(1)
            else:
                raise ValueError(f"Unknown conf_agg: {conf_agg}")
        else:
            conf_v = None

        return pts_v, colors_v, feats_v, dyn_v, conf_v

    else:
        # =========================================================================
        # 模式 B：GPU 動態足跡體素化 (Dynamic Footprint Voxelization + 優先級過濾)
        # =========================================================================
        target_s = torch.clamp(footprints * scale, min_voxel_size, voxel_size)
        k = torch.round(torch.log2(target_s / min_voxel_size)).clamp(min=0).int()
        s_i = torch.clamp(min_voxel_size * (2.0 ** k.float()), min_voxel_size, voxel_size)
        unique_s = torch.sort(torch.unique(s_i)).values

        base_coords = torch.round(pts / min_voxel_size).long()
        base_min = base_coords.min(dim=0).values
        base_coords = base_coords - base_min
        base_dim = base_coords.max(dim=0).values + 1
        base_flat_keys = base_coords[:, 0] + base_coords[:, 1] * base_dim[0] + base_coords[:, 2] * (base_dim[0] * base_dim[1])

        occupied_keys = torch.empty(0, dtype=torch.long, device=dev)
        pts_list, cols_list, feats_list, dyn_list, confs_list = [], [], [], [], []

        for s in unique_s:
            s_val = float(s.item())
            mask = (s_i == s)
            pts_sub = pts[mask]
            cols_sub = colors[mask]
            feat_sub = features[mask] if has_features else None
            dyn_sub = is_dynamic[mask] if has_dynamic else None
            conf_sub = confidences[mask] if has_conf else None
            keys_sub = base_flat_keys[mask]

            if len(pts_sub) == 0:
                continue

            # 細網格佔用過濾
            if s_val > min_voxel_size and len(occupied_keys) > 0:
                keep_mask = ~torch.isin(keys_sub, occupied_keys)
                pts_sub = pts_sub[keep_mask]
                cols_sub = cols_sub[keep_mask]
                if has_features:
                    feat_sub = feat_sub[keep_mask]
                if has_dynamic:
                    dyn_sub = dyn_sub[keep_mask]
                if has_conf:
                    conf_sub = conf_sub[keep_mask]
                if len(pts_sub) == 0:
                    continue

            coords = torch.round(pts_sub / s_val).long()
            c_min = coords.min(dim=0).values
            coords = coords - c_min
            c_dim = coords.max(dim=0).values + 1
            flat_c = coords[:, 0] + coords[:, 1] * c_dim[0] + coords[:, 2] * (c_dim[0] * c_dim[1])

            u_c, inv = torch.unique(flat_c, return_inverse=True)
            M = len(u_c)

            pts_sum = torch.zeros(M, 3, device=dev).scatter_add_(0, inv[:, None].expand(-1, 3), pts_sub)
            cols_sum = torch.zeros(M, 3, device=dev).scatter_add_(0, inv[:, None].expand(-1, 3), cols_sub)
            counts = torch.bincount(inv, minlength=M).unsqueeze(1).float()

            pts_v = pts_sum / counts
            cols_v = cols_sum / counts
            pts_list.append(pts_v)
            cols_list.append(cols_v)

            if has_features:
                f_dim = feat_sub.shape[1]
                feat_v = torch.zeros(M, f_dim, device=dev).scatter_reduce_(
                    0, inv[:, None].expand(-1, f_dim), feat_sub, reduce="amax", include_self=False
                )
                feats_list.append(feat_v)

            if has_dynamic:
                dyn_v = torch.zeros(M, dtype=torch.long, device=dev).scatter_reduce_(
                    0, inv, dyn_sub.long(), reduce="amax", include_self=False
                ).bool()
                dyn_list.append(dyn_v)

            if has_conf:
                if conf_agg == "min":
                    c_v = torch.full((M,), float("inf"), device=dev).scatter_reduce_(
                        0, inv, conf_sub, reduce="amin", include_self=False
                    )
                elif conf_agg == "mean":
                    c_v = torch.zeros(M, device=dev).scatter_add_(0, inv, conf_sub) / counts.squeeze(1)
                else:
                    raise ValueError(f"Unknown conf_agg: {conf_agg}")
                confs_list.append(c_v)

            # 登記至細網格佔用表
            v_base = torch.round(pts_v / min_voxel_size).long() - base_min
            v_flat = v_base[:, 0] + v_base[:, 1] * base_dim[0] + v_base[:, 2] * (base_dim[0] * base_dim[1])
            occupied_keys = torch.cat([occupied_keys, v_flat])

        out_pts = torch.cat(pts_list, dim=0) if pts_list else torch.zeros((0, 3), device=dev)
        out_cols = torch.cat(cols_list, dim=0) if cols_list else torch.zeros((0, 3), device=dev)
        out_feats = torch.cat(feats_list, dim=0) if has_features and feats_list else None
        out_dyn = torch.cat(dyn_list, dim=0) if has_dynamic and dyn_list else None
        out_conf = torch.cat(confs_list, dim=0) if has_conf and confs_list else None

        return out_pts, out_cols, out_feats, out_dyn, out_conf



def _surfel_downsample_gpu(
    pts: torch.Tensor,
    colors: torch.Tensor,
    features: Optional[torch.Tensor] = None,
    normals: Optional[torch.Tensor] = None,
    is_dynamic: Optional[torch.Tensor] = None,
    voxel_size: float = 0.1,
    footprints: Optional[torch.Tensor] = None,
    min_voxel_size: float = 0.01,
    scale: float = 1.0,
    surfel_normal_scale: float = 2.5,
    surfel_angle_deg: float = 25.0,
    confidences: Optional[torch.Tensor] = None,
    conf_agg: str = "min",
    device: torch.device = torch.device("cuda"),
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    PyTorch GPU 向量化多尺度表面基元融合 (Multi-Scale Surfel Point-to-Plane Voxelization - 方案 B)。
    結合動態足跡 (Dynamic Footprint) 與法向射線探訪融合 (Ray-Probe Anisotropic Merging)：
    - 切向保持高解析度足跡 (例如 1cm)
    - 法向依深度自適應擴展容差厚度 (例如 2.5 * s)，將多視角視差重疊點合併為連續平滑表面
    - 轉角邊緣與前後景由夾角與點面距離嚴格保護，邊界天生免疫拉扯
    - 支援信心度 min/mean 融合
    """
    dev = device
    pts = pts.to(dev).float()
    colors = colors.to(dev).float()
    has_features = features is not None
    if has_features:
        features = features.to(dev).float()
    has_dynamic = is_dynamic is not None
    if has_dynamic:
        is_dynamic = is_dynamic.to(dev)
    has_footprints = footprints is not None
    if has_footprints:
        footprints = footprints.to(dev).float()
    has_normals = normals is not None
    if has_normals:
        normals = F.normalize(normals.to(dev).float(), dim=-1, eps=1e-6)
    has_conf = confidences is not None
    if has_conf:
        confidences = confidences.to(dev).float().squeeze()

    # 1. 多尺度等級計算 (動態足跡)
    if has_footprints:
        target_s = torch.clamp(footprints * scale, min_voxel_size, voxel_size)
        k = torch.round(torch.log2(target_s / min_voxel_size)).clamp(min=0).int()
        s_i = torch.clamp(min_voxel_size * (2.0 ** k.float()), min_voxel_size, voxel_size)
        unique_s = torch.sort(torch.unique(s_i)).values
    else:
        s_i = torch.full((len(pts),), voxel_size, device=dev)
        unique_s = torch.tensor([voxel_size], device=dev)

    # 基準細網格 (用於跨尺度佔用過濾)
    base_coords = torch.round(pts / min_voxel_size).long()
    base_min = base_coords.min(dim=0).values
    base_coords = base_coords - base_min
    base_dim = base_coords.max(dim=0).values + 1
    base_flat_keys = base_coords[:, 0] + base_coords[:, 1] * base_dim[0] + base_coords[:, 2] * (base_dim[0] * base_dim[1])

    occupied_keys = torch.empty(0, dtype=torch.long, device=dev)
    pts_list, cols_list, feats_list, dyn_list, confs_list = [], [], [], [], []

    cos_thr = float(np.cos(np.radians(surfel_angle_deg)))

    for s in unique_s:
        s_val = float(s.item())
        mask = (s_i == s)
        pts_sub = pts[mask]
        cols_sub = colors[mask]
        feat_sub = features[mask] if has_features else None
        dyn_sub = is_dynamic[mask] if has_dynamic else None
        norm_sub = normals[mask] if has_normals else None
        conf_sub = confidences[mask] if has_conf else None
        keys_sub = base_flat_keys[mask]

        if len(pts_sub) == 0:
            continue

        # 細網格佔用過濾 (捨棄落在已佔用高解析度區域的粗糙點)
        if s_val > min_voxel_size and len(occupied_keys) > 0:
            keep_mask = ~torch.isin(keys_sub, occupied_keys)
            pts_sub = pts_sub[keep_mask]
            cols_sub = cols_sub[keep_mask]
            if has_features:
                feat_sub = feat_sub[keep_mask]
            if has_dynamic:
                dyn_sub = dyn_sub[keep_mask]
            if has_normals:
                norm_sub = norm_sub[keep_mask]
            if has_conf:
                conf_sub = conf_sub[keep_mask]
            if len(pts_sub) == 0:
                continue

        # 階段一：切向尺度 s_val 初始微基元化 (Micro-Surfel)
        coords = torch.round(pts_sub / s_val).long()
        c_min = coords.min(dim=0).values
        coords = coords - c_min
        c_dim = coords.max(dim=0).values + 1
        flat_c = coords[:, 0] + coords[:, 1] * c_dim[0] + coords[:, 2] * (c_dim[0] * c_dim[1])

        u_c, inv = torch.unique(flat_c, return_inverse=True)
        M = len(u_c)

        pts_sum = torch.zeros(M, 3, device=dev).scatter_add_(0, inv[:, None].expand(-1, 3), pts_sub)
        cols_sum = torch.zeros(M, 3, device=dev).scatter_add_(0, inv[:, None].expand(-1, 3), cols_sub)
        counts = torch.bincount(inv, minlength=M).unsqueeze(1).float()

        pts_v = pts_sum / counts
        cols_v = cols_sum / counts

        if has_normals:
            norm_sum = torch.zeros(M, 3, device=dev).scatter_add_(0, inv[:, None].expand(-1, 3), norm_sub)
            norm_v = F.normalize(norm_sum, dim=-1, eps=1e-6)
        else:
            norm_v = None

        if has_features:
            f_dim = feat_sub.shape[1]
            feat_v = torch.zeros(M, f_dim, device=dev).scatter_reduce_(
                0, inv[:, None].expand(-1, f_dim), feat_sub, reduce="amax", include_self=False
            )
        else:
            feat_v = None

        if has_dynamic:
            dyn_v = torch.zeros(M, dtype=torch.long, device=dev).scatter_reduce_(
                0, inv, dyn_sub.long(), reduce="amax", include_self=False
            ).bool()
        else:
            dyn_v = None

        if has_conf:
            if conf_agg == "min":
                conf_v = torch.full((M,), float("inf"), device=dev).scatter_reduce_(
                    0, inv, conf_sub, reduce="amin", include_self=False
                )
            elif conf_agg == "mean":
                conf_v = torch.zeros(M, device=dev).scatter_add_(0, inv, conf_sub) / counts.squeeze(1)
            else:
                raise ValueError(f"Unknown conf_agg: {conf_agg}")
        else:
            conf_v = None

        # 階段二：沿著法向量方向進行 Point-to-Plane 射線探訪融合
        if has_normals and M > 1:
            max_p2p_dist = surfel_normal_scale * s_val

            dim_x = c_dim[0]
            dim_y = c_dim[1]
            dim_z = c_dim[2]
            stride_y = dim_x
            stride_z = dim_x * dim_y

            ux = u_c % dim_x
            uy = (u_c // dim_x) % dim_y
            uz = u_c // (dim_x * dim_y)
            u_coords = torch.stack([ux, uy, uz], dim=-1)  # [M, 3]

            cluster_parent = torch.arange(M, device=dev)

            # 沿法向前後探訪步長 (例如 ±1, ±2, ±3 格)
            probe_steps = [-3, -2, -1, 1, 2, 3]
            for step_k in probe_steps:
                delta_c = torch.round(step_k * norm_v).long()
                probe_c = u_coords + delta_c

                in_bounds = (
                    (probe_c[:, 0] >= 0) & (probe_c[:, 0] < dim_x) &
                    (probe_c[:, 1] >= 0) & (probe_c[:, 1] < dim_y) &
                    (probe_c[:, 2] >= 0) & (probe_c[:, 2] < dim_z)
                )
                if not in_bounds.any():
                    continue

                probe_flat = torch.zeros(M, dtype=torch.long, device=dev)
                probe_flat[in_bounds] = (
                    probe_c[in_bounds, 0] +
                    probe_c[in_bounds, 1] * stride_y +
                    probe_c[in_bounds, 2] * stride_z
                )

                idx = torch.searchsorted(u_c, probe_flat)
                valid_idx = in_bounds & (idx < M)
                idx_clamped = idx.clamp(max=M - 1)
                hit = valid_idx & (u_c[idx_clamped] == probe_flat)

                if not hit.any():
                    continue

                src_idx = torch.nonzero(hit, as_tuple=True)[0]
                dst_idx = idx_clamped[src_idx]

                diff_mask = (src_idx != dst_idx)
                if not diff_mask.any():
                    continue
                src_idx = src_idx[diff_mask]
                dst_idx = dst_idx[diff_mask]

                # 幾何條件 1：法向量夾角一致性
                dot_prod = (norm_v[src_idx] * norm_v[dst_idx]).sum(dim=-1)
                angle_ok = dot_prod > cos_thr

                # 幾何條件 2：點面距離 (Point-to-Plane) 在容許法向厚度內
                p2p_dist = torch.abs(((pts_v[src_idx] - pts_v[dst_idx]) * norm_v[src_idx]).sum(dim=-1))
                dist_ok = p2p_dist < max_p2p_dist

                merge_pair = angle_ok & dist_ok
                if not merge_pair.any():
                    continue

                s_m = src_idx[merge_pair]
                d_m = dst_idx[merge_pair]

                p_s = cluster_parent[s_m]
                p_d = cluster_parent[d_m]
                min_p = torch.minimum(p_s, p_d)
                cluster_parent[s_m] = min_p
                cluster_parent[d_m] = min_p

            # 兩段路徑壓縮
            cluster_parent = cluster_parent[cluster_parent]
            cluster_parent = cluster_parent[cluster_parent]

            # 聚類合併計算
            u_clusters, inv_cluster = torch.unique(cluster_parent, return_inverse=True)
            M_final = len(u_clusters)

            cluster_counts = torch.bincount(inv_cluster, minlength=M_final).unsqueeze(1).float()
            pts_final = torch.zeros(M_final, 3, device=dev).scatter_add_(
                0, inv_cluster[:, None].expand(-1, 3), pts_v
            ) / cluster_counts
            cols_final = torch.zeros(M_final, 3, device=dev).scatter_add_(
                0, inv_cluster[:, None].expand(-1, 3), cols_v
            ) / cluster_counts

            if has_features:
                feats_final = torch.zeros(M_final, f_dim, device=dev).scatter_reduce_(
                    0, inv_cluster[:, None].expand(-1, f_dim), feat_v, reduce="amax", include_self=False
                )
            else:
                feats_final = None

            if has_dynamic:
                dyn_final = torch.zeros(M_final, dtype=torch.long, device=dev).scatter_reduce_(
                    0, inv_cluster, dyn_v.long(), reduce="amax", include_self=False
                ).bool()
            else:
                dyn_final = None

            if has_conf:
                if conf_agg == "min":
                    confs_final = torch.full((M_final,), float("inf"), device=dev).scatter_reduce_(
                        0, inv_cluster, conf_v, reduce="amin", include_self=False
                    )
                elif conf_agg == "mean":
                    confs_final = torch.zeros(M_final, device=dev).scatter_add_(
                        0, inv_cluster, conf_v
                    ) / cluster_counts.squeeze(1)
            else:
                confs_final = None
        else:
            pts_final = pts_v
            cols_final = cols_v
            feats_final = feat_v
            dyn_final = dyn_v
            confs_final = conf_v

        # 登記至細網格佔用表
        v_base = torch.round(pts_final / min_voxel_size).long() - base_min
        v_flat = v_base[:, 0] + v_base[:, 1] * base_dim[0] + v_base[:, 2] * (base_dim[0] * base_dim[1])
        occupied_keys = torch.cat([occupied_keys, v_flat])

        pts_list.append(pts_final)
        cols_list.append(cols_final)
        if has_features:
            feats_list.append(feats_final)
        if has_dynamic:
            dyn_list.append(dyn_final)
        if has_conf:
            confs_list.append(confs_final)

    out_pts = torch.cat(pts_list, dim=0) if pts_list else torch.zeros((0, 3), device=dev)
    out_cols = torch.cat(cols_list, dim=0) if cols_list else torch.zeros((0, 3), device=dev)
    out_feats = torch.cat(feats_list, dim=0) if has_features and feats_list else None
    out_dyn = torch.cat(dyn_list, dim=0) if has_dynamic and dyn_list else None
    out_conf = torch.cat(confs_list, dim=0) if has_conf and confs_list else None

    return out_pts, out_cols, out_feats, out_dyn, out_conf


def voxel_downsample_unified_with_features(
    pts: Union[np.ndarray, torch.Tensor],
    colors: Union[np.ndarray, torch.Tensor],
    features: Optional[Union[np.ndarray, torch.Tensor]] = None,
    is_dynamic: Optional[Union[np.ndarray, torch.Tensor]] = None,
    normals: Optional[Union[np.ndarray, torch.Tensor]] = None,
    voxel_size: float = 0.1,
    footprints: Optional[Union[np.ndarray, torch.Tensor]] = None,
    min_voxel_size: float = 0.01,
    scale: float = 1.0,
    mode: str = "dynamic",
    surfel_normal_scale: float = 2.5,
    surfel_angle_deg: float = 25.0,
    confidences: Optional[Union[np.ndarray, torch.Tensor]] = None,
    conf_agg: str = "min",
    return_confidences: bool = False,
    device: Optional[Union[str, torch.device]] = None,
) -> Union[
    Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor], Optional[Union[np.ndarray, torch.Tensor]], Optional[Union[np.ndarray, torch.Tensor]]],
    Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor], Optional[Union[np.ndarray, torch.Tensor]], Optional[Union[np.ndarray, torch.Tensor]], Optional[Union[np.ndarray, torch.Tensor]]]
]:
    """
    統一的 3D 空間體素融合下採樣函數 (附帶特徵向量 Max-Pooling 與信心度聚合)。
    支援均勻體素 (uniform)、動態足跡體素 (dynamic) 與多尺度表面基元融合 (surfel)。
    支援 PyTorch GPU 向量化加速，相容 NumPy 陣列與 PyTorch 張量輸入。

    Args:
        pts: [N, 3] 點雲世界座標
        colors: [N, 3] 點雲 RGB 顏色
        features: [N, D] 點雲特徵分數向量 (例如 [S_sobel, S_dct, S_chroma])，可為 None
        is_dynamic: [N] 動態遮罩布林值，可為 None
        normals: [N, 3] 表面單位法向量 (世界座標系)，mode='surfel' 時使用
        voxel_size: 體素最大尺寸 (預設 0.1)
        footprints: [N] 各點反投影之相機像素 3D 足跡 (depth / focal_length)，為 None 則使用均勻體素化
        min_voxel_size: 最小體素尺寸 (預設 0.01)
        scale: 動態足跡縮放係數 (預設 1.0)
        mode: 體素化模式 ('uniform', 'dynamic', 'surfel')
        surfel_normal_scale: surfel 模式法向厚度容差倍率 (預設 2.5)
        surfel_angle_deg: surfel 模式表面夾角門檻 (預設 25.0 度)
        confidences: [N] 點雲信心度分數，可為 None
        conf_agg: 信心度聚合方式 ('min' 或 'mean'，預設 'min')
        return_confidences: 是否回傳融合後體素信心度
        device: 運算裝置 ('cuda' 或 'cpu'，預設自動選取 GPU)

    Returns:
        out_pts: [M, 3] 融合後體素中心座標 (幾何採 Mean)
        out_colors: [M, 3] 融合後體素顏色 (顏色採 Mean)
        out_features: [M, D] 融合後體素特徵向量 (特徵採 Max-Pooling)，若輸入為 None 則回傳 None
        out_is_dyn: [M] 融合後動態布林值 (採 Logical OR)，若輸入為 None 則回傳 None
        out_conf: [M] 融合後體素信心度 (若 return_confidences=True 則回傳)
    """
    has_features = features is not None
    has_dynamic = is_dynamic is not None
    has_conf = confidences is not None
    num_pts = len(pts)

    if num_pts == 0:
        if isinstance(pts, torch.Tensor):
            empty_pts = torch.zeros((0, 3), dtype=torch.float32, device=pts.device)
            empty_cols = torch.zeros((0, 3), dtype=torch.float32, device=pts.device)
            empty_feat = torch.zeros((0, features.shape[1]), dtype=torch.float32, device=pts.device) if has_features else None
            empty_dyn = torch.zeros(0, dtype=torch.bool, device=pts.device) if has_dynamic else None
            empty_conf = torch.zeros(0, dtype=torch.float32, device=pts.device) if has_conf else None
        else:
            empty_pts = np.zeros((0, 3), dtype=np.float32)
            empty_cols = np.zeros((0, 3), dtype=np.float32)
            empty_feat = np.zeros((0, features.shape[1]), dtype=np.float32) if has_features else None
            empty_dyn = np.zeros(0, dtype=bool) if has_dynamic else None
            empty_conf = np.zeros(0, dtype=np.float32) if has_conf else None
        if return_confidences:
            return empty_pts, empty_cols, empty_feat, empty_dyn, empty_conf
        return empty_pts, empty_cols, empty_feat, empty_dyn

    # 判定運算裝置
    is_torch_input = isinstance(pts, torch.Tensor)
    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    # 若為 CUDA 裝置，直接啟用 GPU 向量化加速
    if target_device.type == "cuda":
        pts_t = pts if is_torch_input else torch.from_numpy(pts)
        cols_t = colors if isinstance(colors, torch.Tensor) else torch.from_numpy(colors)
        feats_t = features if (features is None or isinstance(features, torch.Tensor)) else torch.from_numpy(features)
        dyn_t = is_dynamic if (is_dynamic is None or isinstance(is_dynamic, torch.Tensor)) else torch.from_numpy(is_dynamic)
        fps_t = footprints if (footprints is None or isinstance(footprints, torch.Tensor)) else torch.from_numpy(footprints)
        norm_t = normals if (normals is None or isinstance(normals, torch.Tensor)) else torch.from_numpy(normals)
        conf_t = confidences if (confidences is None or isinstance(confidences, torch.Tensor)) else torch.from_numpy(confidences)

        if mode == "surfel" or (normals is not None and mode not in ["uniform", "dynamic"]):
            g_pts, g_cols, g_feats, g_dyn, g_confs = _surfel_downsample_gpu(
                pts=pts_t,
                colors=cols_t,
                features=feats_t,
                normals=norm_t,
                is_dynamic=dyn_t,
                voxel_size=voxel_size,
                footprints=fps_t if mode != "uniform" else None,
                min_voxel_size=min_voxel_size,
                scale=scale,
                surfel_normal_scale=surfel_normal_scale,
                surfel_angle_deg=surfel_angle_deg,
                confidences=conf_t,
                conf_agg=conf_agg,
                device=target_device,
            )
        else:
            effective_footprints = fps_t if mode != "uniform" else None
            g_pts, g_cols, g_feats, g_dyn, g_confs = _voxel_downsample_gpu(
                pts=pts_t,
                colors=cols_t,
                features=feats_t,
                is_dynamic=dyn_t,
                voxel_size=voxel_size,
                footprints=effective_footprints,
                min_voxel_size=min_voxel_size,
                scale=scale,
                confidences=conf_t,
                conf_agg=conf_agg,
                device=target_device,
            )

        if not is_torch_input:
            out_pts = g_pts.cpu().numpy()
            out_colors = g_cols.cpu().numpy()
            out_features = g_feats.cpu().numpy() if g_feats is not None else None
            out_is_dyn = g_dyn.cpu().numpy() if g_dyn is not None else None
            out_confs = g_confs.cpu().numpy() if g_confs is not None else None
            if return_confidences:
                return out_pts, out_colors, out_features, out_is_dyn, out_confs
            return out_pts, out_colors, out_features, out_is_dyn
        else:
            if return_confidences:
                return g_pts, g_cols, g_feats, g_dyn, g_confs
            return g_pts, g_cols, g_feats, g_dyn

    # CPU NumPy Fallback 路徑
    if is_torch_input:
        pts = pts.cpu().numpy()
        colors = colors.cpu().numpy()
        if has_features: features = features.cpu().numpy()
        if has_dynamic: is_dynamic = is_dynamic.cpu().numpy()
        if footprints is not None: footprints = footprints.cpu().numpy()
        if has_conf: confidences = confidences.cpu().numpy()

    if footprints is not None:
        target_s = footprints * scale
        target_s = np.clip(target_s, min_voxel_size, voxel_size)
        k = np.round(np.log2(target_s / min_voxel_size))
        k = np.clip(k, 0, None).astype(np.int32)
        s_i = min_voxel_size * (2 ** k)
        s_i = np.clip(s_i, min_voxel_size, voxel_size)
        unique_s = np.sort(np.unique(s_i))

        downsampled_pts_list = []
        downsampled_colors_list = []
        downsampled_features_list = []
        downsampled_is_dyn_list = []
        downsampled_confs_list = []
        occupied_keys = set()

        for s in unique_s:
            mask = (s_i == s)
            pts_sub = pts[mask]
            colors_sub = colors[mask]
            feat_sub = features[mask] if has_features else None
            is_dyn_sub = is_dynamic[mask] if has_dynamic else None
            conf_sub = confidences[mask] if has_conf else None

            if len(pts_sub) == 0:
                continue

            if s > min_voxel_size and len(occupied_keys) > 0:
                fine_coords = np.round(pts_sub / min_voxel_size).astype(np.int64)
                keep_mask = np.array([
                    (coord[0], coord[1], coord[2]) not in occupied_keys
                    for coord in fine_coords
                ], dtype=bool)

                pts_sub = pts_sub[keep_mask]
                colors_sub = colors_sub[keep_mask]
                if has_features:
                    feat_sub = feat_sub[keep_mask]
                if has_dynamic:
                    is_dyn_sub = is_dyn_sub[keep_mask]
                if has_conf:
                    conf_sub = conf_sub[keep_mask]

                if len(pts_sub) == 0:
                    continue

            coords = np.round(pts_sub / s)
            _, inverse_indices = np.unique(
                coords, axis=0, return_index=False, return_inverse=True
            )

            num_voxels = int(inverse_indices.max()) + 1
            pts_sum = np.zeros((num_voxels, 3), dtype=np.float64)
            colors_sum = np.zeros((num_voxels, 3), dtype=np.float64)

            np.add.at(pts_sum, inverse_indices, pts_sub)
            np.add.at(colors_sum, inverse_indices, colors_sub)
            counts = np.bincount(inverse_indices)

            pts_voxel = (pts_sum / counts[:, None]).astype(np.float32)
            colors_voxel = (colors_sum / counts[:, None]).astype(np.float32)

            downsampled_pts_list.append(pts_voxel)
            downsampled_colors_list.append(colors_voxel)

            if has_features:
                feat_dim = feat_sub.shape[1]
                feat_max = np.zeros((num_voxels, feat_dim), dtype=np.float32)
                np.maximum.at(feat_max, inverse_indices, feat_sub)
                downsampled_features_list.append(feat_max)

            if has_dynamic:
                dyn_max = np.zeros(num_voxels, dtype=bool)
                np.logical_or.at(dyn_max, inverse_indices, is_dyn_sub)
                downsampled_is_dyn_list.append(dyn_max)

            if has_conf:
                if conf_agg == "min":
                    c_v = np.full(num_voxels, np.inf, dtype=np.float32)
                    np.minimum.at(c_v, inverse_indices, conf_sub)
                elif conf_agg == "mean":
                    c_sum = np.zeros(num_voxels, dtype=np.float32)
                    np.add.at(c_sum, inverse_indices, conf_sub)
                    c_v = c_sum / counts
                downsampled_confs_list.append(c_v)

            fine_coords_voxel = np.round(pts_voxel / min_voxel_size).astype(np.int64)
            for coord in fine_coords_voxel:
                occupied_keys.add((coord[0], coord[1], coord[2]))

        out_pts = np.concatenate(downsampled_pts_list, axis=0) if downsampled_pts_list else np.zeros((0, 3), dtype=np.float32)
        out_colors = np.concatenate(downsampled_colors_list, axis=0) if downsampled_colors_list else np.zeros((0, 3), dtype=np.float32)
        out_features = np.concatenate(downsampled_features_list, axis=0) if has_features and downsampled_features_list else None
        out_is_dyn = np.concatenate(downsampled_is_dyn_list, axis=0) if has_dynamic and downsampled_is_dyn_list else None
        out_confs = np.concatenate(downsampled_confs_list, axis=0) if has_conf and downsampled_confs_list else None

        if return_confidences:
            return out_pts, out_colors, out_features, out_is_dyn, out_confs
        return out_pts, out_colors, out_features, out_is_dyn

    else:
        coords = np.round(pts / voxel_size)
        _, inverse_indices = np.unique(
            coords, axis=0, return_index=False, return_inverse=True
        )

        num_voxels = int(inverse_indices.max()) + 1
        pts_sum = np.zeros((num_voxels, 3), dtype=np.float64)
        colors_sum = np.zeros((num_voxels, 3), dtype=np.float64)

        np.add.at(pts_sum, inverse_indices, pts)
        np.add.at(colors_sum, inverse_indices, colors)
        counts = np.bincount(inverse_indices)

        pts_voxel = (pts_sum / counts[:, None]).astype(np.float32)
        colors_voxel = (colors_sum / counts[:, None]).astype(np.float32)

        if has_features:
            feat_dim = features.shape[1]
            features_max = np.zeros((num_voxels, feat_dim), dtype=np.float32)
            np.maximum.at(features_max, inverse_indices, features)
        else:
            features_max = None

        if has_dynamic:
            is_dynamic_max = np.zeros(num_voxels, dtype=bool)
            np.logical_or.at(is_dynamic_max, inverse_indices, is_dynamic)
        else:
            is_dynamic_max = None

        if has_conf:
            if conf_agg == "min":
                confs_voxel = np.full(num_voxels, np.inf, dtype=np.float32)
                np.minimum.at(confs_voxel, inverse_indices, confidences)
            elif conf_agg == "mean":
                confs_sum = np.zeros(num_voxels, dtype=np.float32)
                np.add.at(confs_sum, inverse_indices, confidences)
                confs_voxel = confs_sum / counts
        else:
            confs_voxel = None

        if return_confidences:
            return pts_voxel, colors_voxel, features_max, is_dynamic_max, confs_voxel
        return pts_voxel, colors_voxel, features_max, is_dynamic_max


def unproject_camera_points_with_features(
    rgb: np.ndarray,
    depth: np.ndarray,
    features: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    depth_min: float = 0.05,
    depth_max: float = 500.0,
    valid_mask: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    return_normals: bool = False,
    confidences: Optional[np.ndarray] = None,
    return_confidences: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    """
    將單一相機視角的 RGB、深度圖與 2D 特徵圖反投影至世界座標系 3D 點雲。
    支援可選表面法向量世界座標系旋轉，以及信心度帶入。

    Args:
        rgb: [H, W, 3] 影像，值域 [0, 255] 或 [0, 1]
        depth: [H, W] 絕對度量深度圖 (公尺或世界座標單位)
        features: [H, W, D] 2D 特徵圖
        K: [3, 3] 相機內參矩陣
        c2w: [3, 4] 或 [4, 4] 相機至世界座標之外參變換矩陣
        depth_min, depth_max: 深度有效範圍截斷
        valid_mask: [H, W] 額外有效像素遮罩
        normals: [H, W, 3] 可選相機座標系單位法向量
        return_normals: 是否回傳世界座標系單位法向量
        confidences: [H, W] 可選相機視角信心度圖
        return_confidences: 是否回傳對應 3D 點雲之信心度數值

    Returns:
        pts_w: [N, 3] 世界座標 3D 點雲
        cols: [N, 3] 對應像素之 RGB 顏色 [0, 1]
        feats: [N, D] 對應像素之 2D 特徵向量
        footprints: [N] 像素在該點對應之 3D 足跡 (depth / focal_length)
        norm_w: [N, 3] (若 return_normals=True 則回傳)
        confs_w: [N] (若 return_confidences=True 則回傳)
    """
    H, W = depth.shape
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    focal = 0.5 * (fx + fy)

    # 建立網格座標
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # 深度與有效遮罩
    mask = (depth > depth_min) & (depth < depth_max) & np.isfinite(depth)
    if valid_mask is not None:
        mask = mask & valid_mask

    if not np.any(mask):
        feat_dim = features.shape[-1]
        res = [
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, feat_dim), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        ]
        if return_normals:
            res.append(np.zeros((0, 3), dtype=np.float32))
        if return_confidences:
            res.append(np.zeros(0, dtype=np.float32))
        return tuple(res)

    z = depth[mask]
    x = (u[mask] - cx) * z / fx
    y = (v[mask] - cy) * z / fy

    # 相機坐標 [N, 3]
    pts_c = np.stack([x, y, z], axis=-1)

    # 變換至世界座標系: X_w = R * X_c + T
    R = c2w[:3, :3]
    T = c2w[:3, 3]
    pts_w = (pts_c @ R.T) + T

    # 顏色正規化至 [0, 1]
    cols = rgb[mask].astype(np.float32)
    if cols.max() > 1.0:
        cols = cols / 255.0

    # 特徵向量 [N, D]
    feats = features[mask].astype(np.float32)

    # 3D 足跡 footprint = depth / focal
    footprints = (z / focal).astype(np.float32)

    if return_normals:
        if normals is not None:
            norm_c = normals[mask].astype(np.float32)
            norm_w = (norm_c @ R.T).astype(np.float32)
            norm_len = np.linalg.norm(norm_w, axis=-1, keepdims=True)
            norm_w = (norm_w / np.clip(norm_len, 1e-6, None)).astype(np.float32)
        else:
            norm_w = np.zeros((len(pts_w), 3), dtype=np.float32)
    else:
        norm_w = None

    if return_confidences:
        if confidences is not None:
            confs_w = confidences[mask].astype(np.float32)
        else:
            confs_w = np.ones(len(pts_w), dtype=np.float32) * 10.0
    else:
        confs_w = None

    if return_normals and return_confidences:
        return pts_w.astype(np.float32), cols, feats, footprints, norm_w, confs_w
    elif return_normals:
        return pts_w.astype(np.float32), cols, feats, footprints, norm_w
    elif return_confidences:
        return pts_w.astype(np.float32), cols, feats, footprints, confs_w
    else:
        return pts_w.astype(np.float32), cols, feats, footprints

