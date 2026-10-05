import os
import sys
import math
import struct
import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from plyfile import PlyData
import rerun as rr
import matplotlib.pyplot as plt

from gsplat.rendering import rasterization

class Sandwich(nn.Module):
    def __init__(self, dim=9, outdim=3, ray_dim=3, bias=False):
        super(Sandwich, self).__init__()
        self.mlp1 = nn.Conv2d(9, 6, kernel_size=1, bias=bias)
        self.mlp2 = nn.Conv2d(6, outdim, kernel_size=1, bias=bias)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input, rays, time=None):
        albedo, spec, timefeature = input.chunk(3, dim=1)
        specular = torch.cat([spec, timefeature, rays], dim=1)
        specular = self.mlp1(specular)
        specular = self.relu(specular)
        specular = self.mlp2(specular)
        result = albedo + specular
        result = self.sigmoid(result) 
        return result

def trbfunction(x): 
    return torch.exp(-1 * x.pow(2))

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]
    ], dtype=np.float32)

def read_colmap_cameras(colmap_dir, downscale=2):
    camerasfile = os.path.join(colmap_dir, 'cameras.bin')
    imagesfile = os.path.join(colmap_dir, 'images.bin')
    intrinsics = {}
    with open(camerasfile, 'rb') as f:
        num_cams = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_cams):
            cid, model_id, width, height = struct.unpack('<iiQQ', f.read(24))
            params = struct.unpack('<4d', f.read(32))
            intrinsics[cid] = {
                'width': int(width // downscale),
                'height': int(height // downscale),
                'fx': float(params[0] / downscale),
                'fy': float(params[1] / downscale),
                'cx': float(params[2] / downscale),
                'cy': float(params[3] / downscale),
            }
    cameras = {}
    with open(imagesfile, 'rb') as f:
        num_images = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack('<I', f.read(4))[0]
            q0, q1, q2, q3 = struct.unpack('<4d', f.read(32))
            tx, ty, tz = struct.unpack('<3d', f.read(24))
            camera_id = struct.unpack('<I', f.read(4))[0]
            image_name = ''
            current_char = f.read(1)
            while current_char != b'\x00':
                image_name += current_char.decode('utf-8')
                current_char = f.read(1)
            num_points2D = struct.unpack('<Q', f.read(8))[0]
            f.seek(num_points2D * 24, 1)

            cam_name = image_name.split('.')[0]
            intr = intrinsics[camera_id]
            qvec = np.array([q0, q1, q2, q3], dtype=np.float32)
            tvec = np.array([tx, ty, tz], dtype=np.float32)
            R_w2c = qvec2rotmat(qvec)
            
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = R_w2c.T
            c2w[:3, 3] = - R_w2c.T @ tvec

            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :3] = R_w2c
            w2c[:3, 3] = tvec

            K = np.array([
                [intr['fx'], 0.0, intr['cx']],
                [0.0, intr['fy'], intr['cy']],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)

            cameras[cam_name] = {
                'cam_id': camera_id,
                'image_name': image_name,
                'w2c': w2c,
                'c2w': c2w,
                'K': K,
                'fx': intr['fx'],
                'fy': intr['fy'],
                'cx': intr['cx'],
                'cy': intr['cy'],
                'width': intr['width'],
                'height': intr['height'],
            }

    return cameras

def load_ply_gaussian(ply_path, device='cuda'):
    ply = PlyData.read(ply_path)
    v = ply['vertex']
    means = torch.from_numpy(np.stack([v['x'], v['y'], v['z']], axis=-1).copy()).float().to(device)
    quats = torch.from_numpy(np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], axis=-1).copy()).float().to(device)
    quats = torch.nn.functional.normalize(quats, p=2, dim=-1)
    scales = torch.exp(torch.from_numpy(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=-1).copy()).float().to(device))
    opacities = torch.sigmoid(torch.from_numpy(np.array(v['opacity']).copy()).float().to(device))
    sh_dc = torch.from_numpy(np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=-1).copy()).float().to(device)
    colors = sh_dc.unsqueeze(1)
    return means, quats, scales, opacities, colors

def load_pt_model(ckpt_path, device='cuda'):
    ckpt = torch.load(ckpt_path, map_location=device)
    decoder = Sandwich(9, 3, ray_dim=3, bias=False).to(device)
    decoder.load_state_dict(ckpt['decoder'])
    decoder.eval()

    splats = ckpt['splats']
    means = splats['means']
    quats = splats['quats']
    scales = torch.exp(splats['scales'])
    opacities = torch.sigmoid(splats['opacities'])
    trbfcenter = splats['trbf_center']
    trbfscale = torch.exp(splats['trbf_scale'])
    motion = splats['motion']
    omega = splats['omega']
    feature_color = splats['colors']
    feature_dir = splats['features_dir']
    feature_time = splats['features_time']

    timestamp = 0.0
    pointtimes = torch.ones((means.shape[0], 1), dtype=means.dtype, device=device)
    trbfdistanceoffset = timestamp * pointtimes - trbfcenter
    trbfdistance = trbfdistanceoffset / (math.sqrt(2) * trbfscale)
    trbfoutput = trbfunction(trbfdistance)
    opacity = opacities * trbfoutput.squeeze()

    tforpoly = trbfdistanceoffset.detach()
    means_motion = means
    t_pow = tforpoly
    current_motion_degree = motion.shape[1] // 3
    for i in range(current_motion_degree):
        means_motion = means_motion + motion[:, i*3:(i+1)*3] * t_pow
        t_pow = t_pow * tforpoly

    rotations_feat = quats
    t_pow = tforpoly
    current_rotation_degree = omega.shape[1] // 4
    for i in range(current_rotation_degree):
        rotations_feat = rotations_feat + t_pow * omega[:, i*4:(i+1)*4]
        t_pow = t_pow * tforpoly
    rotations = torch.nn.functional.normalize(rotations_feat)

    colors_precomp = torch.cat((feature_color, feature_dir, tforpoly * feature_time), dim=1)
    return means_motion, rotations, scales, opacity, colors_precomp, decoder

def sh_to_rgb(sh_dc):
    C0 = 0.28209479177387814
    rgb = sh_dc * C0 + 0.5
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

def compute_rays(cinfo, device='cuda'):
    W, H = cinfo['width'], cinfo['height']
    fx, fy, cx, cy = cinfo['fx'], cinfo['fy'], cinfo['cx'], cinfo['cy']
    c2w = cinfo['c2w']
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device) + 0.5,
        torch.arange(W, dtype=torch.float32, device=device) + 0.5,
        indexing='ij'
    )
    dirs_cam = torch.stack([(x_grid - cx) / fx, (y_grid - cy) / fy, torch.ones_like(x_grid)], dim=-1)
    R_c2w = torch.from_numpy(c2w[:3, :3]).float().to(device)
    dirs_world = dirs_cam @ R_c2w.T
    return torch.nn.functional.normalize(dirs_world, p=2.0, dim=-1).permute(2, 0, 1).unsqueeze(0)

def get_visible_box_gaussians(means, opacities, R_w2c, t_w2c, fx, fy, cx, cy,
                               x_min, x_max, y_min, y_max, opa_thresh=0.06, subgrid_res=8):
    R_t = torch.from_numpy(R_w2c).float().to(means.device)
    t_t = torch.from_numpy(t_w2c).float().to(means.device)

    p_cam = (means @ R_t.T) + t_t
    z = p_cam[:, 2]
    valid_z = z > 0.1

    u = fx * (p_cam[:, 0] / z) + cx
    v = fy * (p_cam[:, 1] / z) + cy

    in_box = valid_z & (u >= x_min) & (u <= x_max) & (v >= y_min) & (v <= y_max)
    box_indices = torch.where(in_box)[0]

    if len(box_indices) == 0:
        return []

    box_z = z[box_indices]
    sort_order = torch.argsort(box_z)
    sorted_idx = box_indices[sort_order]

    sub_w = max(1.0, (x_max - x_min) / subgrid_res)
    sub_h = max(1.0, (y_max - y_min) / subgrid_res)
    T_grid = torch.ones((subgrid_res, subgrid_res), device=means.device)

    selected = []
    for idx in sorted_idx:
        u_i = u[idx].item()
        v_i = v[idx].item()
        gx = min(subgrid_res - 1, max(0, int((u_i - x_min) / sub_w)))
        gy = min(subgrid_res - 1, max(0, int((v_i - y_min) / sub_h)))

        alpha_raw = opacities[idx].item()
        T_cur = T_grid[gy, gx].item()
        alpha_eff = alpha_raw * T_cur

        if alpha_eff >= opa_thresh:
            selected.append((idx.item(), alpha_eff, z[idx].item()))

        T_grid[gy, gx] *= max(0.0, 1.0 - alpha_raw)

    return selected

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, default='flame_steak')
    parser.add_argument('--downscale', type=int, default=2)
    parser.add_argument('--save_rrd', type=str, default='visualizations/flame_steak_f0/flame_steak_f0_analysis.rrd')
    args = parser.parse_args()

    scene = args.scene
    downscale = args.downscale
    datasets_dir = 'datasets'
    results_dir = 'results'

    colmap_sparse_dir = os.path.join(datasets_dir, scene, 'colmap_0/sparse/0')
    cameras = read_colmap_cameras(colmap_sparse_dir, downscale=downscale)
    cam_names = sorted(list(cameras.keys()))
    print(f"Loaded {len(cameras)} cameras from {colmap_sparse_dir}")

    ply_colmap_path = f"{results_dir}/{scene}_colmap_init_STG_Strategy/Frame0-49/ckpts/ckpt_10999_rank0_frame0.ply"
    ply_da3_path = f"{results_dir}/{scene}_da3_importance_100k_init_STG_Strategy/Frame0-49/ckpts/ckpt_10999_rank0_frame0.ply"
    pt_colmap_path = f"{results_dir}/{scene}_colmap_init_STG_Strategy/Frame0-49/ckpts/ckpt_10999_rank0.pt"
    pt_da3_path = f"{results_dir}/{scene}_da3_importance_100k_init_STG_Strategy/Frame0-49/ckpts/ckpt_10999_rank0.pt"

    m_col, q_col, s_col, o_col, c_col = load_ply_gaussian(ply_colmap_path)
    m_da3, q_da3, s_da3, o_da3, c_da3 = load_ply_gaussian(ply_da3_path)
    col_pt = load_pt_model(pt_colmap_path)
    da3_pt = load_pt_model(pt_da3_path)

    os.makedirs(os.path.dirname(args.save_rrd), exist_ok=True)
    rr.init("flame_steak_quality_analysis_with_mlp", spawn=False)
    rr.save(args.save_rrd)

    # 1. Log Full Point Clouds
    rgb_col = sh_to_rgb(c_col[:, 0, :].cpu().numpy())
    rgb_da3 = sh_to_rgb(c_da3[:, 0, :].cpu().numpy())
    rr.log("world/3dgs/colmap_init_full", rr.Points3D(m_col.cpu().numpy(), colors=rgb_col, radii=0.006))
    rr.log("world/3dgs/da3_importance_full", rr.Points3D(m_da3.cpu().numpy(), colors=rgb_da3, radii=0.006))

    # 2. Render Cam00 & Dynamically Scan for MLP-based Error Patch
    cam0 = cameras["cam00"]
    W, H = cam0['width'], cam0['height']
    vmat0 = torch.from_numpy(cam0['w2c']).float().unsqueeze(0).cuda()
    K0 = torch.from_numpy(cam0['K']).float().unsqueeze(0).cuda()
    rays0 = compute_rays(cam0)

    # PLY renders
    rend_col0, _, _ = rasterization(m_col, q_col, s_col, o_col, c_col, vmat0, K0, W, H, sh_degree=0)
    rend_da30, _, _ = rasterization(m_da3, q_da3, s_da3, o_da3, c_da3, vmat0, K0, W, H, sh_degree=0)
    img_col0 = rend_col0[0].clamp(0, 1)
    img_da30 = rend_da30[0].clamp(0, 1)

    # MLP renders
    m_pt_c, q_pt_c, s_pt_c, o_pt_c, c_pt_c, dec_c = col_pt
    rend_pt_c0, _, _ = rasterization(m_pt_c, q_pt_c, s_pt_c, o_pt_c, c_pt_c, vmat0, K0, W, H, packed=False)
    with torch.no_grad():
        img_col0_mlp = dec_c(rend_pt_c0.permute(0, 3, 1, 2), rays0, 0.0)[0].permute(1, 2, 0).clamp(0, 1)

    m_pt_d, q_pt_d, s_pt_d, o_pt_d, c_pt_d, dec_d = da3_pt
    rend_pt_d0, _, _ = rasterization(m_pt_d, q_pt_d, s_pt_d, o_pt_d, c_pt_d, vmat0, K0, W, H, packed=False)
    with torch.no_grad():
        img_da30_mlp = dec_d(rend_pt_d0.permute(0, 3, 1, 2), rays0, 0.0)[0].permute(1, 2, 0).clamp(0, 1)

    gt_path0 = os.path.join(datasets_dir, scene, "cam00", "0.png")
    gt_pil0 = Image.open(gt_path0).resize((W, H), Image.BILINEAR)
    gt_img0 = torch.from_numpy(np.array(gt_pil0)).float().cuda() / 255.0

    psnr_full_c_mlp = -10.0 * torch.log10(torch.mean((img_col0_mlp - gt_img0)**2)).item()
    psnr_full_d_mlp = -10.0 * torch.log10(torch.mean((img_da30_mlp - gt_img0)**2)).item()
    print(f"Cam00 Full View PSNR (with MLP): Colmap = {psnr_full_c_mlp:.2f} dB | DA3 = {psnr_full_d_mlp:.2f} dB")

    # Dynamic Scan on MLP Renders
    margin_x = W // 4
    margin_y = H // 4
    patch_size = 80
    stride = 10

    err_col_mlp = (img_col0_mlp - gt_img0) ** 2
    err_da3_mlp = (img_da30_mlp - gt_img0) ** 2

    best_gap = -1e9
    best_box0 = None
    for y in range(margin_y, H - margin_y - patch_size, stride):
        for x in range(margin_x, W - margin_x - patch_size, stride):
            p_err_c = err_col_mlp[y:y+patch_size, x:x+patch_size].mean().item()
            p_err_d = err_da3_mlp[y:y+patch_size, x:x+patch_size].mean().item()
            p_psnr_c = -10.0 * np.log10(max(p_err_c, 1e-10))
            p_psnr_d = -10.0 * np.log10(max(p_err_d, 1e-10))
            gap = p_psnr_c - p_psnr_d
            if gap > best_gap:
                best_gap = gap
                best_box0 = (x, y, patch_size, patch_size, p_psnr_c, p_psnr_d)

    bx0, by0, bw0, bh0, p_c0_mlp, p_d0_mlp = best_box0
    print(f"\n🎯 [新重算 MLP 核心誤差區塊 (Cam00)]:")
    print(f"  位置: X=[{bx0}, {bx0+bw0}], Y=[{by0}, {by0+bh0}] (尺寸: {bw0}x{bh0})")
    print(f"  Colmap PSNR: {p_c0_mlp:.2f} dB | DA3 PSNR: {p_d0_mlp:.2f} dB | 品質差距: {best_gap:+.2f} dB")

    # Save visual comparison for new MLP patch
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    patch_gt = gt_img0[by0:by0+bh0, bx0:bx0+bw0].cpu().numpy()
    patch_col = img_col0_mlp[by0:by0+bh0, bx0:bx0+bw0].cpu().numpy()
    patch_da3 = img_da30_mlp[by0:by0+bh0, bx0:bx0+bw0].cpu().numpy()
    diff_map = np.abs(patch_col - patch_da3).mean(axis=-1)

    axes[0].imshow(patch_gt)
    axes[0].set_title("GT Image", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(patch_col)
    axes[1].set_title(f"COLMAP + MLP ({p_c0_mlp:.2f} dB)", fontsize=12, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(patch_da3)
    axes[2].set_title(f"DA3 + MLP ({p_d0_mlp:.2f} dB)", fontsize=12, fontweight='bold')
    axes[2].axis('off')

    im = axes[3].imshow(diff_map, cmap='magma', vmin=0, vmax=0.35)
    axes[3].set_title(f"|COLMAP - DA3| (Δ = {best_gap:+.2f} dB)", fontsize=12, fontweight='bold')
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    patch_img_path = "visualizations/flame_steak_f0/cam00_mlp_error_patch_breakdown.png"
    plt.savefig(patch_img_path, dpi=200)
    plt.close()

    # Extract Visible Gaussians in this new MLP box
    sel_gs_col0 = get_visible_box_gaussians(
        m_col, o_col, cam0['w2c'][:3, :3], cam0['w2c'][:3, 3],
        cam0['fx'], cam0['fy'], cam0['cx'], cam0['cy'],
        bx0, bx0 + bw0, by0, by0 + bh0, opa_thresh=0.06
    )
    sel_gs_da30 = get_visible_box_gaussians(
        m_da3, o_da3, cam0['w2c'][:3, :3], cam0['w2c'][:3, 3],
        cam0['fx'], cam0['fy'], cam0['cx'], cam0['cy'],
        bx0, bx0 + bw0, by0, by0 + bh0, opa_thresh=0.06
    )

    idx_col0 = [item[0] for item in sel_gs_col0]
    idx_da30 = [item[0] for item in sel_gs_da30]
    pts_col0 = m_col[idx_col0].cpu().numpy()
    pts_da30 = m_da3[idx_da30].cpu().numpy()
    pts_combined = np.vstack([pts_col0, pts_da30])
    print(f"  提取可見 3DGS 數量: COLMAP = {len(idx_col0)} 顆 | DA3 = {len(idx_da30)} 顆 (合併錨點: {len(pts_combined)} 顆)")

    cam_results = {}
    cam_results["cam00"] = {
        'visible': True,
        'box': (bx0, by0, bw0, bh0),
        'psnr_col': p_c0_mlp,
        'psnr_da3': p_d0_mlp,
        'diff': p_c0_mlp - p_d0_mlp,
        'col_idx': idx_col0,
        'da3_idx': idx_da30,
        'img_col': (img_col0.cpu().numpy() * 255).astype(np.uint8),
        'img_col_mlp': (img_col0_mlp.cpu().numpy() * 255).astype(np.uint8),
        'img_da3': (img_da30.cpu().numpy() * 255).astype(np.uint8),
        'img_da3_mlp': (img_da30_mlp.cpu().numpy() * 255).astype(np.uint8),
        'gt_img': (gt_img0.cpu().numpy() * 255).astype(np.uint8),
    }

    # 3. Multi-View Reprojection Across All Other Cameras
    for cam_name in cam_names:
        if cam_name == "cam00":
            continue

        cinfo = cameras[cam_name]
        vmat = torch.from_numpy(cinfo['w2c']).float().unsqueeze(0).cuda()
        K = torch.from_numpy(cinfo['K']).float().unsqueeze(0).cuda()
        rays = compute_rays(cinfo)

        # PLY
        rend_c, _, _ = rasterization(m_col, q_col, s_col, o_col, c_col, vmat, K, W, H, sh_degree=0)
        rend_d, _, _ = rasterization(m_da3, q_da3, s_da3, o_da3, c_da3, vmat, K, W, H, sh_degree=0)
        img_c = rend_c[0].clamp(0, 1)
        img_d = rend_d[0].clamp(0, 1)

        # MLP
        rend_pt_c, _, _ = rasterization(m_pt_c, q_pt_c, s_pt_c, o_pt_c, c_pt_c, vmat, K, W, H, packed=False)
        with torch.no_grad():
            img_c_mlp = dec_c(rend_pt_c.permute(0, 3, 1, 2), rays, 0.0)[0].permute(1, 2, 0).clamp(0, 1)

        rend_pt_d, _, _ = rasterization(m_pt_d, q_pt_d, s_pt_d, o_pt_d, c_pt_d, vmat, K, W, H, packed=False)
        with torch.no_grad():
            img_d_mlp = dec_d(rend_pt_d.permute(0, 3, 1, 2), rays, 0.0)[0].permute(1, 2, 0).clamp(0, 1)

        gt_p = os.path.join(datasets_dir, scene, cam_name, "0.png")
        gt_pil = Image.open(gt_p).resize((W, H), Image.BILINEAR)
        gt_t = torch.from_numpy(np.array(gt_pil)).float().cuda() / 255.0

        p_c_pts = (pts_combined @ cinfo['w2c'][:3, :3].T) + cinfo['w2c'][:3, 3]
        z_pts = p_c_pts[:, 2]

        in_front = z_pts > 0.1
        if in_front.sum() < (len(pts_combined) * 0.3):
            continue

        u_proj = cinfo['fx'] * (p_c_pts[in_front, 0] / z_pts[in_front]) + cinfo['cx']
        v_proj = cinfo['fy'] * (p_c_pts[in_front, 1] / z_pts[in_front]) + cinfo['cy']

        u_min = int(max(0, np.percentile(u_proj, 5) - 20))
        u_max = int(min(W - 1, np.percentile(u_proj, 95) + 20))
        v_min = int(max(0, np.percentile(v_proj, 5) - 20))
        v_max = int(min(H - 1, np.percentile(v_proj, 95) + 20))

        bw = u_max - u_min
        bh = v_max - v_min

        if bw < 20 or bh < 20:
            continue

        p_err_c = ((img_c_mlp[v_min:v_max, u_min:u_max] - gt_t[v_min:v_max, u_min:u_max])**2).mean().item()
        p_err_d = ((img_d_mlp[v_min:v_max, u_min:u_max] - gt_t[v_min:v_max, u_min:u_max])**2).mean().item()
        p_psnr_c = -10.0 * np.log10(max(p_err_c, 1e-10))
        p_psnr_d = -10.0 * np.log10(max(p_err_d, 1e-10))

        sel_c = get_visible_box_gaussians(
            m_col, o_col, cinfo['w2c'][:3, :3], cinfo['w2c'][:3, 3],
            cinfo['fx'], cinfo['fy'], cinfo['cx'], cinfo['cy'],
            u_min, u_max, v_min, v_max, opa_thresh=0.06
        )
        sel_d = get_visible_box_gaussians(
            m_da3, o_da3, cinfo['w2c'][:3, :3], cinfo['w2c'][:3, 3],
            cinfo['fx'], cinfo['fy'], cinfo['cx'], cinfo['cy'],
            u_min, u_max, v_min, v_max, opa_thresh=0.06
        )

        cam_results[cam_name] = {
            'visible': True,
            'box': (u_min, v_min, bw, bh),
            'psnr_col': p_psnr_c,
            'psnr_da3': p_psnr_d,
            'diff': p_psnr_c - p_psnr_d,
            'col_idx': [item[0] for item in sel_c],
            'da3_idx': [item[0] for item in sel_d],
            'img_col': (img_c.cpu().numpy() * 255).astype(np.uint8),
            'img_col_mlp': (img_c_mlp.cpu().numpy() * 255).astype(np.uint8),
            'img_da3': (img_d.cpu().numpy() * 255).astype(np.uint8),
            'img_da3_mlp': (img_d_mlp.cpu().numpy() * 255).astype(np.uint8),
            'gt_img': (gt_t.cpu().numpy() * 255).astype(np.uint8),
        }

    # 4. Log to Rerun
    print(f"Logging {len(cam_results)} cameras with newly recomputed MLP boxes to Rerun...")
    for cam_name, res in cam_results.items():
        if not res['visible']:
            continue

        cinfo = cameras[cam_name]
        c2w = cinfo['c2w']
        cam_t = c2w[:3, 3]
        cam_R = c2w[:3, :3]

        rr.log(
            f"world/cameras/{cam_name}",
            rr.Transform3D(
                translation=cam_t,
                mat3x3=cam_R,
                from_parent=False
            )
        )
        rr.log(
            f"world/cameras/{cam_name}",
            rr.Pinhole(
                resolution=[cinfo['width'], cinfo['height']],
                focal_length=[cinfo['fx'], cinfo['fy']],
                principal_point=[cinfo['cx'], cinfo['cy']]
            )
        )

        rr.log(f"world/cameras/{cam_name}/gt_image", rr.Image(res['gt_img']))
        rr.log(f"world/cameras/{cam_name}/render_colmap_diffuse", rr.Image(res['img_col']))
        rr.log(f"world/cameras/{cam_name}/render_colmap_with_mlp", rr.Image(res['img_col_mlp']))
        rr.log(f"world/cameras/{cam_name}/render_da3_diffuse", rr.Image(res['img_da3']))
        rr.log(f"world/cameras/{cam_name}/render_da3_with_mlp", rr.Image(res['img_da3_mlp']))

        bx, by, bw, bh = res['box']
        label_text = f"MLP: Col={res['psnr_col']:.1f}dB, DA3={res['psnr_da3']:.1f}dB (Δ: {res['diff']:+.1f}dB)"
        rr.log(
            f"world/cameras/{cam_name}/error_box2d",
            rr.Boxes2D(
                mins=[[bx, by]],
                sizes=[[bw, bh]],
                labels=[label_text],
                colors=[[255, 0, 50]]
            )
        )

        if len(res['col_idx']) > 0:
            sub_pts_c = m_col[res['col_idx']].cpu().numpy()
            rr.log(
                f"world/selected_3dgs/{cam_name}/colmap",
                rr.Points3D(sub_pts_c, colors=[[0, 230, 255]], radii=0.012)
            )

        if len(res['da3_idx']) > 0:
            sub_pts_d = m_da3[res['da3_idx']].cpu().numpy()
            rr.log(
                f"world/selected_3dgs/{cam_name}/da3",
                rr.Points3D(sub_pts_d, colors=[[255, 120, 0]], radii=0.012)
            )

    print(f"✅ Rerun database successfully updated at {args.save_rrd}")

if __name__ == '__main__':
    main()
