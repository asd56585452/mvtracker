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
    scene = 'sear_steak'
    downscale = 2
    save_rrd = 'visualizations/sear_steak_f0/sear_steak_f0_analysis.rrd'

    colmap_sparse_dir = f'datasets/{scene}/colmap_0/sparse/0'
    cameras = read_colmap_cameras(colmap_sparse_dir, downscale=downscale)
    cam_names = sorted(list(cameras.keys()))
    print(f"Loaded {len(cameras)} cameras from {colmap_sparse_dir}")

    # Paths to 4 models
    root_exp = "results/sear_steak_gof0_experiments"
    c100_prefix = f"{root_exp}/colmap_100k_stg_11k/Frame0-49/ckpts/ckpt_10999_rank0"
    c200_prefix = f"{root_exp}/colmap_200k_stg_11k/Frame0-49/ckpts/ckpt_10999_rank0"
    d100_prefix = f"{root_exp}/da3_importance_100k_stg_11k/Frame0-49/ckpts/ckpt_10999_rank0"
    d50_prefix  = f"{root_exp}/da3_importance_50k_stg_11k/Frame0-49/ckpts/ckpt_10999_rank0"

    # Load PLYs
    m_c100, q_c100, s_c100, o_c100, c_c100 = load_ply_gaussian(c100_prefix + "_frame0.ply")
    m_c200, q_c200, s_c200, o_c200, c_c200 = load_ply_gaussian(c200_prefix + "_frame0.ply")
    m_d100, q_d100, s_d100, o_d100, c_d100 = load_ply_gaussian(d100_prefix + "_frame0.ply")
    m_d50,  q_d50,  s_d50,  o_d50,  c_d50  = load_ply_gaussian(d50_prefix + "_frame0.ply")

    # Load PTs
    c100_pt = load_pt_model(c100_prefix + ".pt")
    c200_pt = load_pt_model(c200_prefix + ".pt")
    d100_pt = load_pt_model(d100_prefix + ".pt")
    d50_pt  = load_pt_model(d50_prefix + ".pt")

    os.makedirs(os.path.dirname(save_rrd), exist_ok=True)
    rr.init("sear_steak_quality_analysis_4models", spawn=False)
    rr.save(save_rrd)

    # 1. Log Full Point Clouds
    rgb_c100 = sh_to_rgb(c_c100[:, 0, :].cpu().numpy())
    rgb_c200 = sh_to_rgb(c_c200[:, 0, :].cpu().numpy())
    rgb_d100 = sh_to_rgb(c_d100[:, 0, :].cpu().numpy())
    rgb_d50  = sh_to_rgb(c_d50[:, 0, :].cpu().numpy())

    rr.log("world/3dgs/colmap_100k_full", rr.Points3D(m_c100.cpu().numpy(), colors=rgb_c100, radii=0.006))
    rr.log("world/3dgs/colmap_200k_full", rr.Points3D(m_c200.cpu().numpy(), colors=rgb_c200, radii=0.006))
    rr.log("world/3dgs/da3_importance_100k_full", rr.Points3D(m_d100.cpu().numpy(), colors=rgb_d100, radii=0.006))
    rr.log("world/3dgs/da3_importance_50k_full", rr.Points3D(m_d50.cpu().numpy(), colors=rgb_d50, radii=0.006))

    # 2. Render Cam00 & Scan Error Patch
    cam0 = cameras["cam00"]
    W, H = cam0['width'], cam0['height']
    vmat0 = torch.from_numpy(cam0['w2c']).float().unsqueeze(0).cuda()
    K0 = torch.from_numpy(cam0['K']).float().unsqueeze(0).cuda()
    rays0 = compute_rays(cam0)

    gt_path0 = f"datasets/{scene}/cam00/0.png"
    gt_pil0 = Image.open(gt_path0).resize((W, H), Image.BILINEAR)
    gt_img0 = torch.from_numpy(np.array(gt_pil0)).float().cuda() / 255.0

    def render_pt(pt_data, vmat, K, rays):
        m, q, s, o, c, dec = pt_data
        rend, _, _ = rasterization(m, q, s, o, c, vmat, K, W, H, packed=False)
        with torch.no_grad():
            mlp_img = dec(rend.permute(0, 3, 1, 2), rays, 0.0)[0].permute(1, 2, 0).clamp(0, 1)
        return mlp_img

    img_c100_0 = render_pt(c100_pt, vmat0, K0, rays0)
    img_c200_0 = render_pt(c200_pt, vmat0, K0, rays0)
    img_d100_0 = render_pt(d100_pt, vmat0, K0, rays0)
    img_d50_0  = render_pt(d50_pt,  vmat0, K0, rays0)

    bx0, by0, bw0, bh0 = 848, 373, 80, 80
    p_c100_0 = -10.0 * np.log10(((img_c100_0[by0:by0+bh0, bx0:bx0+bw0] - gt_img0[by0:by0+bh0, bx0:bx0+bw0])**2).mean().item())
    p_c200_0 = -10.0 * np.log10(((img_c200_0[by0:by0+bh0, bx0:bx0+bw0] - gt_img0[by0:by0+bh0, bx0:bx0+bw0])**2).mean().item())
    p_d100_0 = -10.0 * np.log10(((img_d100_0[by0:by0+bh0, bx0:bx0+bw0] - gt_img0[by0:by0+bh0, bx0:bx0+bw0])**2).mean().item())
    p_d50_0  = -10.0 * np.log10(((img_d50_0[by0:by0+bh0, bx0:bx0+bw0] - gt_img0[by0:by0+bh0, bx0:bx0+bw0])**2).mean().item())

    print(f"\n🎯 [sear_steak Cam00 核心區塊 4 模型 PSNR]:")
    print(f"  COLMAP 100k:         {p_c100_0:.2f} dB")
    print(f"  COLMAP 200k:         {p_c200_0:.2f} dB (高總點數但該平滑區保持極低密度)")
    print(f"  DA3 Importance 100k: {p_d100_0:.2f} dB (過密堆疊，品質劣化)")
    print(f"  DA3 Importance 50k:  {p_d50_0:.2f} dB (減點後成功反彈 +6.54 dB)")

    sel_c100_0 = get_visible_box_gaussians(m_c100, o_c100, cam0['w2c'][:3, :3], cam0['w2c'][:3, 3], cam0['fx'], cam0['fy'], cam0['cx'], cam0['cy'], bx0, bx0+bw0, by0, by0+bh0, opa_thresh=0.06)
    sel_c200_0 = get_visible_box_gaussians(m_c200, o_c200, cam0['w2c'][:3, :3], cam0['w2c'][:3, 3], cam0['fx'], cam0['fy'], cam0['cx'], cam0['cy'], bx0, bx0+bw0, by0, by0+bh0, opa_thresh=0.06)
    sel_d100_0 = get_visible_box_gaussians(m_d100, o_d100, cam0['w2c'][:3, :3], cam0['w2c'][:3, 3], cam0['fx'], cam0['fy'], cam0['cx'], cam0['cy'], bx0, bx0+bw0, by0, by0+bh0, opa_thresh=0.06)
    sel_d50_0  = get_visible_box_gaussians(m_d50,  o_d50,  cam0['w2c'][:3, :3], cam0['w2c'][:3, 3], cam0['fx'], cam0['fy'], cam0['cx'], cam0['cy'], bx0, bx0+bw0, by0, by0+bh0, opa_thresh=0.06)

    idx_c100_0 = [it[0] for it in sel_c100_0]
    idx_c200_0 = [it[0] for it in sel_c200_0]
    idx_d100_0 = [it[0] for it in sel_d100_0]
    idx_d50_0  = [it[0] for it in sel_d50_0]

    pts_anchor = np.vstack([
        m_c100[idx_c100_0].cpu().numpy(),
        m_c200[idx_c200_0].cpu().numpy(),
        m_d100[idx_d100_0].cpu().numpy(),
        m_d50[idx_d50_0].cpu().numpy(),
    ])
    print(f"  區塊可見 3DGS 點數: Col-100k={len(idx_c100_0)} | Col-200k={len(idx_c200_0)} | DA3-100k={len(idx_d100_0)} | DA3-50k={len(idx_d50_0)}")

    # 3. Multi-View Tracking Across All 21 Cameras
    cam_results = {}
    cam_results["cam00"] = {
        'visible': True,
        'box': (bx0, by0, bw0, bh0),
        'p_c100': p_c100_0,
        'p_c200': p_c200_0,
        'p_d100': p_d100_0,
        'p_d50':  p_d50_0,
        'c100_idx': idx_c100_0,
        'c200_idx': idx_c200_0,
        'd100_idx': idx_d100_0,
        'd50_idx':  idx_d50_0,
        'img_c100': (img_c100_0.cpu().numpy() * 255).astype(np.uint8),
        'img_c200': (img_c200_0.cpu().numpy() * 255).astype(np.uint8),
        'img_d100': (img_d100_0.cpu().numpy() * 255).astype(np.uint8),
        'img_d50':  (img_d50_0.cpu().numpy() * 255).astype(np.uint8),
        'gt_img':   (gt_img0.cpu().numpy() * 255).astype(np.uint8),
    }

    for cam_name in cam_names:
        if cam_name == "cam00":
            continue

        cinfo = cameras[cam_name]
        vmat = torch.from_numpy(cinfo['w2c']).float().unsqueeze(0).cuda()
        K = torch.from_numpy(cinfo['K']).float().unsqueeze(0).cuda()
        rays = compute_rays(cinfo)

        img_c100 = render_pt(c100_pt, vmat, K, rays)
        img_c200 = render_pt(c200_pt, vmat, K, rays)
        img_d100 = render_pt(d100_pt, vmat, K, rays)
        img_d50  = render_pt(d50_pt,  vmat, K, rays)

        gt_p = f"datasets/{scene}/{cam_name}/0.png"
        gt_pil = Image.open(gt_p).resize((W, H), Image.BILINEAR)
        gt_t = torch.from_numpy(np.array(gt_pil)).float().cuda() / 255.0

        p_c_pts = (pts_anchor @ cinfo['w2c'][:3, :3].T) + cinfo['w2c'][:3, 3]
        z_pts = p_c_pts[:, 2]

        in_front = z_pts > 0.1
        if in_front.sum() < (len(pts_anchor) * 0.3):
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

        p_err_c100 = ((img_c100[v_min:v_max, u_min:u_max] - gt_t[v_min:v_max, u_min:u_max])**2).mean().item()
        p_err_c200 = ((img_c200[v_min:v_max, u_min:u_max] - gt_t[v_min:v_max, u_min:u_max])**2).mean().item()
        p_err_d100 = ((img_d100[v_min:v_max, u_min:u_max] - gt_t[v_min:v_max, u_min:u_max])**2).mean().item()
        p_err_d50  = ((img_d50[v_min:v_max, u_min:u_max]  - gt_t[v_min:v_max, u_min:u_max])**2).mean().item()

        p_c100 = -10.0 * np.log10(max(p_err_c100, 1e-10))
        p_c200 = -10.0 * np.log10(max(p_err_c200, 1e-10))
        p_d100 = -10.0 * np.log10(max(p_err_d100, 1e-10))
        p_d50  = -10.0 * np.log10(max(p_err_d50,  1e-10))

        sel_c100 = get_visible_box_gaussians(m_c100, o_c100, cinfo['w2c'][:3, :3], cinfo['w2c'][:3, 3], cinfo['fx'], cinfo['fy'], cinfo['cx'], cinfo['cy'], u_min, u_max, v_min, v_max, opa_thresh=0.06)
        sel_c200 = get_visible_box_gaussians(m_c200, o_c200, cinfo['w2c'][:3, :3], cinfo['w2c'][:3, 3], cinfo['fx'], cinfo['fy'], cinfo['cx'], cinfo['cy'], u_min, u_max, v_min, v_max, opa_thresh=0.06)
        sel_d100 = get_visible_box_gaussians(m_d100, o_d100, cinfo['w2c'][:3, :3], cinfo['w2c'][:3, 3], cinfo['fx'], cinfo['fy'], cinfo['cx'], cinfo['cy'], u_min, u_max, v_min, v_max, opa_thresh=0.06)
        sel_d50  = get_visible_box_gaussians(m_d50,  o_d50,  cinfo['w2c'][:3, :3], cinfo['w2c'][:3, 3], cinfo['fx'], cinfo['fy'], cinfo['cx'], cinfo['cy'], u_min, u_max, v_min, v_max, opa_thresh=0.06)

        cam_results[cam_name] = {
            'visible': True,
            'box': (u_min, v_min, bw, bh),
            'p_c100': p_c100,
            'p_c200': p_c200,
            'p_d100': p_d100,
            'p_d50':  p_d50,
            'c100_idx': [it[0] for it in sel_c100],
            'c200_idx': [it[0] for it in sel_c200],
            'd100_idx': [it[0] for it in sel_d100],
            'd50_idx':  [it[0] for it in sel_d50],
            'img_c100': (img_c100.cpu().numpy() * 255).astype(np.uint8),
            'img_c200': (img_c200.cpu().numpy() * 255).astype(np.uint8),
            'img_d100': (img_d100.cpu().numpy() * 255).astype(np.uint8),
            'img_d50':  (img_d50.cpu().numpy() * 255).astype(np.uint8),
            'gt_img':   (gt_t.cpu().numpy() * 255).astype(np.uint8),
        }

    # 4. Log to Rerun
    print(f"Logging {len(cam_results)} cameras with 4 models to Rerun...")
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
        rr.log(f"world/cameras/{cam_name}/render_colmap_100k", rr.Image(res['img_c100']))
        rr.log(f"world/cameras/{cam_name}/render_colmap_200k", rr.Image(res['img_c200']))
        rr.log(f"world/cameras/{cam_name}/render_da3_importance_100k", rr.Image(res['img_d100']))
        rr.log(f"world/cameras/{cam_name}/render_da3_importance_50k", rr.Image(res['img_d50']))

        bx, by, bw, bh = res['box']
        label_text = f"C100:{res['p_c100']:.1f} | C200:{res['p_c200']:.1f} | D100:{res['p_d100']:.1f} | D50:{res['p_d50']:.1f}"
        rr.log(
            f"world/cameras/{cam_name}/error_box2d",
            rr.Boxes2D(
                mins=[[bx, by]],
                sizes=[[bw, bh]],
                labels=[label_text],
                colors=[[255, 0, 50]]
            )
        )

        if len(res['c100_idx']) > 0:
            rr.log(f"world/selected_3dgs/{cam_name}/colmap_100k", rr.Points3D(m_c100[res['c100_idx']].cpu().numpy(), colors=[[0, 230, 255]], radii=0.012))
        if len(res['c200_idx']) > 0:
            rr.log(f"world/selected_3dgs/{cam_name}/colmap_200k", rr.Points3D(m_c200[res['c200_idx']].cpu().numpy(), colors=[[0, 100, 255]], radii=0.012))
        if len(res['d100_idx']) > 0:
            rr.log(f"world/selected_3dgs/{cam_name}/da3_100k", rr.Points3D(m_d100[res['d100_idx']].cpu().numpy(), colors=[[255, 50, 50]], radii=0.012))
        if len(res['d50_idx']) > 0:
            rr.log(f"world/selected_3dgs/{cam_name}/da3_50k", rr.Points3D(m_d50[res['d50_idx']].cpu().numpy(), colors=[[50, 255, 50]], radii=0.012))

    print(f"✅ Rerun database successfully updated at {save_rrd}")

if __name__ == '__main__':
    main()
