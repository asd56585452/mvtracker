import os, sys, math, struct
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from gsplat.rendering import rasterization

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

def prepare_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cuda')
    decoder = Sandwich(9, 3, ray_dim=3, bias=False).cuda()
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
    pointtimes = torch.ones((means.shape[0], 1), dtype=means.dtype, device='cuda')
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

def main():
    cameras = read_colmap_cameras('datasets/flame_steak/colmap_0/sparse/0', downscale=2)
    cam_names = sorted(list(cameras.keys()))

    col_data = prepare_model('results/flame_steak_colmap_init_STG_Strategy/Frame0-49/ckpts/ckpt_10999_rank0.pt')
    da3_data = prepare_model('results/flame_steak_da3_importance_100k_init_STG_Strategy/Frame0-49/ckpts/ckpt_10999_rank0.pt')

    results = []

    for cname in cam_names:
        cam = cameras[cname]
        W, H = cam['width'], cam['height']
        c2w = cam['c2w']
        w2c = cam['w2c']
        vmat = torch.from_numpy(w2c).float().unsqueeze(0).cuda()
        K = torch.from_numpy(cam['K']).float().unsqueeze(0).cuda()

        fx, fy, cx, cy = cam['fx'], cam['fy'], cam['cx'], cam['cy']
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device='cuda') + 0.5,
            torch.arange(W, dtype=torch.float32, device='cuda') + 0.5,
            indexing='ij'
        )
        dirs_cam = torch.stack([(x_grid - cx) / fx, (y_grid - cy) / fy, torch.ones_like(x_grid)], dim=-1)
        R_c2w = torch.from_numpy(c2w[:3, :3]).float().cuda()
        dirs_world = dirs_cam @ R_c2w.T
        ray_tensor = torch.nn.functional.normalize(dirs_world, p=2.0, dim=-1).permute(2, 0, 1).unsqueeze(0)

        gt_path = f'datasets/flame_steak/{cname}/0.png'
        gt_pil = Image.open(gt_path).resize((W, H), Image.BILINEAR)
        gt_img = torch.from_numpy(np.array(gt_pil)).float().cuda() / 255.0

        # COLMAP
        m, q, s, o, c, dec = col_data
        rend_col, _, _ = rasterization(m, q, s, o, c, vmat, K, W, H, packed=False)
        with torch.no_grad():
            mlp_col = dec(rend_col.permute(0, 3, 1, 2), ray_tensor, 0.0)[0].permute(1, 2, 0).clamp(0, 1)
        psnr_c_mlp = -10.0 * torch.log10(torch.mean((mlp_col - gt_img)**2)).item()
        img_c_nomlp = torch.sigmoid(rend_col[..., :3])[0].clamp(0, 1)
        psnr_c_nomlp = -10.0 * torch.log10(torch.mean((img_c_nomlp - gt_img)**2)).item()

        # DA3
        m, q, s, o, c, dec = da3_data
        rend_da3, _, _ = rasterization(m, q, s, o, c, vmat, K, W, H, packed=False)
        with torch.no_grad():
            mlp_da3 = dec(rend_da3.permute(0, 3, 1, 2), ray_tensor, 0.0)[0].permute(1, 2, 0).clamp(0, 1)
        psnr_d_mlp = -10.0 * torch.log10(torch.mean((mlp_da3 - gt_img)**2)).item()
        img_d_nomlp = torch.sigmoid(rend_da3[..., :3])[0].clamp(0, 1)
        psnr_d_nomlp = -10.0 * torch.log10(torch.mean((img_d_nomlp - gt_img)**2)).item()

        results.append({
            'cam': cname,
            'c_nomlp': psnr_c_nomlp,
            'c_mlp': psnr_c_mlp,
            'd_nomlp': psnr_d_nomlp,
            'd_mlp': psnr_d_mlp
        })

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    x = np.arange(len(cam_names))
    width = 0.2

    ax1.bar(x - 1.5*width, [r['c_nomlp'] for r in results], width, label='COLMAP (No MLP)', color='#90caf9', alpha=0.85)
    ax1.bar(x - 0.5*width, [r['c_mlp'] for r in results], width, label='COLMAP (With MLP Decoder)', color='#1976d2')
    ax1.bar(x + 0.5*width, [r['d_nomlp'] for r in results], width, label='DA3 Importance (No MLP)', color='#ffcc80', alpha=0.85)
    ax1.bar(x + 1.5*width, [r['d_mlp'] for r in results], width, label='DA3 Importance (With MLP Decoder)', color='#f57c00')

    ax1.set_ylabel('PSNR (dB)', fontsize=13, fontweight='bold')
    ax1.set_title('Flame Steak Frame 0: View-Dependent MLP Impact Across All 21 Cameras', fontsize=15, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{c} (Test)" if c == 'cam00' else c for c in cam_names], rotation=45, ha='right', fontsize=10)
    ax1.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.set_ylim(24, 40)

    # Highlight Test vs Train
    col_mlp_all = [r['c_mlp'] for r in results]
    da3_mlp_all = [r['d_mlp'] for r in results]
    gap = np.array(col_mlp_all) - np.array(da3_mlp_all)

    colors_gap = ['#2e7d32' if g > 0 else '#c62828' for g in gap]
    bars = ax2.bar(x, gap, color=colors_gap, width=0.5, edgecolor='black', linewidth=0.8)
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_ylabel('PSNR Advantage: COLMAP - DA3 (dB)', fontsize=13, fontweight='bold')
    ax2.set_title('Generalization (Cam00 Test) vs Overfitting (Cam01-20 Train Views)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{c}\n[TEST]" if c == 'cam00' else c for c in cam_names], rotation=45, ha='right', fontsize=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)

    for bar, val in zip(bars, gap):
        y_pos = bar.get_height() + (0.15 if val >= 0 else -0.45)
        ax2.text(bar.get_x() + bar.get_width()/2.0, y_pos, f"{val:+.1f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    out_img = 'visualizations/flame_steak_f0/mlp_rendering_comparison_21views.png'
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    plt.savefig(out_img, dpi=200)
    print(f"Chart saved to {out_img}")

if __name__ == '__main__':
    main()
