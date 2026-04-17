import logging
import os
import pickle
import sys
from contextlib import ExitStack
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn.functional import interpolate
from torch.utils.data import Dataset
from torchvision import transforms as TF
from tqdm import tqdm

from mvtracker.datasets.utils import Datapoint, transform_scene, align_umeyama, apply_sim3_to_extrinsics


class GenericSceneDataset(Dataset):
    def __init__(
            self,
            dataset_dir,

            use_duster_depths=True,
            use_vggt_depths_with_aligned_cameras=False,
            use_vggt_depths_with_raw_cameras=False,
            use_monofusion_depths=False,
            use_moge2_depths=False,

            skip_depth_computation_if_cached=True,

            drop_first_n_frames=0,

            scene_normalization_mode="auto",  # "auto" | "manual" | "none"
            scene_normalization_auto_conf_thresh=4.8,
            scene_normalization_auto_target_radius=6.3,
            scene_normalization_auto_rescale_by_camera_radius=True,
            scene_normalization_manual_scale=None,  # Optional float
            scene_normalization_manual_rotation=None,  # Optional 3x3 torch.Tensor rotation matrix
            scene_normalization_manual_translation=None,  # Optional 3D torch.Tensor post-scale translation vector
            # E.g., the manual transform that translates up by 1.4 units and scales 2.5 times (was good for EgoExo4D):
            #   scale = 2.5
            #   translate_x = 0
            #   translate_y = 0
            #   translate_z = 1.4 * scale
            #   T = torch.tensor([
            #       [scale, 0.0, 0.0, translate_x],
            #       [0.0, scale, 0.0, translate_y],
            #       [0.0, 0.0, scale, translate_z],
            #       [0.0, 0.0, 0.0, 1.0],
            #   ], dtype=torch.float32)

            stream_viz_to_rerun=False,
    ):
        self.dataset_dir = dataset_dir

        self.use_duster_depths = use_duster_depths
        self.use_vggt_depths_with_aligned_cameras = use_vggt_depths_with_aligned_cameras
        self.use_vggt_depths_with_raw_cameras = use_vggt_depths_with_raw_cameras
        self.use_monofusion_depths = use_monofusion_depths
        self.use_moge2_depths = use_moge2_depths
        # --- Assert exclusive depth-source configuration ---
        # Exactly 0 or 1 of these should be True. (0 => fall back to pkl/dust3r.)
        depth_flags = (int(self.use_duster_depths)
                       + int(self.use_vggt_depths_with_aligned_cameras)
                       + int(self.use_vggt_depths_with_raw_cameras)
                       + int(self.use_monofusion_depths)
                       + int(self.use_moge2_depths))
        assert depth_flags <= 1, (
            "Misconfigured dataset: choose at most one depth source among "
            "`use_monofusion_depths`, `use_moge2_depths`, `use_duster_depths`."
        )

        self.skip_depth_computation_if_cached = skip_depth_computation_if_cached
        self.drop_first_n_frames = drop_first_n_frames

        self.scene_normalization_mode = scene_normalization_mode
        self.scene_normalization_auto_conf_thresh = scene_normalization_auto_conf_thresh
        self.scene_normalization_auto_target_radius = scene_normalization_auto_target_radius
        self.scene_normalization_auto_rescale_by_camera_radius = scene_normalization_auto_rescale_by_camera_radius
        self.scene_normalization_manual_scale = scene_normalization_manual_scale
        self.scene_normalization_manual_rotation = scene_normalization_manual_rotation
        self.scene_normalization_manual_translation = scene_normalization_manual_translation

        self.stream_viz_to_rerun = stream_viz_to_rerun

        self.seq_names = sorted([
            f.replace(".pkl", "")
            for f in os.listdir(dataset_dir)
            if f.endswith(".pkl")
        ])
        assert self.seq_names, f"No sequences found in {dataset_dir}"

    def __len__(self):
        return len(self.seq_names)

    def __getitem__(self, idx):
        seq_name = self.seq_names[idx]
        pkl_path = os.path.join(self.dataset_dir, f"{seq_name}.pkl")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        ego_cam = data.get("ego_cam_name", None)
        rgbs_dict = data["rgbs"]
        intrs_dict = data["intrs"]
        extrs_dict = data["extrs"]
        depths_dict = data.get("depths", None)

        if ego_cam:
            rgbs_dict.pop(ego_cam)
            intrs_dict.pop(ego_cam)
            extrs_dict.pop(ego_cam)
            if depths_dict is not None:
                depths_dict.pop(ego_cam)

        cam_names = sorted(rgbs_dict.keys())
        n_views = len(cam_names)
        n_frames, _, H, W = rgbs_dict[cam_names[0]].shape

        rgbs = torch.stack([torch.from_numpy(rgbs_dict[cam]) for cam in cam_names])  # [V, T, 3, H, W]
        intrs = torch.stack([torch.from_numpy(intrs_dict[cam]) for cam in cam_names])  # [V, 3, 3]
        intrs = intrs[:, None].expand(-1, n_frames, -1, -1)  # [V, T, 3, 3]

        extr_list = []
        for cam in cam_names:
            e = extrs_dict[cam]
            if e.ndim == 2:
                e = np.broadcast_to(e[None, ...], (n_frames, 3, 4))
            extr_list.append(torch.from_numpy(e.copy()))
        extrs = torch.stack(extr_list)  # [V, T, 3, 4]

        # ------- Depth selection & caching -------
        if self.use_duster_depths:
            depth_root = os.path.join(self.dataset_dir, f"duster_depths__{seq_name}")
            if not os.path.exists(os.path.join(depth_root, f"3d_model__{n_frames - 1:05d}__scene.npz")):
                if "../duster" not in sys.path:
                    sys.path.insert(0, "../duster")
                from scripts.egoexo4d_preprocessing import main_estimate_duster_depth
                pkl_path = os.path.join(self.dataset_dir, f"{seq_name}.pkl")

                # Re-enable autograd locally (overrides any surrounding no_grad/inference_mode)
                with ExitStack() as stack:
                    stack.enter_context(torch.inference_mode(False))
                    stack.enter_context(torch.enable_grad())
                    main_estimate_duster_depth(pkl_path, depth_root, self.skip_depth_computation_if_cached)
            duster_depths, duster_confs = [], []
            for t in range(n_frames):
                scene_path = os.path.join(depth_root, f"3d_model__{t:05d}__scene.npz")
                scene = np.load(scene_path)
                d = torch.from_numpy(scene["depths"])  # [V, H', W']
                d = interpolate(d[:, None], size=(H, W), mode="nearest")  # [V, 1, H, W]
                duster_depths.append(d)
                c = torch.from_numpy(scene["confs"])
                c = interpolate(c[:, None], size=(H, W), mode="nearest")
                duster_confs.append(c)
            depths = torch.stack(duster_depths, dim=1)  # [V, T, 1, H, W]
            depth_confs = torch.stack(duster_confs, dim=1)

        elif self.use_vggt_depths_with_aligned_cameras:
            depths, depth_confs, intrs, extrs = _ensure_vggt_aligned_cache_and_load(
                rgbs=rgbs,
                seq_name=seq_name,
                dataset_root=self.dataset_dir,
                extrs_gt=extrs,  # your current GT world->cam
                vggt_cache_subdir="vggt_cache",
                skip_if_cached=self.skip_depth_computation_if_cached,
                model_id="facebook/VGGT-1B",
            )

        elif self.use_vggt_depths_with_raw_cameras:
            # Only use VGGT’s own (raw) cameras and depths
            depths, depth_confs, intrs, extrs = _ensure_vggt_raw_cache_and_load(
                rgbs=rgbs,
                seq_name=seq_name,
                dataset_root=self.dataset_dir,
                vggt_cache_subdir="vggt_cache",
                skip_if_cached=self.skip_depth_computation_if_cached,
                model_id="facebook/VGGT-1B",
            )

        elif self.use_monofusion_depths:
            # MonoFusion (Dust3r + FG/BG-heuristic + MoGE-2) with caching
            final_depths, final_confs = _ensure_monofusion_cache_and_load(
                rgbs=rgbs,
                seq_name=seq_name,
                dataset_root=self.dataset_dir,
                monofusion_cache_subdir="monofusion_cache",
                skip_if_cached=self.skip_depth_computation_if_cached,
            )
            depths = final_depths
            depth_confs = final_confs

        elif self.use_moge2_depths:
            # Raw MoGe-2 (metric) with caching
            depths, depth_confs = _ensure_moge2_cache_and_load(
                rgbs=rgbs,
                seq_name=seq_name,
                dataset_root=self.dataset_dir,
                moge2_cache_subdir="moge2_cache",
                skip_if_cached=self.skip_depth_computation_if_cached,
            )

        elif depths_dict is not None:
            depths = torch.stack([torch.from_numpy(depths_dict[cam]) for cam in cam_names]).unsqueeze(2)
            depth_confs = depths.new_zeros(depths.shape)
            depth_confs[depths > 0] = 1000

        else:
            raise ValueError("No depths available/configured")

        # Sometimes the first frames are noisy, e.g., due to timesync calibration
        if self.drop_first_n_frames:
            assert type(self.drop_first_n_frames) == int
            n_frames -= self.drop_first_n_frames
            rgbs = rgbs[:, self.drop_first_n_frames:]
            depths = depths[:, self.drop_first_n_frames:]
            depth_confs = depth_confs[:, self.drop_first_n_frames:]
            intrs = intrs[:, self.drop_first_n_frames:]
            extrs = extrs[:, self.drop_first_n_frames:]

        if self.scene_normalization_mode == "auto":
            scale, translation = compute_auto_scene_normalization(
                depths, depth_confs, extrs, intrs,
                conf_thresh=self.scene_normalization_auto_conf_thresh,
                target_radius=self.scene_normalization_auto_target_radius,
                rescale_by_camera_radius=self.scene_normalization_auto_rescale_by_camera_radius,
            )
            rot = torch.eye(3, dtype=torch.float32, device=depths.device)
        elif self.scene_normalization_mode == "manual":
            assert self.scene_normalization_manual_scale is not None
            assert self.scene_normalization_manual_rotation is not None
            assert self.scene_normalization_manual_translation is not None
            scale = self.scene_normalization_manual_scale
            rot = self.scene_normalization_manual_rotation.to(depths.device)
            translation = self.scene_normalization_manual_translation.to(depths.device)
        elif self.scene_normalization_mode == "none":
            scale = 1.0
            rot = torch.eye(3, dtype=torch.float32, device=depths.device)
            translation = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=depths.device)
        else:
            raise ValueError(f"Unknown scene_normalization_mode: {self.scene_normalization_mode}")

        depths_trans, extrs_trans, _, _, _ = transform_scene(scale, rot, translation, depths, extrs, None, None, None)

        assert rgbs.shape == (n_views, n_frames, 3, H, W)
        assert depths.shape == (n_views, n_frames, 1, H, W)
        assert depth_confs.shape == (n_views, n_frames, 1, H, W)
        assert intrs.shape == (n_views, n_frames, 3, 3)
        assert extrs.shape == (n_views, n_frames, 3, 4)
        assert extrs_trans.shape == (n_views, n_frames, 3, 4)

        if self.stream_viz_to_rerun:
            import rerun as rr
            from mvtracker.utils.visualizer_rerun import log_pointclouds_to_rerun
            rr.init(f"3dpt", recording_id="v0.16")
            rr.connect_tcp()
            log_pointclouds_to_rerun(f"generic-1-before-norm", idx, rgbs[None], depths[None],
                                     intrs[None], extrs[None], depth_confs[None], [1.0])
            log_pointclouds_to_rerun(f"generic-2-after-norm", idx, rgbs[None], depths[None],
                                     intrs[None], extrs_trans[None], depth_confs[None], [1.0])

        datapoint = Datapoint(
            video=rgbs.float(),
            videodepth=depths_trans.float(),
            videodepthconf=depth_confs.float(),
            feats=None,
            segmentation=torch.ones((n_views, n_frames, 1, H, W), dtype=torch.float32),
            trajectory=None,
            trajectory_3d=None,
            visibility=None,
            valid=None,
            seq_name=seq_name,
            intrs=intrs.float(),
            extrs=extrs_trans.float(),
            query_points=None,
            query_points_3d=None,
            trajectory_category=None,
            track_upscaling_factor=1.0,
            novel_video=None,
            novel_intrs=None,
            novel_extrs=None,
        )

        return datapoint, True


def compute_auto_scene_normalization(
        depths,
        depth_confs,
        extrs,
        intrs,
        conf_thresh=4.8,
        target_radius=6.3,
        rescale_by_camera_radius=True,
):
    V, T, _, H, W = depths.shape
    device = depths.device

    extrs_square = torch.eye(4, device=device)[None, None].repeat(V, T, 1, 1)
    extrs_square[:, :, :3, :] = extrs
    extrs_inv = torch.inverse(extrs_square.float())
    intrs_inv = torch.inverse(intrs.float())

    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    homog = torch.stack([x, y, torch.ones_like(x)], dim=-1).reshape(-1, 3).float()
    homog = homog[None].expand(V, -1, -1)

    pts_all = []
    for v in range(V):
        d = depths[v, 0, 0]
        c = depth_confs[v, 0, 0]
        mask = (c > conf_thresh) & (d > 0)
        if mask.sum() < 100:
            continue

        d_flat = d.flatten()
        conf_mask = mask.flatten()
        intr_inv = intrs_inv[v, 0]
        extr_inv = extrs_inv[v, 0]

        cam_pts = (intr_inv @ homog[v].T).T * d_flat[:, None]
        cam_pts = cam_pts[conf_mask]
        cam_pts_h = torch.cat([cam_pts, torch.ones_like(cam_pts[:, :1])], dim=-1)
        world_pts = (extr_inv @ cam_pts_h.T).T[:, :3]

        pts_all.append(world_pts)

    pts_all = torch.cat(pts_all, dim=0)
    if pts_all.shape[0] < 100:
        raise RuntimeError("Too few valid points for normalization.")

    # --- Center scene ---
    centroid = pts_all.mean(dim=0)
    pts_centered = pts_all - centroid

    # --- Lift scene so floor is at z=0 ---
    floor_z = pts_centered[:, 2].quantile(0.12)  # robust floor estimate
    pts_lifted = pts_centered.clone()
    pts_lifted[:, 2] -= floor_z

    # --- Compute scale ---
    if rescale_by_camera_radius:
        cam_centers = extrs[:, 0, :, 3]  # (V, 3)
        cam_centers_centered = cam_centers - centroid  # shift
        cam_centers_centered[:, 2] -= floor_z  # lift
        cam_dists = cam_centers_centered.norm(dim=1)
        median_dist = cam_dists.median()
        scale = target_radius / median_dist
    else:
        scene_radius = pts_lifted.norm(dim=1).quantile(0.95)
        scale = target_radius / scene_radius

    # --- Compute translation (after scaling) ---
    translate = -scale * centroid
    translate[2] -= scale * floor_z  # lift to z=0

    return scale, translate


def _ensure_moge2_cache_and_load(rgbs, seq_name, dataset_root, moge2_cache_subdir, skip_if_cached=True):
    """
    Raw MoGe-2 depth (metric) with per-sequence caching.
    Returns (depths, confs) shaped [V,T,1,H,W] on CPU.
    """
    V, T, _, H, W = rgbs.shape
    cache_root = os.path.join(dataset_root, moge2_cache_subdir, seq_name)
    os.makedirs(cache_root, exist_ok=True)
    depths_path = os.path.join(cache_root, "moge2_depths.npy")
    confs_path = os.path.join(cache_root, "moge2_confs.npy")

    if skip_if_cached and os.path.isfile(depths_path) and os.path.isfile(confs_path):
        d = torch.from_numpy(np.load(depths_path)).float()  # [V,T,H,W]
        c = torch.from_numpy(np.load(confs_path)).float()  # [V,T,H,W]
        return d.unsqueeze(2), c.unsqueeze(2)

    d = _moge_depths(seq_name, rgbs, cache_root)  # [V,T,H,W], CPU float

    # Simple constant confidence for MoGe-2
    c = torch.full_like(d, 100.0)

    np.save(depths_path, d.numpy())
    np.save(confs_path, c.numpy())
    return d.unsqueeze(2), c.unsqueeze(2)


def _ensure_monofusion_cache_and_load(rgbs, seq_name, dataset_root, monofusion_cache_subdir, skip_if_cached=True):
    """
    MONOFUSION:
      - Background mask: patch-change detector over temporal window (static -> BG)
      - DUSt3R depth: load per frame/view; build static background depth by BG-temporal-average.
      - MoGe-2 monocular depth per frame/view; align to background by affine (a,b).
      - Merge BG (DUSt3R static) with FG (aligned MoGe).
      - Cache final depths & confs.
    """
    V, T, _, H, W = rgbs.shape

    cache_root = os.path.join(dataset_root, monofusion_cache_subdir, seq_name)
    os.makedirs(cache_root, exist_ok=True)
    final_depths_path = os.path.join(cache_root, "final_depths.npy")
    final_confs_path = os.path.join(cache_root, "final_confs.npy")

    if skip_if_cached and os.path.isfile(final_depths_path) and os.path.isfile(final_confs_path):
        fd = torch.from_numpy(np.load(final_depths_path))  # [V,T,H,W]
        fc = torch.from_numpy(np.load(final_confs_path))  # [V,T,H,W]
        return fd.unsqueeze(2), fc.unsqueeze(2)

    # ---- DUSt3R depths per frame/view ----
    depth_root = os.path.join(dataset_root, f"duster_depths__{seq_name}")
    if not os.path.exists(os.path.join(depth_root, f"3d_model__{T - 1:05d}__scene.npz")):
        if "../duster" not in sys.path:
            sys.path.insert(0, "../duster")
        from scripts.egoexo4d_preprocessing import main_estimate_duster_depth
        pkl_path = os.path.join(dataset_root, f"{seq_name}.pkl")

        # Re-enable autograd locally (overrides any surrounding no_grad/inference_mode)
        with ExitStack() as stack:
            stack.enter_context(torch.inference_mode(False))
            stack.enter_context(torch.enable_grad())
            main_estimate_duster_depth(pkl_path, depth_root, skip_if_cached)

    duster_depths = []
    for t in range(T):
        scene_path = os.path.join(depth_root, f"3d_model__{t:05d}__scene.npz")
        scene = np.load(scene_path)
        d = torch.from_numpy(scene["depths"])  # [V, H', W']
        d = interpolate(d[:, None], size=(H, W), mode="nearest")[:, 0]  # [V, H, W]
        duster_depths.append(d)
    duster_depths = torch.stack(duster_depths, dim=1)  # [V, T, H, W]

    # ---- Background mask (patch-change) ----
    compute_device = "cuda" if torch.cuda.is_available() else "cpu"
    bg_mask = _static_bg_mask_from_window(rgbs.to(compute_device)).cpu()  # [V,T,H,W] bool

    # ---- Static background depth per camera via temporal average on BG pixels ----
    V, T, _, _ = duster_depths.shape
    D_bg = torch.zeros((V, H, W), dtype=torch.float32)
    for v in range(V):
        valid = bg_mask[v]  # [T,H,W]
        num = (duster_depths[v] * valid).sum(dim=0)
        den = valid.sum(dim=0).clamp_min(1)
        D_bg[v] = num / den

    # ---- MoGe-2 monocular depths per frame/view ----
    moge_depths = _moge_depths(seq_name, rgbs, cache_root)  # [V,T,H,W]

    # ---- Align MoGe to background (solve a,b on BG pixels) ----
    compute_device = "cuda" if torch.cuda.is_available() else "cpu"
    moge_depths = moge_depths.to(compute_device, dtype=torch.float32)  # [V,T,H,W]
    D_bg_exp = D_bg[:, None].expand_as(moge_depths).to(compute_device)  # [V,T,H,W]
    bg_mask = bg_mask.to(compute_device)  # [V,T,H,W]

    # Valid BG pixels
    valid = bg_mask & torch.isfinite(moge_depths) & (moge_depths > 0) \
            & torch.isfinite(D_bg_exp) & (D_bg_exp > 0)

    # Flatten over pixels
    X = moge_depths.view(V, T, -1)  # [V,T,HW]
    Y = D_bg_exp.view(V, T, -1)  # [V,T,HW]
    M = valid.view(V, T, -1).float()  # [V,T,HW]

    # Count valid pixels
    n = M.sum(dim=-1)  # [V,T]
    min_bg = 200
    if (n < min_bg).any():
        bad = torch.nonzero(n < min_bg, as_tuple=False)
        raise RuntimeError(
            f"Too few background pixels in frames: {[(int(v), int(t)) for v, t in bad.tolist()]}"
        )

    # Sufficient statistics
    sx = (X * M).sum(dim=-1)
    sy = (Y * M).sum(dim=-1)
    sxx = (X * X * M).sum(dim=-1)
    sxy = (X * Y * M).sum(dim=-1)

    # Closed-form least squares for a, b
    eps = 1e-8
    mx = sx / n
    my = sy / n
    varx = sxx / n - mx * mx
    cov = sxy / n - mx * my

    a = cov / (varx + eps)  # [V,T]
    b = my - a * mx

    # Apply alignment
    aligned_moge = (a[..., None] * X + b[..., None]).view(V, T, H, W)

    # Optionally save scale/shift
    scale = a.float().cpu()
    shift = b.float().cpu()

    # ---- Merge FG/BG ----
    final_depths = torch.where(bg_mask, D_bg_exp, aligned_moge)  # [V,T,H,W]

    # ---- Confidence map: high for BG, moderate for FG ----
    final_confs = torch.zeros_like(final_depths)
    final_confs[bg_mask] = 1000.0
    final_confs[~bg_mask] = 10.0

    # ---- Cache results ----
    np.save(final_depths_path, final_depths.cpu().numpy())
    np.save(final_confs_path, final_confs.cpu().numpy())
    np.save(os.path.join(cache_root, "scale.npy"), scale.cpu().numpy())
    np.save(os.path.join(cache_root, "shift.npy"), shift.cpu().numpy())

    return final_depths.unsqueeze(2).cpu(), final_confs.unsqueeze(2).cpu()


def _static_bg_mask_from_window(
        rgbs: torch.Tensor,
        win: int = -1,
        r: int = 7,  # spatial patch radius -> (2r+1)x(2r+1)
        diff_thresh: float = 10.0  # uint8 scale threshold
):
    """
    Fast BG detector using 3D max-pooling over frame-to-frame diffs.
    """
    V, T, C, H, W = rgbs.shape
    device = rgbs.device

    if T == 1:
        return torch.ones((V, T, H, W), dtype=torch.bool, device=device)

    if win == -1:
        win = T

    # 1) Frame-to-frame abs diff (channel-mean): boundaries of length T-1
    x = rgbs.float()
    diffs = (x[:, 1:] - x[:, :-1]).abs().mean(dim=2)  # [V, T-1, H, W]
    diffs = diffs.unsqueeze(1)  # [V, 1, T-1, H, W]  (N,C,D,H,W for 3D pool)

    # 2) 3D max pool over time & space:
    #    - temporal kernel spans (2*win-1) boundaries
    #    - spatial kernel spans (2r+1)x(2r+1) patch
    kt = max(1, 2 * win - 1)
    kh = kw = 2 * r + 1
    pt = (kt - 1) // 2
    ph = pw = r
    pooled = F.max_pool3d(diffs, kernel_size=(kt, kh, kw), stride=1, padding=(pt, ph, pw))
    pooled = pooled[:, 0]  # [V, T-1, H, W]

    # 3) Map boundary maxima back to frame centers (symmetric nearest-window approx)
    change = torch.zeros((V, T, H, W), device=device, dtype=pooled.dtype)
    change[:, 0] = pooled[:, 0]
    change[:, 1:-1] = torch.maximum(pooled[:, :-1], pooled[:, 1:])
    change[:, -1] = pooled[:, -1]

    # 4) Threshold -> background
    bg_mask = (change < diff_thresh)
    return bg_mask


def _moge_depths(seq_name, rgbs, cache_root, resize_to=512, batch_size=18):
    """Runs (and caches) MoGe-2; returns [V,T,H,W] float32 at native resolution."""

    # pip install git+https://github.com/microsoft/MoGe.git
    from moge.model.v2 import MoGeModel as MoGe2Model

    depths_path = os.path.join(cache_root, "moge_depths.npy")
    if os.path.isfile(depths_path):
        logging.info(f"Loading cached MoGe-2 depths for {seq_name} from {depths_path}")
        return torch.from_numpy(np.load(depths_path)).float()

    V, T, C, H, W = rgbs.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MoGe2Model.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device).eval()

    if resize_to is None:
        h1, w1 = H, W
    else:
        if H >= W:
            h1, w1 = int(resize_to), max(1, round(resize_to * W / H))
        else:
            w1, h1 = int(resize_to), max(1, round(resize_to * H / W))

    imgs = rgbs.view(V * T, C, H, W).float()
    if (h1, w1) != (H, W):
        imgs = F.interpolate(imgs, size=(h1, w1), mode="bilinear", align_corners=False)
    imgs = (imgs / 255.0).to(device, non_blocking=True)  # [N,3,h1,w1]

    out_small = torch.empty((V * T, h1, w1), dtype=torch.float32, device=device)

    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                                enabled=(device.type == "cuda")):
        N = imgs.shape[0]
        for i in range(0, N, batch_size):
            chunk = imgs[i:i + batch_size]  # [b,3,h1,w1]
            pred = model.infer(chunk)  # expects batched input
            assert isinstance(pred, dict) and "depth" in pred, "MoGe-2 infer() must return dict with 'depth'."
            d = torch.as_tensor(pred["depth"], device=device)
            assert d.ndim == 3 and d.shape[0] == chunk.shape[0] and tuple(d.shape[1:]) == (h1, w1), \
                f"Depth shape {tuple(d.shape)} != ({chunk.shape[0]},{h1},{w1})"
            out_small[i:i + chunk.shape[0]] = d

    if (h1, w1) != (H, W):
        out = F.interpolate(out_small[:, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0]
    else:
        out = out_small
    out = out.clamp_min(0).view(V, T, H, W).cpu()

    np.save(depths_path, out.numpy())
    return out


def _ensure_vggt_raw_cache_and_load(
        rgbs: torch.Tensor,  # uint8 [V,T,3,H,W]
        seq_name: str,
        dataset_root: str,
        vggt_cache_subdir: str = "vggt_cache",
        skip_if_cached: bool = True,
        model_id: str = "facebook/VGGT-1B",
):
    """
    Run VGGT and cache RAW predictions (no alignment).
    Returns CPU float32 tensors:
      depths_raw   [V,T,1,H,W]
      confs        [V,T,1,H,W]  (constant 100)
      intrs_raw    [V,T,3,3]
      extrs_raw    [V,T,3,4]    (world->cam as predicted by VGGT)
    """
    from mvtracker.models.core.vggt.models.vggt import VGGT
    from mvtracker.models.core.vggt.utils.pose_enc import pose_encoding_to_extri_intri

    assert rgbs.dtype == torch.uint8 and rgbs.ndim == 5 and rgbs.shape[2] == 3, "rgbs must be uint8 [V,T,3,H,W]"
    V, T, _, H, W = rgbs.shape
    cache_root = os.path.join(dataset_root, vggt_cache_subdir, seq_name)
    os.makedirs(cache_root, exist_ok=True)

    f_depths_raw = os.path.join(cache_root, "vggt_depths_raw.npy")  # [V,T,H,W]
    f_confs = os.path.join(cache_root, "vggt_confs.npy")  # [V,T,H,W]
    f_intr_raw = os.path.join(cache_root, "vggt_intrinsics_raw.npy")
    f_extr_raw = os.path.join(cache_root, "vggt_extrinsics_raw.npy")

    all_cached = all(os.path.isfile(p) for p in [f_depths_raw, f_confs, f_intr_raw, f_extr_raw])
    if skip_if_cached and all_cached:
        depths_raw = torch.from_numpy(np.load(f_depths_raw)).float().unsqueeze(2)
        confs = torch.from_numpy(np.load(f_confs)).float().unsqueeze(2)
        intrs_raw = torch.from_numpy(np.load(f_intr_raw)).float()
        extrs_raw = torch.from_numpy(np.load(f_extr_raw)).float()
        return depths_raw, confs, intrs_raw, extrs_raw

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGGT.from_pretrained(model_id).to(device).eval()
    amp_dtype = torch.bfloat16 if (
            device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

    def _compute_pad_to_518(H0: int, W0: int, target: int = 518) -> Tuple[int, int, int, int, int, int]:
        """
        Mirror VGGT's load_and_preprocess_images(mode='pad') padding math so we can undo it.
        Returns: new_h, new_w, pad_top, pad_bottom, pad_left, pad_right
        """
        # Make largest dim target, keep aspect, round smaller dim to /14*14, then pad to (target, target)
        if W0 >= H0:
            new_w = target
            new_h = int(round((H0 * (new_w / W0)) / 14.0) * 14)
            h_pad = max(0, target - new_h)
            w_pad = 0
        else:
            new_h = target
            new_w = int(round((W0 * (new_h / H0)) / 14.0) * 14)
            h_pad = 0
            w_pad = max(0, target - new_w)

        pad_top = h_pad // 2
        pad_bottom = h_pad - pad_top
        pad_left = w_pad // 2
        pad_right = w_pad - pad_left
        return new_h, new_w, pad_top, pad_bottom, pad_left, pad_right

    depths_raw_arr = torch.empty((V, T, H, W), dtype=torch.float32)
    confs_arr = torch.full((V, T, H, W), 100.0, dtype=torch.float32)
    intr_raw_arr = torch.empty((V, T, 3, 3), dtype=torch.float32)
    extr_raw_arr = torch.empty((V, T, 3, 4), dtype=torch.float32)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=amp_dtype):
        for t in tqdm(range(T), desc=f"VGGT RAW {seq_name}", unit="f"):
            image_items = [rgbs[v, t].cpu() for v in range(V)]  # each: [3,H,W] uint8
            images = _vggt_load_and_preprocess_images(image_items, mode="pad").to(device)[None]  # [1,V,3,518,518]

            tokens, ps_idx = model.aggregator(images)
            pose_enc = model.camera_head(tokens)[-1]
            extr_pred, intr_pred = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])  # [1,V,3,4],[1,V,3,3]
            depth_maps, _ = model.depth_head(tokens, images, ps_idx, frames_chunk_size=1)  # [1,V,518,518]

            # per-view: undo pad, resize back to (H0,W0), adjust intrinsics
            d_full_list, K_list = [], []
            for v in range(V):
                H0, W0 = int(rgbs[v, t].shape[-2]), int(rgbs[v, t].shape[-1])
                new_h, new_w, pt, pb, pl, pr = _compute_pad_to_518(H0, W0)

                # crop padding region out of the 518x518 depth
                d_small = depth_maps[0, v:v + 1, pt:518 - pb, pl:518 - pr]  # [1,new_h,new_w]
                d_full_v = F.interpolate(d_small[:, None, :, :, 0], size=(H0, W0), mode="nearest")[:, 0]  # [1,H0,W0]
                d_full_list.append(d_full_v.squeeze(0))

                # adjust intrinsics: subtract removed pad, then scale to (H0,W0)
                K = intr_pred[0, v].detach().cpu().float().clone()
                K[0, 2] -= float(pl)
                K[1, 2] -= float(pt)
                S = torch.tensor([[W0 / float(new_w), 0.0, 0.0],
                                  [0.0, H0 / float(new_h), 0.0],
                                  [0.0, 0.0, 1.0]], dtype=torch.float32)
                K_list.append((S @ K).unsqueeze(0))

            depths_raw_arr[:, t] = torch.stack(d_full_list, dim=0)
            intr_raw_arr[:, t] = torch.cat(K_list, dim=0)
            extr_raw_arr[:, t] = extr_pred[0].detach().cpu().float()  # raw VGGT w2c

    # save raw cache
    # np.save(f_depths_raw, depths_raw_arr.numpy())
    # np.save(f_confs, confs_arr.numpy())
    # np.save(f_intr_raw, intr_raw_arr.numpy())
    # np.save(f_extr_raw, extr_raw_arr.numpy())

    return depths_raw_arr.unsqueeze(2), confs_arr.unsqueeze(2), intr_raw_arr, extr_raw_arr


def _vggt_load_and_preprocess_images(image_items, mode="crop"):
    """
    Same as VGGT loader, but accepts in-memory items as well.
    """
    if len(image_items) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    def _to_pil(item):
        # path
        if isinstance(item, str):
            img = Image.open(item)
            return img
        # numpy HWC
        if isinstance(item, np.ndarray):
            if item.ndim == 3 and item.shape[2] in (3, 4):
                if item.dtype != np.uint8:
                    item = item.astype(np.uint8)
                return Image.fromarray(item)
        # torch CHW
        if torch.is_tensor(item):
            x = item
            if x.ndim == 3 and x.shape[0] in (3, 4):
                if x.dtype == torch.uint8:
                    arr = x.permute(1, 2, 0).cpu().numpy()
                    return Image.fromarray(arr)
                else:
                    # assume float [0,1]
                    arr = (x.clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
                    return Image.fromarray(arr)
        raise ValueError("Unsupported image item type/shape")

    for item in image_items:
        img = _to_pil(item)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y: start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_items) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images


def _ensure_vggt_aligned_cache_and_load(
        rgbs: torch.Tensor,  # uint8 [V,T,3,H,W]
        seq_name: str,
        dataset_root: str,
        extrs_gt: torch.Tensor,  # [V,T,3,4] GT world->cam
        vggt_cache_subdir: str = "vggt_cache",
        skip_if_cached: bool = True,
        model_id: str = "facebook/VGGT-1B",
):
    """
    Ensure RAW VGGT cache exists (running VGGT if needed), then align VGGT cameras to GT via
    Umeyama (pred→gt) per frame. Returns CPU float32:

      depths_aligned  [V,T,1,H,W]   (RAW depths scaled by s)
      confs           [V,T,1,H,W]   (same constant 100 as RAW)
      intr_aligned    [V,T,3,3]     (equal to RAW intrinsics; alignment is Sim3 in world)
      extr_aligned    [V,T,3,4]     (VGGT w2c aligned to GT)
    """
    # 1) Get RAW results (runs VGGT if needed)
    depths_raw, confs_raw, intr_raw, extr_raw = _ensure_vggt_raw_cache_and_load(
        rgbs=rgbs,
        seq_name=seq_name,
        dataset_root=dataset_root,
        vggt_cache_subdir=vggt_cache_subdir,
        skip_if_cached=skip_if_cached,
        model_id=model_id,
    )

    # 2) Aligned cache file paths
    cache_root = os.path.join(dataset_root, vggt_cache_subdir, seq_name)
    f_depths_aln = os.path.join(cache_root, "vggt_depths_aligned.npy")
    f_intr_aln = os.path.join(cache_root, "vggt_intrinsics_aligned.npy")
    f_extr_aln = os.path.join(cache_root, "vggt_extrinsics_aligned.npy")

    # 3) If aligned already cached, return it
    if skip_if_cached and all(os.path.isfile(p) for p in [f_depths_aln, f_intr_aln, f_extr_aln]):
        depths_aln = torch.from_numpy(np.load(f_depths_aln)).float().unsqueeze(2)
        intr_aln = torch.from_numpy(np.load(f_intr_aln)).float()
        extr_aln = torch.from_numpy(np.load(f_extr_aln)).float()
        return depths_aln, confs_raw, intr_aln, extr_aln

    # 4) Compute alignment
    depths_raw_ = depths_raw.squeeze(2)  # [V,T,H,W]
    V, T, H, W = depths_raw_.shape
    assert extrs_gt.shape[:2] == (V, T), "GT extrinsics must be [V,T,3,4]"

    depths_aln = depths_raw_.clone()
    intr_aln = intr_raw.clone()  # intrinsics unchanged by world Sim3
    extr_aln = extr_raw.clone()

    def _camera_center_from_affine_extr(extr):
        extr_sq = np.eye(4, dtype=np.float32)[None].repeat(extr.shape[0], 0)
        extr_sq[:, :3, :4] = extr
        extr_sq_inv = np.linalg.inv(extr_sq)
        return extr_sq_inv[:, :3, 3]

    for t in range(T):
        gt_w2c = extrs_gt[:, t].cpu().numpy()
        pred_w2c = extr_raw[:, t].cpu().numpy()

        s, R_align, t_align = align_umeyama(
            _camera_center_from_affine_extr(gt_w2c),
            _camera_center_from_affine_extr(pred_w2c),
        )
        pred_w2c_aligned = apply_sim3_to_extrinsics(pred_w2c, s, R_align, t_align)

        extr_aln[:, t] = torch.from_numpy(np.array(pred_w2c_aligned)).float()

    # 5) Save aligned cache
    np.save(f_depths_aln, depths_aln.numpy())
    np.save(f_intr_aln, intr_aln.numpy())
    np.save(f_extr_aln, extr_aln.numpy())

    return depths_aln.unsqueeze(2), confs_raw, intr_aln, extr_aln