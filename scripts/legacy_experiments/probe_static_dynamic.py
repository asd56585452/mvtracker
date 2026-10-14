"""
[EXPERIMENT ONLY] Static/Dynamic separability probe for MVTracker.

Question answered:
  If we hand MVTracker *all* t0 query points (DA3 + COLMAP, no RAFT pre-split), do the
  resulting trajectories separate cleanly into static / dynamic?  How does the tracker-based
  statistic compare with an independent photometric (appearance-change) statistic?

Outputs (in --out):
  probe_stats.npz   per-point statistics
  probe_hist.png    histograms
  probe_report.json percentile / agreement numbers
  probe_static_dynamic.ply  green=static red=dynamic (tracker-based, Otsu on log max-disp)
"""
import os, sys, re, glob, json, time, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import numpy as np, torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
from plyfile import PlyData, PlyElement

from scripts.prepare_n3d_mvtracker_track import (
    get_sorted_frame_paths, llff_to_opencv_w2c, select_cams_by_fps,
)
from mvtracker.datasets.generic_scene_dataset import _ensure_da3_aligned_cache_and_load
from mvtracker.models.core.model_utils import init_pointcloud_from_rgbd
from mvtracker.utils.spatial_voxelizer import voxel_downsample_unified_with_features


def otsu_threshold(x, bins=256):
    h, edges = np.histogram(x, bins=bins)
    c = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = ((h * c).sum() - np.cumsum(h * c)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return c[np.argmax(between)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--frames", type=int, default=50)
    p.add_argument("--num_cams", type=int, default=5)
    p.add_argument("--max_queries", type=int, default=20000)
    p.add_argument("--voxel_size", type=float, default=0.1)
    p.add_argument("--da3_model", default="depth-anything/DA3-LARGE-1.1")
    p.add_argument("--da3_res", type=int, default=672)
    p.add_argument("--chunk", type=int, default=4096)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    np.random.seed(72); torch.manual_seed(72)
    dev = "cuda"

    view_dirs = sorted([d for d in glob.glob(os.path.join(args.dir, "cam*")) if os.path.isdir(d)],
                       key=lambda s: int(re.search(r"\d+", os.path.basename(s)).group()))
    s0, T = args.start, args.frames
    W0, H0 = Image.open(get_sorted_frame_paths(view_dirs[0])[0]).size
    pb = np.load(os.path.join(args.dir, "poses_bounds.npy"))[:, :-2].reshape(-1, 3, 5)
    w2c_all, K_all = zip(*[llff_to_opencv_w2c(pb[i], W0, H0) for i in range(len(pb))])
    w2c_all, K_all = np.stack(w2c_all), np.stack(K_all)
    cams = select_cams_by_fps(w2c_all, args.num_cams, 0.8, [0])
    print("cams", cams)
    TW, TH = 768, 576
    sx, sy = TW / W0, TH / H0
    ex = torch.stack([torch.from_numpy(w2c_all[c]).float() for c in cams]).unsqueeze(1).repeat(1, T, 1, 1)
    Ks = []
    for c in cams:
        K = K_all[c].copy(); K[0] *= sx; K[1] *= sy; Ks.append(torch.from_numpy(K).float())
    ix = torch.stack(Ks).unsqueeze(1).repeat(1, T, 1, 1)

    rgbs = []
    for c in tqdm(cams, desc="rgb"):
        fr = get_sorted_frame_paths(view_dirs[c])[s0:s0 + T]
        rgbs.append(torch.stack([(to_tensor(Image.open(f).convert("RGB").resize((TW, TH), Image.Resampling.BILINEAR)) * 255).byte() for f in fr]))
    rgbs = torch.stack(rgbs)  # [V,T,3,H,W]
    V = len(cams)

    tmp_root = os.path.join(args.out, "_da3_tmp")
    with torch.no_grad():
        depths, _, ix_m, ex_m, *_ = _ensure_da3_aligned_cache_and_load(
            rgbs=rgbs, seq_name="probe", dataset_root=tmp_root, extrs_gt=ex, intrs_gt=ix,
            skip_if_cached=False, model_id=args.da3_model, temporal_chunk_size=1,
            enable_layer_offload=None, process_res=args.da3_res)
    ix_m, ex_m = ix.clone(), ex.clone()
    print("depth", depths.shape)

    # ---- queries at t=0 : DA3 all-cam uniform voxel + COLMAP anchors
    with torch.no_grad():
        xyz, rgb = init_pointcloud_from_rgbd(
            fmaps=rgbs[:, 0:1].unsqueeze(0).float(), depths=depths[:, 0:1].unsqueeze(0),
            intrs=ix_m[:, 0:1].unsqueeze(0), extrs=ex_m[:, 0:1].unsqueeze(0), stride=1, level=0)
    valid = depths[:, 0, 0].flatten() > 0
    pts = xyz[0][valid].cpu().numpy(); col = rgb[0][valid].cpu().numpy()
    v_pts, v_col, _, _ = voxel_downsample_unified_with_features(
        pts=pts, colors=col, features=None, is_dynamic=np.zeros(len(pts), bool), normals=None,
        voxel_size=args.voxel_size, footprints=None, min_voxel_size=0.01, scale=1.0, mode="uniform",
        surfel_normal_scale=5.0, surfel_angle_deg=25.0)
    v_pts = v_pts.cpu().numpy() if torch.is_tensor(v_pts) else v_pts
    v_col = v_col.cpu().numpy() if torch.is_tensor(v_col) else v_col
    sel = np.random.permutation(len(v_pts))[:args.max_queries]
    q_da3, c_da3 = v_pts[sel], v_col[sel]
    src = np.zeros(len(q_da3), np.int8)  # 0 = DA3
    try:
        from scripts.generate_uniform_da3_pcds import load_colmap_points
        cp, cc = load_colmap_points(os.path.join(args.dir, f"colmap_{s0}", "sparse", "0", "points3D.bin"))
        q_da3 = np.concatenate([q_da3, cp.astype(np.float32)]); c_da3 = np.concatenate([c_da3, cc.astype(np.float32)])
        src = np.concatenate([src, np.ones(len(cp), np.int8)])  # 1 = COLMAP
        print("colmap anchors", len(cp))
    except Exception as e:
        print("colmap load failed:", e)
    q = torch.from_numpy(q_da3).float()
    N = len(q)
    print("queries", N, "scene bbox", q.min(0).values.tolist(), q.max(0).values.tolist())

    # ---- MVTracker
    mvt = torch.hub.load(".", "mvtracker", source="local", pretrained=True, device=dev)
    amp = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    qp = torch.cat([torch.zeros(N, 1), q], 1)
    trs, vis = [], []
    t_trk = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp):
        for i in tqdm(range(0, N, args.chunk), desc="mvtracker"):
            r = mvt(rgbs=rgbs[None], depths=depths[None], intrs=ix_m[None].to(dev), extrs=ex_m[None].to(dev),
                    query_points_3d=qp[i:i + args.chunk][None].to(dev))
            trs.append(r["traj_e"].cpu()); vis.append(r["vis_e"].cpu()); torch.cuda.empty_cache()
    t_trk = time.time() - t_trk
    tr = torch.cat(trs, 2)[0].numpy()    # [T,N,3]
    vi = torch.cat(vis, 2)[0].numpy()    # [T,N]
    del mvt; torch.cuda.empty_cache()

    # ---- tracker statistics
    d0 = np.linalg.norm(tr - tr[0:1], axis=-1)           # [T,N]
    max_disp = d0.max(0)
    path = np.linalg.norm(np.diff(tr, axis=0), axis=-1).sum(0)
    # robust: median-filter-free "late" displacement (mean of last 10 frames)
    late_disp = d0[-10:].mean(0)
    std_pos = tr.std(0).mean(-1)

    # ---- independent photometric statistic (fixed-position appearance change)
    pw = torch.from_numpy(q_da3).float().to(dev)
    resid = torch.full((V, N), float("nan"), device=dev)
    rg = rgbs.to(dev)
    for v in range(V):
        E = torch.cat([ex_m[v, 0].to(dev), torch.tensor([[0, 0, 0, 1.0]], device=dev)], 0)
        Kv = ix_m[v, 0].to(dev)
        pc = (E[:3, :3] @ pw.T + E[:3, 3:4]).T
        uv = (Kv @ pc.T).T; z = uv[:, 2]; u = uv[:, 0] / z; w = uv[:, 1] / z
        ui, wi = u.round().long(), w.round().long()
        inb = (z > 0.1) & (ui >= 1) & (ui < TW - 1) & (wi >= 1) & (wi < TH - 1)
        ui, wi = ui.clamp(1, TW - 2), wi.clamp(1, TH - 2)
        dz = depths[v, 0, 0].to(dev)[wi, ui]
        ok = inb & ((dz - z).abs() < 0.05 * z)           # visible in cam v at t0 (depth-consistent)
        # 3x3 mean to be robust to sub-pixel misalignment
        img = rg[v].float().mean(1)                       # [T,H,W] gray
        k = torch.nn.functional.avg_pool2d(img[:, None], 3, 1, 1)[:, 0]
        ref = k[0, wi, ui]
        cur = k[:, wi, ui]                                # [T,N]
        r_v = (cur - ref[None]).abs().mean(0) / 255.0
        r_v[~ok] = float("nan")
        resid[v] = r_v
    resid = resid.cpu().numpy()
    photo = np.nanmedian(np.where(np.isnan(resid), np.nan, resid), axis=0)  # over cams
    has_photo = ~np.isnan(photo)

    # ---- separation analysis
    lg = np.log10(np.maximum(max_disp, 1e-4))
    thr_trk = float(10 ** otsu_threshold(lg))
    lp = np.log10(np.maximum(photo[has_photo], 1e-4))
    thr_pho = float(10 ** otsu_threshold(lp))
    dyn_trk = max_disp > thr_trk
    dyn_pho = np.zeros(N, bool); dyn_pho[has_photo] = photo[has_photo] > thr_pho
    both = has_photo
    agree = float((dyn_trk[both] == dyn_pho[both]).mean())
    tp = int((dyn_trk[both] & dyn_pho[both]).sum()); fp = int((dyn_trk[both] & ~dyn_pho[both]).sum())
    fn = int((~dyn_trk[both] & dyn_pho[both]).sum()); tn = int((~dyn_trk[both] & ~dyn_pho[both]).sum())
    pct = [1, 5, 25, 50, 75, 90, 95, 99]
    rep = dict(
        N=int(N), n_da3=int((src == 0).sum()), n_colmap=int((src == 1).sum()), tracker_seconds=t_trk,
        scene_bbox_min=q.min(0).values.tolist(), scene_bbox_max=q.max(0).values.tolist(),
        scene_radius_median=float(np.median(np.linalg.norm(q_da3 - np.median(q_da3, 0), axis=1))),
        max_disp_pct={str(k): float(np.percentile(max_disp, k)) for k in pct},
        photo_pct={str(k): float(np.percentile(photo[has_photo], k)) for k in pct},
        otsu_thr_tracker=thr_trk, otsu_thr_photo=thr_pho,
        frac_dynamic_tracker=float(dyn_trk.mean()), frac_dynamic_photo=float(dyn_pho[both].mean()),
        agreement=agree, confusion=dict(tp=tp, fp=fp, fn=fn, tn=tn),
        colmap_frac_dynamic_tracker=float(dyn_trk[src == 1].mean()) if (src == 1).any() else None,
        mean_vis=float(vi.mean()),
        spearman=float(np.corrcoef(np.argsort(np.argsort(max_disp[both])), np.argsort(np.argsort(photo[both])))[0, 1]),
    )
    json.dump(rep, open(os.path.join(args.out, "probe_report.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(args.out, "probe_stats.npz"), q=q_da3, src=src, max_disp=max_disp,
                        path=path, late=late_disp, std=std_pos, photo=photo, vis=vi.mean(0))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
    ax[0].hist(lg, 100); ax[0].axvline(np.log10(thr_trk), c="r"); ax[0].set_title("log10 tracker max-disp (Otsu red)")
    ax[1].hist(lp, 100); ax[1].axvline(np.log10(thr_pho), c="r"); ax[1].set_title("log10 photometric residual")
    ax[2].scatter(max_disp[both][::5], photo[both][::5], s=1, alpha=.3); ax[2].set_xscale("log"); ax[2].set_yscale("log")
    ax[2].axvline(thr_trk, c="r"); ax[2].axhline(thr_pho, c="r"); ax[2].set_xlabel("tracker max disp"); ax[2].set_ylabel("photometric")
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "probe_hist.png"), dpi=110)

    c8 = np.where(dyn_trk[:, None], np.array([[255, 40, 40]]), np.array([[40, 220, 80]])).astype(np.uint8)
    el = np.empty(N, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    el["x"], el["y"], el["z"] = q_da3[:, 0], q_da3[:, 1], q_da3[:, 2]
    el["red"], el["green"], el["blue"] = c8[:, 0], c8[:, 1], c8[:, 2]
    PlyData([PlyElement.describe(el, "vertex")]).write(os.path.join(args.out, "probe_static_dynamic.ply"))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
