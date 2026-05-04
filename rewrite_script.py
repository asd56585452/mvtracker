import re

with open("scripts/prepare_n3d_mvtracker_track.py", "r") as f:
    code = f.read()

# 1. Update argparse
argparse_target = 'p.add_argument("--export_per_frame_ply", action="store_true", help="是否將每一幀的軌跡儲存為獨立的 PLY 檔以供檢查")'
argparse_replace = argparse_target + '\n    p.add_argument("--segment_frames", type=int, default=50, help="每個追蹤片段的長度 (幀數)")'
code = code.replace(argparse_target, argparse_replace)

# 2. Replace lines 148 to end
split_marker = "# ==========================================\n    # 3. 生成隨機 Query Points (於 t=0)\n    # =========================================="

parts = code.split(split_marker)
if len(parts) == 2:
    new_tail = """# ==========================================
    # 3. 載入 MVTracker 模型
    # ==========================================
    print("🧠 載入 MVTracker...")
    mvtracker = torch.hub.load(".", "mvtracker", source="local", pretrained=True, device=device)
    torch.set_float32_matmul_precision("high")
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

    # ==========================================
    # 4. 分段追蹤 (Chunk-based Tracking)
    # ==========================================
    for start_t in range(0, num_frames, args.segment_frames):
        end_t = min(start_t + args.segment_frames, num_frames)
        if end_t <= start_t:
            continue
            
        chunk_frames = end_t - start_t
        chunk_dir = os.path.join(args.dir, f"{start_t}-{end_t-1}frame")
        os.makedirs(chunk_dir, exist_ok=True)
        
        print(f"\\n" + "="*50)
        print(f"🎬 開始處理片段: {start_t} 到 {end_t-1} 幀 (共 {chunk_frames} 幀)")
        print("="*50)

        # ------------------------------------------
        # 4.1 生成隨機 Query Points (於 t0 = start_t)
        # ------------------------------------------
        t0 = start_t
        print(f"🎯 從 t={t0} 幀「全部相機」萃取並隨機取樣 {args.num_queries} 個追蹤點...")
        with torch.no_grad():
            xyz_full, rgb_full = init_pointcloud_from_rgbd(
                fmaps=rgbs_tensor[:, t0:t0+1].unsqueeze(0).float(),
                depths=depths_tensor[:, t0:t0+1].unsqueeze(0),
                intrs=intrs_model[:, t0:t0+1].unsqueeze(0),
                extrs=extrs_model[:, t0:t0+1].unsqueeze(0),
                stride=1,
                level=0,
            )
        
        pts_full = xyz_full[0]  # [V*H*W, 3] at relative t=0
        colors_full = rgb_full[0] # [V*H*W, 3]
        
        valid_mask = depths_tensor[:, t0, 0].flatten() > 0
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
        pool_pts = pts_full[valid_mask]
        pool_colors = colors_full[valid_mask]
        assert pool_pts.shape[0] > 0, f"在 t={t0} 沒有找到有效的深度點來產生 Queries！"
        
        if pool_pts.shape[0] > args.num_queries:
            idx = torch.randperm(pool_pts.shape[0])[:args.num_queries]
            pts_sampled = pool_pts[idx]
            colors_sampled = pool_colors[idx]
            original_idx = valid_indices[idx]
        else:
            idx = torch.arange(pool_pts.shape[0])
            pts_sampled = pool_pts
            colors_sampled = pool_colors
            original_idx = valid_indices
            
        print(f"✅ 成功提取 {len(pts_sampled)} 個初始點雲！")

        # 匯出初始點雲 (.ply) 供 4DGS 初始化
        pts_np = pts_sampled.cpu().numpy()
        colors_np = colors_sampled.cpu().numpy()
        if colors_np.max() <= 1.0: 
            colors_np = (colors_np * 255).astype(np.uint8)
        times_np = np.full((len(pts_np), 1), float(t0), dtype=np.float32)

        ply_path = os.path.join(chunk_dir, "init_pointcloud_vggt_4d.ply")
        dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('time', 'f4')
        ]
        elements = np.empty(pts_np.shape[0], dtype=dtype)
        elements['x'], elements['y'], elements['z'] = pts_np[:, 0], pts_np[:, 1], pts_np[:, 2]
        elements['red'], elements['green'], elements['blue'] = colors_np[:, 0], colors_np[:, 1], colors_np[:, 2]
        elements['time'] = times_np[:, 0]
        PlyData([PlyElement.describe(elements, 'vertex')]).write(ply_path)
        print(f"🎉 初始點雲已儲存至 {ply_path}")

        # ------------------------------------------
        # 4.2 執行 MVTracker (批次處理)
        # ------------------------------------------
        ts_abs = torch.full((pts_sampled.shape[0], 1), float(t0), device=pts_sampled.device)
        query_points_abs = torch.cat([ts_abs, pts_sampled], dim=1).float()  
        
        ts_rel = torch.full((pts_sampled.shape[0], 1), float(0), device=pts_sampled.device)
        query_points_rel = torch.cat([ts_rel, pts_sampled], dim=1).float()
        
        all_pred_tracks = []
        all_pred_vis = []
        total_queries = query_points_rel.shape[0]
        chunk_size = args.track_chunk_size
        
        print(f"🚀 開始追蹤 {total_queries} 個點，分批大小為 {chunk_size}...")
        start_time = time.time()
        
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=amp_dtype):
            rgbs_input = rgbs_tensor[None, :, start_t:end_t].cpu()
            depths_input = depths_tensor[None, :, start_t:end_t].cpu()
            intrs_input = intrs_model[None, :, start_t:end_t].to(device)
            extrs_input = extrs_model[None, :, start_t:end_t].to(device)
            
            for i in tqdm(range(0, total_queries, chunk_size), desc="MVTracker 追蹤進度"):
                batch_queries = query_points_rel[i:i+chunk_size].to(device)
                results = mvtracker(
                    rgbs=rgbs_input,
                    depths=depths_input,
                    intrs=intrs_input,
                    extrs=extrs_input,
                    query_points_3d=batch_queries[None],
                )
                all_pred_tracks.append(results["traj_e"].cpu())
                all_pred_vis.append(results["vis_e"].cpu())
                
                del results
                torch.cuda.empty_cache()

        end_time = time.time()
        pred_tracks = torch.cat(all_pred_tracks, dim=2)[0]
        pred_vis = torch.cat(all_pred_vis, dim=2)[0]
        print(f"✅ 追蹤完成！ (片段花費時間: {end_time - start_time:.2f} 秒)")

        # ------------------------------------------
        # 4.3 儲存軌跡
        # ------------------------------------------
        tracks_path = os.path.join(chunk_dir, "mvtracker_tracks.npz")
        np.savez(
            tracks_path,
            query_points=query_points_abs.cpu().numpy(),
            pred_tracks=pred_tracks.numpy(),
            pred_vis=pred_vis.numpy(),
        )
        print(f"🎉 軌跡已儲存至 {tracks_path}")

        # ------------------------------------------
        # 4.4 匯出逐幀 PLY
        # ------------------------------------------
        if args.export_per_frame_ply:
            ply_dir = os.path.join(chunk_dir, "mvtracker_plys")
            os.makedirs(ply_dir, exist_ok=True)
            print(f"📁 正在將每一幀的點雲匯出至 {ply_dir}...")
            
            pred_tracks_np = pred_tracks.numpy()
            for t_rel in tqdm(range(chunk_frames), desc="匯出 PLY"):
                abs_t = start_t + t_rel
                ply_out_path = os.path.join(ply_dir, f"frame_{abs_t:04d}.ply")
                
                elements = np.empty(total_queries, dtype=dtype)
                elements['x'], elements['y'], elements['z'] = pred_tracks_np[t_rel, :, 0], pred_tracks_np[t_rel, :, 1], pred_tracks_np[t_rel, :, 2]
                elements['red'], elements['green'], elements['blue'] = colors_np[:, 0], colors_np[:, 1], colors_np[:, 2]
                elements['time'] = np.full(total_queries, float(abs_t), dtype=np.float32)
                PlyData([PlyElement.describe(elements, 'vertex')]).write(ply_out_path)
            print("✅ 逐幀 PLY 匯出完成！")

        # ------------------------------------------
        # 4.5 漂移驗證 (Drift Analysis)
        # ------------------------------------------
        print("🔍 進行漂移驗證 (Drift Analysis)...")
        cam_idx = original_idx // (TARGET_H * TARGET_W)
        rem = original_idx % (TARGET_H * TARGET_W)
        y = rem // TARGET_W
        x = rem % TARGET_W

        N_pts = len(original_idx)
        vggt_tracks = torch.zeros((chunk_frames, N_pts, 3), device=device)
        depths_tensor_dev = depths_tensor[:, start_t:end_t].to(device)
        intrs_model_dev = intrs_model[:, start_t:end_t].to(device)
        extrs_model_dev = extrs_model[:, start_t:end_t].to(device)

        pixel_homo = torch.stack([x.float(), y.float(), torch.ones_like(x).float()], dim=-1).to(device)
        
        for t_rel in range(chunk_frames):
            d_t = depths_tensor_dev[cam_idx, t_rel, 0, y, x] 
            K_t = intrs_model_dev[cam_idx, t_rel] 
            E_t = extrs_model_dev[cam_idx, t_rel] 
            
            K_inv = torch.inverse(K_t) 
            cam_ray = torch.bmm(K_inv, pixel_homo.unsqueeze(-1)).squeeze(-1) 
            cam_xyz = cam_ray * d_t.unsqueeze(-1) 
            
            R = E_t[:, :3, :3] 
            Tr = E_t[:, :3, 3] 
            
            X_c_minus_Tr = cam_xyz - Tr
            R_inv = R.transpose(1, 2)
            world_xyz = torch.bmm(R_inv, X_c_minus_Tr.unsqueeze(-1)).squeeze(-1) 
            vggt_tracks[t_rel] = world_xyz

        pred_tracks_dev = pred_tracks.to(device)
        
        mv_drift = torch.norm(pred_tracks_dev - pred_tracks_dev[0:1], dim=-1) 
        vggt_drift = torch.norm(vggt_tracks - vggt_tracks[0:1], dim=-1) 

        mv_drift_mean = mv_drift.mean(dim=1).cpu().numpy()
        vggt_drift_mean = vggt_drift.mean(dim=1).cpu().numpy()
        
        plt.figure(figsize=(12, 6))
        plt.plot(range(start_t, end_t), mv_drift_mean, label='MVTracker Mean Drift', color='blue', linewidth=2)
        plt.plot(range(start_t, end_t), vggt_drift_mean, label='VGGT Mean Drift', color='orange', linewidth=2, linestyle='--')
        plt.xlabel('Absolute Frame')
        plt.ylabel(f'Mean Displacement from t={start_t} (World Units)')
        plt.title(f'Drift Analysis Over Time (Frames {start_t}-{end_t-1})')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(chunk_dir, "drift_analysis.png")
        plt.savefig(plot_path)
        plt.close()
        
        threshold = 0.5
        mv_bad_pct = (mv_drift[-1] > threshold).float().mean().item() * 100
        vggt_bad_pct = (vggt_drift[-1] > threshold).float().mean().item() * 100
        
        print(f"📊 漂移分析圖表已儲存至 {plot_path}")
        print(f"   - MVTracker 片段最後一幀平均漂移: {mv_drift_mean[-1]:.4f} 單位")
        print(f"   - VGGT      片段最後一幀平均漂移: {vggt_drift_mean[-1]:.4f} 單位")
        print(f"   - MVTracker > {threshold} 單位移動的點比例: {mv_bad_pct:.2f}%")
        print(f"   - VGGT      > {threshold} 單位移動的點比例: {vggt_bad_pct:.2f}%\\n")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", message=".*DtypeTensor constructors are no longer.*", module="pointops.query")
    warnings.filterwarnings("ignore", message=".*Plan failed with a cudnnException.*", module="torch.nn.modules.conv")
    main()
"""
    code = parts[0] + new_tail

    with open("scripts/prepare_n3d_mvtracker_track.py", "w") as f:
        f.write(code)
    print("Replacement successful.")
else:
    print("Could not find split_marker!")

