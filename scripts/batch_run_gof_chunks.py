#!/usr/bin/env python3
import os
import sys
import time
import json
import subprocess
import argparse

ALL_SCENES = [
    "coffee_martini",
    "cook_spinach",
    "cut_roasted_beef",
    "flame_salmon_1",
    "flame_steak",
    "sear_steak",
]


def main():
    parser = argparse.ArgumentParser(description="N3D 4D 分離管線：按 GOF (50幀) 分段批次執行 300 幀")
    parser.add_argument("--scenes", nargs="+", default=ALL_SCENES, help="要處理的場景名稱列表 (預設全部 6 個場景)")
    parser.add_argument("--total_frames", type=int, default=300, help="總影格數 (預設 300)")
    parser.add_argument("--gof_size", type=int, default=50, help="每個 GOF 區塊幀數 (預設 50)")
    parser.add_argument("--frame_step", type=int, default=1, help="動態抽樣影格步長 (預設 1)")
    parser.add_argument("--da3_model", type=str, default="depth-anything/DA3-GIANT-1.1", help="DA3 模型版本")
    parser.add_argument("--conf_thresh", type=float, default=5.0, help="DA3 信心度門檻 (預設 5.0)")
    parser.add_argument("--rrd_first_gof_only", action="store_true", default=True, help="僅第 1 個 GOF 輸出 .rrd 視覺化檔，其餘使用 --no_rrd 提速")
    parser.add_argument("--all_rrd", action="store_true", help="若指定，則所有 GOF 都輸出 .rrd 檔")
    parser.add_argument("--no_rrd", action="store_true", help="關閉所有 GOF 的 Rerun 記錄以最高速度產出點雲")
    parser.add_argument("--test_cam_idx", type=int, default=0, help="排除之測試視角索引 (預設 0 即 cam00)")
    parser.add_argument("--skip_existing", action="store_true", help="若該 GOF 區塊已存在產出則跳過")
    args = parser.parse_args()

    t_start_all = time.time()
    num_chunks = (args.total_frames + args.gof_size - 1) // args.gof_size

    rrd_desc = "關閉所有 Rerun 記錄 (最高速產出)" if args.no_rrd else ("全部 GOF 皆輸出 .rrd" if args.all_rrd else "僅第 1 個 GOF (0-49frame) 輸出 .rrd")
    print("=" * 75)
    print("🎬 [N3D 4D GOF 分段執行] 啟動 300 幀分段分離管線")
    print(f"📁 目標場景: {args.scenes}")
    print(f"⏱️ 總幀數: {args.total_frames} | GOF 區塊大小: {args.gof_size} 幀 (共 {num_chunks} 個 GOF)")
    print(f"🤖 DA3 模型: {args.da3_model} (信心門檻: {args.conf_thresh})")
    print(f"🎨 Rerun 輸出: {rrd_desc}")
    print("=" * 75)

    all_scene_stats = {}

    for s_idx, scene in enumerate(args.scenes, 1):
        s_dir = os.path.join("datasets", scene)
        if not os.path.isdir(s_dir):
            print(f"⚠️ 場景目錄不存在: {s_dir}，跳過。")
            continue

        print(f"\n[{s_idx}/{len(args.scenes)}] 🌟 開始處理場景: {scene}")
        scene_t0 = time.time()
        chunk_results = []

        for c_idx in range(num_chunks):
            start_f = c_idx * args.gof_size
            end_f = min(start_f + args.gof_size, args.total_frames) - 1
            max_f = end_f - start_f + 1

            chunk_name = f"{start_f}-{end_f}frame"
            chunk_dir = os.path.join(s_dir, chunk_name)
            os.makedirs(chunk_dir, exist_ok=True)

            out_ply = os.path.join(chunk_dir, "init_pointcloud_4d_pipeline.ply")
            out_static_ply = os.path.join(chunk_dir, "init_static_3d.ply")
            out_dynamic_ply = os.path.join(chunk_dir, "init_dynamic_4d.ply")

            need_rrd = not args.no_rrd and (args.all_rrd or (start_f == 0 and args.rrd_first_gof_only))
            rrd_path = f"{scene}_4d_separation.rrd" if start_f == 0 else os.path.join(chunk_dir, f"{scene}_{chunk_name}.rrd")

            print(f"\n  ▶️ [GOF {c_idx+1}/{num_chunks}] 區塊: {chunk_name} (幀 {start_f} ~ {end_f}, 共 {max_f} 幀)")
            print(f"     輸出資料夾: {chunk_dir}")

            if args.skip_existing and os.path.exists(out_ply) and os.path.getsize(out_ply) > 1024 * 1024:
                print(f"     ⏩ 偵測到 {out_ply} 已存在，跳過運算。")
                chunk_results.append({
                    "chunk": chunk_name,
                    "start_frame": start_f,
                    "end_frame": end_f,
                    "status": "SKIPPED",
                    "elapsed_sec": 0.0,
                })
                continue

            cmd = [
                sys.executable,
                "scripts/n3d_pipeline_separation_demo.py",
                "--dir", s_dir,
                "--start_frame", str(start_f),
                "--max_frames", str(max_f),
                "--frame_step", str(args.frame_step),
                "--da3_model", args.da3_model,
                "--conf_thresh", str(args.conf_thresh),
                "--test_cam_idx", str(args.test_cam_idx),
                "--export_ply", out_ply,
            ]
            if need_rrd:
                cmd.extend(["--rrd", rrd_path])
            else:
                cmd.append("--no_rrd")

            chunk_start_time = time.time()
            ret = subprocess.run(cmd)
            chunk_elapsed = time.time() - chunk_start_time

            if ret.returncode == 0:
                print(f"     ✅ GOF {chunk_name} 成功完成！耗時: {chunk_elapsed:.1f} 秒")
                chunk_results.append({
                    "chunk": chunk_name,
                    "start_frame": start_f,
                    "end_frame": end_f,
                    "status": "SUCCESS",
                    "elapsed_sec": round(chunk_elapsed, 2),
                    "output_dir": chunk_dir,
                    "unified_ply": out_ply,
                    "static_ply": out_static_ply,
                    "dynamic_ply": out_dynamic_ply,
                    "rrd_file": rrd_path if need_rrd else None,
                })
            else:
                print(f"     ❌ GOF {chunk_name} 失敗！Exit code: {ret.returncode}")
                chunk_results.append({
                    "chunk": chunk_name,
                    "start_frame": start_f,
                    "end_frame": end_f,
                    "status": "FAILED",
                    "elapsed_sec": round(chunk_elapsed, 2),
                    "exit_code": ret.returncode,
                })

        scene_elapsed = time.time() - scene_t0
        all_scene_stats[scene] = {
            "total_elapsed_sec": round(scene_elapsed, 2),
            "chunks": chunk_results,
        }

        # 儲存場景統計報告
        report_path = os.path.join(s_dir, "gof_300frames_timing_summary.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(all_scene_stats[scene], f, indent=2, ensure_ascii=False)
        print(f"\n📊 場景 {scene} 執行完成！總耗時: {scene_elapsed:.1f} 秒 (報告: {report_path})")

    total_all_elapsed = time.time() - t_start_all
    print("\n" + "=" * 75)
    print(f"🎉 全部任務處理完成！總累積耗時: {total_all_elapsed:.1f} 秒 (~{total_all_elapsed/60:.1f} 分鐘)")
    print("=" * 75)
    for scene, res in all_scene_stats.items():
        print(f"📁 場景: {scene} (總計: {res['total_elapsed_sec']:.1f}s)")
        for c in res["chunks"]:
            status_symbol = "✅" if c["status"] == "SUCCESS" else ("⏩" if c["status"] == "SKIPPED" else "❌")
            print(f"   {status_symbol} {c['chunk']:14s}: {c['status']:7s} ({c['elapsed_sec']:.1f}s)")
    print("=" * 75)


if __name__ == "__main__":
    main()
