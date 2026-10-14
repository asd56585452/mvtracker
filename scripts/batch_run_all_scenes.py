#!/usr/bin/env python3
import os
import sys
import time
import subprocess

ALL_SCENES = [
    "coffee_martini",
    "cook_spinach",
    "cut_roasted_beef",
    "flame_salmon_1",
    "flame_steak",
    "sear_steak",
]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="N3D 6 場景前 50 幀 4D 靜動態分離管線批次執行")
    parser.add_argument("--scenes", nargs="+", default=ALL_SCENES, help="要處理的場景名稱列表")
    parser.add_argument("--max_frames", type=int, default=50, help="處理影格數 (預設 50)")
    parser.add_argument("--frame_step", type=int, default=1, help="動態抽樣影格步長 (預設 1)")
    parser.add_argument("--da3_model", type=str, default="depth-anything/DA3-GIANT-1.1", help="DA3 模型版本 (預設 depth-anything/DA3-GIANT-1.1)")
    parser.add_argument("--skip_existing", action="store_true", help="若該場景之 .rrd 檔案已存在且有效則跳過")
    args = parser.parse_args()

    t_start_all = time.time()
    print("=" * 70)
    print(f"🎬 [N3D 4D 分離批次執行] 開始處理 {len(args.scenes)} 個場景 (前 {args.max_frames} 幀)...")
    print(f"📁 目標場景清單: {args.scenes}")
    print(f"🤖 DA3 模型: {args.da3_model}")
    print("=" * 70)

    summary = []
    for idx, scene in enumerate(args.scenes, 1):
        s_dir = os.path.join("datasets", scene)
        rrd_file = f"{scene}_4d_separation.rrd"
        ply_file = os.path.join(s_dir, "init_pointcloud_4d_pipeline.ply")

        print(f"\n[{idx}/{len(args.scenes)}] 🚀 開始處理場景: {scene}")
        if args.skip_existing and os.path.exists(rrd_file) and os.path.getsize(rrd_file) > 100 * 1024 * 1024:
            print(f"  ⏩ 偵測到 {rrd_file} 已存在 ({os.path.getsize(rrd_file) / (1024*1024):.1f} MB)，跳過運算。")
            summary.append((scene, "SKIPPED", 0.0))
            continue

        t0 = time.time()
        cmd = [
            sys.executable,
            "scripts/n3d_pipeline_separation_demo.py",
            "--dir", s_dir,
            "--max_frames", str(args.max_frames),
            "--frame_step", str(args.frame_step),
            "--da3_model", args.da3_model,
            "--rrd", rrd_file,
            "--export_ply", ply_file,
        ]

        ret = subprocess.run(cmd)
        elapsed = time.time() - t0
        if ret.returncode == 0:
            print(f"  ✅ 場景 {scene} 成功完成！耗時: {elapsed:.1f} 秒")
            summary.append((scene, "SUCCESS", elapsed))
        else:
            print(f"  ❌ 場景 {scene} 執行失敗！Exit Code: {ret.returncode}")
            summary.append((scene, "FAILED", elapsed))

    print("\n" + "=" * 70)
    print(f"🎉 全部場景批次處理完成！總耗時: {time.time() - t_start_all:.1f} 秒")
    print("=" * 70)
    for s, status, el in summary:
        print(f"  - {s:20s}: {status:7s} ({el:.1f}s)")
    print("=" * 70)


if __name__ == "__main__":
    main()
