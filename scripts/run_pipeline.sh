#!/bin/bash
# 偵測到的 6 個 N3D dataset 名稱
datasets=(
  "coffee_martini"
  "cook_spinach"
  "cut_roasted_beef"
  "flame_salmon_1"
  "flame_steak"
  "sear_steak"
)

# 依序執行 (Sequential Execution)
for name in "${datasets[@]}"; do
  echo "========================================"
  echo "正在處理資料集: ${name}"
  echo "========================================"
  docker compose run --user $(id -u):$(id -g) --rm mvtracker python scripts/extract_frames.py --dir datasets/${name}
  docker compose run --user $(id -u):$(id -g) --rm mvtracker python scripts/prepare_n3d_mvtracker_track.py --dir datasets/${name} --num_cams 5 --use_raft_mask --use_da3 --da3_chunk_size 1 --use_dynamic_voxel --voxel_size 0.8 --dynamic_voxel_scale 3 --min_voxel_size 0.02 --export_vggt_all_frame_ply --skip_mvtracker
done

echo "所有資料集處理完成！"

