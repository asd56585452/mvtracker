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

# 1. 重新求解 3DGS 空間特徵權重 (使用當前 DA3 Surfel 相機挑選)
echo "========================================"
echo "步驟 1: 使用 DA3 Surfel 相機挑選求解 3DGS 空間特徵最佳化權重"
echo "========================================"
docker compose run --user $(id -u):$(id -g) --rm mvtracker python scripts/optimize_spatial_weights_3dgs.py \
  --dataset_dir datasets/sear_steak \
  --ply_path datasets/sear_steak/colmap_0/sparse/0/points3D_total1_sec_fps30.ply \
  --frame_idx 0 \
  --voxel_mode surfel \
  --surfel_normal_scale 5.0 \
  --surfel_angle_deg 25.0 \
  --min_voxel_size 0.01 \
  --dynamic_voxel_scale 1.0 \
  --num_cams 5 \
  --cam_select_mode da3_surfel \
  --da3_model depth-anything/DA3-LARGE-1.1 \
  --process_res 672 \
  --output_json datasets/sear_steak/weights_sear_steak_f0_surfel.json

# 2. 依序執行 6 個資料集處理 (Sequential Execution)
echo "========================================"
echo "步驟 2: 依序生成 6 個資料集之 4D 重要度採樣點雲"
echo "========================================"
for name in "${datasets[@]}"; do
  echo "========================================"
  echo "正在處理資料集: ${name}"
  echo "========================================"

  # 執行 4D 完整點雲與 RGB 分數重要度採樣點雲生成 (使用 sear_steak 最佳化權重，跳過 MVTracker 追蹤)
  docker compose run --user $(id -u):$(id -g) --rm mvtracker python scripts/prepare_n3d_mvtracker_track.py \
    --dir datasets/${name} \
    --num_cams 5 \
    --cam_select_mode da3_surfel \
    --use_da3 \
    --da3_chunk_size 1 \
    --da3_model depth-anything/DA3-LARGE-1.1 \
    --da3_process_res 672 \
    --voxel_mode surfel \
    --feature_weights_json datasets/sear_steak/weights_sear_steak_f0_surfel.json \
    --export_all_frame_ply \
    --export_importance_all_frame_ply \
    --importance_total_points 100000 \
    --importance_vis_style rgb \
    --skip_mvtracker

done

echo "所有資料集處理完成！"


