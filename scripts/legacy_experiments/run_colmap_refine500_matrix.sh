#!/bin/bash
# ==============================================================================
# 🚀 6 Scenes x 6 Keyframes 純 COLMAP 基線補全 (預熱 500 步, Reset 3000)
#
# 目標:
# 驗證「預熱步數 500、Reset 週期 3000」下純 COLMAP 的表現，
# 與先前跑的純 COLMAP (預熱 300 步) 以及 DA3 組別 (預熱 500 步) 進行嚴格的控制變因對比！
#
# 特色:
# - 自動檢查 stats/val_step4999.json，已完成者秒速跳過
# - 即時統計已完成與新訓練數量
# ==============================================================================

set -e

PROJECT_DIR="/home/tsaichenghan/GSCodec_Studio"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.yml"
TRACKER_DIR="/home/tsaichenghan/mvtracker"

ALL_SCENES=(
  "coffee_martini"
  "cook_spinach"
  "cut_roasted_beef"
  "flame_salmon_1"
  "flame_steak"
  "sear_steak"
)

ALL_FRAMES=(25 75 125 175 225 275)

if [ "$#" -gt 0 ]; then
  SCENES=("$@")
else
  SCENES=("${ALL_SCENES[@]}")
fi

echo "======================================================================"
echo "🎯 啟動 6 Scenes x 6 Keyframes 純 COLMAP (預熱 500 步, Reset 3000) 排程"
echo "   目標場景: ${SCENES[*]}"
echo "   目標幀數: ${ALL_FRAMES[*]}"
echo "   當前時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"

total_completed=0
total_skipped=0

for scene in "${SCENES[@]}"; do
  echo ""
  echo "======================================================================"
  echo "🎬 開始處理場景: ${scene}"
  echo "======================================================================"

  mkdir -p "$PROJECT_DIR/results/144exp/${scene}"

  for frame in "${ALL_FRAMES[@]}"; do
    colmap_dir="datasets/${scene}/colmap_${frame}"
    out_dir="results/144exp/${scene}/f${frame}_3dgs_colmap_refine500_reset3000"

    if [ -f "$PROJECT_DIR/$out_dir/stats/val_step4999.json" ]; then
      echo "  ⏩ [純 COLMAP (refine 500, reset 3000)] ${scene} f${frame} 已存在，跳過"
      total_skipped=$((total_skipped + 1))
    else
      echo "  🚀 [純 COLMAP (refine 500, reset 3000)] ${scene} f${frame} 開始訓練..."
      docker compose --project-directory "$PROJECT_DIR" -f "$COMPOSE_FILE" run --user $(id -u):$(id -g) --rm gscodec \
        python examples/simple_trainer.py default \
        --data-dir "$colmap_dir" \
        --data-factor 2 \
        --test-views cam00 \
        --disable-viewer \
        --max-steps 5000 \
        --strategy.refine-start-iter 500 \
        --strategy.refine-stop-iter 3000 \
        --strategy.refine-every 100 \
        --strategy.reset-every 3000 \
        --eval-steps 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 \
        --save-steps 5000 \
        --result-dir "$out_dir"
      echo "  ✅ [純 COLMAP (refine 500, reset 3000)] ${scene} f${frame} 完成！"
      total_completed=$((total_completed + 1))
    fi
  done

  echo ""
  echo "📊 [${scene}] 最新指標更新:"
  python3 "$TRACKER_DIR/scripts/evaluate_6scenes_matrix_ablation.py" || true
done

echo ""
echo "======================================================================"
echo "🎉 純 COLMAP (refine 500, reset 3000) 全量排程完成！"
echo "   新完成訓練: $total_completed 組"
echo "   已存在跳過: $total_skipped 組"
echo "   結束時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"
python3 "$TRACKER_DIR/scripts/evaluate_6scenes_matrix_ablation.py"
