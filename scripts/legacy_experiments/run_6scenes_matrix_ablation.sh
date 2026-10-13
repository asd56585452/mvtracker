#!/bin/bash
# ==============================================================================
# 🚀 6 Scenes x 6 Keyframes 3DGS 消融訓練總排程腳本 (共 180 Runs)
#
# 涵蓋 5 組對照方法 (純 COLMAP + 4 組 DA3 全新去噪點雲消融):
# 1. 純 COLMAP (~7k):               reset=3000, 300 步預熱 (已全數完成，秒速跳過)
# 2. 組別 B (14k Importance):        reset=3000, 500 步預熱 (COLMAP 7k + 7k DA3 Imp) - 全新
# 3. 組別 C (14k Uniform):          reset=3000, 500 步預熱 (COLMAP 7k + 7k DA3 Uni) - 舊的重跑
# 4. 50k Importance (reset=3000):   reset=3000, 500 步預熱 (COLMAP 7k + ~43k DA3 Imp) - 舊的重跑
# 5. 50k Uniform (reset=3000):      reset=3000, 500 步預熱 (COLMAP 7k + ~43k DA3 Uni) - 全新
#
# 特色:
# - 自動檢查 val_step4999.json，已完成者秒速跳過 (支援中斷續跑)
# - 支援自訂場景傳參，例如: ./run_6scenes_matrix_ablation.sh coffee_martini
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

# 若使用者有傳參場景，則使用傳入的場景；否則使用全部場景
if [ "$#" -gt 0 ]; then
  SCENES=("$@")
else
  SCENES=("${ALL_SCENES[@]}")
fi

echo "======================================================================"
echo "🎯 啟動 6 Scenes x 6 Keyframes 3DGS 消融實驗排程"
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
    echo ""
    echo "----------------------------------------------------------------------"
    echo "📌 場景: ${scene} | 關鍵幀: Frame ${frame}"
    echo "----------------------------------------------------------------------"

    colmap_dir="datasets/${scene}/colmap_${frame}"

    # ------------------------------------------------------------------
    # 1. 純 COLMAP (~7k): reset=3000, refine_start=300
    # ------------------------------------------------------------------
    out_1="results/144exp/${scene}/f${frame}_3dgs_colmap_reset3000"
    if [ -f "$PROJECT_DIR/$out_1/stats/val_step4999.json" ]; then
      echo "  ⏩ [1. 純 COLMAP reset=3000 (refine 300)] 已存在，跳過"
      total_skipped=$((total_skipped + 1))
    else
      echo "  🚀 [1. 純 COLMAP reset=3000 (refine 300)] 開始訓練..."
      docker compose --project-directory "$PROJECT_DIR" -f "$COMPOSE_FILE" run --user $(id -u):$(id -g) --rm gscodec \
        python examples/simple_trainer.py default \
        --data-dir "$colmap_dir" \
        --data-factor 2 \
        --test-views cam00 \
        --disable-viewer \
        --max-steps 5000 \
        --strategy.refine-start-iter 300 \
        --strategy.refine-stop-iter 3000 \
        --strategy.refine-every 100 \
        --strategy.reset-every 3000 \
        --eval-steps 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 \
        --save-steps 5000 \
        --result-dir "$out_1"
      echo "  ✅ [1. 純 COLMAP reset=3000 (refine 300)] 完成！"
      total_completed=$((total_completed + 1))
    fi

    # ------------------------------------------------------------------
    # 2. 組別 B (14k Importance): reset=3000, refine_start=500
    # ------------------------------------------------------------------
    out_2="results/144exp/${scene}/f${frame}_3dgs_giant_colmap_14k_imp_refine500_reset3000"
    init_2="$colmap_dir/da3_giant_colmap_importance_14k.ply"
    if [ -f "$PROJECT_DIR/$out_2/stats/val_step4999.json" ]; then
      echo "  ⏩ [2. 組別 B (14k Importance, refine 500)] 已存在，跳過"
      total_skipped=$((total_skipped + 1))
    else
      if [ ! -f "$PROJECT_DIR/$init_2" ]; then
        echo "  ❌ 找不到初始點雲: $PROJECT_DIR/$init_2，請先生成點雲！"
        exit 1
      fi
      echo "  🚀 [2. 組別 B (14k Importance, refine 500)] 開始訓練 ($init_2)..."
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
        --init-ply-path "$init_2" \
        --result-dir "$out_2"
      echo "  ✅ [2. 組別 B (14k Importance, refine 500)] 完成！"
      total_completed=$((total_completed + 1))
    fi

    # ------------------------------------------------------------------
    # 3. 組別 C (14k Uniform): reset=3000, refine_start=500
    # ------------------------------------------------------------------
    out_3="results/144exp/${scene}/f${frame}_3dgs_giant_colmap_14k_uni_refine500_reset3000"
    init_3="$colmap_dir/da3_giant_colmap_uniform_14k.ply"
    if [ -f "$PROJECT_DIR/$out_3/stats/val_step4999.json" ]; then
      echo "  ⏩ [3. 組別 C (14k Uniform, refine 500)] 已存在，跳過"
      total_skipped=$((total_skipped + 1))
    else
      if [ ! -f "$PROJECT_DIR/$init_3" ]; then
        echo "  ❌ 找不到初始點雲: $PROJECT_DIR/$init_3，請先生成點雲！"
        exit 1
      fi
      echo "  🚀 [3. 組別 C (14k Uniform, refine 500)] 開始訓練 ($init_3)..."
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
        --init-ply-path "$init_3" \
        --result-dir "$out_3"
      echo "  ✅ [3. 組別 C (14k Uniform, refine 500)] 完成！"
      total_completed=$((total_completed + 1))
    fi

    # ------------------------------------------------------------------
    # 4. 50k Importance: reset=3000, refine_start=500
    # ------------------------------------------------------------------
    out_4="results/144exp/${scene}/f${frame}_3dgs_giant_colmap_50k_imp_refine500_reset3000"
    init_4="$colmap_dir/da3_giant_colmap_importance_50k.ply"
    if [ -f "$PROJECT_DIR/$out_4/stats/val_step4999.json" ]; then
      echo "  ⏩ [4. 50k Importance (reset=3000, refine 500)] 已存在，跳過"
      total_skipped=$((total_skipped + 1))
    else
      if [ ! -f "$PROJECT_DIR/$init_4" ]; then
        echo "  ❌ 找不到初始點雲: $PROJECT_DIR/$init_4，請先生成點雲！"
        exit 1
      fi
      echo "  🚀 [4. 50k Importance (reset=3000, refine 500)] 開始訓練 ($init_4)..."
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
        --init-ply-path "$init_4" \
        --result-dir "$out_4"
      echo "  ✅ [4. 50k Importance (reset=3000, refine 500)] 完成！"
      total_completed=$((total_completed + 1))
    fi

    # ------------------------------------------------------------------
    # 5. 50k Uniform: reset=3000, refine_start=500
    # ------------------------------------------------------------------
    out_5="results/144exp/${scene}/f${frame}_3dgs_giant_colmap_50k_uni_refine500_reset3000"
    init_5="$colmap_dir/da3_giant_colmap_uniform_50k.ply"
    if [ -f "$PROJECT_DIR/$out_5/stats/val_step4999.json" ]; then
      echo "  ⏩ [5. 50k Uniform (reset=3000, refine 500)] 已存在，跳過"
      total_skipped=$((total_skipped + 1))
    else
      if [ ! -f "$PROJECT_DIR/$init_5" ]; then
        echo "  ❌ 找不到初始點雲: $PROJECT_DIR/$init_5，請先生成點雲！"
        exit 1
      fi
      echo "  🚀 [5. 50k Uniform (reset=3000, refine 500)] 開始訓練 ($init_5)..."
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
        --init-ply-path "$init_5" \
        --result-dir "$out_5"
      echo "  ✅ [5. 50k Uniform (reset=3000, refine 500)] 完成！"
      total_completed=$((total_completed + 1))
    fi

  done

  echo ""
  echo "📊 [${scene}] 訓練進度更新，即時統計成績:"
  python3 "$TRACKER_DIR/scripts/evaluate_6scenes_matrix_ablation.py" || true
done

echo ""
echo "======================================================================"
echo "🎉 實驗矩陣排程全部完成！"
echo "   新完成訓練: $total_completed 組"
echo "   已存在跳過: $total_skipped 組"
echo "   結束時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"
python3 "$TRACKER_DIR/scripts/evaluate_6scenes_matrix_ablation.py"
