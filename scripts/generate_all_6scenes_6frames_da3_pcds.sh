#!/bin/bash
# ======================================================================
# 🚀 N3D 完整 6 場景 x 6 影格 (共 36 個關鍵幀) DA3-GIANT 解耦點雲批次生成
#
# 6 個場景:
#   1. coffee_martini
#   2. cook_spinach
#   3. cut_roasted_beef
#   4. flame_salmon_1
#   5. flame_steak
#   6. sear_steak
#
# 6 個評估影格: 25, 75, 125, 175, 225, 275 (共 36 幀)
#
# 點數預算與規格:
#   - 14000 (14k):
#       * da3_giant_colmap_importance_14k.ply (7k COLMAP + 7k DA3 Importance) -> 組別 B
#       * da3_giant_colmap_uniform_14k.ply    (7k COLMAP + 7k DA3 Uniform)    -> 組別 C
#       * da3_giant_colmap_importance_7k.ply  (3.5k COLMAP + 3.5k DA3 Importance) -> 7k 相容組
#   - 50000 (50k):
#       * da3_giant_colmap_importance_50k.ply (7k COLMAP + 43k DA3 Importance) -> 50k 組
#       * da3_giant_colmap_uniform_50k.ply    (7k COLMAP + 43k DA3 Uniform)
#
# 核心參數 (零插值原生幾何管線):
#   - 相機挑選模式: fps (最遠點採樣均勻視角，穩定可靠且避開特定窗戶玻璃法向偽影)
#   - DA3 幾何模型: depth-anything/DA3-GIANT-1.1 (啟用分層推論 Sequential Block Offload)
#   - 原生解析度: process_res=728 (輸出 546x728，無任何深度/信心/法向插值失真)
#   - 玻璃反光去噪: 2D 前置信心過濾 (conf_thresh >= 5.0)，徹底杜絕窗戶/鏡面懸浮噪點
#   - 融合策略: decoupled (100% 保留 COLMAP 錨點)
# ======================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

SCENES=(
  "coffee_martini"
  "cook_spinach"
  "cut_roasted_beef"
  "flame_salmon_1"
  "flame_steak"
  "sear_steak"
)

FRAMES=(25 75 125 175 225 275)

echo "======================================================================"
echo "🎯 開始執行 6 場景 x 6 影格 (共 36 幀) DA3-GIANT 點雲批次生成"
echo "   時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "   場景: ${SCENES[*]}"
echo "   影格: ${FRAMES[*]}"
echo "   相機挑選模式: fps (最遠點採樣均勻視角)"
echo "   模型: DA3-GIANT-1.1 (res=728 原生推論)"
echo "   預算: 14000 (14k: 7k COLMAP + 7k DA3), 50000 (50k)"
echo "======================================================================"

docker compose run --user $(id -u):$(id -g) --rm mvtracker python3 scripts/generate_uniform_da3_pcds.py \
  --scenes "${SCENES[@]}" \
  --frames "${FRAMES[@]}" \
  --budgets 14000 50000 \
  --cam_select_mode fps \
  --process_res 728 \
  --da3_model depth-anything/DA3-GIANT-1.1 \
  --conf_thresh 5.0 \
  --conf_filter_mode 2d \
  --colmap_strategy decoupled \
  --overwrite

echo ""
echo "======================================================================"
echo "🎉 6 場景 x 6 影格 (共 36 幀) 全部點雲生成完成！"
echo "   時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"
