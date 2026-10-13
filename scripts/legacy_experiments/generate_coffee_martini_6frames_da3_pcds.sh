#!/bin/bash
# ======================================================================
# 🚀 coffee_martini 6 個關鍵幀 DA3-GIANT 解耦點雲批次生成腳本
#
# 目標幀列表: 25, 75, 125, 175, 225, 275
#
# 生成預算與點雲規格:
#   1. 14000 (14k):
#      - da3_giant_colmap_importance_14k.ply (7k COLMAP + 7k DA3 Importance) -> 組別 B 新版
#      - da3_giant_colmap_uniform_14k.ply    (7k COLMAP + 7k DA3 Uniform)    -> 組別 C
#      - da3_giant_colmap_importance_7k.ply  (3.5k COLMAP + 3.5k DA3 Importance) -> 組別 B 舊版相容
#   2. 50000 (50k):
#      - da3_giant_colmap_importance_50k.ply (7k COLMAP + 43k DA3 Importance) -> 50k 組
#      - da3_giant_colmap_uniform_50k.ply    (7k COLMAP + 43k DA3 Uniform)
#
# 核心參數 (DA3-GIANT 零插值原生推論):
#   - 模型: depth-anything/DA3-GIANT-1.1 (啟用分層推論 Sequential Block Offload)
#   - 推理解析度: process_res=728 (原生輸出 546x728，無任何深度/信心/法向插值失真)
#   - 相機選擇模式: fps (外參最遠點採樣，挑選 5 視角: cam01, cam06, cam09, cam14, cam20)
#   - 信心過濾門檻: 2D 前置過濾 >= 5.0 (徹底杜絕玻璃深層懸浮噪點)
#   - 特徵權重檔: datasets/coffee_martini/weights_coffee_martini_f25_surfel.json
# ======================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "======================================================================"
echo "🎯 開始批次生成 coffee_martini 6 個關鍵幀之 DA3-GIANT 點雲"
echo "   時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "   影格: 25 75 125 175 225 275"
echo "   預算: 14000 (14k: 7k COLMAP + 7k DA3), 50000 (50k)"
echo "======================================================================"

docker compose run --user $(id -u):$(id -g) --rm mvtracker python3 scripts/generate_uniform_da3_pcds.py \
  --scenes coffee_martini \
  --frames 25 75 125 175 225 275 \
  --budgets 14000 50000 \
  --weights_json datasets/coffee_martini/weights_coffee_martini_f25_surfel.json \
  --cam_select_mode fps \
  --process_res 728 \
  --da3_model depth-anything/DA3-GIANT-1.1 \
  --conf_thresh 5.0 \
  --conf_filter_mode 2d \
  --colmap_strategy decoupled \
  --overwrite

echo ""
echo "======================================================================"
echo "🎉 6 個影格全部點雲生成完成！"
echo "======================================================================"
