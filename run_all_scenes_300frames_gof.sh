#!/bin/bash
# ==============================================================================
# N3D 全 6 場景完整 300 幀 4D 靜動態分離批次執行腳本 (GOF 50 幀分段，不產出 RRD)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "=============================================================================="
echo "🚀 [N3D 4D Separation Pipeline] 開始全 6 場景 300 幀批次執行 (GOF 50 幀/區塊)"
echo "   場景清單: coffee_martini cook_spinach cut_roasted_beef flame_salmon_1 flame_steak sear_steak"
echo "   模型版本: depth-anything/DA3-GIANT-1.1 (Sequential Block Offload)"
echo "   Rerun 模式: 關閉 RRD 輸出 (--no_rrd，極速產出點雲與統計報告)"
echo "   既有略過: 已啟用 --skip_existing (若該 GOF 點雲已存在則自動跳過)"
echo "=============================================================================="

env UID=$(id -u) GID=$(id -g) docker compose run --rm mvtracker \
    python3 scripts/batch_run_gof_chunks.py \
    --scenes coffee_martini cook_spinach cut_roasted_beef flame_salmon_1 flame_steak sear_steak \
    --total_frames 300 \
    --gof_size 50 \
    --frame_step 1 \
    --da3_model depth-anything/DA3-GIANT-1.1 \
    --conf_thresh 5.0 \
    --no_rrd

echo "=============================================================================="
echo "🎉 全 6 場景 300 幀 GOF 分段產出與耗時統計全部完成！"
echo "=============================================================================="
