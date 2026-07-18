#!/usr/bin/env bash
# 启动真 RTSP 推流识别（raw BGR 环缓 + Producer/Consumer，configs/default_config.yaml）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RTSP="${1:-}"
OUT="${2:-output/rtsp_stream.txt}"
ROI_SAVE="${3:-output/basket_roi_rtsp.json}"
ROI_LOAD="${4:-}"

CONFIG="configs/default_config.yaml"
EXTRA=()
if [[ -n "$ROI_LOAD" && -f "$ROI_LOAD" ]]; then
  EXTRA+=(--basket-roi-json "$ROI_LOAD")
fi
if [[ -n "$RTSP" ]]; then
  EXTRA+=(--rtsp "$RTSP")
fi

echo "[run] 推流识别 → $OUT"
echo "[run] 配置: $CONFIG"
if [[ ${#EXTRA[@]} -gt 0 ]]; then
  echo "[run] 参数: ${EXTRA[*]}"
fi

export DISPLAY="${DISPLAY:-:0}"
export OPENCV_FFMPEG_LOGLEVEL="${OPENCV_FFMPEG_LOGLEVEL:-8}"

python -u main_basket_stream.py \
  --excel "input/视频中的商品信息表.xlsx" \
  --out "$OUT" \
  --save-basket-roi "$ROI_SAVE" \
  --config "$CONFIG" \
  "${EXTRA[@]}" \
  2>&1 | tee "${OUT%.txt}.log"
