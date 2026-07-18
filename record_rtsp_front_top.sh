#!/usr/bin/env bash
# 独立脚本：将 front_top RTSP 流录制为本地 MP4
# 默认保存到 /home/baitian/桌面/
#
# 用法:
#   ./record_rtsp_front_top.sh           # 一直录，Ctrl+C 停止
#   ./record_rtsp_front_top.sh 60        # 录 60 秒
#   ./record_rtsp_front_top.sh 120 a.mp4 # 录 120 秒，指定文件名

set -euo pipefail

RTSP_URL="rtsp://192.168.3.140:8554/front_top_test"
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${RECORD_OUT_DIR:-$ROOT/output/recordings}"
RTSP_TRANSPORT="tcp"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "错误: 未找到 ffmpeg，请先安装: sudo apt install ffmpeg" >&2
  exit 1
fi

DURATION=""
OUT_NAME=""

if [[ $# -ge 1 && "$1" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  DURATION="$1"
  OUT_NAME="${2:-}"
elif [[ $# -ge 1 ]]; then
  OUT_NAME="$1"
else
  :
fi

mkdir -p "$OUT_DIR"

if [[ -n "$OUT_NAME" ]]; then
  OUT_FILE="${OUT_DIR%/}/${OUT_NAME}"
else
  STAMP="$(date +%Y%m%d_%H%M%S)"
  OUT_FILE="${OUT_DIR%/}/front_top_test_${STAMP}.mp4"
fi

echo "========================================"
echo " RTSP 录制"
echo " 地址: ${RTSP_URL}"
echo " 输出: ${OUT_FILE}"
if [[ -n "$DURATION" ]]; then
  echo " 时长: ${DURATION} 秒"
else
  echo " 时长: 手动停止 (Ctrl+C)"
fi
echo "========================================"

FFMPEG_ARGS=(
  -hide_banner
  -loglevel info
  -rtsp_transport "$RTSP_TRANSPORT"
  -stimeout 5000000
  -i "$RTSP_URL"
  -c copy
  -movflags +faststart
)

if [[ -n "$DURATION" ]]; then
  FFMPEG_ARGS+=(-t "$DURATION")
fi

FFMPEG_ARGS+=(-y "$OUT_FILE")

trap 'echo; echo "正在停止录制…"; exit 130' INT TERM

ffmpeg "${FFMPEG_ARGS[@]}"
echo "完成: ${OUT_FILE}"
