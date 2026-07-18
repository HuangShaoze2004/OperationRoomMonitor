#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "=== 耗材推流识别 + 医生身份识别 — 环境检查 ==="

if ! python3 -c "import tkinter" 2>/dev/null; then
  echo "警告: 未检测到 python3-tk，框选篮子 ROI 会失败。"
  echo "  Ubuntu/Debian: sudo apt install python3-tk"
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "提示: 未检测到 ffmpeg（RTSP 录制脚本需要，推流识别可选）。"
fi

echo ""
echo "=== Python 依赖 ==="
echo "1. 按 https://pytorch.org 安装与 CUDA 匹配的 torch / torchvision"
echo "2. pip install -r requirements.txt"
echo ""
if command -v conda >/dev/null 2>&1; then
  echo "检测到 conda，推荐:"
  echo "  conda activate yolo"
  echo "  pip install -r requirements.txt"
else
  echo "未检测到 conda，可使用 venv:"
  echo "  python3 -m venv .venv && source .venv/bin/activate"
  echo "  pip install -U pip && pip install -r requirements.txt"
fi

echo ""
echo "=== 权重与数据检查 ==="
missing=0
for w in hand_detect.pt goodbad_frame.pt haocai_classify.pt; do
  if test -f "weights/$w"; then
    echo "  OK weights/$w"
  else
    echo "  缺失 weights/$w"
    missing=1
  fi
done
if test -f "yolo11n.pt"; then
  echo "  OK yolo11n.pt（医生人体检测）"
else
  echo "  缺失 yolo11n.pt"
  missing=1
fi
if test -f "doctor_identity_package/doctor_reid_best.pth"; then
  echo "  OK doctor_identity_package/doctor_reid_best.pth（医生 ReID）"
else
  echo "  缺失 doctor_identity_package/doctor_reid_best.pth"
  missing=1
fi
if test -f "doctor_identity_package/labels.csv"; then
  echo "  OK doctor_identity_package/labels.csv"
else
  echo "  缺失 doctor_identity_package/labels.csv"
  missing=1
fi
if test -f "input/视频中的商品信息表.xlsx"; then
  echo "  OK input/视频中的商品信息表.xlsx"
else
  echo "  缺失 input/视频中的商品信息表.xlsx"
  missing=1
fi

echo ""
if [[ "$missing" -eq 0 ]]; then
  echo "资源检查通过。启动推流识别:"
  echo "  bash scripts/run_rtsp_stream.sh"
else
  echo "存在缺失文件，请先补齐后再运行。"
fi
echo ""
echo "详细说明见 README.md"
