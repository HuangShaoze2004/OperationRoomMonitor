#!/usr/bin/env python3
"""推流篮子耗材识别：弹窗框选 ROI → RTSP 逐帧触发 → 缓存 [contact+1,contact+6] → 耗材识别。"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACK_ROOT / "src"))

from paths import ensure_code_on_path

ensure_code_on_path(PACK_ROOT)

from config import load_run_config
from stream_orchestrator import run_stream_pipeline


def main() -> int:
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")
    ap = argparse.ArgumentParser(description="推流篮子耗材识别（无撕膜）")
    ap.add_argument(
        "--rtsp",
        type=str,
        default=None,
        help="RTSP/摄像头 URL；本地 mp4 也可用于测试",
    )
    ap.add_argument(
        "--excel",
        type=Path,
        required=True,
        help="商品表 Excel（C 列白名单 + 产品编码）",
    )
    ap.add_argument("--out", type=Path, required=True, help="输出 TSV（实时追加）")
    ap.add_argument(
        "--config",
        type=Path,
        default=PACK_ROOT / "configs" / "default_config.yaml",
        help="配置文件",
    )
    ap.add_argument(
        "--save-basket-roi",
        type=Path,
        default=None,
        help="框选后将 ROI 保存为 JSON（可选；每次运行仍会先弹窗标框）",
    )
    ap.add_argument(
        "--segment-start-offset-sec",
        type=float,
        default=None,
        help="段起点相对 contact 偏移（默认读 yaml，与 basket 一致 → contact+1）",
    )
    ap.add_argument(
        "--segment-end-offset-sec",
        type=float,
        default=None,
        help="段终点相对 contact 偏移（默认读 yaml，与 basket 一致 → contact+6，窗口 5s）",
    )
    ap.add_argument(
        "--min-segment-sec",
        type=float,
        default=None,
        help="段长不足此值则丢弃（默认 4.0）",
    )
    ap.add_argument(
        "--ring-buffer-sec",
        type=float,
        default=None,
        help="帧环形缓存时长（秒，默认 15）",
    )
    ap.add_argument(
        "--stream-fps",
        type=float,
        default=None,
        help="RTSP 无 FPS 元数据时的假定帧率（默认 25）",
    )
    ap.add_argument(
        "--infer-workers",
        type=int,
        default=None,
        help="段级耗材推理线程数（默认 1）",
    )
    ap.add_argument(
        "--basket-roi-json",
        type=Path,
        default=None,
        help="从 JSON 加载篮子 ROI，跳过弹窗（与 --save-basket-roi 配套复用）",
    )
    ap.add_argument(
        "--warmup-skip-frames",
        type=int,
        default=None,
        help="RTSP 预热丢弃帧数（默认读 yaml stream.warmup_skip_frames，仅实时流）",
    )
    args = ap.parse_args()

    cfg_path = args.config.resolve()
    if not cfg_path.is_file():
        print("找不到配置:", cfg_path, file=sys.stderr)
        return 1

    run_cfg = load_run_config(PACK_ROOT, cfg_path)
    run_cfg.excel = args.excel.resolve()
    run_cfg.out = args.out.resolve()

    rtsp = args.rtsp or getattr(run_cfg, "stream_rtsp", None)
    if not rtsp:
        print("请指定 --rtsp 或在 yaml stream.rtsp 中配置", file=sys.stderr)
        return 1
    run_cfg.stream_rtsp = str(rtsp)

    if args.basket_roi_json is not None:
        run_cfg.basket_load_roi_json = args.basket_roi_json.resolve()
        run_cfg.basket_skip_roi_select = True
    else:
        run_cfg.basket_load_roi_json = None
        run_cfg.basket_skip_roi_select = False
    if args.save_basket_roi is not None:
        run_cfg.basket_save_roi_json = args.save_basket_roi.resolve()
    if args.segment_start_offset_sec is not None:
        run_cfg.stream_segment_start_offset_sec = float(args.segment_start_offset_sec)
    if args.segment_end_offset_sec is not None:
        run_cfg.stream_segment_end_offset_sec = float(args.segment_end_offset_sec)
    if args.min_segment_sec is not None:
        run_cfg.stream_min_segment_sec = float(args.min_segment_sec)
    if args.ring_buffer_sec is not None:
        run_cfg.stream_ring_buffer_sec = float(args.ring_buffer_sec)
    if args.stream_fps is not None:
        run_cfg.stream_fps = float(args.stream_fps)
    if args.infer_workers is not None:
        run_cfg.stream_infer_workers = int(args.infer_workers)
    if args.warmup_skip_frames is not None:
        run_cfg.stream_warmup_skip_frames = int(args.warmup_skip_frames)

    return int(run_stream_pipeline(run_cfg))


if __name__ == "__main__":
    raise SystemExit(main())
