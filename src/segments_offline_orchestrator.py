"""按 TSV 时间段对离线视频做手检 → 好帧门控 → 耗材识别（无分段、无撕膜）。"""
from __future__ import annotations

import gc
from argparse import Namespace
from pathlib import Path
from typing import Any

import cv2
import run_haocai_actionformer_consumables_e2e as e2e
from pipeline.segment_processor import (
    HaocaiOnlyClassifier,
    process_segment_haocai_from_cap_with_gate_retries,
)
from ultralytics import YOLO

from pack_utils import log, resolve_allowed_class_idx
from stream_orchestrator import (
    _format_result_row,
    _maybe_free_gpu,
    _resolve_haocai_min_conf_retry,
)
from tsv_segments import load_segments_from_result_tsv


def _validate_haocai_weights(args: Namespace) -> bool:
    for p, lab in (
        (args.hand_model, "手部检测"),
        (args.goodbad_model, "好坏帧"),
        (args.haocai_model, "耗材分类"),
    ):
        if not Path(p).is_file():
            log(f"缺少{lab}: {p}")
            return False
    return True


def run_segments_offline_pipeline(args: Namespace) -> int:
    video_path = Path(args.video).resolve()
    if not video_path.is_file():
        log(f"找不到视频: {video_path}")
        return 1

    excel_path = Path(args.excel).resolve()
    if not excel_path.is_file():
        log(f"找不到 Excel: {excel_path}")
        return 1

    tsv_path = Path(args.segments_tsv).resolve()
    if not tsv_path.is_file():
        log(f"找不到时间段 TSV: {tsv_path}")
        return 1

    if not _validate_haocai_weights(args):
        return 1

    segs = load_segments_from_result_tsv(
        tsv_path,
        skip_empty_top1=bool(getattr(args, "segments_skip_empty", False)),
    )
    if not segs:
        log("TSV 未解析到任何有效时间段")
        return 1

    product_map = e2e.load_product_code_map(excel_path)
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    predict_kw: dict[str, Any] = {"device": args.device}
    if args.half:
        predict_kw["half"] = True

    log("[segments-offline] 加载 YOLO（手 / 好坏帧 / 耗材）…")
    from hand_detector import create_hand_detector

    det = create_hand_detector(args)
    gb = YOLO(str(args.goodbad_model))
    cls_m = YOLO(str(args.haocai_model))
    hc = HaocaiOnlyClassifier(
        cls_m,
        cls_names=cls_m.names,
        imgsz_cls=int(args.imgsz_cls),
        predict_kw=predict_kw,
        gb=gb,
        gb_names=gb.names,
    )
    try:
        allowed_idx = resolve_allowed_class_idx(args, excel_path, cls_m.names)
    except FileNotFoundError as exc:
        log(str(exc))
        return 1
    if getattr(args, "use_whitelist", True):
        log(f"[segments-offline] 白名单启用，{len(allowed_idx or ())} 个类参与投票")
    else:
        log("[segments-offline] 白名单已关闭，使用全 41 类")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log("无法打开视频")
        return 1

    header = "\t".join(
        [
            "rank",
            "start_sec",
            "end_sec",
            "product_id_top1",
            "top1_name",
            "top1_conf",
            "product_id_top2",
            "top2_name",
            "top2_conf",
            "product_id_top3",
            "top3_name",
            "top3_conf",
        ]
    )
    lines_out = [header]

    try:
        for rank, (t0, t1, _sc) in enumerate(segs, start=1):
            log(f"[segments-offline] rank={rank} [{t0:.3f},{t1:.3f}] …")
            info = process_segment_haocai_from_cap_with_gate_retries(
                cap,
                det,
                hc,
                start_sec=t0,
                end_sec=t1,
                seek_margin_sec=float(args.seek_margin_sec),
                det_conf=float(args.det_conf),
                pad_ratio=float(args.pad_ratio),
                imgsz_det=int(args.imgsz_det),
                frame_stride=max(1, int(args.frame_stride)),
                haocai_min_conf=float(args.haocai_min_conf),
                haocai_min_conf_retry=_resolve_haocai_min_conf_retry(args),
                good_top1_conf_threshold=float(args.good_top1_conf_threshold),
                good_top1_retry_threshold=float(args.good_top1_retry_threshold),
                cls_names=cls_m.names,
                allowed_class_idx=allowed_idx,
                predict_kw=predict_kw,
                log_fn=log,
                log_prefix=f"[segments-offline] rank={rank}: ",
            )
            lines_out.append(
                _format_result_row(
                    rank,
                    t0,
                    t1,
                    info,
                    product_map,
                    legacy_12_col=bool(args.legacy_12_col_only),
                )
            )
            _maybe_free_gpu()
    finally:
        cap.release()
        gc.collect()

    out_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    log(f"[segments-offline] 完成，共 {len(segs)} 段，结果: {out_path}")
    return 0
