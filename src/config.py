"""加载 configs/*.yaml，解析为运行参数 Namespace。"""
from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import yaml


def _rel(pack_root: Path, raw: str | None) -> Path | None:
    if raw is None:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return (pack_root / path).resolve()


def load_run_config(pack_root: Path, config_path: Path) -> Namespace:
    pack_root = pack_root.resolve()
    data: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    io = data["io"]
    w = data.get("weights", {})
    rt = data.get("runtime", {})
    dev = data.get("device", {})
    p2 = data["phase2"]
    cl = data["classification"]
    gm = data.get("gap_merge", {})
    outopt = data.get("output", {})
    did = data.get("doctor_identity", {})
    bk = data.get("basket", {})
    st = data.get("stream", {})

    py = rt.get("python")
    python_exe = sys.executable if py is None or str(py).strip() == "" else str(py)

    whitelist_raw = io.get("whitelist_json")
    whitelist_path = _rel(pack_root, whitelist_raw) if whitelist_raw else None

    work_raw = rt.get("work_dir")
    work_dir = _rel(pack_root, work_raw) if work_raw else None

    doctor_ckpt_raw = did.get("checkpoint", "doctor_identity_package/doctor_reid_best.pth")
    doctor_labels_raw = did.get("labels_csv", "doctor_identity_package/labels.csv")

    basket_save_raw = bk.get("save_roi_json")
    basket_load_raw = bk.get("load_roi_json")
    basket_roi_frame = bk.get("roi_frame", "middle")
    if isinstance(basket_roi_frame, (int, float)):
        basket_roi_frame = float(basket_roi_frame)
    else:
        basket_roi_frame = str(basket_roi_frame)

    legacy_contact_iou = float(bk.get("contact_iou_threshold", 0.05))
    on_raw = bk.get("contact_iou_on")
    off_raw = bk.get("contact_iou_off")
    basket_contact_iou_on = float(on_raw) if on_raw is not None else legacy_contact_iou
    basket_contact_iou_off = (
        float(off_raw) if off_raw is not None else max(legacy_contact_iou * 0.6, 0.01)
    )
    if basket_contact_iou_off >= basket_contact_iou_on:
        basket_contact_iou_off = max(basket_contact_iou_on - 0.02, 0.01)

    pad_bottom = float(p2.get("pad_bottom_ratio", p2.get("pad_ratio", 0.5)))

    # 篮子/推流默认不用；main.py（ActionFormer）或撕膜合并可在 yaml 中另行配置
    actionformer_raw = w.get("actionformer")
    tear_raw = w.get("tear")
    p1 = data.get("phase1", {})
    tm = data.get("tear_merge", {})
    hd = data.get("hand", {})
    hand_backend = str(hd.get("backend", "yolo")).strip().lower()
    hand_mp_task_raw = hd.get(
        "mediapipe_task", "weights/hand_landmarker.task"
    )

    return Namespace(
        pack_root=pack_root,
        video=_rel(pack_root, io["video"]),
        excel=_rel(pack_root, io["excel"]),
        out=_rel(pack_root, io["out"]),
        whitelist_json=whitelist_path,
        use_whitelist=bool(io.get("use_whitelist", True)),
        work_dir=work_dir,
        keep_work_dir=bool(rt.get("keep_work_dir", False)),
        python=python_exe,
        actionformer_ckpt=_rel(pack_root, actionformer_raw) if actionformer_raw else None,
        hand_model=_rel(pack_root, w["hand"]),
        hand_backend=hand_backend,
        hand_mediapipe_task=_rel(pack_root, hand_mp_task_raw),
        hand_mediapipe_num_hands=int(hd.get("mediapipe_num_hands", 2)),
        hand_mediapipe_min_detection_confidence=float(
            hd.get("mediapipe_min_detection_confidence", 0.3)
        ),
        hand_mediapipe_min_presence_confidence=float(
            hd.get("mediapipe_min_presence_confidence", 0.3)
        ),
        hand_mediapipe_min_tracking_confidence=float(
            hd.get("mediapipe_min_tracking_confidence", 0.3)
        ),
        hand_mediapipe_bbox_margin=float(hd.get("mediapipe_bbox_margin", 0.05)),
        goodbad_model=_rel(pack_root, w["goodbad"]),
        haocai_model=_rel(pack_root, w["haocai"]),
        tear_model=_rel(pack_root, tear_raw) if tear_raw else None,
        device=str(dev.get("type", "cuda")),
        half=bool(dev.get("half", False)),
        af_min_score=float(p1.get("af_min_score", 0.1)),
        af_min_seg_seconds=float(p1.get("af_min_seg_seconds", 2.0)),
        feat_batch_size=int(p1.get("feat_batch_size", 1)),
        seek_margin_sec=float(p2["seek_margin_sec"]),
        frame_stride=int(p2["frame_stride"]),
        det_conf=float(p2["det_conf"]),
        pad_bottom_ratio=pad_bottom,
        pad_ratio=pad_bottom,
        imgsz_det=int(p2["imgsz_det"]),
        merge_iou_gt=float(p2["merge_iou_gt"]),
        merge_center_dist_max_px=(
            float(p2["merge_center_dist_max_px"])
            if p2.get("merge_center_dist_max_px") is not None
            else None
        ),
        merge_center_dist_max_frac_diag=(
            float(p2["merge_center_dist_max_frac_diag"])
            if p2.get("merge_center_dist_max_frac_diag") is not None
            else None
        ),
        tracking_alpha=float(p2.get("tracking_alpha", 0.6)),
        tracking_max_lost_frames=int(p2.get("tracking_max_lost_frames", 0)),
        imgsz_cls=int(cl["imgsz_cls"]),
        good_top1_conf_threshold=float(cl["good_top1_conf_threshold"]),
        good_top1_retry_threshold=float(cl.get("good_top1_retry_threshold", 0)),
        haocai_min_conf=float(cl["haocai_min_conf"]),
        haocai_min_conf_retry=(
            float(cl["haocai_min_conf_retry"])
            if cl.get("haocai_min_conf_retry") is not None
            else None
        ),
        empty_cache_every=int(cl.get("empty_cache_every", 0)),
        legacy_12_col_only=bool(outopt.get("legacy_12_col_only", True)),
        merge_adjacent_tear=bool(tm.get("merge_adjacent_tear", False)),
        tear_merge_weights=_rel(pack_root, tm["tear_merge_weights"])
        if tm.get("tear_merge_weights")
        else None,
        tear_merge_class=str(tm.get("tear_merge_class", "tearing")),
        tear_merge_head_sec=float(tm.get("tear_merge_head_sec", 3.0)),
        tear_merge_prob=float(tm.get("tear_merge_prob", 0.9)),
        tear_merge_min_frames=int(tm.get("tear_merge_min_frames", 6)),
        tear_merge_verbose=bool(tm.get("tear_merge_verbose", False)),
        tear_merge_full_frame=bool(tm.get("tear_merge_full_frame", False)),
        gap_merge_enabled=bool(gm.get("enabled", False)),
        gap_merge_max_gap_sec=float(gm.get("max_gap_sec", 2.0)),
        doctor_identity_enabled=bool(did.get("enabled", True)),
        doctor_identity_stream_enabled=bool(did.get("stream_enabled", True)),
        doctor_identity_checkpoint=_rel(pack_root, doctor_ckpt_raw),
        doctor_identity_labels_csv=_rel(pack_root, doctor_labels_raw),
        doctor_identity_person_yolo_weights=_rel(
            pack_root, did.get("person_yolo_weights", "yolo11n.pt")
        ),
        doctor_identity_person_det_conf=float(
            did.get(
                "person_det_conf",
                did.get("pose_min_detection_confidence", 0.65),
            )
        ),
        doctor_identity_person_det_imgsz=int(did.get("person_det_imgsz", 1280)),
        doctor_identity_min_identity_confidence=float(did.get("min_identity_confidence", 0.0)),
        doctor_identity_middle_seconds=float(did.get("middle_seconds", 10.0)),
        doctor_identity_sample_fps=float(did.get("sample_fps", 3.0)),
        doctor_identity_segment_sample_fps=float(
            did.get("segment_sample_fps", did.get("sample_fps", 3.0))
        ),
        doctor_identity_segment_window_sec=float(did.get("segment_window_sec", 3.0)),
        doctor_identity_pad_frac=float(did.get("pad_frac", 0.15)),
        basket_det_conf=float(bk.get("det_conf", p2["det_conf"])),
        basket_contact_iou_threshold=legacy_contact_iou,
        basket_contact_iou_on=basket_contact_iou_on,
        basket_contact_iou_off=basket_contact_iou_off,
        basket_confirm_seconds=float(bk.get("confirm_seconds", 0.4)),
        basket_cooldown_seconds=float(bk.get("cooldown_seconds", 5.0)),
        basket_segment_start_offset_sec=float(bk.get("segment_start_offset_sec", 1.0)),
        basket_segment_end_offset_sec=float(bk.get("segment_end_offset_sec", 5.0)),
        basket_min_segment_sec=float(bk.get("min_segment_sec", 4.0)),
        basket_scan_frame_stride=int(bk.get("scan_frame_stride", 1)),
        basket_roi_frame=basket_roi_frame,
        basket_save_roi_json=_rel(pack_root, basket_save_raw) if basket_save_raw else None,
        basket_load_roi_json=_rel(pack_root, basket_load_raw) if basket_load_raw else None,
        basket_skip_roi_select=bool(bk.get("skip_roi_select", False)),
        basket_roi_backend=str(bk.get("roi_backend", "tkinter")),
        basket_contact_tracking_enabled=bool(bk.get("contact_tracking_enabled", True)),
        basket_contact_tracker=str(bk.get("contact_tracker", "bytetrack")).strip().lower(),
        basket_contact_track_buffer=int(bk.get("contact_track_buffer", 30)),
        stream_rtsp=st.get("rtsp"),
        stream_rtsp_transport=str(st.get("rtsp_transport", "tcp")).strip().lower(),
        stream_rtsp_buffer_size=int(st.get("rtsp_buffer_size", 1)),
        stream_ring_buffer_sec=float(st.get("ring_buffer_sec", 10.0)),
        stream_fps=float(st.get("fps", 25.0)),
        stream_ffmpeg_low_latency=bool(st.get("ffmpeg_low_latency", True)),
        stream_infer_workers=int(st.get("infer_workers", 1)),
        stream_segment_start_offset_sec=float(
            st.get("segment_start_offset_sec", bk.get("segment_start_offset_sec", 1.0))
        ),
        stream_segment_end_offset_sec=float(
            st.get("segment_end_offset_sec", bk.get("segment_end_offset_sec", 6.0))
        ),
        stream_min_segment_sec=float(
            st.get("min_segment_sec", bk.get("min_segment_sec", 4.0))
        ),
        stream_infer_source=str(st.get("infer_source", "file")).strip().lower(),
        stream_infer_fallback=str(st.get("infer_fallback", "cache")).strip().lower(),
        stream_warmup_skip_frames=int(st.get("warmup_skip_frames", 0)),
    )
