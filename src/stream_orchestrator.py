"""RTSP 推流篮子耗材识别编排（无撕膜模型 / 无 tear_merge）。"""
from __future__ import annotations

import gc
import threading
from argparse import Namespace
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import run_haocai_actionformer_consumables_e2e as e2e
from action_trigger_logic import ActionTriggerLogic
from pipeline.segment_processor import (
    HaocaiOnlyClassifier,
    REASON_NO_HANDS_IN_SEGMENT,
    REASON_NO_VALID_HAOCAI_FRAMES,
    process_segment_haocai_from_cap,
    process_segment_haocai_from_frames,
)
from ultralytics import YOLO

from basket_segmenter import (
    _roi_xyxy_from_select,
    _scale_frame_for_display,
    _select_basket_roi_tkinter,
    load_basket_roi_json,
    save_basket_roi_json,
)
from doctor_face_identity import DoctorFaceIdentityService
from doctor_identity import (
    DoctorIdentityService,
    stream_doctor_enabled,
    vote_doctor_from_segment_results,
)
from pack_utils import log, resolve_allowed_class_idx
from stream_basket_session import CachedClip, StreamBasketSession
from stream_capture import (
    is_local_media,
    is_rtsp_url,
    open_stream_capture,
    probe_capture,
    skip_warmup_frames,
)
from stream_frame_buffer import RawFrameRingBuffer
from stream_ingest import StreamIngestPipeline

_HAOCAI_COLS = [
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


def _stream_segment_window_sec(args: Namespace) -> float:
    """段识别窗口时长 end_offset - start_offset（当前配置为 7s）。"""
    end_off = float(
        getattr(
            args,
            "stream_segment_end_offset_sec",
            getattr(args, "basket_segment_end_offset_sec", 10.0),
        )
    )
    start_off = float(
        getattr(
            args,
            "stream_segment_start_offset_sec",
            getattr(args, "basket_segment_start_offset_sec", 3.0),
        )
    )
    return max(0.0, end_off - start_off)


class StreamTsvWriter:
    """
    推流 TSV：每段耗材识别完立即追加一行；医生仅内存投票，结束时写末行汇总。
    """

    def __init__(self, out_path: Path) -> None:
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._header = "\t".join(_HAOCAI_COLS)
        self.body_lines: list[str] = []
        self.out_path.write_text(self._header + "\n", encoding="utf-8")

    def append_segment(self, line: str, *, rank: int) -> None:
        with self.out_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        self.body_lines.append(line)
        log(f"[stream] rank={rank} 耗材已写入 {self.out_path}")

    def finalize(
        self,
        doctor_votes: list[dict[str, Any] | None],
        *,
        args: Namespace,
        collect_doctor: bool,
    ) -> int:
        min_dur = _stream_segment_window_sec(args)
        dropped_last = False
        votes = list(doctor_votes)
        body = list(self.body_lines)

        if body:
            parts = body[-1].split("\t")
            if len(parts) >= 3:
                t0, t1 = float(parts[1]), float(parts[2])
                if t1 - t0 < min_dur - 1e-9:
                    log(
                        f"[stream] 末段 [{t0:.3f},{t1:.3f}] 时长 {t1 - t0:.3f}s < {min_dur:g}s，"
                        "判为误触已丢弃"
                    )
                    body = body[:-1]
                    if votes:
                        votes = votes[:-1]
                    dropped_last = True

        doctor_line: str | None = None
        if collect_doctor:
            summary = vote_doctor_from_segment_results(votes)
            doctor_line = f"医生信息：{summary}"
            log(f"[stream] 医生投票汇总：{summary}")

        if dropped_last:
            lines = [self._header] + body
            if doctor_line is not None:
                lines.append(doctor_line)
            self.out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        elif doctor_line is not None:
            with self.out_path.open("a", encoding="utf-8") as f:
                f.write(doctor_line + "\n")

        n = len(body)
        if dropped_last:
            log(f"[stream] 有效段数 {n}（已丢弃末段误触）")
        return n


def _finalize_stream_output(
    out_path: Path,
    body_lines: list[str],
    doctor_votes: list[dict[str, Any] | None],
    *,
    args: Namespace,
    collect_doctor: bool,
) -> int:
    """兼容测试：一次性写盘后追加医生汇总。"""
    writer = StreamTsvWriter(out_path)
    writer.body_lines = list(body_lines)
    writer.out_path.write_text(
        writer._header + "\n" + "\n".join(body_lines) + ("\n" if body_lines else ""),
        encoding="utf-8",
    )
    return writer.finalize(
        doctor_votes, args=args, collect_doctor=collect_doctor
    )


def _validate_stream_weights(args: Namespace) -> bool:
    from hand_detector import validate_hand_assets

    ok, hand_lab = validate_hand_assets(args)
    if not ok:
        log(hand_lab)
        return False
    for p, lab in (
        (args.goodbad_model, "好坏帧"),
        (args.haocai_model, "耗材分类"),
    ):
        if not Path(p).is_file():
            log(f"缺少{lab}: {p}")
            return False
    return True


def _resolve_basket_roi(
    args: Namespace,
    first_frame,
    *,
    t_sec: float = 0.0,
) -> list[float]:
    load_json = getattr(args, "basket_load_roi_json", None)
    if load_json is not None and Path(load_json).is_file():
        roi = load_basket_roi_json(Path(load_json))
        log(f"[stream] 从 JSON 加载篮子 ROI: {load_json} xyxy={roi}")
        return roi

    backend = str(getattr(args, "basket_roi_backend", "tkinter")).strip().lower()
    if backend != "tkinter":
        log(f"[stream] 推流框选暂仅支持 tkinter，当前 {backend!r} 将回退 tkinter")
    disp, scale = _scale_frame_for_display(first_frame, 1920)
    log("[stream] 请在弹窗中框选篮子 ROI…")
    rx, ry, rw, rh = _select_basket_roi_tkinter(
        disp, t_sec=t_sec, title="框选耗材篮子（推流）"
    )
    if scale != 1.0:
        rx, ry, rw, rh = rx / scale, ry / scale, rw / scale, rh / scale
    roi = _roi_xyxy_from_select(int(round(rx)), int(round(ry)), int(round(rw)), int(round(rh)))
    log(f"[stream] 篮子 ROI xyxy={roi}")

    save_json = getattr(args, "basket_save_roi_json", None)
    if save_json is not None:
        save_basket_roi_json(Path(save_json), roi)
        log(f"[stream] ROI 已保存: {save_json}")
    return roi


def _doctor_row_cells(doc: dict[str, Any] | None) -> list[str]:
    if doc is None or not doc.get("ok"):
        return ["", "", ""]
    conf = doc.get("doctor_conf")
    conf_s = f"{float(conf):.6f}" if conf is not None else ""
    return [str(doc.get("doctor_id", "")), str(doc.get("doctor_name", "")), conf_s]


def _format_result_row(
    rank: int,
    t0: float,
    t1: float,
    info: dict[str, Any],
    product_map: dict[str, str],
    *,
    legacy_12_col: bool,
    include_doctor_cols: bool = False,
    doctor: dict[str, Any] | None = None,
) -> str:
    sep = "\t"
    if not info.get("ok"):
        reason = str(info.get("reason", ""))
        row = [
            str(rank),
            f"{t0:.6f}",
            f"{t1:.6f}",
            "",
            reason,
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ]
        if not legacy_12_col:
            row.extend(["", ""])
        if include_doctor_cols:
            row.extend(_doctor_row_cells(doctor))
        return sep.join(row)

    n1, n2, n3 = info["top_names"]
    c1, c2, c3 = info["top_confs"]
    id1 = product_map.get(n1, "") if n1 else ""
    id2 = product_map.get(n2, "") if n2 else ""
    id3 = product_map.get(n3, "") if n3 else ""
    for nm, pid in ((n1, id1), (n2, id2), (n3, id3)):
        if nm and not pid:
            log(f"警告: 商品表无名称「{nm}」，产品编码置空。")

    row = [
        str(rank),
        f"{t0:.6f}",
        f"{t1:.6f}",
        id1,
        n1,
        f"{c1:.6f}" if n1 else "",
        id2,
        n2,
        f"{c2:.6f}" if n2 else "",
        id3,
        n3,
        f"{c3:.6f}" if n3 else "",
    ]
    if not legacy_12_col:
        row.extend(["", ""])
    if include_doctor_cols:
        row.extend(_doctor_row_cells(doctor))
    return sep.join(row)


def _log_doctor_result(rank: int, doc: dict[str, Any] | None) -> None:
    if doc is None:
        return
    if doc.get("ok"):
        name = doc.get("doctor_name") or doc.get("doctor_id", "")
        conf = doc.get("doctor_conf", 0.0)
        low = " [低置信度]" if doc.get("low_confidence") else ""
        log(f"[stream] rank={rank} 医生: {name} (id={doc.get('doctor_id')}, conf={conf:.4f}){low}")
    else:
        log(f"[stream] rank={rank} 医生识别失败: {doc.get('reason', '')}")


def _process_one_clip(
    rank: int,
    clip: CachedClip,
    *,
    det: Any,
    hc: HaocaiOnlyClassifier,
    infer_cap: cv2.VideoCapture | None,
    use_file_infer: bool,
    is_file: bool,
    source: str,
    args: Namespace,
    cls_names: dict,
    allowed_idx: frozenset[int] | None,
    predict_kw: dict[str, Any],
    product_map: dict[str, str],
    doctor_svc: DoctorIdentityService | None,
    collect_doctor_vote: bool,
    face_doctor_svc: DoctorFaceIdentityService | None = None,
    ring: Any = None,
) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
    log(
        f"[stream] 识别 rank={rank} [{clip.start_sec:.3f},{clip.end_sec:.3f}] "
        f"({len(clip.frames)} 帧)…"
    )
    frames_copy = list(clip.frames) if doctor_svc is not None and not use_file_infer else None
    video_path = Path(source).resolve() if is_file else None

    doc: dict[str, Any] | None = None
    if doctor_svc is None:
        info = _infer_clip(
            clip,
            det=det,
            hc=hc,
            cap=infer_cap,
            use_file_infer=use_file_infer,
            args=args,
            cls_names=cls_names,
            allowed_idx=allowed_idx,
            predict_kw=predict_kw,
            rank=rank,
        )
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_h = pool.submit(
                _infer_clip,
                clip,
                det=det,
                hc=hc,
                cap=infer_cap,
                use_file_infer=use_file_infer,
                args=args,
                cls_names=cls_names,
                allowed_idx=allowed_idx,
                predict_kw=predict_kw,
                rank=rank,
            )
            fut_d = pool.submit(
                doctor_svc.infer_segment,
                start_sec=clip.start_sec,
                end_sec=clip.end_sec,
                video_path=video_path,
                use_file_source=use_file_infer and is_file,
                frames=frames_copy,
            )
            info = fut_h.result()
            doc = fut_d.result()

    # ---- 医生人脸识别（InsightFace 图库，与 ReID 独立） ----
    face_doc: dict[str, Any] | None = None
    if face_doctor_svc is not None and ring is not None:
        try:
            # 从环缓切片医生窗口: [contact_t, contact_t + 5s], 1fps → 5 帧
            doctor_window_end = clip.contact_t + 5.0
            doctor_frames = ring.slice_frames(clip.contact_t, doctor_window_end)
            if doctor_frames:
                # 降采样到约 1fps（最多取 5 帧）
                n = len(doctor_frames)
                if n >= 5:
                    step = max(1, n // 5)
                    sampled = [doctor_frames[i][1] for i in range(0, n, step)][:5]
                else:
                    sampled = [f[1] for f in doctor_frames]
                face_doc = face_doctor_svc.identify_from_frames(sampled)
            else:
                face_doc = {"ok": False, "reason": "医生人脸窗口环缓无帧"}
        except Exception as exc:  # noqa: BLE001
            face_doc = {"ok": False, "reason": f"医生人脸识别异常: {exc}"}

        if face_doc and face_doc.get("ok"):
            fname = face_doc.get("doctor_name") or face_doc.get("doctor_id", "")
            fconf = face_doc.get("doctor_conf", 0.0)
            flow = " [低置信度]" if face_doc.get("low_confidence") else ""
            log(f"[stream] rank={rank} 医生人脸: {fname} (id={face_doc.get('doctor_id')}, conf={fconf:.4f}){flow}")
        elif face_doc:
            log(f"[stream] rank={rank} 医生人脸识别失败: {face_doc.get('reason', '')}")

    line = _format_result_row(
        rank,
        clip.start_sec,
        clip.end_sec,
        info,
        product_map,
        legacy_12_col=True,
        include_doctor_cols=False,
        doctor=None,
    )
    if collect_doctor_vote:
        _log_doctor_result(rank, doc)
    return line, doc if collect_doctor_vote else None, face_doc


def _maybe_free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# 推流段内重试：阈值各降 0.1 后反复重试，直至有结果或无法再降
_STREAM_CONF_RETRY_DELTA = 0.1


def _resolve_haocai_min_conf_retry(args: Namespace) -> float | None:
    h_retry = getattr(args, "haocai_min_conf_retry", None)
    if h_retry is None:
        return None
    h_retry = float(h_retry)
    if h_retry <= 0:
        return None
    if h_retry >= float(args.haocai_min_conf) - 1e-12:
        return None
    return h_retry


def _haocai_infer_kwargs(
    args: Namespace,
    cls_names: dict,
    allowed_idx: frozenset[int] | None,
    predict_kw: dict[str, Any],
) -> dict[str, Any]:
    return {
        "det_conf": float(args.det_conf),
        "pad_ratio": float(args.pad_ratio),
        "imgsz_det": int(args.imgsz_det),
        "frame_stride": max(1, int(args.frame_stride)),
        "cls_names": cls_names,
        "allowed_class_idx": allowed_idx,
        "predict_kw": predict_kw,
    }


def _run_haocai_segment_infer(
    clip: CachedClip,
    *,
    det: Any,
    hc: HaocaiOnlyClassifier,
    cap: cv2.VideoCapture | None,
    use_file_infer: bool,
    args: Namespace,
    infer_kw: dict[str, Any],
    good_top1_conf_threshold: float,
    haocai_min_conf: float,
) -> dict[str, Any]:
    if use_file_infer and cap is not None:
        return process_segment_haocai_from_cap(
            cap,
            det,
            hc,
            start_sec=clip.start_sec,
            end_sec=clip.end_sec,
            seek_margin_sec=float(args.seek_margin_sec),
            good_top1_conf_threshold=good_top1_conf_threshold,
            haocai_min_conf=haocai_min_conf,
            **infer_kw,
        )
    return process_segment_haocai_from_frames(
        clip.frames,
        det,
        hc,
        start_sec=clip.start_sec,
        end_sec=clip.end_sec,
        good_top1_conf_threshold=good_top1_conf_threshold,
        haocai_min_conf=haocai_min_conf,
        **infer_kw,
    )


def _infer_clip_with_stream_retries(
    info: dict[str, Any],
    *,
    det_conf: float,
    good_thr: float,
    haocai_thr: float,
    run: Any,
    log_prefix: str,
) -> dict[str, Any]:
    d, g, h = float(det_conf), float(good_thr), float(haocai_thr)

    while (
        not info.get("ok")
        and str(info.get("reason", "")) == REASON_NO_HANDS_IN_SEGMENT
    ):
        d2 = max(0.0, d - _STREAM_CONF_RETRY_DELTA)
        if d2 >= d - 1e-12:
            break
        log(
            f"{log_prefix}以 det_conf={d2:.4g} 重试本段（段内未检测到手部）…"
        )
        d = d2
        info = run(d, g, h)

    while (
        not info.get("ok")
        and str(info.get("reason", "")) == REASON_NO_VALID_HAOCAI_FRAMES
    ):
        g2 = max(0.0, g - _STREAM_CONF_RETRY_DELTA)
        h2 = max(0.0, h - _STREAM_CONF_RETRY_DELTA)
        if g2 >= g - 1e-12 and h2 >= h - 1e-12:
            break
        parts: list[str] = []
        if g2 < g - 1e-12:
            parts.append(f"good_top1_conf_threshold={g2:.4g}")
        if h2 < h - 1e-12:
            parts.append(f"haocai_min_conf={h2:.4g}")
        log(f"{log_prefix}以 {'、'.join(parts)} 重试本段（无有效耗材帧）…")
        g, h = g2, h2
        info = run(d, g, h)

    return info


def _use_file_infer_for_stream(args: Namespace, *, is_file: bool) -> bool:
    """本地可 seek 文件且 infer_source=file 时，段内识别回源 4K。"""
    if not is_file:
        return False
    mode = str(getattr(args, "stream_infer_source", "file")).strip().lower()
    return mode in ("file", "auto", "source")


def _infer_clip(
    clip: CachedClip,
    *,
    det: Any,
    hc: HaocaiOnlyClassifier,
    cap: cv2.VideoCapture | None,
    use_file_infer: bool,
    args: Namespace,
    cls_names: dict,
    allowed_idx: frozenset[int] | None,
    predict_kw: dict[str, Any],
    rank: int | None = None,
) -> dict[str, Any]:
    log_prefix = f"[stream] rank={rank}: " if rank is not None else "[stream] "
    infer_kw = _haocai_infer_kwargs(args, cls_names, allowed_idx, predict_kw)
    det_conf = float(args.det_conf)
    good_thr = float(args.good_top1_conf_threshold)
    haocai_thr = float(args.haocai_min_conf)
    try:
        def run(
            hand_det_conf: float,
            good_top1_conf_threshold: float,
            haocai_min_conf: float,
        ) -> dict[str, Any]:
            kw = {**infer_kw, "det_conf": float(hand_det_conf)}
            return _run_haocai_segment_infer(
                clip,
                det=det,
                hc=hc,
                cap=cap,
                use_file_infer=use_file_infer,
                args=args,
                infer_kw=kw,
                good_top1_conf_threshold=good_top1_conf_threshold,
                haocai_min_conf=haocai_min_conf,
            )

        info = run(det_conf, good_thr, haocai_thr)
        info = _infer_clip_with_stream_retries(
            info,
            det_conf=det_conf,
            good_thr=good_thr,
            haocai_thr=haocai_thr,
            run=run,
            log_prefix=log_prefix,
        )
        return info
    finally:
        clip.frames.clear()
        _maybe_free_gpu()


class StreamBasketOrchestrator:
    def __init__(self, args: Namespace) -> None:
        self.args = args

    def run(self) -> int:
        args = self.args
        source = str(getattr(args, "stream_rtsp", "") or getattr(args, "rtsp", "")).strip()
        if not source:
            log("缺少推流地址：--rtsp 或 yaml stream.rtsp")
            return 1

        excel_path = Path(args.excel).resolve()
        if not excel_path.is_file():
            log(f"找不到 Excel: {excel_path}")
            return 1
        if not _validate_stream_weights(args):
            return 1

        product_map = e2e.load_product_code_map(excel_path)
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        predict_kw: dict[str, Any] = {"device": args.device}
        if args.half:
            predict_kw["half"] = True

        from hand_detector import create_hand_contact_tracker, create_hand_detector

        hand_lab = str(getattr(args, "hand_backend", "yolo"))
        log(f"[stream] 加载手部检测（{hand_lab}）与 YOLO（好坏帧 / 耗材）…")
        det = create_hand_detector(args)
        gb = YOLO(str(args.goodbad_model))
        cls_m = YOLO(str(args.haocai_model))
        cls_names = cls_m.names
        hc = HaocaiOnlyClassifier(
            cls_m,
            cls_names=cls_names,
            imgsz_cls=int(args.imgsz_cls),
            predict_kw=predict_kw,
            gb=gb,
            gb_names=gb.names,
        )
        try:
            allowed_idx = resolve_allowed_class_idx(args, excel_path, cls_names)
        except FileNotFoundError as exc:
            log(str(exc))
            return 1
        if getattr(args, "use_whitelist", True):
            log(f"[stream] 白名单启用，{len(allowed_idx or ())} 个类参与投票")
        else:
            log("[stream] 白名单已关闭，使用全 41 类")

        collect_doctor_vote = stream_doctor_enabled(args)
        doctor_svc: DoctorIdentityService | None = None
        if collect_doctor_vote:
            try:
                doctor_svc = DoctorIdentityService(args)
                win = float(getattr(args, "doctor_identity_segment_window_sec", 3.0))
                log(
                    f"[stream] 医生身份识别已启用（每段居中 {win:g}s 采样 + 末行投票汇总）"
                )
            except Exception as exc:  # noqa: BLE001
                log(f"[stream] 医生识别初始化失败，本 run 不写入医生汇总: {exc}")
                collect_doctor_vote = False
                doctor_svc = None

        # 医生人脸识别（基于 InsightFace 图库，与 ReID 独立）
        collect_face_doctor = bool(getattr(args, "doctor_face_identity_enabled", False))
        face_doctor_svc: DoctorFaceIdentityService | None = None
        if collect_face_doctor:
            try:
                face_doctor_svc = DoctorFaceIdentityService(args)
                log("[stream] 医生人脸识别已启用（InsightFace + 图库 KNN）")
            except Exception as exc:  # noqa: BLE001
                log(f"[stream] 医生人脸识别初始化失败: {exc}")
                collect_face_doctor = False
                face_doctor_svc = None

        ffmpeg_low_latency = bool(getattr(args, "stream_ffmpeg_low_latency", True))
        cap = open_stream_capture(
            source,
            rtsp_transport=str(getattr(args, "stream_rtsp_transport", "tcp")),
            buffer_size=int(getattr(args, "stream_rtsp_buffer_size", 1)),
            ffmpeg_low_latency=ffmpeg_low_latency,
        )
        if not cap.isOpened():
            log(f"[stream] 无法打开流: {source}")
            return 1

        is_file = is_local_media(source)
        is_live = is_rtsp_url(source) or not is_file
        use_file_infer = _use_file_infer_for_stream(args, is_file=is_file)
        infer_cap: cv2.VideoCapture | None = None
        if use_file_infer:
            infer_cap = open_stream_capture(
                source,
                rtsp_transport=str(getattr(args, "stream_rtsp_transport", "tcp")),
                ffmpeg_low_latency=ffmpeg_low_latency,
            )
            if not infer_cap.isOpened():
                log("[stream] 无法打开回源推理用 VideoCapture，回退 raw BGR 缓存识别")
                use_file_infer = False
                infer_cap = None

        meta = probe_capture(cap)
        w, h = int(meta["width"]), int(meta["height"])
        fps = float(meta["fps"])
        if fps <= 1e-3:
            fps = float(getattr(args, "stream_fps", 25.0))

        if is_live:
            log(
                f"[stream] 实时流 {w}x{h} @fps≈{fps:g} "
                f"(transport={getattr(args, 'stream_rtsp_transport', 'tcp')})"
            )
        else:
            log(f"[stream] 本地文件 {w}x{h} @fps≈{fps:g}")

        warmup_skip = 0
        if is_live:
            warmup_n = int(getattr(args, "stream_warmup_skip_frames", 0))
            if warmup_n > 0:
                warmup_skip = skip_warmup_frames(cap, warmup_n)
                log(
                    f"[stream] 预热丢弃前 {warmup_skip}/{warmup_n} 帧"
                    "（避免 RTSP 初帧花屏）"
                )

        ok0, first = cap.read()
        if not ok0 or first is None:
            log("[stream] 无法读取首帧")
            cap.release()
            return 1

        t0 = (
            float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if is_file
            else warmup_skip / fps
        )
        basket_roi = _resolve_basket_roi(args, first, t_sec=t0)

        seg_start_off = float(
            getattr(
                args,
                "stream_segment_start_offset_sec",
                getattr(args, "basket_segment_start_offset_sec", 1.0),
            )
        )
        seg_end_off = float(
            getattr(
                args,
                "stream_segment_end_offset_sec",
                getattr(args, "basket_segment_end_offset_sec", 6.0),
            )
        )
        ring_sec = float(getattr(args, "stream_ring_buffer_sec", 10.0))

        ring_lock = threading.Lock()
        ring = RawFrameRingBuffer(
            max_seconds=ring_sec,
            fps=fps,
            lock=ring_lock,
        )

        trigger = ActionTriggerLogic(
            fps=fps,
            confirm_seconds=float(getattr(args, "basket_confirm_seconds", 0.12)),
            cooldown_seconds=float(getattr(args, "basket_cooldown_seconds", 2.5)),
            threshold_on=float(getattr(args, "basket_contact_iou_on", 0.04)),
            threshold_off=float(getattr(args, "basket_contact_iou_off", 0.02)),
        )

        pack_root = getattr(args, "pack_root", Path(__file__).resolve().parents[1])
        hand_tracker = create_hand_contact_tracker(
            args,
            det,
            det_conf=float(getattr(args, "basket_det_conf", args.det_conf)),
            imgsz_det=int(args.imgsz_det),
            predict_kw=predict_kw,
            pack_root=pack_root,
        )
        if hand_tracker is not None:
            log("[stream] 接触判定已启用 YOLO ByteTrack")

        session = StreamBasketSession(
            basket_roi,
            det,
            trigger,
            ring,
            segment_start_offset_sec=seg_start_off,
            segment_end_offset_sec=seg_end_off,
            min_segment_sec=float(getattr(args, "stream_min_segment_sec", 4.0)),
            det_conf=float(getattr(args, "basket_det_conf", args.det_conf)),
            imgsz_det=int(args.imgsz_det),
            predict_kw=predict_kw,
            hand_tracker=hand_tracker,
            log_fn=log,
        )

        infer_workers = max(1, int(getattr(args, "stream_infer_workers", 1)))

        log(
            f"[stream] 帧缓存: ring={ring_sec:g}s, raw BGR {w}x{h}（无损，Producer/Consumer 解耦）"
        )
        if use_file_infer:
            log("[stream] 段内识别: 回源本地文件 4K（infer_source=file，与 TSV 离线一致）")
        else:
            fallback = str(getattr(args, "stream_infer_fallback", "cache"))
            log(f"[stream] 段内识别: raw BGR 环缓（infer_fallback={fallback}）")
        log(f"[stream] 段级推理线程池 workers={infer_workers}")

        doctor_votes: list[dict[str, Any] | None] = []
        face_doctor_votes: list[dict[str, Any] | None] = []
        tsv_writer = StreamTsvWriter(out_path)
        rank = 0
        rank_lock = threading.Lock()
        infer_futures: list[Future] = []
        state_lock = threading.Lock()
        infer_pool = ThreadPoolExecutor(
            max_workers=infer_workers, thread_name_prefix="stream-infer"
        )

        def handle_clips(clips: list[CachedClip]) -> None:
            nonlocal rank
            for clip in clips:
                with rank_lock:
                    rank += 1
                    clip_rank = rank
                fut = infer_pool.submit(
                    _process_one_clip,
                    clip_rank,
                    clip,
                    det=det,
                    hc=hc,
                    infer_cap=infer_cap,
                    use_file_infer=use_file_infer,
                    is_file=is_file,
                    source=source,
                    args=args,
                    cls_names=cls_names,
                    allowed_idx=allowed_idx,
                    predict_kw=predict_kw,
                    product_map=product_map,
                    doctor_svc=doctor_svc,
                    collect_doctor_vote=collect_doctor_vote,
                    face_doctor_svc=face_doctor_svc,
                    ring=ring,
                )
                infer_futures.append(fut)

                def _done(f: Future, r: int = clip_rank) -> None:
                    try:
                        line, doc, face_doc = f.result()
                    except Exception as exc:  # noqa: BLE001
                        log(f"[stream] rank={r} 推理异常: {exc}")
                        return
                    with state_lock:
                        tsv_writer.append_segment(line, rank=r)
                        if collect_doctor_vote:
                            doctor_votes.append(doc)
                        if collect_face_doctor:
                            face_doctor_votes.append(face_doc)

                fut.add_done_callback(_done)

        pipeline = StreamIngestPipeline(
            cap,
            session,
            ring,
            fps=fps,
            is_file=is_file,
            warmup_frame_idx=warmup_skip,
            on_clips_ready=handle_clips,
            log_fn=log,
        )

        pipeline.enqueue_frame(t0, first)

        log(f"[stream] 开始读流: {source} (fps≈{fps:g}, {w}x{h})")
        pipeline.start()
        try:
            pipeline.wait_until_stopped()
        except KeyboardInterrupt:
            log("[stream] 用户中断")
            pipeline.stop()
            pipeline.wait_until_stopped()
        finally:
            pipeline.stop()
            pipeline.wait_until_stopped()
            handle_clips(pipeline.flush_ready_clips())
            for fut in infer_futures:
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001
                    log(f"[stream] 推理任务异常: {exc}")
            infer_pool.shutdown(wait=True)
            cap.release()
            if infer_cap is not None:
                infer_cap.release()
            if doctor_svc is not None:
                doctor_svc.close()
            if face_doctor_svc is not None:
                face_doctor_svc.close()
            if hasattr(det, "close"):
                det.close()

        # 医生汇总：优先使用人脸识别结果，否则回退 ReID
        if collect_face_doctor:
            # 对 face_doctor_votes 做众数投票
            face_summary = vote_doctor_from_segment_results(face_doctor_votes)
            # 如果人脸识别全部失败但 ReID 可用，回退 ReID
            if "识别失败" in face_summary and collect_doctor_vote:
                n_written = tsv_writer.finalize(
                    doctor_votes,
                    args=args,
                    collect_doctor=True,
                )
            else:
                # 用 face_doctor_votes 借 finalize 的投票逻辑写入
                n_written = tsv_writer.finalize(
                    face_doctor_votes,
                    args=args,
                    collect_doctor=True,
                )
        else:
            n_written = tsv_writer.finalize(
                doctor_votes,
                args=args,
                collect_doctor=collect_doctor_vote,
            )
        log(f"[stream] 结束，共 {n_written} 段，结果: {out_path}")
        return 0 if n_written > 0 or is_file else 0


def run_stream_pipeline(args: Namespace) -> int:
    return StreamBasketOrchestrator(args).run()
