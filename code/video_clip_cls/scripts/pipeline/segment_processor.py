"""
单段时间范围内的流式解码：多手部 ROI → 好帧门控 → 耗材 + 撕膜分类，汇总投票样本。

不将整段视频载入内存；每帧处理后可 del 大图与 ROI（由调用方循环内负责）。
"""
from __future__ import annotations

import gc
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

for _repo in Path(__file__).resolve().parents:
    if (_repo / "repo_root.py").is_file() and (_repo / "dataset.py").is_file():
        CODE_ROOT = _repo
        if str(_repo) not in sys.path:
            sys.path.insert(0, str(_repo))
        break
else:
    raise RuntimeError("未定位到仓库 code/ 根目录")

_SCRIPTS = CODE_ROOT / "video_clip_cls" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

_INF = CODE_ROOT / "video_clip_cls" / "infer_single_0506"
if str(_INF) not in sys.path:
    sys.path.insert(0, str(_INF))

import run_segments_consumable_vote as _rsv  # noqa: E402
from run_haocai_actionformer_consumables_e2e import (  # noqa: E402
    aggregate_top3_votes,
    mask_probs_whitelist,
)
from ultralytics import YOLO  # noqa: E402

from pipeline.hand_roi_merge import HandRoiGrouper, two_largest_hands, union_xyxy  # noqa: E402

# 与 run_haocai_actionformer_consumables_e2e 段内失败 return 文案一致，供 Phase2 重试判断


def _detect_hands_on_frame(
    det: Any,
    fr: np.ndarray,
    det_conf: float,
    imgsz_det: int,
    predict_kw: dict[str, Any] | None,
) -> list[list[float]]:
    try:
        from hand_detector import detect_hands_xyxy

        return detect_hands_xyxy(
            det,
            fr,
            det_conf=det_conf,
            imgsz_det=imgsz_det,
            predict_kw=predict_kw,
        )
    except ImportError:
        pred_kw = dict(predict_kw or {})
        r0 = det.predict(
            fr, conf=det_conf, imgsz=imgsz_det, verbose=False, **pred_kw
        )[0]
        return collect_hand_boxes(det, r0.boxes) if r0.boxes else []

REASON_NO_HANDS_IN_SEGMENT = "（段内未检测到手部）"
REASON_NO_VALID_HAOCAI_FRAMES = "（无有效耗材帧：好帧/白名单/耗材置信度未全部满足）"
# 推流 / TSV 离线（无好坏帧门控）
REASON_NO_VALID_HAOCAI_FRAMES_STREAM = "（无有效耗材帧：白名单/耗材置信度未满足）"

collect_hand_boxes = _rsv.collect_hand_boxes
pad_box = _rsv.pad_box_bottom_only
_cls_name = _rsv._cls_name


def _float_top1conf(pr: Any) -> float:
    tc = pr.top1conf
    if tc is None:
        return 0.0
    if isinstance(tc, (float, int, np.floating)):
        return float(tc)
    return float(tc.detach().float().cpu().item())


def passes_good_gate_top1_conf_kw(
    gb_model: YOLO,
    crop: np.ndarray,
    gb_names: dict,
    imgsz: int,
    top1_conf_must_exceed: float,
    predict_kw: dict[str, Any],
) -> bool:
    """与 run_segments_consumable_vote 一致，但向 predict 透传 half/device。"""
    if crop.size == 0:
        return False
    r = gb_model.predict(crop, imgsz=imgsz, verbose=False, **predict_kw)[0]
    pr = r.probs
    if pr is None:
        return False
    tid = int(pr.top1)
    label = str(gb_names.get(tid, "")).strip().lower()
    conf = _float_top1conf(pr)
    return label == "good" and conf > top1_conf_must_exceed


def aggregate_top2_votes(
    pairs: list[tuple[str, float]],
) -> tuple[list[str], list[float]]:
    """与 aggregate_top3 相同思想，取前二类及次数归一化置信度。"""
    empty = (["", ""], [0.0, 0.0])
    if not pairs:
        return empty
    cnt = Counter(p[0] for p in pairs)
    ranked = sorted(cnt.items(), key=lambda x: (-x[1], x[0]))
    top = ranked[:2]
    if not top:
        return empty
    total = float(sum(c for _, c in top))
    if total <= 0:
        return empty
    out_names: list[str] = ["", ""]
    out_conf: list[float] = [0.0, 0.0]
    for i, (nm, c) in enumerate(top):
        out_names[i] = nm
        out_conf[i] = float(c) / total
    return out_names, out_conf


def _clip_xyxy(box: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """
    将 xyxy 框裁剪到图像边界，并保证 x2>x1, y2>y1。
    """
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(x1, img_w - 1.0))
    y1 = max(0.0, min(y1, img_h - 1.0))
    x2 = max(0.0, min(x2, img_w - 1.0))
    y2 = max(0.0, min(y2, img_h - 1.0))
    if x2 <= x1:
        x2 = min(img_w - 1.0, x1 + 1.0)
    if y2 <= y1:
        y2 = min(img_h - 1.0, y1 + 1.0)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _fuse_hands_to_one_box(hands: list[list[float]], img_w: int, img_h: int) -> np.ndarray | None:
    """
    多手框融合为一个大框（x1,y1,x2,y2），用于段内时序平滑与短时补帧。
    """
    if not hands:
        return None
    arr = np.asarray(hands, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 4:
        return None
    x1 = float(np.min(arr[:, 0]))
    y1 = float(np.min(arr[:, 1]))
    x2 = float(np.max(arr[:, 2]))
    y2 = float(np.max(arr[:, 3]))
    fused = np.array([x1, y1, x2, y2], dtype=np.float32)
    return _clip_xyxy(fused, img_w, img_h)


def _crop_two_hands_union(
    fr: np.ndarray,
    hands: list[list[float]],
    pad_ratio: float,
) -> np.ndarray | None:
    """至少两只手时取最大两只 union 并 pad；否则 None（跳过该帧）。"""
    if len(hands) < 2:
        return None
    img_h, img_w = fr.shape[:2]
    h1, h2 = two_largest_hands(hands)
    uni = union_xyxy(h1, h2)
    x1, y1, x2, y2 = pad_box(uni, img_w, img_h, pad_ratio)
    return fr[y1:y2, x1:x2]


class FineGrainedClassifier:
    """好坏帧 / 耗材 / 撕膜：薄封装 Ultralytics cls.predict，便于统一 half/device。"""

    def __init__(
        self,
        gb: YOLO,
        cls_m: YOLO,
        tear_m: YOLO,
        *,
        gb_names: dict,
        cls_names: dict,
        tear_names: dict,
        imgsz_cls: int,
        predict_kw: dict[str, Any],
    ) -> None:
        self.gb = gb
        self.cls_m = cls_m
        self.tear_m = tear_m
        self.gb_names = gb_names
        self.cls_names = cls_names
        self.tear_names = tear_names
        self.imgsz_cls = imgsz_cls
        self.predict_kw = predict_kw

    def passes_good(
        self,
        crop: np.ndarray,
        good_top1_conf_threshold: float,
    ) -> bool:
        return passes_good_gate_top1_conf_kw(
            self.gb,
            crop,
            self.gb_names,
            self.imgsz_cls,
            good_top1_conf_threshold,
            self.predict_kw,
        )

    def haocai_label_top_prob(
        self,
        crop: np.ndarray,
        n_cls: int,
        allowed_class_idx: frozenset[int] | None,
        haocai_min_conf: float,
    ) -> tuple[str, float] | None:
        if crop.size == 0:
            return None
        r = self.cls_m.predict(crop, imgsz=self.imgsz_cls, verbose=False, **self.predict_kw)[0]
        pr = r.probs
        if pr is None or pr.data is None:
            return None
        v = pr.data.detach().float().cpu().numpy().astype(np.float64).ravel()
        if v.size < n_cls:
            v = np.resize(v, n_cls)
        v = v[:n_cls].copy()
        s = float(np.sum(v))
        if s <= 1e-12:
            return None
        if abs(s - 1.0) > 0.08:
            v = v - float(np.max(v))
            e = np.exp(np.clip(v, -40.0, 40.0))
            vec_raw = e / float(np.sum(e))
        else:
            vec_raw = v / s
        if allowed_class_idx is not None:
            vec = mask_probs_whitelist(vec_raw, allowed_class_idx, n_cls)
        else:
            vec = vec_raw
        if vec is None:
            return None
        top_prob = float(np.max(vec))
        if top_prob <= haocai_min_conf:
            return None
        label = int(np.argmax(vec))
        return _cls_name(self.cls_names, label), top_prob

    def tear_label_top_conf(self, crop: np.ndarray) -> tuple[str, float] | None:
        if crop.size == 0:
            return None
        r = self.tear_m.predict(crop, imgsz=self.imgsz_cls, verbose=False, **self.predict_kw)[0]
        pr = r.probs
        if pr is None:
            return None
        tid = int(pr.top1)
        conf = _float_top1conf(pr)
        return str(self.tear_names.get(tid, str(tid))).strip(), conf


def _maybe_cuda_empty_cache(every: int, frame_idx: int) -> None:
    if every <= 0:
        return
    if frame_idx % every != 0:
        return
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def process_segment_multi_hand_tear(
    cap: cv2.VideoCapture,
    det: YOLO,
    fg: FineGrainedClassifier,
    grouper: HandRoiGrouper,
    *,
    start_sec: float,
    end_sec: float,
    seek_margin_sec: float,
    det_conf: float,
    imgsz_det: int,
    frame_stride: int,
    good_top1_conf_threshold: float,
    haocai_min_conf: float,
    cls_names: dict,
    allowed_class_idx: frozenset[int] | None,
    tracking_alpha: float = 0.6,
    tracking_max_lost_frames: int = 0,
    empty_cache_every: int = 0,
) -> dict[str, Any]:
    """
    与 process_segment_e2e 相同 seek 策略；每帧最多两 ROI，逐 ROI做好帧+耗材+撕膜门控。
    """
    probe_from = float(max(0.0, start_sec - seek_margin_sec))
    cap.set(cv2.CAP_PROP_POS_MSEC, probe_from * 1000.0)
    synced_frame: np.ndarray | None = None
    synced_t: float | None = None
    tol = 0.04
    while True:
        ok0, grab = cap.read()
        if not ok0 or grab is None:
            synced_frame, synced_t = None, None
            break
        t0 = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
        if t0 + tol >= start_sec:
            synced_frame, synced_t = grab, t0
            break

    n_cls_key_max = max(int(k) for k in cls_names.keys())
    n_cls = n_cls_key_max + 1

    n_hand_frames = 0
    n_gate_pass = 0
    # pairs_h 存放段内耗材候选 (类名, 置信度)，后续会做“按置信度加权”的段内投票聚合。
    # 仅记录通过门控的样本；失败分支仍按是否为空来判定，不改变既有逻辑。
    pairs_h: list[tuple[str, float]] = []
    pairs_t: list[tuple[str, float]] = []
    frames_read_in_segment = 0

    def one_frame(fr: np.ndarray) -> None:
        nonlocal frames_read_in_segment, n_hand_frames, n_gate_pass, pairs_h, pairs_t
        frames_read_in_segment += 1
        idx_local = frames_read_in_segment
        _maybe_cuda_empty_cache(empty_cache_every, idx_local)

        if frame_stride > 1 and (frames_read_in_segment - 1) % frame_stride != 0:
            return

        hands = _detect_hands_on_frame(
            det, fr, det_conf, imgsz_det, fg.predict_kw
        )
        if len(hands) < 2:
            return

        n_hand_frames += 1
        rois = grouper.frame_to_rois(fr, hands)
        if not rois:
            return
        for crop in rois:
            if not fg.passes_good(crop, good_top1_conf_threshold):
                del crop
                continue
            n_gate_pass += 1
            hc = fg.haocai_label_top_prob(
                crop, n_cls, allowed_class_idx, haocai_min_conf
            )
            tr = fg.tear_label_top_conf(crop)
            del crop
            if hc is not None:
                pairs_h.append(hc)
            if tr is not None:
                pairs_t.append(tr)

    if synced_frame is not None and synced_t is not None and synced_t <= end_sec + 0.08:
        one_frame(synced_frame)
        del synced_frame
        synced_frame = None

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        t = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
        if t > end_sec + 0.08:
            del frame
            break
        if t + 1e-6 < start_sec:
            del frame
            continue
        one_frame(frame)
        del frame

    gc.collect()
    if empty_cache_every > 0:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    if n_hand_frames == 0:
        return {"ok": False, "reason": "（段内未检测到手部）", "pairs_h": [], "pairs_t": [], "n_gate_pass": 0}
    if not pairs_h:
        return {
            "ok": False,
            "reason": REASON_NO_VALID_HAOCAI_FRAMES,
            "pairs_h": [],
            "pairs_t": pairs_t,
            "n_hand_frames": n_hand_frames,
            "n_gate_pass": n_gate_pass,
        }

    n1, c1 = aggregate_top3_votes(pairs_h)
    t1, t2 = aggregate_top2_votes(pairs_t)
    return {
        "ok": True,
        "top_names": n1,
        "top_confs": c1,
        "tear_top_names": t1,
        "tear_top_confs": t2,
        "pairs_h": pairs_h,
        "pairs_t": pairs_t,
        "n_hand_frames": n_hand_frames,
        "n_gate_pass": n_gate_pass,
        "n_valid_haocai": len(pairs_h),
    }


def process_segment_multi_hand_tear_with_gate_retries(
    cap: cv2.VideoCapture,
    det: YOLO,
    fg: FineGrainedClassifier,
    grouper: HandRoiGrouper,
    *,
    start_sec: float,
    end_sec: float,
    seek_margin_sec: float,
    det_conf: float,
    imgsz_det: int,
    frame_stride: int,
    good_top1_conf_threshold: float,
    good_top1_retry_threshold: float,
    haocai_min_conf: float,
    haocai_min_conf_retry: float | None,
    cls_names: dict,
    allowed_class_idx: frozenset[int] | None,
    empty_cache_every: int = 0,
    log_fn: Callable[[str], None] | None = None,
    log_prefix: str | None = None,
    tracking_alpha: float = 0.6,
    tracking_max_lost_frames: int = 0,
) -> dict[str, Any]:
    """
    先跑段内推理；若仍为「无有效耗材帧」则：
    1) 可放宽好帧 top1 阈值（good_top1_retry_threshold）再试；
    2) 再放宽耗材置信阈值（haocai_min_conf_retry）再试。
    log_fn / log_prefix：重试时各打一行（如 log_prefix='段落 rank=3: '）。
    """

    def run(good_thr: float, haocai_thr: float) -> dict[str, Any]:
        return process_segment_multi_hand_tear(
            cap,
            det,
            fg,
            grouper,
            start_sec=start_sec,
            end_sec=end_sec,
            seek_margin_sec=seek_margin_sec,
            det_conf=det_conf,
            imgsz_det=imgsz_det,
            frame_stride=frame_stride,
            tracking_alpha=tracking_alpha,
            tracking_max_lost_frames=tracking_max_lost_frames,
            good_top1_conf_threshold=good_thr,
            haocai_min_conf=haocai_thr,
            cls_names=cls_names,
            allowed_class_idx=allowed_class_idx,
            empty_cache_every=empty_cache_every,
        )

    good_thr = float(good_top1_conf_threshold)
    haocai_thr = float(haocai_min_conf)
    info = run(good_thr, haocai_thr)

    rgb = float(good_top1_retry_threshold)
    if (
        not info.get("ok")
        and str(info.get("reason", "")) == REASON_NO_VALID_HAOCAI_FRAMES
        and rgb > 0
        and rgb < good_thr - 1e-12
    ):
        if log_fn and log_prefix:
            log_fn(
                f"{log_prefix}以 good_top1_conf_threshold={rgb} 重试本段（无有效耗材帧）…"
            )
        good_thr = rgb
        info = run(good_thr, haocai_thr)

    if (
        haocai_min_conf_retry is not None
        and haocai_min_conf_retry > 1e-12
        and haocai_min_conf_retry < haocai_thr - 1e-12
    ):
        if (
            not info.get("ok")
            and str(info.get("reason", "")) == REASON_NO_VALID_HAOCAI_FRAMES
        ):
            h2 = float(haocai_min_conf_retry)
            if log_fn and log_prefix:
                log_fn(
                    f"{log_prefix}以 haocai_min_conf={h2} 重试本段（无有效耗材帧）…"
                )
            info = run(good_thr, h2)

    return info


class HaocaiOnlyClassifier:
    """耗材分类（推流/TSV 离线）；可选好坏帧门控，无撕膜。"""

    def __init__(
        self,
        cls_m: YOLO,
        *,
        cls_names: dict,
        imgsz_cls: int,
        predict_kw: dict[str, Any],
        gb: YOLO | None = None,
        gb_names: dict | None = None,
    ) -> None:
        self.cls_m = cls_m
        self.cls_names = cls_names
        self.imgsz_cls = imgsz_cls
        self.predict_kw = predict_kw
        self.gb = gb
        self.gb_names = gb_names or {}

    @property
    def use_good_gate(self) -> bool:
        return self.gb is not None

    def passes_good(self, crop: np.ndarray, good_top1_conf_threshold: float) -> bool:
        if self.gb is None:
            return True
        return passes_good_gate_top1_conf_kw(
            self.gb,
            crop,
            self.gb_names,
            self.imgsz_cls,
            good_top1_conf_threshold,
            self.predict_kw,
        )

    def haocai_label_top_prob(
        self,
        crop: np.ndarray,
        n_cls: int,
        allowed_class_idx: frozenset[int] | None,
        haocai_min_conf: float,
    ) -> tuple[str, float] | None:
        if crop.size == 0:
            return None
        r = self.cls_m.predict(crop, imgsz=self.imgsz_cls, verbose=False, **self.predict_kw)[0]
        pr = r.probs
        if pr is None or pr.data is None:
            return None
        v = pr.data.detach().float().cpu().numpy().astype(np.float64).ravel()
        if v.size < n_cls:
            v = np.resize(v, n_cls)
        v = v[:n_cls].copy()
        s = float(np.sum(v))
        if s <= 1e-12:
            return None
        if abs(s - 1.0) > 0.08:
            v = v - float(np.max(v))
            e = np.exp(np.clip(v, -40.0, 40.0))
            vec_raw = e / float(np.sum(e))
        else:
            vec_raw = v / s
        if allowed_class_idx is not None:
            vec = mask_probs_whitelist(vec_raw, allowed_class_idx, n_cls)
        else:
            vec = vec_raw
        if vec is None:
            return None
        top_prob = float(np.max(vec))
        if top_prob <= haocai_min_conf:
            return None
        label = int(np.argmax(vec))
        return _cls_name(self.cls_names, label), top_prob


def _haocai_fail_reason(hc: HaocaiOnlyClassifier) -> str:
    if hc.use_good_gate:
        return REASON_NO_VALID_HAOCAI_FRAMES
    return REASON_NO_VALID_HAOCAI_FRAMES_STREAM


def process_segment_haocai_from_frames(
    frames: list[tuple[float, np.ndarray]],
    det: YOLO,
    hc: HaocaiOnlyClassifier,
    *,
    start_sec: float,
    end_sec: float,
    det_conf: float,
    pad_ratio: float,
    imgsz_det: int,
    frame_stride: int,
    haocai_min_conf: float,
    good_top1_conf_threshold: float = 0.9,
    cls_names: dict,
    allowed_class_idx: frozenset[int] | None,
    predict_kw: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    对内存中的帧列表做耗材识别（手 → 可选好帧 → haocai），不含撕膜。
    frames: [(t_sec, bgr), ...] 已按时间过滤到 [start_sec, end_sec]。
    """
    if not frames:
        return {"ok": False, "reason": "（段内无帧）", "pairs": [], "n_gate_pass": 0}

    pred_kw = dict(predict_kw or {})
    n_cls_key_max = max(int(k) for k in cls_names.keys())
    n_cls = n_cls_key_max + 1

    n_hand_frames = 0
    n_gate_pass = 0
    pairs: list[tuple[str, float]] = []
    frames_in_segment = 0

    def one_frame(fr: np.ndarray) -> None:
        nonlocal frames_in_segment, n_hand_frames, n_gate_pass, pairs
        frames_in_segment += 1
        if frame_stride > 1 and (frames_in_segment - 1) % frame_stride != 0:
            return

        hands = _detect_hands_on_frame(det, fr, det_conf, imgsz_det, pred_kw)
        crop = _crop_two_hands_union(fr, hands, pad_ratio)
        if crop is None:
            return

        n_hand_frames += 1
        if hc.use_good_gate and not hc.passes_good(crop, good_top1_conf_threshold):
            del crop
            return
        n_gate_pass += 1
        label_prob = hc.haocai_label_top_prob(
            crop, n_cls, allowed_class_idx, haocai_min_conf
        )
        del crop
        if label_prob is not None:
            pairs.append(label_prob)

    lo = float(start_sec)
    hi = float(end_sec)
    for t, fr in frames:
        if t + 1e-6 < lo:
            continue
        if t > hi + 0.08:
            break
        one_frame(fr)

    if n_hand_frames == 0:
        return {"ok": False, "reason": REASON_NO_HANDS_IN_SEGMENT, "pairs": [], "n_gate_pass": 0}
    if not pairs:
        return {
            "ok": False,
            "reason": _haocai_fail_reason(hc),
            "pairs": [],
            "n_hand_frames": n_hand_frames,
            "n_gate_pass": n_gate_pass,
        }

    n1, c1 = aggregate_top3_votes(pairs)
    return {
        "ok": True,
        "top_names": n1,
        "top_confs": c1,
        "pairs": pairs,
        "n_hand_frames": n_hand_frames,
        "n_gate_pass": n_gate_pass,
        "n_valid_haocai": len(pairs),
    }


def process_segment_haocai_from_cap(
    cap: cv2.VideoCapture,
    det: YOLO,
    hc: HaocaiOnlyClassifier,
    *,
    start_sec: float,
    end_sec: float,
    seek_margin_sec: float,
    det_conf: float,
    pad_ratio: float,
    imgsz_det: int,
    frame_stride: int,
    haocai_min_conf: float,
    good_top1_conf_threshold: float = 0.9,
    cls_names: dict,
    allowed_class_idx: frozenset[int] | None,
    predict_kw: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """从视频逐帧解码做耗材识别（手 → 可选好帧 → haocai），不含撕膜。"""
    probe_from = float(max(0.0, start_sec - seek_margin_sec))
    cap.set(cv2.CAP_PROP_POS_MSEC, probe_from * 1000.0)
    synced_frame: np.ndarray | None = None
    synced_t: float | None = None
    tol = 0.04
    while True:
        ok0, grab = cap.read()
        if not ok0 or grab is None:
            synced_frame, synced_t = None, None
            break
        t0 = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
        if t0 + tol >= start_sec:
            synced_frame, synced_t = grab, t0
            break

    pred_kw = dict(predict_kw or {})
    n_cls_key_max = max(int(k) for k in cls_names.keys())
    n_cls = n_cls_key_max + 1

    n_hand_frames = 0
    n_gate_pass = 0
    pairs: list[tuple[str, float]] = []
    frames_in_segment = 0

    def one_frame(fr: np.ndarray) -> None:
        nonlocal frames_in_segment, n_hand_frames, n_gate_pass, pairs
        frames_in_segment += 1
        if frame_stride > 1 and (frames_in_segment - 1) % frame_stride != 0:
            return

        img_h, img_w = fr.shape[:2]
        hands = _detect_hands_on_frame(det, fr, det_conf, imgsz_det, pred_kw)
        crop = _crop_two_hands_union(fr, hands, pad_ratio)
        if crop is None:
            return

        n_hand_frames += 1
        if hc.use_good_gate and not hc.passes_good(crop, good_top1_conf_threshold):
            del crop
            return
        n_gate_pass += 1
        label_prob = hc.haocai_label_top_prob(
            crop, n_cls, allowed_class_idx, haocai_min_conf
        )
        del crop
        if label_prob is not None:
            pairs.append(label_prob)

    lo = float(start_sec)
    hi = float(end_sec)

    if synced_frame is not None and synced_t is not None and synced_t <= hi + 0.08:
        if synced_t + 1e-6 >= lo:
            one_frame(synced_frame)

    while True:
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        t = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
        if t > hi + 0.08:
            break
        if t + 1e-6 < lo:
            continue
        one_frame(fr)

    if n_hand_frames == 0:
        return {"ok": False, "reason": REASON_NO_HANDS_IN_SEGMENT, "pairs": [], "n_gate_pass": 0}
    if not pairs:
        return {
            "ok": False,
            "reason": _haocai_fail_reason(hc),
            "pairs": [],
            "n_hand_frames": n_hand_frames,
            "n_gate_pass": n_gate_pass,
        }

    n1, c1 = aggregate_top3_votes(pairs)
    return {
        "ok": True,
        "top_names": n1,
        "top_confs": c1,
        "pairs": pairs,
        "n_hand_frames": n_hand_frames,
        "n_gate_pass": n_gate_pass,
        "n_valid_haocai": len(pairs),
    }


def _apply_haocai_gate_retries(
    run: Callable[[float, float], dict[str, Any]],
    *,
    hc: HaocaiOnlyClassifier,
    good_top1_conf_threshold: float,
    good_top1_retry_threshold: float,
    haocai_min_conf: float,
    haocai_min_conf_retry: float | None,
    log_fn: Callable[[str], None] | None = None,
    log_prefix: str | None = None,
) -> dict[str, Any]:
    fail_reason = _haocai_fail_reason(hc)
    good_thr = float(good_top1_conf_threshold)
    haocai_thr = float(haocai_min_conf)
    info = run(good_thr, haocai_thr)

    if hc.use_good_gate:
        rgb = float(good_top1_retry_threshold)
        if (
            not info.get("ok")
            and str(info.get("reason", "")) == fail_reason
            and rgb > 0
            and rgb < good_thr - 1e-12
        ):
            if log_fn and log_prefix:
                log_fn(
                    f"{log_prefix}以 good_top1_conf_threshold={rgb} 重试本段（无有效耗材帧）…"
                )
            good_thr = rgb
            info = run(good_thr, haocai_thr)

    if (
        haocai_min_conf_retry is not None
        and haocai_min_conf_retry > 1e-12
        and haocai_min_conf_retry < haocai_thr - 1e-12
    ):
        if not info.get("ok") and str(info.get("reason", "")) == fail_reason:
            h2 = float(haocai_min_conf_retry)
            if log_fn and log_prefix:
                log_fn(
                    f"{log_prefix}以 haocai_min_conf={h2} 重试本段（无有效耗材帧）…"
                )
            info = run(good_thr, h2)

    return info


def process_segment_haocai_from_frames_with_gate_retries(
    frames: list[tuple[float, np.ndarray]],
    det: YOLO,
    hc: HaocaiOnlyClassifier,
    *,
    start_sec: float,
    end_sec: float,
    det_conf: float,
    pad_ratio: float,
    imgsz_det: int,
    frame_stride: int,
    haocai_min_conf: float,
    haocai_min_conf_retry: float | None,
    good_top1_conf_threshold: float = 0.9,
    good_top1_retry_threshold: float = 0.5,
    cls_names: dict,
    allowed_class_idx: frozenset[int] | None,
    predict_kw: dict[str, Any] | None = None,
    log_fn: Callable[[str], None] | None = None,
    log_prefix: str | None = None,
) -> dict[str, Any]:
    """推流帧列表：好帧门控 + 耗材阈值；失败时先放宽好帧再放宽耗材。"""

    def run(good_thr: float, haocai_thr: float) -> dict[str, Any]:
        return process_segment_haocai_from_frames(
            frames,
            det,
            hc,
            start_sec=start_sec,
            end_sec=end_sec,
            det_conf=det_conf,
            pad_ratio=pad_ratio,
            imgsz_det=imgsz_det,
            frame_stride=frame_stride,
            haocai_min_conf=haocai_thr,
            good_top1_conf_threshold=good_thr,
            cls_names=cls_names,
            allowed_class_idx=allowed_class_idx,
            predict_kw=predict_kw,
        )

    return _apply_haocai_gate_retries(
        run,
        hc=hc,
        good_top1_conf_threshold=good_top1_conf_threshold,
        good_top1_retry_threshold=good_top1_retry_threshold,
        haocai_min_conf=haocai_min_conf,
        haocai_min_conf_retry=haocai_min_conf_retry,
        log_fn=log_fn,
        log_prefix=log_prefix,
    )


def process_segment_haocai_from_cap_with_gate_retries(
    cap: cv2.VideoCapture,
    det: YOLO,
    hc: HaocaiOnlyClassifier,
    *,
    start_sec: float,
    end_sec: float,
    seek_margin_sec: float,
    det_conf: float,
    pad_ratio: float,
    imgsz_det: int,
    frame_stride: int,
    haocai_min_conf: float,
    haocai_min_conf_retry: float | None,
    good_top1_conf_threshold: float = 0.9,
    good_top1_retry_threshold: float = 0.5,
    cls_names: dict,
    allowed_class_idx: frozenset[int] | None,
    predict_kw: dict[str, Any] | None = None,
    log_fn: Callable[[str], None] | None = None,
    log_prefix: str | None = None,
) -> dict[str, Any]:
    """离线视频逐帧解码：手 → 可选好帧 → haocai，含门控重试。"""

    def run(good_thr: float, haocai_thr: float) -> dict[str, Any]:
        return process_segment_haocai_from_cap(
            cap,
            det,
            hc,
            start_sec=start_sec,
            end_sec=end_sec,
            seek_margin_sec=seek_margin_sec,
            det_conf=det_conf,
            pad_ratio=pad_ratio,
            imgsz_det=imgsz_det,
            frame_stride=frame_stride,
            haocai_min_conf=haocai_thr,
            good_top1_conf_threshold=good_thr,
            cls_names=cls_names,
            allowed_class_idx=allowed_class_idx,
            predict_kw=predict_kw,
        )

    return _apply_haocai_gate_retries(
        run,
        hc=hc,
        good_top1_conf_threshold=good_top1_conf_threshold,
        good_top1_retry_threshold=good_top1_retry_threshold,
        haocai_min_conf=haocai_min_conf,
        haocai_min_conf_retry=haocai_min_conf_retry,
        log_fn=log_fn,
        log_prefix=log_prefix,
    )
