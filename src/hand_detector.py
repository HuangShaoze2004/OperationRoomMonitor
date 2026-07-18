"""手部检测统一入口：YOLO hand_detect 或 MediaPipe Hands。"""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np


def _collect_hand_boxes(model: Any, boxes) -> list[list[float]]:
    if not boxes:
        return []
    names = model.names
    out: list[list[float]] = []
    for box in boxes:
        cid = int(box.cls[0])
        label = names.get(cid, "")
        if label == "hand":
            out.append(box.xyxy[0].tolist())
    return out


def create_hand_detector(args: Namespace) -> Any:
    backend = str(getattr(args, "hand_backend", "yolo")).strip().lower()
    if backend == "mediapipe":
        from mediapipe_hand_detector import MediapipeHandDetector

        task = Path(getattr(args, "hand_mediapipe_task"))
        return MediapipeHandDetector(
            task,
            num_hands=int(getattr(args, "hand_mediapipe_num_hands", 2)),
            min_detection_confidence=float(
                getattr(args, "hand_mediapipe_min_detection_confidence", 0.3)
            ),
            min_presence_confidence=float(
                getattr(args, "hand_mediapipe_min_presence_confidence", 0.3)
            ),
            min_tracking_confidence=float(
                getattr(args, "hand_mediapipe_min_tracking_confidence", 0.3)
            ),
            bbox_margin=float(getattr(args, "hand_mediapipe_bbox_margin", 0.05)),
        )
    from ultralytics import YOLO

    return YOLO(str(args.hand_model))


def _resolve_tracker_yaml(args: Namespace, pack_root: Path | None = None) -> str:
    """解析 ByteTrack 配置路径：优先项目内 bytetrack_hand.yaml。"""
    name = str(getattr(args, "basket_contact_tracker", "bytetrack")).strip().lower()
    if name in ("bytetrack", "byte_track"):
        if pack_root is not None:
            custom = pack_root / "configs" / "bytetrack_hand.yaml"
            if custom.is_file():
                return str(custom)
        return "bytetrack.yaml"
    if name.endswith(".yaml"):
        return name
    return f"{name}.yaml"


class YoloHandByteTracker:
    """YOLO 手部 ByteTrack：跨帧维持手框，弥补接触判定阶段的短暂漏检。"""

    def __init__(
        self,
        model: Any,
        *,
        tracker_yaml: str = "bytetrack.yaml",
        det_conf: float = 0.6,
        imgsz_det: int = 640,
        predict_kw: dict[str, Any] | None = None,
    ) -> None:
        self._model = model
        self._tracker_yaml = str(tracker_yaml)
        self._det_conf = float(det_conf)
        self._imgsz_det = int(imgsz_det)
        self._predict_kw = dict(predict_kw or {})

    def update(self, frame: np.ndarray) -> list[list[float]]:
        r0 = self._model.track(
            frame,
            persist=True,
            tracker=self._tracker_yaml,
            conf=self._det_conf,
            imgsz=self._imgsz_det,
            verbose=False,
            **self._predict_kw,
        )[0]
        return _collect_hand_boxes(self._model, r0.boxes) if r0.boxes else []

    def reset(self) -> None:
        predictor = getattr(self._model, "predictor", None)
        if predictor is not None:
            predictor.trackers = None
            self._model.predictor = None


def create_hand_contact_tracker(
    args: Namespace,
    model: Any,
    *,
    det_conf: float | None = None,
    imgsz_det: int | None = None,
    predict_kw: dict[str, Any] | None = None,
    pack_root: Path | None = None,
) -> YoloHandByteTracker | None:
    """
    接触判定阶段手部跟踪器工厂。
    mediapipe 或 contact_tracking_enabled=false 时返回 None。
    """
    backend = str(getattr(args, "hand_backend", "yolo")).strip().lower()
    if backend != "yolo":
        return None
    if hasattr(model, "detect_xyxy") and not hasattr(model, "track"):
        return None
    if not bool(getattr(args, "basket_contact_tracking_enabled", True)):
        return None

    tracker_yaml = _resolve_tracker_yaml(args, pack_root=pack_root)
    conf = float(
        det_conf if det_conf is not None else getattr(args, "basket_det_conf", 0.6)
    )
    imgsz = int(imgsz_det if imgsz_det is not None else getattr(args, "imgsz_det", 640))
    return YoloHandByteTracker(
        model,
        tracker_yaml=tracker_yaml,
        det_conf=conf,
        imgsz_det=imgsz,
        predict_kw=predict_kw,
    )


def detect_hands_xyxy(
    det: Any,
    frame: np.ndarray,
    *,
    det_conf: float = 0.6,
    imgsz_det: int = 640,
    predict_kw: dict[str, Any] | None = None,
) -> list[list[float]]:
    if hasattr(det, "detect_xyxy"):
        return det.detect_xyxy(frame)
    from run_segments_consumable_vote import collect_hand_boxes

    r0 = det.predict(
        frame,
        conf=det_conf,
        imgsz=imgsz_det,
        verbose=False,
        **(predict_kw or {}),
    )[0]
    return collect_hand_boxes(det, r0.boxes) if r0.boxes else []


def validate_hand_assets(args: Namespace) -> tuple[bool, str]:
    backend = str(getattr(args, "hand_backend", "yolo")).strip().lower()
    if backend == "mediapipe":
        p = Path(getattr(args, "hand_mediapipe_task"))
        if not p.is_file():
            return False, f"缺少 MediaPipe 手部模型: {p}"
        return True, "MediaPipe Hands"
    p = Path(getattr(args, "hand_model"))
    if not p.is_file():
        return False, f"缺少手部检测权重: {p}"
    return True, "YOLO hand_detect"
