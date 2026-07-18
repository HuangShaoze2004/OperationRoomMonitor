"""MediaPipe Hand Landmarker：输出与 YOLO 手检兼容的 xyxy 框列表。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class MediapipeHandDetector:
    """每只手一个外接框；供 detect_hands_xyxy / 双手 union 使用。"""

    names: dict[int, str] = {0: "hand"}

    def __init__(
        self,
        task_path: Path,
        *,
        num_hands: int = 2,
        min_detection_confidence: float = 0.3,
        min_presence_confidence: float = 0.3,
        min_tracking_confidence: float = 0.3,
        bbox_margin: float = 0.05,
    ) -> None:
        task_path = Path(task_path).resolve()
        if not task_path.is_file():
            raise FileNotFoundError(f"MediaPipe 手部模型不存在: {task_path}")

        self.task_path = task_path
        self.num_hands = int(num_hands)
        self.bbox_margin = float(bbox_margin)
        base_options = python.BaseOptions(model_asset_path=str(task_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=self.num_hands,
            min_hand_detection_confidence=float(min_detection_confidence),
            min_hand_presence_confidence=float(min_presence_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None  # type: ignore[assignment]

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def detect_xyxy(self, frame_bgr: np.ndarray) -> list[list[float]]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        margin = self.bbox_margin
        boxes: list[list[float]] = []
        for lms in result.hand_landmarks:
            xs = [lm.x for lm in lms]
            ys = [lm.y for lm in lms]
            x1 = max(0.0, (min(xs) - margin) * w)
            y1 = max(0.0, (min(ys) - margin) * h)
            x2 = min(float(w), (max(xs) + margin) * w)
            y2 = min(float(h), (max(ys) + margin) * h)
            if x2 > x1 + 1.0 and y2 > y1 + 1.0:
                boxes.append([x1, y1, x2, y2])
        return boxes

    def predict(
        self,
        frame: np.ndarray,
        conf: float | None = None,
        imgsz: int | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> list[_YoloLikeResult]:
        del conf, imgsz, verbose, kwargs
        return [_YoloLikeResult(self.detect_xyxy(frame))]


class _YoloLikeResult:
    def __init__(self, hands: list[list[float]]) -> None:
        self.boxes = _YoloLikeBoxes(hands) if hands else None


class _YoloLikeBoxes:
    def __init__(self, hands: list[list[float]]) -> None:
        self._hands = hands

    def __iter__(self):
        for xyxy in self._hands:
            yield _YoloLikeBox(xyxy)

    def __len__(self) -> int:
        return len(self._hands)


class _YoloLikeBox:
    def __init__(self, xyxy: list[float]) -> None:
        import torch

        self.cls = torch.tensor([0.0])
        self.conf = torch.tensor([1.0])
        self.xyxy = torch.tensor([xyxy], dtype=torch.float32)
