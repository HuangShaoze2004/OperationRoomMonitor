"""推流篮子会话：逐帧手部检测 + ActionTriggerLogic + 待识别片段队列。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from action_trigger_logic import ActionTriggerLogic
from hand_detector import YoloHandByteTracker, detect_hands_xyxy
from stream_frame_buffer import RawFrameRingBuffer, TimestampedFrame


@dataclass
class CachedClip:
    """收满窗口后可送耗材识别的片段（raw BGR 帧，识别后应显式释放）。"""

    contact_t: float
    start_sec: float
    end_sec: float
    frames: list[tuple[float, np.ndarray]]

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass
class _PendingClip:
    contact_t: float
    start_sec: float
    end_sec: float


class StreamBasketSession:
    """
    process_frame：手部检测 + ActionTriggerLogic → 可选 start。
    环缓写入由 Producer 负责；poll_ready_clips 从共享 RawFrameRingBuffer 切片。
    """

    def __init__(
        self,
        basket_xyxy: list[float],
        hand_model: Any,
        trigger: ActionTriggerLogic,
        buffer: RawFrameRingBuffer,
        *,
        segment_start_offset_sec: float = 1.0,
        segment_end_offset_sec: float = 6.0,
        min_segment_sec: float = 4.0,
        det_conf: float = 0.6,
        imgsz_det: int = 640,
        predict_kw: dict[str, Any] | None = None,
        hand_tracker: YoloHandByteTracker | None = None,
        log_fn: Callable[[str], None] | None = None,
    ) -> None:
        self.basket_xyxy = [float(v) for v in basket_xyxy]
        self.hand_model = hand_model
        self._hand_tracker = hand_tracker
        self.trigger = trigger
        self.buffer = buffer
        self.segment_start_offset = float(segment_start_offset_sec)
        self.segment_end_offset = float(segment_end_offset_sec)
        self.min_segment_sec = float(min_segment_sec)
        self.det_conf = float(det_conf)
        self.imgsz_det = int(imgsz_det)
        self.predict_kw = dict(predict_kw or {})
        self.log_fn = log_fn
        self._pending: list[_PendingClip] = []
        self._current_t = 0.0

    def process_frame(self, t_sec: float, frame: np.ndarray) -> float | None:
        """处理一帧；若触发 start 返回 contact 时间戳。"""
        t = float(t_sec)
        self._current_t = t

        if self._hand_tracker is not None:
            hands = self._hand_tracker.update(frame)
        else:
            hands = detect_hands_xyxy(
                self.hand_model,
                frame,
                det_conf=self.det_conf,
                imgsz_det=self.imgsz_det,
                predict_kw=self.predict_kw,
            )
        start_t = self.trigger.process_frame(t, hands, self.basket_xyxy)

        if start_t is not None:
            contact = float(start_t)
            seg0 = contact + self.segment_start_offset
            seg1 = contact + self.segment_end_offset
            self._pending.append(
                _PendingClip(contact_t=contact, start_sec=seg0, end_sec=seg1)
            )
            if self.log_fn:
                self.log_fn(
                    f"[stream] 接触上升沿 t={contact:.3f}s → 窗口 [{seg0:.3f}, {seg1:.3f}]s"
                )
            return contact

        return None

    def process_timestamped(self, item: TimestampedFrame) -> float | None:
        return self.process_frame(item.t_sec, item.frame)

    def push_frame(self, t_sec: float, frame: np.ndarray) -> float | None:
        """单线程模式兼容：写环缓 + 处理。"""
        self.buffer.append(t_sec, frame)
        self.buffer.prune_before(float(t_sec) - self.buffer.max_seconds)
        return self.process_frame(t_sec, frame)

    def poll_ready_clips(self) -> list[CachedClip]:
        """返回当前时刻已收满窗口、且满足最小时长的片段。"""
        ready: list[CachedClip] = []
        still_pending: list[_PendingClip] = []

        for pc in self._pending:
            if self._current_t + 1e-6 < pc.end_sec:
                still_pending.append(pc)
                continue

            frames = self.buffer.slice_frames(pc.start_sec, pc.end_sec)
            duration = pc.end_sec - pc.start_sec
            if duration + 1e-9 < self.min_segment_sec:
                if self.log_fn:
                    self.log_fn(
                        f"[stream] 丢弃短段 [{pc.start_sec:.3f},{pc.end_sec:.3f}] "
                        f"时长 {duration:.3f}s < {self.min_segment_sec:g}s"
                    )
                continue
            if not frames:
                if self.log_fn:
                    self.log_fn(
                        f"[stream] 丢弃空段 [{pc.start_sec:.3f},{pc.end_sec:.3f}]（缓存无帧）"
                    )
                continue

            ready.append(
                CachedClip(
                    contact_t=pc.contact_t,
                    start_sec=pc.start_sec,
                    end_sec=pc.end_sec,
                    frames=frames,
                )
            )

        self._pending = still_pending
        return ready
