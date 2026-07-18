"""推流帧环形缓存：原始 BGR 无损存储，按时间戳截取片段。"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimestampedFrame:
    t_sec: float
    frame: np.ndarray  # BGR uint8 copy


class RawFrameRingBuffer:
    """
    保留最近 max_seconds 内的 (t, raw BGR)。
    线程安全；append 时 frame.copy()，slice 时再 copy 供 Phase2 使用。
    """

    def __init__(
        self,
        *,
        max_seconds: float = 10.0,
        fps: float = 25.0,
        lock: threading.Lock | None = None,
    ) -> None:
        self.max_seconds = max(1.0, float(max_seconds))
        self.fps = max(1.0, float(fps))
        self._lock = lock if lock is not None else threading.Lock()
        self._items: deque[TimestampedFrame] = deque()
        self._latest_t = 0.0

    @property
    def latest_t(self) -> float:
        with self._lock:
            return self._latest_t

    def append(self, t_sec: float, frame: np.ndarray) -> TimestampedFrame:
        t = float(t_sec)
        item = TimestampedFrame(t, frame.copy())
        with self._lock:
            self._latest_t = t
            self._items.append(item)
            self._prune_locked(t - self.max_seconds)
        return item

    def _prune_locked(self, t_min: float) -> None:
        cutoff = float(t_min)
        while self._items and self._items[0].t_sec < cutoff - 1e-9:
            self._items.popleft()

    def prune_before(self, t_min: float) -> None:
        with self._lock:
            self._prune_locked(float(t_min))

    def slice_frames(self, t0: float, t1: float) -> list[tuple[float, np.ndarray]]:
        """返回 t0 <= t <= t1 的帧副本列表。"""
        lo = float(t0)
        hi = float(t1)
        if hi < lo:
            lo, hi = hi, lo
        out: list[tuple[float, np.ndarray]] = []
        with self._lock:
            for it in self._items:
                if lo - 1e-6 <= it.t_sec <= hi + 1e-6:
                    out.append((it.t_sec, it.frame.copy()))
        return out


# 向后兼容别名
FrameRingBuffer = RawFrameRingBuffer
