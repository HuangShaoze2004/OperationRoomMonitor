"""RTSP 多线程采集：Producer 专职 cap.read，Consumer FIFO 接触判定。"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable

import cv2

from stream_basket_session import CachedClip, StreamBasketSession
from stream_frame_buffer import RawFrameRingBuffer, TimestampedFrame


class StreamIngestPipeline:
    """
    Producer 线程：tight loop cap.read → ring.append + pending 入队。
    Consumer 线程：FIFO 取帧 → session.process_timestamped → poll_ready_clips。
    """

    def __init__(
        self,
        cap: cv2.VideoCapture,
        session: StreamBasketSession,
        ring: RawFrameRingBuffer,
        *,
        fps: float,
        is_file: bool,
        warmup_frame_idx: int = 0,
        on_clips_ready: Callable[[list[CachedClip]], None] | None = None,
        log_fn: Callable[[str], None] | None = None,
    ) -> None:
        self._cap = cap
        self._session = session
        self._ring = ring
        self._fps = max(1e-3, float(fps))
        self._is_file = bool(is_file)
        self._warmup_frame_idx = int(warmup_frame_idx)
        self._on_clips_ready = on_clips_ready
        self._log = log_fn

        self._pending: deque[TimestampedFrame] = deque()
        self._pending_lock = threading.Lock()
        self._stop = threading.Event()
        self._producer_done = threading.Event()
        self._frame_idx = self._warmup_frame_idx + 1

        self._producer: threading.Thread | None = None
        self._consumer: threading.Thread | None = None

    def enqueue_frame(self, t_sec: float, frame) -> TimestampedFrame:
        item = self._ring.append(t_sec, frame)
        with self._pending_lock:
            self._pending.append(item)
        return item

    def start(self) -> None:
        if self._producer is not None:
            return
        self._producer = threading.Thread(
            target=self._producer_loop, name="stream-producer", daemon=True
        )
        self._consumer = threading.Thread(
            target=self._consumer_loop, name="stream-consumer", daemon=True
        )
        self._producer.start()
        self._consumer.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        if self._producer is not None:
            self._producer.join(timeout=timeout)
        if self._consumer is not None:
            self._consumer.join(timeout=timeout)

    def wait_until_stopped(self) -> None:
        """阻塞至 Producer 结束且 Consumer 排空 pending。"""
        if self._producer is not None:
            self._producer.join()
        if self._consumer is not None:
            self._consumer.join()

    def _producer_loop(self) -> None:
        try:
            while not self._stop.is_set():
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    if self._is_file:
                        break
                    if self._log:
                        self._log("[stream] 读帧失败，0.5s 后重试…")
                    time.sleep(0.5)
                    continue

                if self._is_file:
                    t_sec = float(self._cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                else:
                    t_sec = self._frame_idx / self._fps

                item = self._ring.append(t_sec, frame)
                with self._pending_lock:
                    self._pending.append(item)

                self._frame_idx += 1
                del frame
        finally:
            self._producer_done.set()

    def _consumer_loop(self) -> None:
        while True:
            item: TimestampedFrame | None = None
            with self._pending_lock:
                if self._pending:
                    item = self._pending.popleft()

            if item is None:
                if self._producer_done.is_set() or self._stop.is_set():
                    with self._pending_lock:
                        if not self._pending:
                            break
                time.sleep(0.001)
                continue

            self._session.process_timestamped(item)
            self._emit_ready_clips()

        self._emit_ready_clips()

    def _emit_ready_clips(self) -> None:
        if self._on_clips_ready is None:
            return
        clips = self._session.poll_ready_clips()
        if clips:
            self._on_clips_ready(clips)

    def flush_ready_clips(self) -> list[CachedClip]:
        return self._session.poll_ready_clips()
