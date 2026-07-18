"""RTSP / 本地文件 VideoCapture 统一打开（低延迟、TCP 传输）。"""
from __future__ import annotations

import os
from pathlib import Path

import cv2


def is_rtsp_url(source: str) -> bool:
    s = str(source).strip().lower()
    return s.startswith(("rtsp://", "rtsps://"))


def is_local_media(source: str) -> bool:
    if is_rtsp_url(source):
        return False
    return Path(source).is_file()


def open_stream_capture(
    source: str,
    *,
    rtsp_transport: str = "tcp",
    buffer_size: int = 1,
    ffmpeg_low_latency: bool = True,
) -> cv2.VideoCapture:
    """
    打开 RTSP 或本地媒体。
    RTSP 默认 TCP + 小缓冲 + 可选 nobuffer/low_delay，降低延迟与丢包。
    """
    src = str(source).strip()
    if is_rtsp_url(src):
        transport = str(rtsp_transport or "tcp").strip().lower()
        opts = f"rtsp_transport;{transport}|stimeout;5000000"
        if ffmpeg_low_latency:
            opts += "|fflags;nobuffer|flags;low_delay"
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = opts
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if buffer_size > 0:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, int(buffer_size))
    else:
        cap = cv2.VideoCapture(src)
    return cap


def probe_capture(cap: cv2.VideoCapture) -> dict[str, float | int]:
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    return {"width": w, "height": h, "fps": fps}


def skip_warmup_frames(cap: cv2.VideoCapture, n: int) -> int:
    """丢弃流开头若干帧（RTSP 解码预热，避免花屏）。返回实际丢弃帧数。"""
    n = max(0, int(n))
    skipped = 0
    for _ in range(n):
        ok, _ = cap.read()
        if not ok:
            break
        skipped += 1
    return skipped
