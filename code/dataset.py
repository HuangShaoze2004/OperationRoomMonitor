#!/usr/bin/env python3
"""
从 ~/data/haocai/ 递归扫描「叶子会话目录」（含 mp4 + xlsx，且子目录中不再含 mp4），
按 Excel 中的时间段从对应视频抽帧，输出到「输出根/images/<商品名称>/<规格>/」并生成 JSON 元数据。
输出分辨率默认与源视频帧一致；可用 --max-width / --max-height 限制最大尺寸（仅缩小、不放大）。
可选 --sample-every N：按全局成功保存顺序，每第 N 张在 JSON 中标记 sample=true（便于抽检）。
可选 --limit N：最多生成 N 条（图片或片段），用于快速检查 JSON 格式；0 表示不限制。
可选 --extract-backend：抽帧方式。默认 auto（有 ffmpeg 则用 ffmpeg）。默认精确 seek（-ss 在 -i 之后）；
  可加 --ffmpeg-fast-seek 换快 seek（部分 HEVC/H.265 文件会得到全灰无效帧，脚本会自动改回精确 seek 重试）。
  建议安装 ffprobe 与 ffmpeg，时长/帧率以 ffprobe 为准。
可选 --detect-bbox：用 Grounding DINO（transformers + torch）检测人体并输出 bbox 到 JSON。
可选 --save-vis：在输出根下单独目录（默认 vis/）生成与 images 同结构的 *_vis.jpg，框与英文类别叠加在图上。

列约定（与样本数据一致）：
- 单个 xlsx、两个视频：约 A–J，表头含「视频1」「视频2」时间段列（常见为第 9、10 列）。
- 单个 xlsx、一个视频：约 A–I，最后一列为「视频内时间段」。
- 两个 xlsx、两个视频：每个文件 A–I，最后一列为该视频「视频内时间段」；按文件名中的 01/02 与视频配对。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import cv2
import numpy as np
import pandas as pd

# 临时 / 锁文件
_IGNORE_XLSX = re.compile(r"^~\$|^\._|^\.\~", re.I)


def _log(msg: str) -> None:
    """运行日志（stderr，立即刷新）。"""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


@dataclass
class ImageRecord:
    name: str
    path: str
    label_category: str  # 商品名称
    size: str  # 规格
    sample: bool = False  # 每第 N 张（见 --sample-every）为 True
    # YOLO 格式 [x_center, y_center, w, h] 归一化 0–1；未启用检测或未检出时为 None
    bbox_xywhn: Optional[list[float]] = None
    detection_score: Optional[float] = None


@dataclass
class VideoMeta:
    """视频流元数据；优先来自 ffprobe（比 OpenCV 对 HEVC/VFR 更可靠）。"""

    width: int
    height: int
    fps: float
    duration_sec: float
    frame_count: int = 0


def _parse_fraction(s: str) -> float:
    s = (s or "").strip()
    if not s or s == "0/0":
        return 0.0
    if "/" in s:
        a, b = s.split("/", 1)
        try:
            den = float(b)
            return float(a) / den if den else 0.0
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _ffprobe_video_meta(path: Path, ffprobe_bin: str) -> Optional[VideoMeta]:
    if not shutil.which(ffprobe_bin):
        return None
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, check=False
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if p.returncode != 0 or not p.stdout:
        return None
    try:
        data = json.loads(p.stdout)
    except json.JSONDecodeError:
        return None
    streams = data.get("streams") or []
    if not streams:
        return None
    st = streams[0]
    w = int(st.get("width") or 0)
    h = int(st.get("height") or 0)
    if w < 2 or h < 2:
        return None
    fps = _parse_fraction(str(st.get("avg_frame_rate") or ""))
    if fps <= 0:
        fps = _parse_fraction(str(st.get("r_frame_rate") or ""))
    dur_s = float(st.get("duration") or 0.0)
    fmt = data.get("format") or {}
    if dur_s <= 0:
        dur_s = float(fmt.get("duration") or 0.0)
    nbf = st.get("nb_frames")
    frame_count = 0
    if nbf is not None and str(nbf).strip() and str(nbf).upper() != "N/A":
        try:
            frame_count = int(nbf)
        except (TypeError, ValueError):
            frame_count = 0
    if frame_count <= 0 and dur_s > 0 and fps > 0:
        frame_count = int(round(dur_s * fps))
    if fps <= 0 and dur_s > 0 and frame_count > 0:
        fps = frame_count / dur_s
    if fps <= 0:
        fps = 25.0
    return VideoMeta(
        width=w,
        height=h,
        fps=float(fps),
        duration_sec=float(dur_s),
        frame_count=frame_count,
    )


def _opencv_video_meta(path: Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return VideoMeta(0, 0, 25.0, 0.0, 0)
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = (
            (frame_count / fps) if fps > 0 and frame_count > 0 else 0.0
        )
        return VideoMeta(
            width=w, height=h, fps=fps, duration_sec=duration, frame_count=frame_count
        )
    finally:
        cap.release()


# 同一视频在一张表里会抽多次帧；缓存 ffprobe 结果，避免每个时间点都跑一遍 ffprobe。
_VIDEO_META_CACHE: dict[tuple[str, str], VideoMeta] = {}


def get_video_meta(path: Path, ffprobe_bin: str = "ffprobe") -> VideoMeta:
    key = (str(Path(path).resolve()), ffprobe_bin)
    if key in _VIDEO_META_CACHE:
        return _VIDEO_META_CACHE[key]
    m = _ffprobe_video_meta(path, ffprobe_bin)
    if m is not None:
        _VIDEO_META_CACHE[key] = m
        return m
    m = _opencv_video_meta(path)
    _VIDEO_META_CACHE[key] = m
    return m


def _clamp_time_sec(t_sec: float, meta: VideoMeta) -> float:
    if meta.duration_sec > 0:
        margin = 1.0 / max(meta.fps, 1.0)
        return float(
            min(max(0.0, t_sec), max(0.0, meta.duration_sec - margin))
        )
    return max(0.0, t_sec)


def _time_to_frame_index(t_sec: float, meta: VideoMeta) -> int:
    fps = meta.fps if meta.fps > 0 else 25.0
    t = _clamp_time_sec(t_sec, meta)
    idx = int(round(t * fps))
    if meta.frame_count > 0:
        idx = min(idx, meta.frame_count - 1)
    return max(0, idx)


def _expand_root(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def _is_real_xlsx(path: Path) -> bool:
    if path.suffix.lower() not in (".xlsx", ".xls"):
        return False
    name = path.name
    if name.startswith("~$") or name.startswith(".~"):
        return False
    if _IGNORE_XLSX.search(name):
        return False
    return True


def _is_real_mp4(path: Path) -> bool:
    if path.suffix.lower() != ".mp4":
        return False
    if ".crdownload" in path.name.lower():
        return False
    return True


def _dir_has_mp4_recursive(d: Path) -> bool:
    if not d.is_dir():
        return False
    try:
        for p in d.rglob("*.mp4"):
            if _is_real_mp4(p):
                return True
    except OSError:
        pass
    return False


def iter_leaf_session_dirs(root: Path) -> Iterator[Path]:
    """叶子目录：直接包含至少一个有效 mp4 与 xlsx，且其子目录内不再出现 mp4。"""
    import os

    root = root.resolve()
    if not root.is_dir():
        return

    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        p = Path(dirpath)
        mp4s = [p / f for f in filenames if _is_real_mp4(p / f)]
        xlsxs = [p / f for f in filenames if _is_real_xlsx(p / f)]
        if not mp4s or not xlsxs:
            continue
        sub_has_mp4 = False
        for sub in dirnames:
            if _dir_has_mp4_recursive(p / sub):
                sub_has_mp4 = True
                break
        if sub_has_mp4:
            continue
        yield p


def _video_sort_key(path: Path) -> tuple:
    stem = path.stem
    m = re.search(r"(\d+)", stem)
    n = int(m.group(1)) if m else 10**9
    return (n, stem.lower())


def list_videos(session_dir: Path) -> list[Path]:
    vids = [p for p in session_dir.iterdir() if p.is_file() and _is_real_mp4(p)]
    return sorted(vids, key=_video_sort_key)


def list_excels(session_dir: Path) -> list[Path]:
    xs = [p for p in session_dir.iterdir() if p.is_file() and _is_real_xlsx(p)]
    return sorted(xs, key=lambda p: p.name.lower())


def _excel_pair_key(path: Path) -> tuple:
    m = re.search(r"(\d+)", path.stem)
    n = int(m.group(1)) if m else 10**9
    return (n, path.name.lower())


def _normalize_header(s: Any) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return str(s).strip()


def _find_col(df: pd.DataFrame, *candidates: str) -> str | None:
    cols = [str(c).strip() for c in df.columns]
    for want in candidates:
        for c in df.columns:
            h = _normalize_header(c)
            if h == want or want in h:
                return c
    return None


def normalize_haocai_class_name(name: str) -> str:
    """
    与 build_haocai_dataset_hand_crops.row_product 保持一致的类名归一。
    Excel 与训练类名在个别耗材上同物异名，此处合并为同一条目。
    """
    s = (name or "").strip()
    if s == "一次性使用灭菌棉签":
        return "一次性医用灭菌棉签"
    if s in (
        "一次性使用手术衣",
        "一次性使用手术单（一次性医用垫单）",
        "一次性医用垫单",
    ):
        return "一次性使用手术单"
    return s


def parse_time_range(text: Any) -> tuple[float, float] | None:
    """
    支持：
    - 1.23-2.23 → 1 分 23 秒 到 2 分 23 秒
    - 0.05-0.11 → 0 分 5 秒 到 0 分 11 秒（点后为两位秒）
    - 00：10-00：16 / 00:10-00:16 → mm:ss
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    s = str(text).strip()
    if not s or s.lower() == "nan":
        return None

    # 全角冒号
    s = s.replace("：", ":")

    # mm:ss - mm:ss
    m = re.match(
        r"^\s*(\d{1,2}):(\d{2})\s*[-–—~～]\s*(\d{1,2}):(\d{2})\s*$",
        s,
    )
    if m:
        h1, m1, h2, m2 = m.groups()
        a = int(h1) * 60 + int(m1)
        b = int(h2) * 60 + int(m2)
        return (float(min(a, b)), float(max(a, b)))

    # M.SS - M.SS（分.秒，秒为 1～2 位时按两位秒理解）
    m = re.match(
        r"^\s*(\d+)\s*\.\s*(\d{1,2})\s*[-–—~～]\s*(\d+)\s*\.\s*(\d{1,2})\s*$",
        s,
    )
    if m:
        mm1, ss1, mm2, ss2 = m.groups()
        ss1 = ss1.zfill(2)[:2]
        ss2 = ss2.zfill(2)[:2]
        a = int(mm1) * 60 + int(ss1)
        b = int(mm2) * 60 + int(ss2)
        return (float(min(a, b)), float(max(a, b)))

    return None


def _midpoint_seconds(start: float, end: float) -> float:
    return max(0.0, (start + end) / 2.0)


def _sample_time_in_tear_segment(
    start: float,
    end: float,
    *,
    mode: str = "tear_first_half",
) -> float:
    """
    在 Excel 标注的「撕」时间段 [start, end] 内选取抽帧时刻。

    - tear_first_half（默认）：落在区间**前半段**，取该半段内 3/4 分位
      t = start + 0.375 * (end - start)，与「后半段 3/4」对称。
    - tear_second_half：整段的后 3/4 分位 t = start + 0.75 * (end - start)。
    - midpoint：取 (start+end)/2。
    """
    if end <= start:
        return max(0.0, start)
    span = end - start
    if mode == "midpoint":
        return _midpoint_seconds(start, end)
    if mode == "tear_second_half":
        return max(0.0, start + 0.75 * span)
    # tear_first_half
    return max(0.0, start + 0.375 * span)


def resize_frame_to_max(
    frame: Any,
    max_width: int,
    max_height: int,
) -> Any:
    """
    将帧限制在 max_width×max_height 以内，保持宽高比。
    max_width / max_height 为 0 表示该方向不限制；二者均为 0 则返回原帧（原始分辨率）。
    仅缩小不放大。
    """
    if frame is None:
        return None
    if max_width <= 0 and max_height <= 0:
        return frame
    h, w = frame.shape[:2]
    scales: list[float] = []
    if max_width > 0:
        scales.append(max_width / w)
    if max_height > 0:
        scales.append(max_height / h)
    if not scales:
        return frame
    scale = min(scales)
    scale = min(scale, 1.0)
    if scale >= 1.0:
        return frame
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)


def save_frame_jpeg(
    frame: Any,
    out_path: Path,
    jpeg_quality: int = 85,
    max_width: int = 0,
    max_height: int = 0,
) -> tuple[bool, Optional[np.ndarray]]:
    """按 max_width/max_height 可选缩小后以 JPEG 写出；返回 (是否成功, 与磁盘一致的 BGR 图)。"""
    img = resize_frame_to_max(frame, max_width, max_height)
    if img is None:
        return False, None
    params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    ok = bool(cv2.imwrite(str(out_path), img, params))
    return ok, img if ok else None


def save_bbox_vis_jpeg(
    img_bgr: np.ndarray,
    out_path: Path,
    bbox_xywhn: Optional[list[float]],
    detection_score: Optional[float],
    jpeg_quality: int = 85,
) -> bool:
    """在副本上画框后保存为 JPEG。bbox_xywhn 为 YOLO 格式归一化 [cx, cy, w, h]。"""
    vis = img_bgr.copy()
    h, w = vis.shape[:2]
    if bbox_xywhn and len(bbox_xywhn) == 4:
        cx, cy, bw, bh = bbox_xywhn
        x1 = int(round((cx - bw / 2) * w))
        y1 = int(round((cy - bh / 2) * h))
        x2 = int(round((cx + bw / 2) * w))
        y2 = int(round((cy + bh / 2) * h))
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 0), max(1, min(w, h) // 400))
        cap = f"{detection_score:.2f}" if detection_score is not None else "det"
        (tw, th), _ = cv2.getTextSize(cap, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(y1 - 4, th + 4)
        cv2.rectangle(vis, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), (0, 220, 0), -1)
        cv2.putText(vis, cap, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    else:
        cv2.putText(vis, "no detection", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    return bool(cv2.imwrite(str(out_path), vis, params))


def _write_vis_if_enabled(
    vis_out_root: Optional[Path],
    label_category: str,
    size: str,
    fname: str,
    img_bgr: np.ndarray,
    bbox_xywhn: Optional[list[float]],
    detection_score: Optional[float],
) -> None:
    if vis_out_root is None:
        return
    vis_dir = _product_image_dir(vis_out_root, label_category, size)
    vis_dir.mkdir(parents=True, exist_ok=True)
    vis_path = vis_dir / f"{Path(fname).stem}_vis.jpg"
    save_bbox_vis_jpeg(img_bgr, vis_path, bbox_xywhn, detection_score)


def _clip_xyxy_xyxy(
    xyxy: list[float], w: int, h: int
) -> list[float]:
    x1, y1, x2, y2 = xyxy
    x1 = float(max(0, min(x1, w - 1)))
    x2 = float(max(0, min(x2, w)))
    y1 = float(max(0, min(y1, h - 1)))
    y2 = float(max(0, min(y2, h)))
    if x2 <= x1:
        x2 = min(x1 + 1.0, float(w))
    if y2 <= y1:
        y2 = min(y1 + 1.0, float(h))
    return [x1, y1, x2, y2]


def _xyxy_to_xywhn(xyxy: list[float], w: int, h: int) -> list[float]:
    """xyxy 像素 → YOLO [x_center, y_center, width, height] 归一化 0–1。"""
    x1, y1, x2, y2 = xyxy
    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return [cx / w, cy / h, bw / w, bh / h]


class GroundingDinoDetector:
    """
    使用 Grounding DINO（HuggingFace transformers）做开放词汇检测。
    返回得分最高的一个框：YOLO 格式 [cx, cy, w, h] 归一化 + 分数。
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        prompt: str = "person .",
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
    ) -> None:
        import torch
        from PIL import Image as _PILImage  # noqa: F401
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._torch = torch
        self._PILImage = _PILImage
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self._device)
        self._model.eval()
        self.prompt = prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        _log(f"GroundingDinoDetector loaded: {model_id} on {self._device}")

    def detect(self, img_bgr: np.ndarray) -> tuple[
        Optional[list[float]],
        Optional[float],
    ]:
        h, w = img_bgr.shape[:2]
        if w < 2 or h < 2:
            return None, None

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = self._PILImage.fromarray(rgb)

        with self._torch.no_grad():
            inputs = self._processor(images=pil, text=self.prompt, return_tensors="pt").to(self._device)
            outputs = self._model(**inputs)
            target_sizes = self._torch.tensor([[h, w]], device=self._device)
            try:
                results = self._processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    target_sizes=target_sizes,
                )[0]
            except TypeError:
                results = self._processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    target_sizes=target_sizes,
                )[0]

        if results is None or len(results["boxes"]) == 0:
            return None, None

        best_idx = int(results["scores"].argmax().item())
        b = results["boxes"][best_idx].tolist()
        score = float(results["scores"][best_idx].item())
        xyxy = _clip_xyxy_xyxy([float(b[0]), float(b[1]), float(b[2]), float(b[3])], w, h)
        xywhn = _xyxy_to_xywhn(xyxy, w, h)
        return xywhn, score


def _is_degenerate_gray_frame(img: np.ndarray) -> bool:
    """ffmpeg 快 seek 在部分 HEVC 码流上可能输出近似中性灰、几乎无纹理的无效帧。"""
    if img is None or img.size == 0:
        return True
    m = float(np.mean(img))
    s = float(np.std(img))
    return 118.0 <= m <= 138.0 and s < 8.0


def extract_frame_ffmpeg(
    video_path: Path,
    t_sec: float,
    *,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
    accurate_seek: bool = True,
    timeout_sec: float = 600.0,
) -> np.ndarray | None:
    """
    使用 ffmpeg 解码单帧。时间戳 clamp 优先用 ffprobe，避免 OpenCV 对 HEVC 的 fps/时长偏差。

    accurate_seek=True（默认）：-ss 在 -i 之后，解码正确，长视频较慢。
    accurate_seek=False：-ss 在 -i 之前，快，少数文件仍可能异常。
    """
    if not shutil.which(ffmpeg_bin):
        return None
    meta = get_video_meta(video_path, ffprobe_bin)
    if meta.width < 2 or meta.height < 2:
        return None
    t_clamped = _clamp_time_sec(t_sec, meta)
    w, h = meta.width, meta.height
    expected_raw = w * h * 3

    def _run_ffmpeg(cmd: list[str]) -> tuple[Optional[bytes], Optional[str]]:
        try:
            p = subprocess.run(
                cmd,
                capture_output=True,
                timeout=timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None, "timeout"
        err = (p.stderr or b"").decode("utf-8", errors="replace")[:800]
        if p.returncode != 0:
            return None, err or f"exit {p.returncode}"
        if not p.stdout:
            return None, err or "empty stdout"
        return p.stdout, None

    def _decode_png(data: bytes) -> Optional[np.ndarray]:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img

    # 1) 精确 seek + PNG（通用）
    if accurate_seek:
        cmd_png = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-ss",
            f"{t_clamped:.6f}",
            "-frames:v",
            "1",
            "-an",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-",
        ]
    else:
        cmd_png = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{t_clamped:.6f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-an",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-",
        ]
    out, err = _run_ffmpeg(cmd_png)
    if out is not None:
        img = _decode_png(out)
        if img is not None and img.size > 0:
            if not accurate_seek and _is_degenerate_gray_frame(img):
                _log(
                    f"快 seek 输出疑似灰帧，改用精确 seek: {video_path.name} t={t_clamped:.2f}s"
                )
                return extract_frame_ffmpeg(
                    video_path,
                    t_sec,
                    ffmpeg_bin=ffmpeg_bin,
                    ffprobe_bin=ffprobe_bin,
                    accurate_seek=True,
                    timeout_sec=timeout_sec,
                )
            return img
        if err and err != "timeout":
            _log(f"ffmpeg PNG 解码失败: {video_path.name}: {err[:200]}")

    # 2) 精确 seek + raw BGR（避免 PNG 编解码；尺寸来自 ffprobe）
    cmd_raw = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-ss",
        f"{t_clamped:.6f}",
        "-frames:v",
        "1",
        "-an",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{w}x{h}",
        "-",
    ]
    if not accurate_seek:
        cmd_raw = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{t_clamped:.6f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-an",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{w}x{h}",
            "-",
        ]
    out2, err2 = _run_ffmpeg(cmd_raw)
    if out2 is not None and len(out2) == expected_raw:
        img = np.frombuffer(out2, dtype=np.uint8).reshape((h, w, 3)).copy()
        if not accurate_seek and _is_degenerate_gray_frame(img):
            _log(
                f"快 seek raw 疑似灰帧，改用精确 seek: {video_path.name} t={t_clamped:.2f}s"
            )
            return extract_frame_ffmpeg(
                video_path,
                t_sec,
                ffmpeg_bin=ffmpeg_bin,
                ffprobe_bin=ffprobe_bin,
                accurate_seek=True,
                timeout_sec=timeout_sec,
            )
        return img
    if err2 and err2 != "timeout":
        _log(f"ffmpeg rawvideo 失败: {video_path.name}: {err2[:200]}")

    return None


def extract_frame_opencv_sequential(
    video_path: Path,
    t_sec: float,
    ffprobe_bin: str = "ffprobe",
) -> Any | None:
    """
    从第 0 帧顺序读到目标帧；帧索引由 ffprobe 元数据计算（比仅用 OpenCV fps 更稳）。
    """
    meta = get_video_meta(video_path, ffprobe_bin)
    target_idx = _time_to_frame_index(t_sec, meta)
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return None
    try:
        frame: Any | None = None
        for _ in range(target_idx + 1):
            ok, frame = cap.read()
            if not ok or frame is None:
                return None
        return frame
    finally:
        cap.release()


def make_extract_frame_fn(
    backend: str,
    ffmpeg_bin: str,
    ffprobe_bin: str,
    accurate_seek: bool,
) -> tuple[Callable[[Path, float], Any | None], str]:
    """
    返回 (抽帧函数, 实际后端说明)。
    auto：有 ffmpeg 用 ffmpeg，否则 OpenCV 顺序解码。
    """
    b = backend.strip().lower()
    if b == "auto":
        b = "ffmpeg" if shutil.which(ffmpeg_bin) else "opencv"
    if b == "ffmpeg" and not shutil.which(ffmpeg_bin):
        _log(f"未找到 {ffmpeg_bin!r}，改用 OpenCV 顺序解码（较慢）")
        b = "opencv"
    if b == "ffmpeg":

        def fn_ffmpeg(p: Path, t: float) -> Any | None:
            img = extract_frame_ffmpeg(
                p,
                t,
                ffmpeg_bin=ffmpeg_bin,
                ffprobe_bin=ffprobe_bin,
                accurate_seek=accurate_seek,
            )
            if img is None:
                return extract_frame_opencv_sequential(p, t, ffprobe_bin)
            return img

        mode = "ffmpeg_accurate" if accurate_seek else "ffmpeg_fast"
        return fn_ffmpeg, mode
    def fn_cv_only(p: Path, t: float) -> Any | None:
        return extract_frame_opencv_sequential(p, t, ffprobe_bin)

    return fn_cv_only, "opencv_sequential"


def _unique_image_name(
    session_rel: str,
    row_idx: int,
    video_tag: str,
    time_raw: str,
    ext: str = ".jpg",
) -> str:
    h = hashlib.sha1(
        f"{session_rel}|{row_idx}|{video_tag}|{time_raw}".encode("utf-8")
    ).hexdigest()[:16]
    safe = re.sub(r"[^\w\u4e00-\u9fff\-]+", "_", session_rel)[-80:]
    return f"{safe}__r{row_idx}_{video_tag}_{h}{ext}"


def _sanitize_dir_segment(text: Any, fallback: str) -> str:
    """目录名：去掉路径非法字符，过长截断；空则用 fallback。"""
    if text is None:
        return fallback
    if isinstance(text, float) and pd.isna(text):
        return fallback
    t = str(text).strip()
    if not t:
        return fallback
    t = re.sub(r'[/\\:\0<>"|?*]+', "_", t)
    t = t.strip(" .")
    if not t or all(c == "." for c in t):
        return fallback
    max_len = 180
    if len(t) > max_len:
        t = t[:max_len].rstrip()
    return t or fallback


def _product_image_dir(
    images_out: Path, label_category: str, size: str
) -> Path:
    """images/<商品名称>/<规格>/"""
    d_name = _sanitize_dir_segment(label_category, "未命名商品")
    d_spec = _sanitize_dir_segment(size, "未填规格")
    return images_out / d_name / d_spec


def _read_excel(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, header=0)


def _limit_reached(records: list[ImageRecord], limit: int) -> bool:
    """limit>0 且已保存条数达到上限时返回 True。"""
    return limit > 0 and len(records) >= limit


def _record_saved(
    records: list[ImageRecord],
    global_idx: list[int],
    sample_every: int,
    fname: str,
    out_path: Path,
    label_category: str,
    size: str,
    bbox_xywhn: Optional[list[float]] = None,
    detection_score: Optional[float] = None,
) -> None:
    """global_idx[0] 为已成功保存张数；每第 sample_every 张标记 sample（N=10 → 第 10、20… 张）。"""
    global_idx[0] += 1
    sample = bool(
        sample_every > 0 and global_idx[0] % sample_every == 0
    )
    records.append(
        ImageRecord(
            name=fname,
            path=str(out_path.resolve()),
            label_category=label_category,
            size=size,
            sample=sample,
            bbox_xywhn=bbox_xywhn,
            detection_score=detection_score,
        )
    )


def _bbox_from_detector(
    detector: Optional[GroundingDinoDetector],
    img_bgr: Optional[np.ndarray],
) -> tuple[Optional[list[float]], Optional[float]]:
    if detector is None or img_bgr is None:
        return None, None
    return detector.detect(img_bgr)


def process_session(
    session_dir: Path,
    data_root: Path,
    images_out: Path,
    records: list[ImageRecord],
    global_idx: list[int],
    sample_every: int,
    limit: int = 0,
    max_width: int = 0,
    max_height: int = 0,
    bbox_detector: Optional[GroundingDinoDetector] = None,
    vis_out_root: Optional[Path] = None,
    extract_frame_fn: Callable[[Path, float], Any | None] = extract_frame_opencv_sequential,
    time_sample_mode: str = "tear_first_half",
) -> int:
    """处理一个叶子目录，返回成功写入的图片数量。limit>0 时最多再写入到总条数达 limit。"""
    videos = list_videos(session_dir)
    excels = list_excels(session_dir)
    if not videos or not excels:
        return 0

    session_rel = str(session_dir.relative_to(data_root))
    n_ok = 0

    def row_product(row: pd.Series, df: pd.DataFrame) -> tuple[str, str]:
        c_name = _find_col(df, "商品名称")
        c_spec = _find_col(df, "规格")
        name = ""
        spec = ""
        if c_name is not None:
            v = row.get(c_name)
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                name = str(v).strip()
        if c_spec is not None:
            v = row.get(c_spec)
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                spec = str(v).strip()
        return normalize_haocai_class_name(name), spec

    # 两个 Excel + 两个视频：各读各表，按行与对应视频抽帧
    if len(excels) >= 2 and len(videos) >= 2:
        excel_list = sorted(excels, key=_excel_pair_key)
        vid_list = sorted(videos, key=_video_sort_key)
        pairs = min(len(excel_list), len(vid_list), 2)
        for pi in range(pairs):
            df = _read_excel(excel_list[pi])
            vid = vid_list[pi]
            time_col = _find_col(
                df,
                "视频内时间段",
                "视频01内时间段",
                "视频02内时间段",
            )
            if time_col is None:
                # 最后一列常为时间
                time_col = df.columns[-1]
            for ri, (_, row) in enumerate(df.iterrows()):
                if _limit_reached(records, limit):
                    return n_ok
                tr = row.get(time_col)
                pr = parse_time_range(tr)
                if pr is None:
                    continue
                t0, t1 = pr
                label, size = row_product(row, df)
                if not label and not size:
                    continue
                t_mid = _sample_time_in_tear_segment(
                    t0, t1, mode=time_sample_mode
                )
                frame = extract_frame_fn(vid, t_mid)
                if frame is None:
                    continue
                fname = _unique_image_name(
                    session_rel, ri, f"v{pi + 1}", str(tr)
                )
                out_dir = _product_image_dir(images_out, label, size)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / fname
                saved, img_out = save_frame_jpeg(
                    frame,
                    out_path,
                    max_width=max_width,
                    max_height=max_height,
                )
                if saved:
                    bx, ds = _bbox_from_detector(bbox_detector, img_out)
                    _record_saved(
                        records, global_idx, sample_every,
                        fname, out_path, label, size,
                        bbox_xywhn=bx, detection_score=ds,
                    )
                    _write_vis_if_enabled(
                        vis_out_root, label, size, fname, img_out, bx, ds,
                    )
                    n_ok += 1
                    if _limit_reached(records, limit):
                        return n_ok
        return n_ok

    # 单个 Excel
    if len(excels) == 1:
        df = _read_excel(excels[0])
        c_v1 = _find_col(df, "视频1内时间段", "视频01内时间段")
        c_v2 = _find_col(df, "视频2内时间段", "视频02内时间段")

        if len(videos) >= 2 and c_v1 is not None and c_v2 is not None:
            vid_list = sorted(videos, key=_video_sort_key)[:2]
            for ri, (_, row) in enumerate(df.iterrows()):
                for vi, (c_time, vid) in enumerate(
                    zip([c_v1, c_v2], vid_list)
                ):
                    if _limit_reached(records, limit):
                        return n_ok
                    tr = row.get(c_time)
                    pr = parse_time_range(tr)
                    if pr is None:
                        continue
                    t_mid = _sample_time_in_tear_segment(
                        *pr, mode=time_sample_mode
                    )
                    frame = extract_frame_fn(vid, t_mid)
                    if frame is None:
                        continue
                    label, size = row_product(row, df)
                    fname = _unique_image_name(
                        session_rel, ri, f"v{vi + 1}", str(tr)
                    )
                    out_dir = _product_image_dir(images_out, label, size)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / fname
                    saved, img_out = save_frame_jpeg(
                        frame,
                        out_path,
                        max_width=max_width,
                        max_height=max_height,
                    )
                    if saved:
                        bx, ds = _bbox_from_detector(bbox_detector, img_out)
                        _record_saved(
                            records, global_idx, sample_every,
                            fname, out_path, label, size,
                            bbox_xywhn=bx, detection_score=ds,
                        )
                        _write_vis_if_enabled(
                            vis_out_root, label, size, fname, img_out, bx, ds,
                        )
                        n_ok += 1
                        if _limit_reached(records, limit):
                            return n_ok
            return n_ok

        # 单视频：最后一列或「视频内时间段」
        time_col = _find_col(df, "视频内时间段", "视频1内时间段")
        if time_col is None:
            time_col = df.columns[-1]
        vid = vid_list[0] if (vid_list := sorted(videos, key=_video_sort_key)) else None
        if vid is None:
            return 0
        for ri, (_, row) in enumerate(df.iterrows()):
            if _limit_reached(records, limit):
                return n_ok
            tr = row.get(time_col)
            pr = parse_time_range(tr)
            if pr is None:
                continue
            t_mid = _sample_time_in_tear_segment(
                *pr, mode=time_sample_mode
            )
            frame = extract_frame_fn(vid, t_mid)
            if frame is None:
                continue
            label, size = row_product(row, df)
            fname = _unique_image_name(session_rel, ri, "v1", str(tr))
            out_dir = _product_image_dir(images_out, label, size)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / fname
            saved, img_out = save_frame_jpeg(
                frame,
                out_path,
                max_width=max_width,
                max_height=max_height,
            )
            if saved:
                bx, ds = _bbox_from_detector(bbox_detector, img_out)
                _record_saved(
                    records, global_idx, sample_every,
                    fname, out_path, label, size,
                    bbox_xywhn=bx, detection_score=ds,
                )
                _write_vis_if_enabled(
                    vis_out_root, label, size, fname, img_out, bx, ds,
                )
                n_ok += 1
                if _limit_reached(records, limit):
                    return n_ok
        return n_ok

    # 其余情况：尝试用第一个 Excel + 第一个视频
    if excels and videos:
        df = _read_excel(excels[0])
        time_col = _find_col(df, "视频内时间段") or df.columns[-1]
        vid = sorted(videos, key=_video_sort_key)[0]
        for ri, (_, row) in enumerate(df.iterrows()):
            if _limit_reached(records, limit):
                return n_ok
            tr = row.get(time_col)
            pr = parse_time_range(tr)
            if pr is None:
                continue
            t_mid = _sample_time_in_tear_segment(
                *pr, mode=time_sample_mode
            )
            frame = extract_frame_fn(vid, t_mid)
            if frame is None:
                continue
            label, size = row_product(row, df)
            fname = _unique_image_name(session_rel, ri, "v1", str(tr))
            out_dir = _product_image_dir(images_out, label, size)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / fname
            saved, img_out = save_frame_jpeg(
                frame,
                out_path,
                max_width=max_width,
                max_height=max_height,
            )
            if saved:
                bx, ds = _bbox_from_detector(bbox_detector, img_out)
                _record_saved(
                    records, global_idx, sample_every,
                    fname, out_path, label, size,
                    bbox_xywhn=bx, detection_score=ds,
                )
                _write_vis_if_enabled(
                    vis_out_root, label, size, fname, img_out, bx, ds,
                )
                n_ok += 1
                if _limit_reached(records, limit):
                    return n_ok
    return n_ok


def main() -> int:
    parser = argparse.ArgumentParser(description="浩材视频抽帧数据集生成")
    parser.add_argument(
        "--data-root",
        type=str,
        default="~/data/haocai",
        help="数据根目录（默认 ~/data/haocai）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./haocai_dataset",
        help="输出根目录（图片与 JSON 放在其下）",
    )
    parser.add_argument(
        "--json-name",
        type=str,
        default="dataset.json",
        help="JSON 文件名（位于 output-dir 下）",
    )
    parser.add_argument(
        "--images-subdir",
        type=str,
        default="images",
        help="图片子目录名（位于 output-dir 下）",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=0,
        metavar="N",
        help="全局按保存顺序计数，每第 N 张在 JSON 中 sample=true（0 表示全部 sample=false）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        metavar="N",
        help="最多生成 N 条记录（与 JSON 条目数一致），用于试跑检查格式；0 表示不限制",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=0,
        metavar="PX",
        help="输出 JPEG 最大宽度（像素），0=不限制（默认，保持原始分辨率）",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=0,
        metavar="PX",
        help="输出 JPEG 最大高度（像素），0=不限制（默认）。与 --max-width 同时生效时缩放到可放入矩形内",
    )
    parser.add_argument(
        "--detect-bbox",
        action="store_true",
        help="用 Grounding DINO 检测人体并写 bbox 到 JSON（需 pip install transformers torch pillow）",
    )
    parser.add_argument(
        "--dino-model-id",
        type=str,
        default="IDEA-Research/grounding-dino-base",
        metavar="ID",
        help="Grounding DINO HuggingFace 模型 ID",
    )
    parser.add_argument(
        "--dino-prompt",
        type=str,
        default="person .",
        metavar="TEXT",
        help="Grounding DINO 检测 prompt（默认 'person .'）",
    )
    parser.add_argument(
        "--dino-box-threshold",
        type=float,
        default=0.30,
        metavar="F",
        help="Grounding DINO box 置信度阈值（默认 0.30）",
    )
    parser.add_argument(
        "--dino-text-threshold",
        type=float,
        default=0.25,
        metavar="F",
        help="Grounding DINO text 置信度阈值（默认 0.25）",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="在 output-dir 下写入可视化图（默认子目录 vis/），与 images 同目录结构，文件名为 <原名>_vis.jpg",
    )
    parser.add_argument(
        "--vis-subdir",
        type=str,
        default="vis",
        help="可视化 JPEG 所在子目录名（位于 output-dir 下，默认 vis）",
    )
    parser.add_argument(
        "--extract-backend",
        type=str,
        choices=("auto", "ffmpeg", "opencv"),
        default="auto",
        help="抽帧：auto=有 ffmpeg 则用 ffmpeg（推荐，HEVC 不易花屏）；"
        "ffmpeg=必须可用 ffmpeg；opencv=顺序解码，无 ffmpeg 时可用但较慢",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        type=str,
        default="ffmpeg",
        metavar="CMD",
        help="ffmpeg 可执行文件名或绝对路径（默认 ffmpeg）",
    )
    parser.add_argument(
        "--ffprobe-bin",
        type=str,
        default="ffprobe",
        metavar="CMD",
        help="ffprobe 可执行文件名（用于时长/帧率/分辨率；默认 ffprobe）",
    )
    parser.add_argument(
        "--ffmpeg-fast-seek",
        action="store_true",
        help="快 seek：-ss 在 -i 之前，长视频抽帧快很多；默认精确 seek 从开头解码到目标时刻，故很慢",
    )
    parser.add_argument(
        "--sample-midpoint",
        action="store_true",
        help="时间段内抽帧取中点；默认取「撕」区间前半段（半段内 3/4 分位）",
    )
    parser.add_argument(
        "--tear-second-half",
        action="store_true",
        help="撕时间段内用整段后半 3/4 分位（旧默认）；与默认前半段二选一",
    )
    args = parser.parse_args()

    if args.sample_every < 0:
        print("--sample-every 须 >= 0", file=sys.stderr)
        return 2
    if args.limit < 0:
        print("--limit 须 >= 0", file=sys.stderr)
        return 2
    if args.max_width < 0 or args.max_height < 0:
        print("--max-width / --max-height 须 >= 0", file=sys.stderr)
        return 2
    bbox_detector: Optional[GroundingDinoDetector] = None
    if args.detect_bbox:
        try:
            _log("Grounding DINO bbox detection enabled")
            _log(
                f"model={args.dino_model_id}, prompt={args.dino_prompt!r}, "
                f"box_threshold={args.dino_box_threshold}, "
                f"text_threshold={args.dino_text_threshold}"
            )
            bbox_detector = GroundingDinoDetector(
                model_id=args.dino_model_id,
                prompt=args.dino_prompt,
                box_threshold=args.dino_box_threshold,
                text_threshold=args.dino_text_threshold,
            )
        except Exception as e:
            print(
                f"启用 --detect-bbox 失败: {type(e).__name__}: {e}\n"
                "请确认已安装: pip install transformers torch pillow",
                file=sys.stderr,
            )
            return 2

    data_root = _expand_root(args.data_root)
    out_root = _expand_root(args.output_dir)
    images_out = out_root / args.images_subdir
    images_out.mkdir(parents=True, exist_ok=True)

    vis_out_root: Optional[Path] = None
    if args.save_vis:
        vis_out_root = out_root / args.vis_subdir
        vis_out_root.mkdir(parents=True, exist_ok=True)

    records: list[ImageRecord] = []
    global_idx = [0]
    total = 0
    sessions = list(iter_leaf_session_dirs(data_root))
    if not sessions:
        print(f"未找到叶子会话目录（需同时含 mp4 与 xlsx）: {data_root}", file=sys.stderr)

    if not shutil.which(args.ffprobe_bin):
        _log(
            f"未找到 {args.ffprobe_bin!r}，时长/帧率将仅用 OpenCV（HEVC 可能偏差）；"
            "建议: conda install ffmpeg 或 apt install ffmpeg"
        )
    extract_frame_fn, extract_mode = make_extract_frame_fn(
        args.extract_backend,
        args.ffmpeg_bin,
        args.ffprobe_bin,
        accurate_seek=not args.ffmpeg_fast_seek,
    )
    _log(f"抽帧后端: {extract_mode}")
    if args.sample_midpoint:
        time_sample_mode = "midpoint"
    elif args.tear_second_half:
        time_sample_mode = "tear_second_half"
    else:
        time_sample_mode = "tear_first_half"
    _log(
        "时间段采样: "
        + (
            "中点（--sample-midpoint）"
            if time_sample_mode == "midpoint"
            else (
                "撕区间后半段 3/4（--tear-second-half）"
                if time_sample_mode == "tear_second_half"
                else "撕区间前半段（默认，半段内 3/4 分位）"
            )
        )
    )
    if extract_mode.startswith("ffmpeg") and not args.ffmpeg_fast_seek:
        _log(
            "精确 seek（默认）在长视频、大时间戳时很慢：每次抽帧都会从文件开头解码到目标时刻。"
            "若可接受略快 seek，请加 --ffmpeg-fast-seek 加速。"
        )

    for sd in sorted(sessions):
        if _limit_reached(records, args.limit):
            break
        n = process_session(
            sd,
            data_root,
            images_out,
            records,
            global_idx,
            args.sample_every,
            args.limit,
            args.max_width,
            args.max_height,
            bbox_detector,
            vis_out_root,
            extract_frame_fn=extract_frame_fn,
            time_sample_mode=time_sample_mode,
        )
        total += n
        print(f"{sd.relative_to(data_root)}: {n} 张")

    json_path = out_root / args.json_name
    payload = [asdict(r) for r in records]
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lim_note = f"（limit={args.limit}）" if args.limit > 0 else ""
    vis_note = (
        f"，可视化目录: {vis_out_root}"
        if vis_out_root is not None
        else ""
    )
    print(
        f"共写入 {total} 张图片{lim_note}，JSON 条目 {len(records)}，元数据: {json_path}{vis_note}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
