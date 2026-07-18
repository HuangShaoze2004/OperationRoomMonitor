"""从 Excel 时间段列加载段列表，供 debug 主流程替代 ActionFormer。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import cv2
import pandas as pd

from pack_utils import log


def parse_mm_ss_to_seconds(value: str) -> float:
    text = str(value).strip()
    if not text:
        raise ValueError("empty time value")
    if "." in text:
        left, right = text.split(".", 1)
        minutes = int(left) if left else 0
        seconds = int(right) if right else 0
        if seconds >= 60:
            raise ValueError(f"invalid mm.ss seconds >= 60: {text}")
        return float(minutes * 60 + seconds)
    return float(int(text))


def _is_legacy_mm_dot_ss(token: str) -> bool:
    if "." not in token:
        return False
    a, b = token.split(".", 1)
    if not a.isdigit() or not b.isdigit():
        return False
    return 1 <= len(b) <= 2


def parse_time_token(t: str) -> float:
    t = str(t).strip().replace("：", ":")
    if not t:
        raise ValueError("empty token")
    if ":" in t:
        parts = [float(x) for x in t.split(":")]
        if len(parts) == 3:
            return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
        if len(parts) == 2:
            return parts[0] * 60.0 + parts[1]
        raise ValueError(f"bad colon time: {t}")
    if _is_legacy_mm_dot_ss(t):
        return parse_mm_ss_to_seconds(t)
    return float(t)


def parse_cell_to_segments_v2(cell: object) -> List[Tuple[float, float]]:
    """解析单元格内多段「开始-结束」（冒号 / 分.秒 / 纯秒）。"""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    text = str(cell).strip()
    if not text:
        return []
    text = (
        text.replace("；", ";")
        .replace("，", ",")
        .replace("、", ",")
        .replace("\n", ";")
        .replace("：", ":")
        .replace(" ", "")
    )
    chunks = re.split(r"[;,]+", text)
    segments: List[Tuple[float, float]] = []
    for ch in chunks:
        if not ch:
            continue
        m = re.match(r"^(.+?)\-(.+)$", ch)
        if not m:
            continue
        left, right = m.group(1), m.group(2)
        try:
            s = parse_time_token(left)
            e = parse_time_token(right)
        except (ValueError, TypeError):
            continue
        if e > s:
            segments.append((s, e))
    return segments


def _video_duration_sec(video_path: Path | None) -> float | None:
    if video_path is None:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps > 0 and nfr > 0:
        return nfr / fps
    return None


def load_segments_from_excel_column_i(
    excel_path: Path,
    *,
    col_index: int = 8,
    sheet_name: int | str = 0,
    video_path: Path | None = None,
    default_score: float = 1.0,
) -> list[tuple[float, float, float]]:
    """
    从 Excel 指定列（默认 I 列 index=8）汇总所有行的时间段，返回 (start, end, score)。
    """
    excel_path = excel_path.resolve()
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=0)

    if df.shape[1] > col_index:
        time_series = df.iloc[:, col_index]
        time_col_name = str(df.columns[col_index])
    else:
        cand_cols = [c for c in df.columns if "时间段" in str(c)]
        if not cand_cols:
            raise ValueError(
                f"Excel 列数不足且未找到含「时间段」的列: {excel_path} (cols={df.shape[1]})"
            )
        time_col_name = str(cand_cols[0])
        time_series = df[time_col_name]

    duration = _video_duration_sec(video_path)
    raw_pairs: list[tuple[float, float]] = []
    invalid_cnt = 0

    for cell in time_series.tolist():
        segs = parse_cell_to_segments_v2(cell)
        for s, e in segs:
            cs, ce = s, e
            if duration is not None:
                cs = max(0.0, min(s, duration))
                ce = max(0.0, min(e, duration))
            if ce <= cs:
                invalid_cnt += 1
                continue
            raw_pairs.append((cs, ce))

    raw_pairs.sort(key=lambda x: (x[0], x[1]))
    segs_out = [(s, e, float(default_score)) for s, e in raw_pairs]

    log(
        f"[debug] Excel 时间段列「{time_col_name}」(index={col_index}) "
        f"→ {len(segs_out)} 段"
        + (f"，丢弃无效 {invalid_cnt} 段" if invalid_cnt else "")
    )
    if duration is not None:
        log(f"[debug] 视频时长 {duration:.3f}s，段已裁剪到 [0, duration]")

    return segs_out
