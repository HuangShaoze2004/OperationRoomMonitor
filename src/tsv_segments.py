"""从推流/离线结果 TSV 加载时间段列表。"""
from __future__ import annotations

from pathlib import Path

from pack_utils import log


def load_segments_from_result_tsv(
    tsv_path: Path,
    *,
    skip_empty_top1: bool = False,
) -> list[tuple[float, float, float]]:
    """
    解析 rank/start_sec/end_sec 列，返回 (start, end, score=1.0) 列表。
    skip_empty_top1: 跳过 top1_name 为空或为失败原因文案的行。
    """
    tsv_path = tsv_path.resolve()
    text = tsv_path.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        log(f"[segments] TSV 无数据行: {tsv_path}")
        return []

    header = lines[0].split("\t")
    col = {name.strip(): i for i, name in enumerate(header)}
    for req in ("start_sec", "end_sec"):
        if req not in col:
            raise ValueError(f"TSV 缺少列 {req!r}: {tsv_path}")

    top1_idx = col.get("top1_name")
    segs: list[tuple[float, float, float]] = []
    skipped = 0

    for ln in lines[1:]:
        parts = ln.split("\t")
        if len(parts) <= col["end_sec"]:
            continue
        try:
            t0 = float(parts[col["start_sec"]].strip())
            t1 = float(parts[col["end_sec"]].strip())
        except ValueError:
            skipped += 1
            continue
        if t1 <= t0:
            skipped += 1
            continue
        if skip_empty_top1 and top1_idx is not None and len(parts) > top1_idx:
            name = parts[top1_idx].strip()
            if not name or name.startswith("（"):
                skipped += 1
                continue
        segs.append((t0, t1, 1.0))

    log(f"[segments] 从 TSV 加载 {len(segs)} 段: {tsv_path}")
    if skipped:
        log(f"[segments] 跳过无效/空行 {skipped} 条")
    return segs
