"""相邻成功段 gap 小于阈值时合并，pairs_h 拼接后 aggregate_top3_votes。"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable

from run_haocai_actionformer_consumables_e2e import aggregate_top3_votes

from .tear_gate_merge import E2eRow

_GAP_EPS = 1e-9


def span_key(t0: float, t1: float) -> tuple[float, float]:
    return (round(float(t0), 6), round(float(t1), 6))


def group_rows_by_gap(
    rows: list[E2eRow],
    max_gap_sec: float = 2.0,
) -> list[list[E2eRow]]:
    """左→右贪心分组；失败行单独成组且不跨组合并。"""
    groups: list[list[E2eRow]] = []
    i = 0
    while i < len(rows):
        row = rows[i]
        if not row.is_success():
            groups.append([row])
            i += 1
            continue
        grp = [row]
        j = i + 1
        while j < len(rows):
            nxt = rows[j]
            if not nxt.is_success():
                break
            gap = float(nxt.start_sec) - float(grp[-1].end_sec)
            if gap < float(max_gap_sec) - _GAP_EPS:
                grp.append(nxt)
                j += 1
            else:
                break
        groups.append(grp)
        i = j
    return groups


def e2e_row_from_pairs(
    start_sec: float,
    end_sec: float,
    pairs: list[tuple[str, float]],
    product_map: dict[str, str],
    *,
    rank: int = 0,
) -> E2eRow:
    names, confs = aggregate_top3_votes(pairs)
    n1, n2, n3 = (names + ["", "", ""])[:3]
    c1, c2, c3 = (confs + [0.0, 0.0, 0.0])[:3]
    id1 = product_map.get(n1, "") if n1 else ""
    id2 = product_map.get(n2, "") if n2 else ""
    id3 = product_map.get(n3, "") if n3 else ""

    def _cf(nm: str, c: float) -> str:
        return f"{c:.6f}" if nm else ""

    return E2eRow(
        rank=rank,
        start_sec=float(start_sec),
        end_sec=float(end_sec),
        id1=id1,
        n1=n1,
        c1=_cf(n1, c1),
        id2=id2,
        n2=n2,
        c2=_cf(n2, c2),
        id3=id3,
        n3=n3,
        c3=_cf(n3, c3),
    )


def merge_all_by_gap(
    rows: list[E2eRow],
    span_to_pairs: dict[tuple[float, float], list[tuple[str, float]]],
    product_map: dict[str, str],
    *,
    max_gap_sec: float = 2.0,
    log_fn: Callable[[str], None] | None = None,
) -> list[E2eRow]:
    """按 gap 分组合并；组内拼接 pairs_h 后重新 aggregate top3。"""
    merged: list[E2eRow] = []
    for grp in group_rows_by_gap(rows, max_gap_sec):
        if len(grp) == 1:
            merged.append(grp[0])
            continue

        all_pairs: list[tuple[str, float]] = []
        pair_counts: list[int] = []
        missing = False
        for r in grp:
            sk = span_key(r.start_sec, r.end_sec)
            pairs = span_to_pairs.get(sk)
            if pairs is None:
                missing = True
                break
            pair_counts.append(len(pairs))
            all_pairs.extend(pairs)

        if missing or not all_pairs:
            if log_fn and missing:
                ranks = ",".join(str(r.rank) for r in grp)
                log_fn(f"[gap_merge] 跳过合并 rank={ranks}（缺少 pairs_h 缓存）")
            merged.extend(grp)
            continue

        out_row = e2e_row_from_pairs(
            grp[0].start_sec,
            grp[-1].end_sec,
            all_pairs,
            product_map,
        )
        if log_fn:
            cnt_str = "+".join(str(n) for n in pair_counts)
            ranks = "~".join(str(r.rank) for r in grp)
            log_fn(
                f"[gap_merge] 合并 rank={ranks} "
                f"[{out_row.start_sec:.3f},{out_row.end_sec:.3f}] "
                f"pairs 帧数 {cnt_str}={len(all_pairs)}"
            )
        merged.append(out_row)

    return [replace(r, rank=i) for i, r in enumerate(merged, start=1)]
