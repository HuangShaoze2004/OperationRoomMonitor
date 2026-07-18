"""
相邻成功行若 top1 相同：在下一段开头 head_sec 内统计「撕膜」高置信帧数；
>= tear_min_frames 视为两次耗材（不合并），否则合并为一段。

main_pipeline 内：默认在门控窗口内 **手检 → 双手 ROI（与 Phase2 相同合并策略）→ 撕膜分类**；
若未传入 det/grouper 则退化为 **整帧** 撕膜（与旧 pack merge 脚本一致）。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import cv2
from ultralytics import YOLO

from .hand_roi_merge import HandRoiGrouper

try:
    from run_segments_consumable_vote import collect_hand_boxes
except ImportError:  # 脚本独立运行时无 path
    collect_hand_boxes = None  # type: ignore[misc, assignment]


@dataclass
class E2eRow:
    rank: int
    start_sec: float
    end_sec: float
    id1: str
    n1: str
    c1: str
    id2: str
    n2: str
    c2: str
    id3: str
    n3: str
    c3: str

    def is_success(self) -> bool:
        if not self.n1.strip():
            return False
        try:
            float(self.c1.strip())
            return True
        except ValueError:
            return False

    def to_line12(self, rank: int) -> str:
        r = replace(self, rank=rank)
        return "\t".join(
            [
                str(r.rank),
                f"{r.start_sec:.6f}",
                f"{r.end_sec:.6f}",
                r.id1,
                r.n1,
                r.c1,
                r.id2,
                r.n2,
                r.c2,
                r.id3,
                r.n3,
                r.c3,
            ]
        )


def parse_e2e_rows_from_body_lines(lines: list[str]) -> list[E2eRow]:
    rows: list[E2eRow] = []
    for i, line in enumerate(lines, start=2):
        if not line.strip():
            continue
        parts_line = line.split("\t")
        while len(parts_line) < 12:
            parts_line.append("")
        parts_line = parts_line[:12]
        try:
            rank = int(parts_line[0])
            s = float(parts_line[1])
            e = float(parts_line[2])
        except ValueError as ex:
            raise ValueError(f"第{i}行解析失败: {line[:80]}...") from ex
        rows.append(
            E2eRow(
                rank=rank,
                start_sec=s,
                end_sec=e,
                id1=parts_line[3],
                n1=parts_line[4],
                c1=parts_line[5],
                id2=parts_line[6],
                n2=parts_line[7],
                c2=parts_line[8],
                id3=parts_line[9],
                n3=parts_line[10],
                c3=parts_line[11],
            )
        )
    return rows


def tear_class_index(model: YOLO, class_name: str) -> int:
    names: dict[int, str] = model.names  # type: ignore[assignment]
    for k, v in names.items():
        if str(v).strip() == class_name:
            return int(k)
    lower = {str(v).strip().lower(): int(k) for k, v in names.items()}
    if lower.get(class_name.lower()) is not None:
        return lower[class_name.lower()]
    raise ValueError(f"模型中无类别「{class_name}」，names={names}")


def count_tearing_frames(
    cap: cv2.VideoCapture,
    window_start: float,
    window_end: float,
    yolo: YOLO,
    tear_cls: int,
    tear_prob: float,
    imgsz: int,
    *,
    predict_kw: dict[str, Any] | None = None,
    det: YOLO | None = None,
    grouper: HandRoiGrouper | None = None,
    imgsz_det: int = 640,
    det_conf: float = 0.5,
) -> int:
    """[window_start, window_end) 内逐帧统计：P(tear_cls) >= tear_prob 的帧数。

    若提供 det+grouper：每帧先检测手，再对每个 ROI 跑撕膜；**任一 ROI** 达到阈值则该帧计 1。
    否则对 **整帧** 跑一次撕膜（与旧 merge_e2e 一致）。
    """
    pred_tear: dict[str, Any] = {"imgsz": imgsz, "verbose": False}
    pred_det: dict[str, Any] = {"imgsz": imgsz_det, "verbose": False}
    if predict_kw:
        pred_tear.update(predict_kw)
        pred_det.update(predict_kw)
    use_hand = (
        det is not None
        and grouper is not None
        and collect_hand_boxes is not None
    )
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, window_start) * 1000.0)
    cnt = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        t = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
        if t >= window_end - 1e-9:
            break
        if t + 1e-6 < window_start:
            continue
        if use_hand:
            r0 = det.predict(  # type: ignore[union-attr]
                frame, conf=det_conf, **pred_det
            )[0]
            hands = collect_hand_boxes(det, r0.boxes) if r0.boxes else []  # type: ignore[arg-type]
            if not hands:
                continue
            rois = grouper.frame_to_rois(frame, hands)  # type: ignore[union-attr]
            frame_hit = False
            for crop in rois:
                if crop is None or crop.size == 0:
                    continue
                res = yolo.predict(crop, **pred_tear)[0]
                if res.probs is None:
                    continue
                prob_tear = float(res.probs.data[tear_cls].item())
                if prob_tear >= tear_prob - 1e-12:
                    frame_hit = True
                    break
            if frame_hit:
                cnt += 1
        else:
            res = yolo.predict(frame, **pred_tear)[0]
            if res.probs is None:
                continue
            prob_tear = float(res.probs.data[tear_cls].item())
            if prob_tear >= tear_prob - 1e-12:
                cnt += 1
    return cnt


def merge_two_segments(a: E2eRow, b: E2eRow) -> E2eRow:
    n1 = a.n1.strip()
    fc1 = max(float(a.c1.strip()), float(b.c1.strip()))
    c1s = f"{fc1:.6f}"

    id1 = a.id1.strip() or b.id1.strip()

    top1_name = n1
    cands: list[tuple[str, float, str]] = []
    for row in (a, b):
        for nm, cf, pid in (
            (row.n2.strip(), row.c2.strip(), row.id2.strip()),
            (row.n3.strip(), row.c3.strip(), row.id3.strip()),
        ):
            if not nm or not cf:
                continue
            try:
                cff = float(cf)
            except ValueError:
                continue
            if nm == top1_name:
                continue
            cands.append((nm, cff, pid))

    cands.sort(key=lambda x: -x[1])
    seen: set[str] = set()
    picked: list[tuple[str, float, str]] = []
    for nm, cff, pid in cands:
        if nm in seen:
            continue
        seen.add(nm)
        picked.append((nm, cff, pid))
        if len(picked) >= 2:
            break

    id2 = n2 = c2 = id3 = n3 = c3 = ""
    if len(picked) >= 1:
        n2, c2f, id2 = picked[0]
        c2 = f"{c2f:.6f}"
    if len(picked) >= 2:
        n3, c3f, id3 = picked[1]
        c3 = f"{c3f:.6f}"

    return E2eRow(
        rank=0,
        start_sec=a.start_sec,
        end_sec=b.end_sec,
        id1=id1,
        n1=n1,
        c1=c1s,
        id2=id2,
        n2=n2,
        c2=c2,
        id3=id3,
        n3=n3,
        c3=c3,
    )


def one_pass_merge(
    rows: list[E2eRow],
    cap: cv2.VideoCapture,
    yolo: YOLO,
    tear_cls: int,
    *,
    head_sec: float,
    tear_prob: float,
    tear_min_frames: int,
    imgsz: int,
    predict_kw: dict[str, Any] | None,
    verbose: bool,
    det: YOLO | None = None,
    grouper: HandRoiGrouper | None = None,
    imgsz_det: int = 640,
    det_conf: float = 0.5,
) -> tuple[list[E2eRow], bool]:
    out: list[E2eRow] = []
    i = 0
    changed = False
    while i < len(rows):
        a = rows[i]
        if i + 1 >= len(rows):
            out.append(a)
            break
        b = rows[i + 1]
        same_top1 = (
            a.is_success()
            and b.is_success()
            and a.n1.strip() == b.n1.strip()
        )
        if same_top1:
            w0 = b.start_sec
            w1 = min(b.start_sec + head_sec, b.end_sec)
            n_high = count_tearing_frames(
                cap,
                w0,
                w1,
                yolo,
                tear_cls,
                tear_prob,
                imgsz,
                predict_kw=predict_kw,
                det=det,
                grouper=grouper,
                imgsz_det=imgsz_det,
                det_conf=det_conf,
            )
            if verbose:
                mode = "hand_roi" if det is not None and grouper is not None else "full_frame"
                print(
                    f"[tear_gate:{mode}] 窗口 [{w0:.3f},{w1:.3f})s（下一段起点起 head_sec={head_sec:g}s，截断至本段 end） "
                    f"P(tearing)>={tear_prob} 计数={n_high} (保留两段需>={tear_min_frames})",
                    flush=True,
                )
            if n_high >= tear_min_frames:
                out.append(a)
                out.append(b)
            else:
                out.append(merge_two_segments(a, b))
                changed = True
            i += 2
        else:
            out.append(a)
            i += 1
    return out, changed


def merge_all(
    rows: list[E2eRow],
    cap: cv2.VideoCapture,
    yolo: YOLO,
    tear_cls: int,
    *,
    head_sec: float,
    tear_prob: float,
    tear_min_frames: int,
    imgsz: int,
    predict_kw: dict[str, Any] | None = None,
    verbose: bool = False,
    det: YOLO | None = None,
    grouper: HandRoiGrouper | None = None,
    imgsz_det: int = 640,
    det_conf: float = 0.5,
) -> list[E2eRow]:
    cur = rows
    while True:
        cur, changed = one_pass_merge(
            cur,
            cap,
            yolo,
            tear_cls,
            head_sec=head_sec,
            tear_prob=tear_prob,
            tear_min_frames=tear_min_frames,
            imgsz=imgsz,
            predict_kw=predict_kw,
            verbose=verbose,
            det=det,
            grouper=grouper,
            imgsz_det=imgsz_det,
            det_conf=det_conf,
        )
        if not changed:
            break
    return cur
