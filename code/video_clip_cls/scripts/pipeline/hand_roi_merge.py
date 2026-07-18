"""
双手检测框分组：检测到至少两只手时合并为单个 ROI；不足两只手则跳过该帧。

坐标系：全部在原图像素空间（与 Ultralytics xyxy 一致）。
内存：仅产出 numpy 切片的 .copy() 小图，避免长时间引用整帧。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HandMergeConfig:
    """两手是否合并为单个外接 ROI 的判定（OR 关系，满足任一即合并）。"""

    # IoU 严格大于该值则合并；默认 0 表示只要有交叠（IoU>0）即合并
    merge_iou_gt: float = 0.0
    # 两框中心欧氏距离（像素）不超过该值则合并；None 表示不启用该项
    merge_center_dist_max_px: float | None = None
    # 中心距不超过 frame_diag * 该比例则合并；None 表示不启用（对角线 sqrt(W^2+H^2)）
    merge_center_dist_max_frac_diag: float | None = None


def bbox_area_xyxy(b: list[float]) -> float:
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou_xyxy(a: list[float], b: list[float]) -> float:
    """轴对齐框 IoU。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = bbox_area_xyxy(a) + bbox_area_xyxy(b) - inter
    if ua <= 1e-12:
        return 0.0
    return inter / ua


def bbox_center(xyxy: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def bbox_center_distance(a: list[float], b: list[float]) -> float:
    cx1, cy1 = bbox_center(a)
    cx2, cy2 = bbox_center(b)
    dx = cx1 - cx2
    dy = cy1 - cy2
    return float((dx * dx + dy * dy) ** 0.5)


def union_xyxy(a: list[float], b: list[float]) -> list[float]:
    """两框轴对齐最小外接矩形（仍在原图坐标）。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return [
        min(ax1, bx1),
        min(ay1, by1),
        max(ax2, bx2),
        max(ay2, by2),
    ]


def two_largest_hands(hands: list[list[float]]) -> tuple[list[float], list[float]]:
    """按面积取最大的两只手（ hands 已非空且至少 2 个）。"""
    sorted_h = sorted(hands, key=bbox_area_xyxy, reverse=True)
    return sorted_h[0], sorted_h[1]


def hands_should_merge(
    h1: list[float],
    h2: list[float],
    cfg: HandMergeConfig,
    frame_diag: float,
) -> bool:
    iou = bbox_iou_xyxy(h1, h2)
    if iou > cfg.merge_iou_gt + 1e-12:
        return True
    d = bbox_center_distance(h1, h2)
    if cfg.merge_center_dist_max_px is not None and d <= cfg.merge_center_dist_max_px + 1e-12:
        return True
    if (
        cfg.merge_center_dist_max_frac_diag is not None
        and d <= cfg.merge_center_dist_max_frac_diag * frame_diag + 1e-12
    ):
        return True
    return False


class HandRoiGrouper:
    """根据配置把手框列表转为 1~2 张 ROI（带 padding 的裁剪图）。"""

    def __init__(
        self,
        merge_cfg: HandMergeConfig,
        pad_box_fn,
        pad_ratio: float,
    ) -> None:
        self.merge_cfg = merge_cfg
        self.pad_box_fn = pad_box_fn
        self.pad_ratio = pad_ratio

    def frame_to_rois(
        self,
        frame: np.ndarray,
        hands: list[list[float]],
    ) -> list[np.ndarray]:
        """
        从整帧与手框列表得到本帧用于分类的小图列表。
        至少两只手：取面积最大的两只，合并外接框后 1 张；否则返回空（跳过该帧）。
        """
        h, w = frame.shape[:2]
        if len(hands) < 2:
            return []

        h1, h2 = two_largest_hands(hands)
        uni = union_xyxy(h1, h2)
        x1, y1, x2, y2 = self.pad_box_fn(uni, w, h, self.pad_ratio)
        crop = np.ascontiguousarray(frame[y1:y2, x1:x2].copy())
        return [crop]
