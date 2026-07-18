"""手篮接触 ActionTriggerLogic：帧防抖 + 上升沿 + 绝对冷却三道锁。"""
from __future__ import annotations

from pipeline.hand_roi_merge import bbox_iou_xyxy


def max_hand_basket_iou(
    hand_boxes: list[list[float]], basket_xyxy: list[float]
) -> float:
    """任意一只手与篮子的最大 IoU；无手则 0.0。"""
    if not hand_boxes:
        return 0.0
    basket = [float(v) for v in basket_xyxy]
    return max(bbox_iou_xyxy(hb, basket) for hb in hand_boxes)


def resolve_contact_iou_thresholds(
    *,
    contact_iou_threshold: float | None = None,
    contact_iou_on: float | None = None,
    contact_iou_off: float | None = None,
) -> tuple[float, float]:
    """由 legacy 单阈值或显式 on/off 解析 IoU 滞回参数。"""
    legacy = float(contact_iou_threshold if contact_iou_threshold is not None else 0.05)
    iou_on = float(contact_iou_on if contact_iou_on is not None else legacy)
    iou_off = float(
        contact_iou_off if contact_iou_off is not None else max(legacy * 0.6, 0.01)
    )
    if iou_off >= iou_on:
        iou_off = max(iou_on - 0.02, 0.01)
    return iou_on, iou_off


class ActionTriggerLogic:
    """
    基于 2D 防区的动作触发状态机。

    三道锁：
    1. 帧级防抖 — 连续 confirm_frames 帧滞回判定为接触才确认
    2. 上升沿 — 单次接触会话仅触发一次 Start
    3. 绝对冷却 — 触发后 cooldown_seconds 内忽略一切信号
    """

    def __init__(
        self,
        fps: float = 25,
        confirm_seconds: float = 0.4,
        cooldown_seconds: float = 5.0,
        threshold_on: float = 0.08,
        threshold_off: float = 0.03,
    ) -> None:
        self.fps = float(fps)
        self.confirm_seconds = float(confirm_seconds)
        self.cooldown_seconds = float(cooldown_seconds)
        self.threshold_on = float(threshold_on)
        self.threshold_off = float(threshold_off)
        if self.threshold_off >= self.threshold_on:
            self.threshold_off = max(self.threshold_on - 0.02, 0.01)

        self._confirm_frames = max(1, int(round(self.confirm_seconds * self.fps)))
        self._overlap_counter = 0
        self._debounce_start_t: float | None = None
        self._hysteresis_inside = False
        self._armed = True
        self._last_trigger_t = float("-inf")

    def reset(self) -> None:
        """换视频或换篮子时清空内部状态。"""
        self._overlap_counter = 0
        self._debounce_start_t = None
        self._hysteresis_inside = False
        self._armed = True
        self._last_trigger_t = float("-inf")

    def _is_contacting(self, current_iou: float) -> bool:
        if not self._hysteresis_inside:
            return current_iou > self.threshold_on + 1e-12
        return current_iou > self.threshold_off + 1e-12

    def step_iou(self, current_timestamp: float, current_iou: float) -> float | None:
        """以预计算 IoU 驱动状态机（供单元测试）；返回 Start 时间戳或 None。"""
        t = float(current_timestamp)
        iou = float(current_iou)

        if t - self._last_trigger_t < self.cooldown_seconds - 1e-12:
            self._overlap_counter = 0
            self._debounce_start_t = None
            return None

        is_contacting = self._is_contacting(iou)

        if is_contacting:
            if self._overlap_counter == 0:
                self._debounce_start_t = t
            self._overlap_counter += 1
            self._hysteresis_inside = True
        else:
            self._overlap_counter = 0
            self._debounce_start_t = None
            self._hysteresis_inside = False
            self._armed = True

        if self._overlap_counter >= self._confirm_frames and self._armed:
            self._armed = False
            self._last_trigger_t = t
            start_t = self._debounce_start_t if self._debounce_start_t is not None else t
            return start_t

        return None

    def process_frame(
        self,
        current_timestamp: float,
        hand_boxes: list[list[float]],
        basket_box: tuple[float, float, float, float] | list[float],
    ) -> float | None:
        """逐帧处理；任意一只手满足条件即可。触发成功返回 Start 时间戳。"""
        basket = [float(v) for v in basket_box]
        current_iou = max_hand_basket_iou(hand_boxes, basket)
        return self.step_iou(current_timestamp, current_iou)
