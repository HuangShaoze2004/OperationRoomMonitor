"""篮子 ROI 交互选取 + 手篮接触上升沿扫描 → 固定窗口段列表。"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Callable

import cv2
from ultralytics import YOLO

from action_trigger_logic import ActionTriggerLogic, resolve_contact_iou_thresholds
from pipeline.hand_roi_merge import bbox_iou_xyxy
from hand_detector import YoloHandByteTracker, create_hand_contact_tracker, detect_hands_xyxy


def _roi_xyxy_from_select(x: int, y: int, w: int, h: int) -> list[float]:
    if w <= 0 or h <= 0:
        raise ValueError("未框选有效区域（宽高须 > 0）")
    return [float(x), float(y), float(x + w), float(y + h)]


def _read_frame_at(cap: cv2.VideoCapture, *, mode: str | float) -> tuple[Any, float]:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = n_frames / fps if n_frames > 0 and fps > 0 else 0.0

    if isinstance(mode, (int, float)):
        t_sec = float(mode)
    elif mode == "first":
        t_sec = 0.0
    elif mode == "middle":
        t_sec = max(0.0, duration * 0.5)
    else:
        raise ValueError(f"未知 roi_frame 模式: {mode!r}")

    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("无法从视频读取用于框选 ROI 的帧")
        t_sec = 0.0
    else:
        t_sec = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
    return frame, t_sec


def save_basket_roi_json(path: Path, roi: list[float], *, video_path: Path | None = None) -> None:
    payload: dict[str, Any] = {"basket_xyxy": [float(v) for v in roi]}
    if video_path is not None:
        payload["video"] = str(video_path.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_basket_roi_json(path: Path) -> list[float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    roi = data.get("basket_xyxy")
    if not isinstance(roi, list) or len(roi) != 4:
        raise ValueError(f"无效的篮子 ROI JSON: {path}")
    return [float(v) for v in roi]


def _scale_frame_for_display(frame, max_display_px: int) -> tuple[Any, float]:
    orig_h, orig_w = frame.shape[:2]
    scale = 1.0
    disp = frame
    if max(orig_w, orig_h) > max_display_px:
        scale = max_display_px / float(max(orig_w, orig_h))
        disp = cv2.resize(
            frame,
            (int(round(orig_w * scale)), int(round(orig_h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        print(
            f"[basket] 4K 预览缩放 scale={scale:.4f} "
            f"({orig_w}x{orig_h} -> {disp.shape[1]}x{disp.shape[0]})"
        )
    return disp, scale


def _select_basket_roi_tkinter(
    disp_bgr,
    *,
    t_sec: float,
    title: str,
) -> tuple[float, float, float, float]:
    """Tkinter 弹窗：按住左键拖动画框，点顶部【确认】提交。"""
    import tkinter as tk
    from tkinter import messagebox

    from PIL import Image, ImageTk

    rgb = cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    img_w, img_h = pil.size

    root = tk.Tk()
    root.title(title)
    root.attributes("-topmost", True)
    root.after(300, lambda: root.attributes("-topmost", False))

    sw = int(root.winfo_screenwidth() or 1920)
    sh = int(root.winfo_screenheight() or 1080)
    # 预留顶部说明+按钮、窗口边框；画布不超过屏幕可用高度
    max_canvas_w = max(640, sw - 48)
    max_canvas_h = max(360, sh - 220)
    ui_scale = min(max_canvas_w / img_w, max_canvas_h / img_h, 1.0)
    show_w = int(round(img_w * ui_scale))
    show_h = int(round(img_h * ui_scale))
    if (show_w, show_h) != (img_w, img_h):
        pil = pil.resize((show_w, show_h), Image.Resampling.LANCZOS)

    state: dict[str, float | None] = {"x1": None, "y1": None, "x2": None, "y2": None}
    start: dict[str, int | None] = {"x": None, "y": None}
    rect_holder: dict[str, int | None] = {"id": None}
    cancelled = {"v": False}

    def to_disp_coords(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
        inv = 1.0 / ui_scale
        return x1 * inv, y1 * inv, x2 * inv, y2 * inv

    def on_confirm() -> None:
        if state["x1"] is None:
            messagebox.showwarning(title, "请先在图片上按住左键拖动，框选篮子区域")
            return
        root.quit()
        root.destroy()

    def on_cancel() -> None:
        cancelled["v"] = True
        root.quit()
        root.destroy()

    top = tk.Frame(root, padx=12, pady=8)
    top.pack(side=tk.TOP, fill=tk.X)

    tk.Label(
        top,
        text=(
            f"参考帧 t={t_sec:.2f}s  |  按住左键在图片上拖动画框  |  完成后点【确认】或按 Enter"
        ),
        font=("", 12),
        justify=tk.LEFT,
        anchor=tk.W,
    ).pack(fill=tk.X)

    status = tk.Label(top, text="尚未框选", font=("", 11), fg="gray", anchor=tk.W)
    status.pack(fill=tk.X, pady=(4, 8))

    btn_frame = tk.Frame(top)
    btn_frame.pack(fill=tk.X)
    confirm_btn = tk.Button(
        btn_frame,
        text="确认",
        command=on_confirm,
        font=("", 15, "bold"),
        width=14,
        height=1,
        bg="#4CAF50",
        fg="white",
        activebackground="#43A047",
    )
    confirm_btn.pack(side=tk.LEFT, padx=(0, 10))
    tk.Button(
        btn_frame,
        text="取消",
        command=on_cancel,
        font=("", 14),
        width=12,
    ).pack(side=tk.LEFT)

    photo = ImageTk.PhotoImage(pil)
    canvas = tk.Canvas(
        root,
        width=show_w,
        height=show_h,
        cursor="crosshair",
        highlightthickness=1,
        highlightbackground="#cccccc",
    )
    canvas.pack(side=tk.TOP, padx=10, pady=(0, 10))
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    def on_press(event: tk.Event) -> None:
        start["x"], start["y"] = int(event.x), int(event.y)
        if rect_holder["id"] is not None:
            canvas.delete(rect_holder["id"])
        rect_holder["id"] = canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline="red", width=3
        )
        status.config(text="正在框选…（松开左键完成矩形）", fg="orange")

    def on_drag(event: tk.Event) -> None:
        if rect_holder["id"] is not None and start["x"] is not None and start["y"] is not None:
            canvas.coords(rect_holder["id"], start["x"], start["y"], event.x, event.y)

    def on_release(event: tk.Event) -> None:
        if start["x"] is None or start["y"] is None:
            return
        x1, y1 = min(start["x"], event.x), min(start["y"], event.y)
        x2, y2 = max(start["x"], event.x), max(start["y"], event.y)
        if x2 - x1 < 8 or y2 - y1 < 8:
            status.config(text="框太小，请重新按住左键拖动", fg="red")
            state["x1"] = state["y1"] = state["x2"] = state["y2"] = None
            return
        dx1, dy1, dx2, dy2 = to_disp_coords(float(x1), float(y1), float(x2), float(y2))
        state["x1"], state["y1"], state["x2"], state["y2"] = dx1, dy1, dx2, dy2
        status.config(
            text=f"已框选 {int(dx2 - dx1)}×{int(dy2 - dy1)} 像素 — 请点击上方绿色【确认】或按 Enter",
            fg="green",
        )

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    root.bind("<Return>", lambda _e: on_confirm())
    root.bind("<Escape>", lambda _e: on_cancel())
    confirm_btn.focus_set()

    # 居中并限制窗口不超过屏幕
    win_w = min(sw - 20, show_w + 24)
    win_h = min(sh - 20, show_h + 180)
    x0 = max(0, (sw - win_w) // 2)
    y0 = max(0, (sh - win_h) // 2)
    root.geometry(f"{win_w}x{win_h}+{x0}+{y0}")
    root.minsize(min(win_w, 720), min(win_h, 480))

    print("[basket] 已打开框选窗口：顶部有绿色【确认】按钮；拖框后点确认或按 Enter")
    root.mainloop()

    if cancelled["v"]:
        raise ValueError("用户取消框选")
    if state["x1"] is None or state["x2"] is None or state["y1"] is None or state["y2"] is None:
        raise ValueError("未框选有效区域：请按住左键拖动画出矩形后点【确认】")
    x1, y1, x2, y2 = state["x1"], state["y1"], state["x2"], state["y2"]
    return float(x1), float(y1), float(x2 - x1), float(y2 - y1)


def _select_basket_roi_matplotlib(
    disp_bgr,
    *,
    t_sec: float,
    title: str,
) -> tuple[float, float, float, float]:
    """matplotlib 弹窗框选；关闭窗口即确认。"""
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector

    rgb = cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    fig_w = min(16.0, max(8.0, w / 120.0))
    fig_h = min(9.0, max(4.5, h / 120.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(rgb)
    ax.set_title(
        f"{title}\n参考帧 t={t_sec:.2f}s | 鼠标左键拖框 | 可拖拽调整 | 关闭窗口确认",
        fontsize=11,
    )
    ax.axis("off")

    box: dict[str, float | None] = {"x1": None, "y1": None, "x2": None, "y2": None}

    def onselect(eclick, erelease) -> None:
        if eclick.xdata is None or erelease.xdata is None:
            return
        if eclick.ydata is None or erelease.ydata is None:
            return
        box["x1"] = float(min(eclick.xdata, erelease.xdata))
        box["y1"] = float(min(eclick.ydata, erelease.ydata))
        box["x2"] = float(max(eclick.xdata, erelease.xdata))
        box["y2"] = float(max(eclick.ydata, erelease.ydata))

    RectangleSelector(
        ax,
        onselect,
        useblit=False,
        button=[1],
        minspanx=10,
        minspany=10,
        spancoords="data",
        interactive=True,
    )
    fig.canvas.manager.set_window_title(title)
    plt.tight_layout()
    print("[basket] 已打开 matplotlib 框选窗口：按住左键拖动画框，关闭窗口确认")
    plt.show()

    if box["x1"] is None or box["x2"] is None or box["y1"] is None or box["y2"] is None:
        raise ValueError("未框选有效区域：请用鼠标拖出一个矩形后关闭窗口")
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    if x2 - x1 < 1 or y2 - y1 < 1:
        raise ValueError("框选区域过小，请重新运行并框选篮子")
    return x1, y1, x2 - x1, y2 - y1


def _select_basket_roi_opencv(
    disp_bgr,
    *,
    title: str,
) -> tuple[float, float, float, float]:
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, disp_bgr.shape[1], disp_bgr.shape[0])
    rx, ry, rw, rh = cv2.selectROI(title, disp_bgr, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)
    cv2.destroyAllWindows()
    return float(rx), float(ry), float(rw), float(rh)


def select_basket_roi(
    video_path: Path,
    *,
    roi_frame: str | float = "middle",
    window_title: str = "框选耗材篮子",
    max_display_px: int = 1920,
    roi_backend: str = "tkinter",
) -> list[float]:
    """弹窗框选篮子 ROI。默认 tkinter（按住拖动 + 确认按钮）。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    try:
        frame, t_sec = _read_frame_at(cap, mode=roi_frame)
        disp, scale = _scale_frame_for_display(frame, max_display_px)

        backend = str(roi_backend).strip().lower()
        if backend == "tkinter":
            rx, ry, rw, rh = _select_basket_roi_tkinter(disp, t_sec=t_sec, title=window_title)
        elif backend == "matplotlib":
            rx, ry, rw, rh = _select_basket_roi_matplotlib(
                disp, t_sec=t_sec, title=window_title
            )
        elif backend == "opencv":
            print(f"[basket] 框选参考帧 t={t_sec:.2f}s，Enter/Space 确认，Esc 取消")
            rx, ry, rw, rh = _select_basket_roi_opencv(disp, title=window_title)
        else:
            raise ValueError(f"未知 roi_backend: {roi_backend!r}，可选 tkinter / matplotlib / opencv")

        if scale != 1.0:
            rx, ry, rw, rh = rx / scale, ry / scale, rw / scale, rh / scale
        roi = _roi_xyxy_from_select(int(round(rx)), int(round(ry)), int(round(rw)), int(round(rh)))
        print(f"[basket] 篮子 ROI xyxy={roi}")
        return roi
    finally:
        cap.release()
        cv2.destroyAllWindows()


def hands_contact_basket(
    hand_boxes: list[list[float]],
    basket_xyxy: list[float],
    iou_threshold: float,
) -> bool:
    """任意一只手框与篮子 IoU 严格大于阈值即视为接触。"""
    thr = float(iou_threshold)
    for hb in hand_boxes:
        if bbox_iou_xyxy(hb, basket_xyxy) > thr + 1e-12:
            return True
    return False



def filter_near_contact_starts(
    starts: list[float],
    min_interval_sec: float,
    *,
    log_fn: Callable[[str], None] | None = None,
) -> list[float]:
    """
    合并时间上过于接近的接触上升沿，保留每簇中的第一个。
    用于抑制手框抖动导致的重复触发（如 71.0s 与 71.9s）。
    """
    gap = float(min_interval_sec)
    if gap <= 0 or not starts:
        return list(starts)
    kept: list[float] = []
    for t in sorted(starts):
        if kept and t - kept[-1] < gap - 1e-9:
            if log_fn:
                log_fn(
                    f"[basket] 忽略近距离上升沿 t={t:.3f}s "
                    f"（距上次 {t - kept[-1]:.3f}s < {gap:g}s）"
                )
            continue
        kept.append(t)
    return kept


def video_duration_sec(cap: cv2.VideoCapture) -> float:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n_frames > 0 and fps > 0:
        return n_frames / fps
    ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)
    end_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
    cap.set(cv2.CAP_PROP_POS_MSEC, ms)
    return max(0.0, end_ms / 1000.0)


def warn_if_hevc(video_path: Path) -> None:
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=nw=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        codec = (out.stdout or "").strip().split("=", 1)[-1].lower()
        if codec in ("hevc", "h265"):
            print(
                "[basket] 警告: 检测到 HEVC 编码，VideoSwin 不受影响但 OpenCV 解码可能不稳定；"
                "建议先运行 scripts/remux_hevc.sh 转 H.264"
            )
    except FileNotFoundError:
        pass


def scan_contact_segments(
    video_path: Path,
    det_model: YOLO | str | Path,
    basket_xyxy: list[float],
    *,
    contact_iou_threshold: float = 0.05,
    contact_iou_on: float | None = None,
    contact_iou_off: float | None = None,
    confirm_seconds: float = 0.4,
    cooldown_seconds: float = 5.0,
    segment_start_offset_sec: float = 1.0,
    segment_end_offset_sec: float = 5.0,
    min_segment_sec: float = 4.0,
    scan_frame_stride: int = 1,
    det_conf: float = 0.6,
    imgsz_det: int = 640,
    device: str = "cuda",
    half: bool = False,
    args: Any | None = None,
    pack_root: Path | None = None,
    log_fn: Callable[[str], None] | None = print,
) -> list[tuple[float, float, float]]:
    """
    全片扫描手篮接触上升沿，每段 [contact+start_offset, contact+end_offset]（末尾截断至视频时长）。
    截断后段长短于 min_segment_sec 的段会被丢弃。
    接触判定经 ActionTriggerLogic（滞回 + 帧防抖 + 上升沿 + 绝对冷却）。
    返回 (start_sec, end_sec, score) 列表，score 固定 1.0。
    """
    iou_on, iou_off = resolve_contact_iou_thresholds(
        contact_iou_threshold=contact_iou_threshold,
        contact_iou_on=contact_iou_on,
        contact_iou_off=contact_iou_off,
    )
    model = det_model if isinstance(det_model, YOLO) else YOLO(str(det_model))
    predict_kw: dict[str, Any] = {"device": device}
    if half:
        predict_kw["half"] = True

    stride = max(1, int(scan_frame_stride))
    t_start_off = float(segment_start_offset_sec)
    t_end_off = float(segment_end_offset_sec)
    if t_end_off <= t_start_off + 1e-9:
        raise ValueError(
            f"segment_end_offset_sec ({t_end_off}) 须大于 segment_start_offset_sec ({t_start_off})"
        )
    basket = [float(v) for v in basket_xyxy]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    trigger = ActionTriggerLogic(
        fps=fps,
        confirm_seconds=float(confirm_seconds),
        cooldown_seconds=float(cooldown_seconds),
        threshold_on=iou_on,
        threshold_off=iou_off,
    )

    starts: list[float] = []
    frame_idx = 0
    hand_tracker: YoloHandByteTracker | None = None
    if args is not None:
        hand_tracker = create_hand_contact_tracker(
            args,
            model,
            det_conf=det_conf,
            imgsz_det=imgsz_det,
            predict_kw=predict_kw,
            pack_root=pack_root,
        )
        if hand_tracker is not None and log_fn:
            log_fn("[basket] 接触判定已启用 YOLO ByteTrack")

    try:
        duration = video_duration_sec(cap)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_idx += 1
            if stride > 1 and (frame_idx - 1) % stride != 0:
                continue

            t_sec = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if hand_tracker is not None:
                hands = hand_tracker.update(frame)
            else:
                hands = detect_hands_xyxy(
                    model,
                    frame,
                    det_conf=det_conf,
                    imgsz_det=imgsz_det,
                    predict_kw=predict_kw,
                )
            event_t = trigger.process_frame(t_sec, hands, basket)
            if event_t is not None:
                starts.append(event_t)
                if log_fn:
                    log_fn(f"[basket] 接触上升沿 t={event_t:.3f}s")
    finally:
        if hand_tracker is not None:
            hand_tracker.reset()
        cap.release()

    starts = filter_near_contact_starts(
        starts, float(cooldown_seconds), log_fn=log_fn
    )

    segs: list[tuple[float, float, float]] = []
    min_seg = float(min_segment_sec)
    for t_contact in starts:
        t0 = t_contact + t_start_off
        t1 = t_contact + t_end_off
        if duration > 0:
            t1 = min(t1, duration)
        if t1 <= t0 + 1e-9:
            continue
        seg_len = t1 - t0
        if min_seg > 0 and seg_len < min_seg - 1e-9:
            if log_fn:
                log_fn(
                    f"[basket] 丢弃截断短段 [{t0:.3f}, {t1:.3f}] "
                    f"时长 {seg_len:.3f}s < {min_seg:g}s"
                )
            continue
        segs.append((t0, t1, 1.0))

    confirm_frames = max(1, int(round(float(confirm_seconds) * fps)))
    if log_fn:
        log_fn(
            f"[basket] 扫描完成: {len(segs)} 段 "
            f"([contact+{t_start_off:g}, contact+{t_end_off:g}]s, "
            f"IoU on>{iou_on:g} off<={iou_off:g}, "
            f"confirm={float(confirm_seconds):g}s (~{confirm_frames} frames), "
            f"cooldown={float(cooldown_seconds):g}s"
            + (f", min_segment>={min_seg:g}s" if min_seg > 0 else "")
            + ")"
        )
    return segs


def build_segments_from_basket(
    video_path: Path,
    hand_model: Path,
    *,
    basket_roi_json: Path | None = None,
    save_roi_json: Path | None = None,
    skip_roi_select: bool = False,
    roi_frame: str | float = "middle",
    roi_backend: str = "tkinter",
    contact_iou_threshold: float = 0.05,
    contact_iou_on: float | None = None,
    contact_iou_off: float | None = None,
    confirm_seconds: float = 0.4,
    cooldown_seconds: float = 5.0,
    segment_start_offset_sec: float = 1.0,
    segment_end_offset_sec: float = 5.0,
    min_segment_sec: float = 4.0,
    scan_frame_stride: int = 1,
    det_conf: float = 0.6,
    imgsz_det: int = 640,
    device: str = "cuda",
    half: bool = False,
    args: Any | None = None,
    pack_root: Path | None = None,
    log_fn: Callable[[str], None] | None = print,
) -> tuple[list[tuple[float, float, float]], list[float]]:
    """解析/框选 ROI 并扫描接触段。返回 (segments, basket_xyxy)。"""
    warn_if_hevc(video_path)

    if basket_roi_json is not None and basket_roi_json.is_file():
        roi = load_basket_roi_json(basket_roi_json)
        if log_fn:
            log_fn(f"[basket] 从 JSON 加载 ROI: {basket_roi_json}")
    elif skip_roi_select:
        raise ValueError("skip_roi_select 需要有效的 --basket-roi-json")
    else:
        roi = select_basket_roi(video_path, roi_frame=roi_frame, roi_backend=roi_backend)

    if save_roi_json is not None:
        save_basket_roi_json(save_roi_json, roi, video_path=video_path)
        if log_fn:
            log_fn(f"[basket] ROI 已保存: {save_roi_json}")

    segs = scan_contact_segments(
        video_path,
        hand_model,
        roi,
        contact_iou_threshold=contact_iou_threshold,
        contact_iou_on=contact_iou_on,
        contact_iou_off=contact_iou_off,
        confirm_seconds=confirm_seconds,
        cooldown_seconds=cooldown_seconds,
        segment_start_offset_sec=segment_start_offset_sec,
        segment_end_offset_sec=segment_end_offset_sec,
        min_segment_sec=min_segment_sec,
        scan_frame_stride=scan_frame_stride,
        det_conf=det_conf,
        imgsz_det=imgsz_det,
        device=device,
        half=half,
        args=args,
        pack_root=pack_root,
        log_fn=log_fn,
    )
    return segs, roi
