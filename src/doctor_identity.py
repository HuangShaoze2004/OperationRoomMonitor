"""医生身份识别：离线整片 + 推流段内（与耗材并行）。"""
from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

PACK_ROOT = Path(__file__).resolve().parent.parent


def _load_doctor_module(script_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("doctor_identity_runtime", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载医生识别脚本: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _doctor_script_path() -> Path:
    return PACK_ROOT / "doctor_identity_package" / "infer_doctor_from_video.py"


class DoctorIdentityService:
    """一次性加载 YOLO 人体检测 + ReID，供推流每段复用。"""

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self._mod: Any | None = None
        self._person_detector: Any | None = None
        self._reid_model: Any | None = None
        self._reid_device: torch.device | None = None
        self._label_to_pid: dict[int, str] | None = None
        self._transform: Any | None = None
        self._name_map: dict[str, str] = {}

    def _ensure_loaded(self) -> Any:
        if self._mod is not None:
            return self._mod

        script_path = _doctor_script_path()
        if not script_path.is_file():
            raise FileNotFoundError(f"缺少脚本: {script_path}")

        pack_dir = script_path.parent
        if str(pack_dir) not in sys.path:
            sys.path.insert(0, str(pack_dir))

        mod = _load_doctor_module(script_path)
        checkpoint = Path(self.args.doctor_identity_checkpoint).resolve()
        labels_csv = Path(self.args.doctor_identity_labels_csv).resolve()
        yolo_weights = Path(self.args.doctor_identity_person_yolo_weights).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"缺少权重: {checkpoint}")
        if not labels_csv.is_file():
            raise FileNotFoundError(f"缺少标签映射: {labels_csv}")

        self._person_detector = mod.create_person_detector(
            yolo_weights,
            min_conf=float(self.args.doctor_identity_person_det_conf),
            pad_frac=float(self.args.doctor_identity_pad_frac),
            imgsz=int(self.args.doctor_identity_person_det_imgsz),
        )
        self._reid_model, self._reid_device, self._label_to_pid, self._transform = (
            mod.load_reid_model(checkpoint)
        )
        self._name_map = mod.load_name_mapping(labels_csv)
        self._mod = mod
        return mod

    def close(self) -> None:
        self._person_detector = None

    def _format_result(self, raw_pid: str, conf: float) -> dict[str, Any]:
        min_conf = float(self.args.doctor_identity_min_identity_confidence)
        name = self._name_map.get(str(raw_pid), "")
        low = conf < min_conf
        return {
            "ok": True,
            "doctor_id": str(raw_pid),
            "doctor_name": name,
            "doctor_conf": conf,
            "low_confidence": low,
        }

    def infer_segment(
        self,
        *,
        start_sec: float,
        end_sec: float,
        video_path: Path | None = None,
        use_file_source: bool = False,
        frames: list[tuple[float, np.ndarray]] | None = None,
    ) -> dict[str, Any]:
        """段内医生识别。失败返回 ok=False 与 reason。"""
        try:
            mod = self._ensure_loaded()
            sample_fps = float(
                getattr(
                    self.args,
                    "doctor_identity_segment_sample_fps",
                    getattr(self.args, "doctor_identity_sample_fps", 3.0),
                )
            )
            win_sec = float(getattr(self.args, "doctor_identity_segment_window_sec", 3.0))
            doc_t0, doc_t1 = segment_doctor_infer_window(start_sec, end_sec, win_sec)

            if use_file_source and video_path is not None and video_path.is_file():
                best_crop = mod.pick_best_person_crop_in_window(
                    video_path,
                    self._person_detector,
                    doc_t0,
                    doc_t1,
                    sample_fps,
                )
            elif frames:
                best_crop = mod.pick_best_person_crop_from_frames(
                    frames,
                    self._person_detector,
                    doc_t0,
                    doc_t1,
                )
            else:
                return {"ok": False, "reason": "无可用视频源或缓存帧"}

            raw_pid, conf = mod.run_inference_preloaded(
                best_crop,
                self._reid_model,
                self._reid_device,
                self._label_to_pid,
                self._transform,
            )
            return self._format_result(raw_pid, conf)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": str(exc)}

    def infer_whole_video(self, video_path: Path) -> str:
        """离线全片：取视频中间窗口识别，返回展示用文本。"""
        if not bool(getattr(self.args, "doctor_identity_enabled", True)):
            return "未启用"

        try:
            mod = self._ensure_loaded()
            best_crop = mod.pick_best_person_crop(
                video_path=video_path,
                detector=self._person_detector,
                middle_seconds=float(self.args.doctor_identity_middle_seconds),
                sample_fps=float(self.args.doctor_identity_sample_fps),
            )
            raw_pid, conf = mod.run_inference_preloaded(
                best_crop,
                self._reid_model,
                self._reid_device,
                self._label_to_pid,
                self._transform,
            )
            res = self._format_result(raw_pid, conf)
            suffix = " [低置信度]" if res.get("low_confidence") else ""
            name = res.get("doctor_name") or ""
            if name:
                return f"{name} (id={raw_pid}, conf={conf:.4f}){suffix}"
            return f"doctor_id={raw_pid} (conf={conf:.4f}){suffix}"
        except Exception as exc:  # noqa: BLE001
            return f"识别失败（{exc}）"


def segment_doctor_infer_window(
    start_sec: float,
    end_sec: float,
    window_sec: float,
) -> tuple[float, float]:
    """段内医生 ReID 采样窗：全长不超过 window_sec，相对段窗口居中。"""
    t0 = float(start_sec)
    t1 = float(end_sec)
    dur = t1 - t0
    w = max(0.1, float(window_sec))
    if dur <= w + 1e-9:
        return t0, t1
    mid = (t0 + t1) * 0.5
    half = w * 0.5
    return mid - half, mid + half


def stream_doctor_enabled(args: Namespace) -> bool:
    return bool(getattr(args, "doctor_identity_enabled", True)) and bool(
        getattr(args, "doctor_identity_stream_enabled", True)
    )


def vote_doctor_from_segment_results(
    segment_results: list[dict[str, Any] | None],
) -> str:
    """
    对各段医生识别结果投票：按 doctor_id 众数取 Top1，展示名与置信度取该 id 下最高 conf 的一段。
    """
    ok = [r for r in segment_results if r is not None and r.get("ok")]
    if not ok:
        reasons = [
            str(r.get("reason", ""))
            for r in segment_results
            if r is not None and not r.get("ok")
        ]
        hint = reasons[0] if reasons else "无有效段"
        return f"识别失败（{hint}）"

    counts = Counter(str(r["doctor_id"]) for r in ok)
    top_id, _ = counts.most_common(1)[0]
    candidates = [r for r in ok if str(r["doctor_id"]) == top_id]
    best = max(candidates, key=lambda r: float(r.get("doctor_conf") or 0.0))
    conf = float(best.get("doctor_conf") or 0.0)
    suffix = " [低置信度]" if best.get("low_confidence") else ""
    name = best.get("doctor_name") or ""
    if name:
        return f"{name} (id={top_id}, conf={conf:.4f}){suffix}"
    return f"doctor_id={top_id} (conf={conf:.4f}){suffix}"


def infer_doctor_text_offline(args: Namespace, video_path: Path) -> str:
    """离线入口：校验资源后返回医生信息文本。"""
    if not bool(getattr(args, "doctor_identity_enabled", True)):
        return "未启用"

    checkpoint = Path(args.doctor_identity_checkpoint).resolve()
    labels_csv = Path(args.doctor_identity_labels_csv).resolve()
    if not checkpoint.is_file():
        return f"识别失败（缺少权重: {checkpoint}）"
    if not labels_csv.is_file():
        return f"识别失败（缺少标签映射: {labels_csv}）"
    if not _doctor_script_path().is_file():
        return f"识别失败（缺少脚本: {_doctor_script_path()}）"

    svc = DoctorIdentityService(args)
    try:
        return svc.infer_whole_video(video_path.resolve())
    finally:
        svc.close()
