#!/usr/bin/env python3
"""批量医生身份识别：使用 7.3 包内 YOLO11n + ReID 流程处理目录下视频。"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACK_ROOT / "src"))

from config import load_run_config
from doctor_identity import DoctorIdentityService

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="批量医生身份识别（YOLO11n 人体框 + ReID）",
    )
    ap.add_argument(
        "--video-dir",
        type=Path,
        default=Path("/home/baitian/flh_devlop/runs"),
        help="待识别视频目录（递归扫描）",
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=PACK_ROOT / "configs" / "default_config.yaml",
        help="配置文件（读取 doctor_identity 段参数）",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=PACK_ROOT / "output" / "doctor_identity_runs.tsv",
        help="结果 TSV 输出路径",
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="递归扫描子目录（默认开启）",
    )
    ap.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="仅扫描顶层目录",
    )
    return ap.parse_args()


def collect_videos(root: Path, *, recursive: bool) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"视频目录不存在: {root}")

    if recursive:
        candidates = root.rglob("*")
    else:
        candidates = root.iterdir()

    videos = sorted(
        p.resolve()
        for p in candidates
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )
    return videos


def infer_one_video(svc: DoctorIdentityService, video_path: Path) -> dict:
    try:
        mod = svc._ensure_loaded()
        best_crop = mod.pick_best_person_crop(
            video_path=video_path,
            detector=svc._person_detector,
            middle_seconds=float(svc.args.doctor_identity_middle_seconds),
            sample_fps=float(svc.args.doctor_identity_sample_fps),
        )
        raw_pid, conf = mod.run_inference_preloaded(
            best_crop,
            svc._reid_model,
            svc._reid_device,
            svc._label_to_pid,
            svc._transform,
        )
        return svc._format_result(raw_pid, conf)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc)}


def write_results(out_path: Path, rows: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_path",
        "video_name",
        "ok",
        "doctor_id",
        "doctor_name",
        "doctor_conf",
        "low_confidence",
        "reason",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> int:
    args = parse_args()
    video_dir = args.video_dir.resolve()
    cfg = load_run_config(PACK_ROOT, args.config.resolve())

    try:
        videos = collect_videos(video_dir, recursive=args.recursive)
    except FileNotFoundError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    if not videos:
        print(f"[error] 未找到视频: {video_dir}", file=sys.stderr)
        return 2

    print(f"[batch] 视频目录: {video_dir}")
    print(f"[batch] 共 {len(videos)} 个视频")
    print(f"[batch] 配置: {args.config}")
    print(f"[batch] ReID 权重: {cfg.doctor_identity_checkpoint}")
    print(f"[batch] YOLO 权重: {cfg.doctor_identity_person_yolo_weights}")
    print(f"[batch] 输出: {args.out}")
    print()

    svc = DoctorIdentityService(cfg)
    rows: list[dict] = []
    ok_count = 0

    try:
        for idx, video in enumerate(videos, start=1):
            print(f"[{idx}/{len(videos)}] {video.name} ... ", end="", flush=True)
            result = infer_one_video(svc, video)
            row = {
                "video_path": str(video),
                "video_name": video.name,
                "ok": result.get("ok", False),
                "doctor_id": result.get("doctor_id", ""),
                "doctor_name": result.get("doctor_name", ""),
                "doctor_conf": (
                    f"{float(result['doctor_conf']):.4f}"
                    if result.get("doctor_conf") is not None
                    else ""
                ),
                "low_confidence": result.get("low_confidence", ""),
                "reason": result.get("reason", ""),
            }
            rows.append(row)

            if result.get("ok"):
                ok_count += 1
                name = result.get("doctor_name") or result.get("doctor_id", "")
                conf = float(result.get("doctor_conf") or 0.0)
                low = " [低置信度]" if result.get("low_confidence") else ""
                print(f"{name} (conf={conf:.4f}){low}")
            else:
                print(f"失败: {result.get('reason', 'unknown')}")
    finally:
        svc.close()

    write_results(args.out.resolve(), rows)
    print()
    print(f"[batch] 完成: {ok_count}/{len(videos)} 识别成功")
    print(f"[batch] 结果已写入: {args.out.resolve()}")
    return 0 if ok_count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
