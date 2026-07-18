"""Phase1：VideoSwin 特征 + ActionFormer 时段（与仓库 main_pipeline.ActionSegmenter 一致）。"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import cv2

import run_haocai_actionformer_consumables_e2e as e2e
from pack_utils import log


class ActionSegmenter:
    @staticmethod
    def build_segments(
        *,
        video_path: Path,
        stem: str,
        work: Path,
        actionformer_ckpt: Path,
        af_min_score: float,
        af_min_seg_seconds: float,
        python_exe: str,
        feat_batch_size: int,
        device: str,
    ) -> list[tuple[float, float, float]]:
        inp = work / "input"
        feat_dir = work / "features"
        inp.mkdir(parents=True, exist_ok=True)
        feat_dir.mkdir(parents=True, exist_ok=True)
        for stale in inp.glob("*.mp4"):
            stale.unlink(missing_ok=True)

        single_video = inp / video_path.name
        if single_video.resolve() != video_path.resolve():
            shutil.copy2(video_path, single_video)

        meta_path = feat_dir / "meta.json"
        e2e.run_feature_extraction(
            python_exe=python_exe,
            data_root=inp,
            output_dir=feat_dir,
            meta_file=meta_path,
            device=device,
            batch_size=max(1, feat_batch_size),
        )

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        duration, fps = e2e.duration_fps_from_meta(meta, stem)
        if stem not in meta.get("videos", {}):
            log("meta 中未找到 video_id=stem，使用 OpenCV 估 duration…")
            cap0 = cv2.VideoCapture(str(video_path))
            if cap0.isOpened():
                fps = float(cap0.get(cv2.CAP_PROP_FPS)) or fps
                nfr = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT))
                cap0.release()
                if fps > 0 and nfr > 0:
                    duration = nfr / fps

        npy_path = feat_dir / f"{stem}.npy"
        if not npy_path.is_file():
            raise FileNotFoundError(f"特征文件不存在: {npy_path}")

        json_path = work / "infer_single.json"
        e2e.write_infer_json(json_path, stem, duration, fps)

        yaml_path = work / "infer_single.yaml"
        e2e.write_infer_yaml(yaml_path, json_path.resolve(), feat_dir.resolve())

        pkl_dest = work / "eval_results.pkl"
        e2e.run_actionformer_eval(
            python_exe=python_exe,
            yaml_path=yaml_path.resolve(),
            ckpt_path=actionformer_ckpt.resolve(),
            copy_pkl_to=pkl_dest,
        )

        raw_segs = e2e.parse_actionformer_pkl(pkl_dest, stem)
        raw_segs = [(s, e, sc) for s, e, sc in raw_segs if sc > af_min_score]
        segs = e2e.greedy_mutual_exclusive(raw_segs)
        n_exclusive = len(segs)
        min_seg = float(af_min_seg_seconds)
        if min_seg > 0:
            segs = [(s, e, sc) for s, e, sc in segs if (e - s) >= min_seg - 1e-9]
        if min_seg > 0:
            log(
                f"ActionFormer 候选 {len(raw_segs)} -> 互斥后 {n_exclusive} 段 -> "
                f"剔除短于 {min_seg:g}s 后 {len(segs)} 段（score>{af_min_score}）"
            )
        else:
            log(
                f"ActionFormer 候选 {len(raw_segs)} -> 互斥后 {n_exclusive} 段（score>{af_min_score}）"
            )
        return segs
