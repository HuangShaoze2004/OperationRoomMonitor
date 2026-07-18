#!/usr/bin/env python3
"""
单视频端到端：VideoSwin 特征 → ActionFormer 划段 → 分数引导边界切割+score 过滤 →
手检 + 好帧(>阈值) + 白名单裁剪 + 耗材(softmax max>阈值) → 段内在有效帧上对类名计数，取 **票数前三**，
再以这三类出现次数 **归一化** 为 top1~3 置信度（三项和为 1；不足三类则空位补 0）。
商品 id 来自 Excel「产品编码」。
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

for _repo in Path(__file__).resolve().parents:
    if (_repo / "repo_root.py").is_file() and (_repo / "dataset.py").is_file():
        if str(_repo) not in sys.path:
            sys.path.insert(0, str(_repo))
        break
else:
    raise RuntimeError("未定位到仓库 code/ 根目录")

from repo_root import CODE_ROOT  # noqa: E402

# 单文件夹打包：由 run.py 设置 HAOCAI_E2E_BUNDLE=解压根目录，权重/Excel 走包内路径，ActionFormer 在 <bundle>/actionformer_release
_BUNDLE_ENV = os.environ.get("HAOCAI_E2E_BUNDLE", "").strip()
_BUNDLE_ROOT: Path | None = Path(_BUNDLE_ENV).resolve() if _BUNDLE_ENV else None

if _BUNDLE_ROOT is not None:
    _DEFAULT_EXCEL = _BUNDLE_ROOT / "data" / "视频中的商品信息表.xlsx"
    _DEFAULT_AF_CKPT = _BUNDLE_ROOT / "models" / "actionformer_epoch_045.pth.tar"
    _DEFAULT_HAND = _BUNDLE_ROOT / "models" / "hand_detect.pt"
    _DEFAULT_GOODBAD = _BUNDLE_ROOT / "models" / "goodbad_frame.pt"
    _DEFAULT_HAOCAI = _BUNDLE_ROOT / "models" / "haocai_classify.pt"
else:
    _DEFAULT_EXCEL = CODE_ROOT.parent / "data/haocai/视频中的商品信息表.xlsx"
    _DEFAULT_AF_CKPT = (
        CODE_ROOT
        / "video_clip_cls/runs/actionformer_ckpt/haocai_main_perspective_videoswin_haocai_main_perspective_videoswin/epoch_045.pth.tar"
    )
    _DEFAULT_HAND = CODE_ROOT / "hand_detection/runs/hand_det_y11s_multiframe-better/weights/best.pt"
    _DEFAULT_GOODBAD = CODE_ROOT / "goodORbad_frame/runs/goodbad_frame_y11m_e50/weights/best.pt"
    _DEFAULT_HAOCAI = (
        CODE_ROOT / "haocai_classify/runs/haocai_cls_41cls_goodframe_lastest-0.95/weights/best.pt"
    )


def _actionformer_release_dir() -> Path:
    if _BUNDLE_ROOT is not None:
        return _BUNDLE_ROOT / "actionformer_release"
    return CODE_ROOT / "actionformer_release"


# 耗材投票：复用片段推理工具（infer_single_0506 为平铺目录，非 package）
_SYS_INSERT = str(CODE_ROOT / "video_clip_cls" / "infer_single_0506")
if _SYS_INSERT not in sys.path:
    sys.path.insert(0, _SYS_INSERT)
import run_segments_consumable_vote as _rsv  # noqa: E402

collect_hand_boxes = _rsv.collect_hand_boxes
haocai_softmax_probs = _rsv.haocai_softmax_probs
largest_hand = _rsv.largest_hand
pad_box = _rsv.pad_box_bottom_only
passes_good_gate_top1_conf = _rsv.passes_good_gate_top1_conf
_cls_name = _rsv._cls_name

try:
    import pandas as pd
except ImportError as e:
    raise SystemExit("需要 pandas / openpyxl 读取 Excel：pip install pandas openpyxl") from e

# ---------- 与训练/曾用 infer 对齐的 VideoSwin 参数 ----------
FEAT_STRIDE_FRAMES = 8
CLIP_LEN = 16
FRAME_STRIDE = 1
INPUT_DIM = 768


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_product_code_map(excel_path: Path) -> dict[str, str]:
    """商品名称 -> 产品编码。"""
    df = pd.read_excel(excel_path, sheet_name=0, header=0)
    col_code = "产品编码"
    col_name = "商品名称"
    if col_code not in df.columns or col_name not in df.columns:
        df = pd.read_excel(excel_path, sheet_name=0, header=None)
        col_code, col_name = df.columns[1], df.columns[2]
    m: dict[str, str] = {}
    for _, row in df.iterrows():
        name = row[col_name]
        code = row[col_code]
        if pd.isna(name) or str(name).strip() == "":
            continue
        name_s = str(name).strip()
        if name_s not in m:
            m[name_s] = str(code) if not pd.isna(code) else ""
    return m


def mask_probs_whitelist(
    probs: np.ndarray,
    allowed: frozenset[int],
    n_cls: int,
) -> np.ndarray | None:
    v = np.asarray(probs, dtype=np.float64).ravel()
    if v.size < n_cls:
        v = np.resize(v, n_cls)
    v = v[:n_cls].copy()
    out = np.zeros_like(v)
    for i in allowed:
        if 0 <= i < n_cls:
            out[i] = v[i]
    s = float(np.sum(out))
    if s < 1e-12:
        return None
    return out / s


def allowed_indices_from_json_names(
    allowed_names: list[str], cls_names: dict
) -> frozenset[int] | None:
    """None 表示不按名称裁剪（全类）。"""
    if not allowed_names:
        return None
    idx_by_name: dict[str, int] = {}
    for k, v in cls_names.items():
        nm = str(v).strip()
        if nm and nm not in idx_by_name:
            idx_by_name[nm] = int(k)
    out: set[int] = set()
    for n in allowed_names:
        ns = str(n).strip()
        if ns in idx_by_name:
            out.add(idx_by_name[ns])
    if not out:
        log("警告: allowed_names 与模型类名无交集，白名单裁剪将不生效（等同全类）。")
        return None
    return frozenset(out)


def load_whitelist_json(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "allowed_names" in data:
        raw = data["allowed_names"]
    elif isinstance(data, list):
        raw = data
    else:
        raise ValueError("白名单 JSON 应为 {\"allowed_names\": [...]} 或名称数组")
    return [str(x).strip() for x in raw if str(x).strip()]


def run_feature_extraction(
    *,
    python_exe: str,
    data_root: Path,
    output_dir: Path,
    meta_file: Path,
    device: str,
    batch_size: int,
) -> None:
    ext_script = CODE_ROOT / "video_clip_cls" / "extract_videoswin_features.py"
    cmd = [
        python_exe,
        str(ext_script),
        "--data-root",
        str(data_root),
        "--output-dir",
        str(output_dir),
        "--meta-file",
        str(meta_file),
        "--device",
        device,
        "--clip-len",
        str(CLIP_LEN),
        "--frame-stride",
        str(FRAME_STRIDE),
        "--feat-stride-frames",
        str(FEAT_STRIDE_FRAMES),
        "--batch-size",
        str(batch_size),
        "--max-videos",
        "1",
    ]
    log("运行 VideoSwin 特征提取…")
    env = os.environ.copy()
    env.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")
    r = subprocess.run(cmd, cwd=str(CODE_ROOT), env=env, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"特征提取失败，exit={r.returncode}")


def write_infer_json(
    out_path: Path,
    video_id: str,
    duration: float,
    fps: float,
) -> None:
    payload = {
        "version": "haocai_infer_single_v1",
        "taxonomy": [{"nodeName": "Action", "nodeId": 0}],
        "database": {
            video_id: {
                "subset": "val",
                "duration": float(duration),
                "fps": float(fps),
                "annotations": [
                    {"segment": [0.0, min(1.0, duration)], "label": "Action", "label_id": 0}
                ],
            }
        },
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_infer_yaml(out_path: Path, json_file: Path, feat_folder: Path) -> None:
    jf = str(json_file.resolve())
    ff = str(feat_folder.resolve())
    nf = CLIP_LEN * FRAME_STRIDE
    text = f"""dataset_name: thumos
devices: [0]
train_split: ['train']
val_split: ['val']

dataset:
  json_file: "{jf}"
  feat_folder: "{ff}"
  file_prefix: null
  file_ext: ".npy"
  num_classes: 1
  input_dim: {INPUT_DIM}
  feat_stride: {FEAT_STRIDE_FRAMES}
  num_frames: {nf}
  default_fps: null
  downsample_rate: 1
  trunc_thresh: 0.5
  crop_ratio: [0.9, 1.0]
  max_seq_len: 2304
  force_upsampling: false

model:
  fpn_type: identity
  max_buffer_len_factor: 6.0
  n_mha_win_size: 19
  n_head: 4
  embd_dim: 256
  fpn_dim: 256
  head_dim: 256
  use_abs_pe: false

loader:
  batch_size: 1
  num_workers: 2

test_cfg:
  voting_thresh: 0.75
  pre_nms_topk: 4000
  max_seg_num: 600
  min_score: 0.001
  iou_threshold: 0.1
  duration_thresh: 0.05
  nms_method: soft
  nms_sigma: 0.5
  multiclass_nms: true
"""
    out_path.write_text(text, encoding="utf-8")


def run_actionformer_eval(
    *,
    python_exe: str,
    yaml_path: Path,
    ckpt_path: Path,
    copy_pkl_to: Path,
) -> None:
    af_dir = _actionformer_release_dir()
    eval_py = af_dir / "eval.py"
    cmd = [python_exe, str(eval_py), str(yaml_path), str(ckpt_path), "--saveonly"]
    log("运行 ActionFormer eval（saveonly）…")
    r = subprocess.run(cmd, cwd=str(af_dir), check=False)
    if r.returncode != 0:
        raise RuntimeError(f"ActionFormer eval 失败，exit={r.returncode}")
    src_pkl = ckpt_path.parent / "eval_results.pkl"
    if not src_pkl.is_file():
        raise FileNotFoundError(f"未找到输出: {src_pkl}")
    shutil.copy2(src_pkl, copy_pkl_to)
    log(f"已复制 eval_results.pkl -> {copy_pkl_to}")


def segments_overlap(s0: float, e0: float, s1: float, e1: float) -> bool:
    inter = min(e0, e1) - max(s0, s1)
    return inter > 1e-6


def greedy_mutual_exclusive(
    items: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """items: (t_start, t_end, score)。按 score 降序；与已选段重叠则整段丢弃。"""
    sorted_items = sorted(items, key=lambda x: -x[2])
    picked: list[tuple[float, float, float]] = []
    for s, e, sc in sorted_items:
        if any(segments_overlap(s, e, ps, pe) for ps, pe, _ in picked):
            continue
        picked.append((s, e, sc))
    picked.sort(key=lambda x: x[0])
    return picked


_INTERVAL_EPS = 1e-6
_IOU_NMS_THRESHOLD = 0.4
_HYBRID_MIN_LEN = 1.5


def segment_iou_1d(s0: float, e0: float, s1: float, e1: float) -> float:
    """一维时间段 IoU；无交集或 union<=0 时返回 0.0。"""
    inter = max(0.0, min(e0, e1) - max(s0, s1))
    if inter <= _INTERVAL_EPS:
        return 0.0
    union = max(e0, e1) - min(s0, s1)
    if union <= _INTERVAL_EPS:
        return 0.0
    return inter / union


def _subtract_interval(
    s: float, e: float, ps: float, pe: float
) -> list[tuple[float, float]]:
    """从 [s,e] 挖掉 blocker [ps,pe]，返回 0~2 个不重叠子区间。"""
    if min(e, pe) - max(s, ps) <= _INTERVAL_EPS:
        return [(s, e)]
    out: list[tuple[float, float]] = []
    if ps - s > _INTERVAL_EPS:
        out.append((s, min(e, ps)))
    if e - pe > _INTERVAL_EPS:
        out.append((max(s, pe), e))
    return out


def hybrid_nms_and_trimming(
    items: list[tuple[float, float, float]],
    iou_threshold: float = _IOU_NMS_THRESHOLD,
    min_len: float = _HYBRID_MIN_LEN,
) -> list[tuple[float, float, float]]:
    """混合后处理：IoU NMS 去重 → 边界切割 → 最短片段过滤。"""
    sorted_items = sorted(items, key=lambda x: -x[2])
    picked: list[tuple[float, float, float]] = []
    for s, e, sc in sorted_items:
        if e - s <= _INTERVAL_EPS:
            continue
        if any(
            segment_iou_1d(s, e, ps, pe) > iou_threshold + _INTERVAL_EPS
            for ps, pe, _ in picked
        ):
            continue
        frags: list[tuple[float, float]] = [(s, e)]
        for ps, pe, _ in picked:
            nxt: list[tuple[float, float]] = []
            for fs, fe in frags:
                nxt.extend(_subtract_interval(fs, fe, ps, pe))
            frags = nxt
            if not frags:
                break
        for fs, fe in frags:
            if fe - fs >= min_len - _INTERVAL_EPS:
                picked.append((fs, fe, sc))
    picked.sort(key=lambda x: x[0])
    return picked


def parse_actionformer_pkl(
    pkl_path: Path, video_id: str
) -> list[tuple[float, float, float]]:
    with pkl_path.open("rb") as f:
        data: dict[str, Any] = pickle.load(f)
    vids = data["video-id"]
    t0 = np.asarray(data["t-start"]).reshape(-1)
    t1 = np.asarray(data["t-end"]).reshape(-1)
    scores = np.asarray(data["score"]).reshape(-1)
    # 兼容 str / bytes
    def _norm(x: object) -> str:
        if isinstance(x, bytes):
            return x.decode("utf-8", errors="replace")
        return str(x)

    mask = np.array([_norm(v) == video_id for v in np.asarray(vids).reshape(-1)])
    out: list[tuple[float, float, float]] = []
    for i in np.where(mask)[0]:
        out.append((float(t0[i]), float(t1[i]), float(scores[i])))
    return out


def aggregate_top3_votes(
    pairs: list[tuple[str, float]],
) -> tuple[list[str], list[float]]:
    """
    pairs: (类名, 该帧 max softmax)；按置信度做段内加权累计。
    按累计分数取前三类（同分按类名字典序稳定次序），再以这三类累计分数之和归一化为 top1~3 置信度。
    """
    empty = (["", "", ""], [0.0, 0.0, 0.0])
    if not pairs:
        return empty

    # 1) 初始化“积分池”：key=类名，value=该类在段内累计得到的置信度积分。
    score_pool: defaultdict[str, float] = defaultdict(float)
    # 2) 逐帧累加积分：同一类在不同帧的 top_prob 按加和方式累计。
    for name, conf in pairs:
        score_pool[name] += float(conf)

    # 3) 按累计积分降序排序（同分用类名字典序保证结果稳定），取 Top3。
    ranked = sorted(score_pool.items(), key=lambda x: (-x[1], x[0]))
    top = ranked[:3]
    if not top:
        return empty

    # 4) 仅对 Top3 的累计积分做归一化，得到 top1~top3 置信度（和为 1）。
    total = float(sum(score for _, score in top))
    if total <= 0:
        return empty
    out_names: list[str] = ["", "", ""]
    out_conf: list[float] = [0.0, 0.0, 0.0]
    for i, (nm, score) in enumerate(top):
        out_names[i] = nm
        out_conf[i] = float(score) / total
    return out_names, out_conf


def process_segment_e2e(
    cap: cv2.VideoCapture,
    det: YOLO,
    gb: YOLO,
    cls_m: YOLO,
    *,
    start_sec: float,
    end_sec: float,
    seek_margin_sec: float,
    det_conf: float,
    pad_ratio: float,
    imgsz_det: int,
    imgsz_cls: int,
    frame_stride: int,
    good_top1_conf_threshold: float,
    haocai_min_conf: float,
    gb_names: dict,
    cls_names: dict,
    allowed_class_idx: frozenset[int] | None,
) -> dict[str, Any]:
    probe_from = float(max(0.0, start_sec - seek_margin_sec))
    cap.set(cv2.CAP_PROP_POS_MSEC, probe_from * 1000.0)
    synced_frame: np.ndarray | None = None
    synced_t: float | None = None
    tol = 0.04
    while True:
        ok0, grab = cap.read()
        if not ok0 or grab is None:
            synced_frame, synced_t = None, None
            break
        t0 = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
        if t0 + tol >= start_sec:
            synced_frame, synced_t = grab, t0
            break

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n_cls_key_max = max(int(k) for k in cls_names.keys())
    n_cls = n_cls_key_max + 1

    n_hand_frames = 0
    n_gate_pass = 0
    pairs: list[tuple[str, float]] = []
    frames_read_in_segment = 0

    def one_frame(fr: np.ndarray) -> None:
        nonlocal frames_read_in_segment, n_hand_frames, n_gate_pass, pairs
        frames_read_in_segment += 1
        if frame_stride > 1 and (frames_read_in_segment - 1) % frame_stride != 0:
            return

        r0 = det.predict(fr, conf=det_conf, imgsz=imgsz_det, verbose=False)[0]
        hands = collect_hand_boxes(det, r0.boxes) if r0.boxes else []
        if not hands:
            return

        n_hand_frames += 1
        xyxy = largest_hand(hands)
        x1, y1, x2, y2 = pad_box(xyxy, w, h, pad_ratio)
        crop = fr[y1:y2, x1:x2]
        if not passes_good_gate_top1_conf(
            gb, crop, gb_names, imgsz_cls, good_top1_conf_threshold
        ):
            return
        n_gate_pass += 1
        vec_raw = haocai_softmax_probs(cls_m, crop, imgsz_cls, n_cls)
        if vec_raw is None:
            return
        if allowed_class_idx is not None:
            vec = mask_probs_whitelist(vec_raw, allowed_class_idx, n_cls)
        else:
            vec = vec_raw
        if vec is None:
            return
        top_prob = float(np.max(vec))
        if top_prob <= haocai_min_conf:
            return
        label = int(np.argmax(vec))
        pairs.append((_cls_name(cls_names, label), top_prob))

    if synced_frame is not None and synced_t is not None and synced_t <= end_sec + 0.08:
        one_frame(synced_frame)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        t = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
        if t > end_sec + 0.08:
            break
        if t + 1e-6 < start_sec:
            continue
        one_frame(frame)

    if n_hand_frames == 0:
        return {"ok": False, "reason": "（段内未检测到手部）", "pairs": [], "n_gate_pass": 0}
    if not pairs:
        return {
            "ok": False,
            "reason": "（无有效耗材帧：好帧/白名单/耗材置信度未全部满足）",
            "pairs": [],
            "n_hand_frames": n_hand_frames,
            "n_gate_pass": n_gate_pass,
        }

    n1, c1 = aggregate_top3_votes(pairs)
    return {
        "ok": True,
        "top_names": n1,
        "top_confs": c1,
        "pairs": pairs,
        "n_hand_frames": n_hand_frames,
        "n_gate_pass": n_gate_pass,
        "n_valid": len(pairs),
    }


def duration_fps_from_meta(meta: dict, video_id: str) -> tuple[float, float]:
    v = meta.get("videos", {}).get(video_id, {})
    if v:
        fps = float(v.get("fps", 25.0))
        tf = int(v.get("total_frames", 0))
        if tf > 0 and fps > 0:
            return tf / fps, fps
    return 300.0, 25.0


def main() -> int:
    ap = argparse.ArgumentParser(description="ActionFormer 划段 + 耗材端到端（单视频）")
    ap.add_argument("--video", type=Path, required=True, help="输入 MP4")
    ap.add_argument("--whitelist-json", type=Path, required=True, help='{"allowed_names":["..."]}')
    ap.add_argument(
        "--excel",
        type=Path,
        default=_DEFAULT_EXCEL,
        help="商品名称→产品编码",
    )
    ap.add_argument("--out", type=Path, required=True, help="输出制表符 TXT")
    ap.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="工作目录（默认临时目录；加 --keep-work-dir 可保留）",
    )
    ap.add_argument("--keep-work-dir", action="store_true")
    ap.add_argument(
        "--actionformer-ckpt",
        type=Path,
        default=_DEFAULT_AF_CKPT,
    )
    ap.add_argument(
        "--hand-model",
        type=Path,
        default=_DEFAULT_HAND,
    )
    ap.add_argument(
        "--goodbad-model",
        type=Path,
        default=_DEFAULT_GOODBAD,
    )
    ap.add_argument(
        "--haocai-model",
        type=Path,
        default=_DEFAULT_HAOCAI,
    )
    ap.add_argument("--good-top1-conf-threshold", type=float, default=0.9)
    ap.add_argument("--haocai-min-conf", type=float, default=0.8)
    ap.add_argument("--af-min-score", type=float, default=0.1, help="划段保留 score 下限（不含等于）")
    ap.add_argument("--det-conf", type=float, default=0.5)
    ap.add_argument("--pad-ratio", type=float, default=0.30)
    ap.add_argument("--imgsz-det", type=int, default=640)
    ap.add_argument("--imgsz-cls", type=int, default=224)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--seek-margin-sec", type=float, default=3.0)
    ap.add_argument("--feat-batch-size", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="子进程 Python（建议 conda yolo 环境的 python）",
    )
    args = ap.parse_args()

    video_path = args.video.resolve()
    if not video_path.is_file():
        log(f"找不到视频: {video_path}")
        return 1
    if not args.excel.is_file():
        log(f"找不到 Excel: {args.excel}")
        return 1
    if not args.whitelist_json.is_file():
        log(f"找不到白名单 JSON: {args.whitelist_json}")
        return 1
    for p, name in (
        (args.actionformer_ckpt, "ActionFormer ckpt"),
        (args.hand_model, "hand"),
        (args.goodbad_model, "goodbad"),
        (args.haocai_model, "haocai"),
    ):
        if not Path(p).is_file():
            log(f"缺少{name}: {p}")
            return 1

    stem = video_path.stem
    tmp_ctx: tempfile.TemporaryDirectory | None = None
    if args.work_dir is not None:
        work = Path(args.work_dir).resolve()
        work.mkdir(parents=True, exist_ok=True)
    elif args.keep_work_dir:
        work = Path(tempfile.mkdtemp(prefix="haocai_e2e_"))
        log(f"工作目录（保留）: {work}")
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="haocai_e2e_")
        work = Path(tmp_ctx.name)

    try:
        product_map = load_product_code_map(args.excel.resolve())
        allowed_names = load_whitelist_json(args.whitelist_json.resolve())

        inp = work / "input"
        feat_dir = work / "features"
        inp.mkdir(parents=True, exist_ok=True)
        feat_dir.mkdir(parents=True, exist_ok=True)

        single_video = inp / video_path.name
        if single_video.resolve() != video_path.resolve():
            shutil.copy2(video_path, single_video)

        meta_path = feat_dir / "meta.json"
        run_feature_extraction(
            python_exe=args.python,
            data_root=inp,
            output_dir=feat_dir,
            meta_file=meta_path,
            device=args.device,
            batch_size=max(1, args.feat_batch_size),
        )

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        duration, fps = duration_fps_from_meta(meta, stem)
        if stem not in meta.get("videos", {}):
            # 回退：用文件名 stem 对应 npy
            log("meta 中未找到 video_id=stem，使用 ffprobe 估 duration…")
            cap0 = cv2.VideoCapture(str(video_path))
            if cap0.isOpened():
                fps = float(cap0.get(cv2.CAP_PROP_FPS)) or fps
                nfr = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT))
                cap0.release()
                if fps > 0 and nfr > 0:
                    duration = nfr / fps

        npy_path = feat_dir / f"{stem}.npy"
        if not npy_path.is_file():
            log(f"特征文件不存在: {npy_path}")
            return 1

        json_path = work / "infer_single.json"
        write_infer_json(json_path, stem, duration, fps)

        yaml_path = work / "infer_single.yaml"
        write_infer_yaml(yaml_path, json_path.resolve(), feat_dir.resolve())

        pkl_dest = work / "eval_results.pkl"
        run_actionformer_eval(
            python_exe=args.python,
            yaml_path=yaml_path.resolve(),
            ckpt_path=args.actionformer_ckpt.resolve(),
            copy_pkl_to=pkl_dest,
        )

        raw_segs = parse_actionformer_pkl(pkl_dest, stem)
        raw_segs = [(s, e, sc) for s, e, sc in raw_segs if sc > args.af_min_score]
        segs = greedy_mutual_exclusive(raw_segs)
        log(f"ActionFormer 候选 {len(raw_segs)} -> 互斥后 {len(segs)} 段（score>{args.af_min_score}）")

        log("加载 YOLO 模型…")
        det = YOLO(str(args.hand_model))
        gb = YOLO(str(args.goodbad_model))
        cls_m = YOLO(str(args.haocai_model))
        gb_names = gb.names
        cls_names = cls_m.names
        allowed_idx = allowed_indices_from_json_names(allowed_names, cls_names)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            log("无法打开视频")
            return 1

        sep = "\t"
        header = sep.join(
            [
                "rank",
                "start_sec",
                "end_sec",
                "product_id_top1",
                "top1_name",
                "top1_conf",
                "product_id_top2",
                "top2_name",
                "top2_conf",
                "product_id_top3",
                "top3_name",
                "top3_conf",
            ]
        )
        lines_out = [header]

        try:
            for rank, (t0, t1, af_sc) in enumerate(segs, start=1):
                log(f"段落 rank={rank} [{t0:.3f},{t1:.3f}] score={af_sc:.4f} …")
                info = process_segment_e2e(
                    cap,
                    det,
                    gb,
                    cls_m,
                    start_sec=t0,
                    end_sec=t1,
                    seek_margin_sec=args.seek_margin_sec,
                    det_conf=args.det_conf,
                    pad_ratio=args.pad_ratio,
                    imgsz_det=args.imgsz_det,
                    imgsz_cls=args.imgsz_cls,
                    frame_stride=max(1, args.frame_stride),
                    good_top1_conf_threshold=args.good_top1_conf_threshold,
                    haocai_min_conf=args.haocai_min_conf,
                    gb_names=gb_names,
                    cls_names=cls_names,
                    allowed_class_idx=allowed_idx,
                )
                if not info.get("ok"):
                    reason = str(info.get("reason", ""))
                    lines_out.append(
                        sep.join(
                            [
                                str(rank),
                                f"{t0:.6f}",
                                f"{t1:.6f}",
                                "",
                                reason,
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                            ]
                        )
                    )
                    continue

                n1, n2, n3 = info["top_names"]
                c1, c2, c3 = info["top_confs"]
                id1 = product_map.get(n1, "") if n1 else ""
                id2 = product_map.get(n2, "") if n2 else ""
                id3 = product_map.get(n3, "") if n3 else ""
                for nm, pid in ((n1, id1), (n2, id2), (n3, id3)):
                    if nm and not pid:
                        log(f"警告: 商品表无名称「{nm}」，产品编码置空。")

                lines_out.append(
                    sep.join(
                        [
                            str(rank),
                            f"{t0:.6f}",
                            f"{t1:.6f}",
                            id1,
                            n1,
                            f"{c1:.6f}" if n1 else "",
                            id2,
                            n2,
                            f"{c2:.6f}" if n2 else "",
                            id3,
                            n3,
                            f"{c3:.6f}" if n3 else "",
                        ]
                    )
                )
        finally:
            cap.release()

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
        log(f"已写出: {args.out.resolve()}")
        if args.work_dir is not None or (args.keep_work_dir and args.work_dir is None):
            log(f"工作目录: {work}")
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
