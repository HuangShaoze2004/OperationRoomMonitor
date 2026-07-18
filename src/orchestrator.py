"""主流程编排：与仓库 main_pipeline.PipelineManager 逻辑一致，参数来自 YAML（SimpleNamespace）。"""
from __future__ import annotations

import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Any

import cv2
import run_haocai_actionformer_consumables_e2e as e2e
from actionformer_utils import ActionSegmenter
from excel_segments import load_segments_from_excel_column_i
from pipeline.hand_roi_merge import HandMergeConfig, HandRoiGrouper
from pipeline.segment_processor import (
    HaocaiOnlyClassifier,
    process_segment_haocai_from_cap_with_gate_retries,
)
from pipeline.gap_adjacent_merge import merge_all_by_gap
from pipeline.tear_gate_merge import (
    merge_all,
    parse_e2e_rows_from_body_lines,
    tear_class_index,
)
from run_segments_consumable_vote import pad_box_bottom_only as _pad_box
from ultralytics import YOLO

from basket_segmenter import build_segments_from_basket
from doctor_identity import infer_doctor_text_offline
from pack_utils import load_allowed_names_from_excel, log, resolve_allowed_class_idx
from stream_orchestrator import _haocai_infer_kwargs


def _resolve_allowed_names(args: Namespace, excel_path: Path) -> list[str] | None:
    if not getattr(args, "use_whitelist", True):
        return []
    if args.whitelist_json is not None:
        if not args.whitelist_json.is_file():
            log(f"找不到白名单 JSON: {args.whitelist_json}")
            return None
        return e2e.load_whitelist_json(args.whitelist_json.resolve())
    return load_allowed_names_from_excel(excel_path)


def _validate_phase2_weights(args: Namespace, *, require_actionformer: bool) -> bool:
    from hand_detector import validate_hand_assets

    ok, msg = validate_hand_assets(args)
    if not ok:
        log(msg)
        return False
    checks: list[tuple[Any, str]] = [
        (args.goodbad_model, "好坏帧"),
        (args.haocai_model, "耗材分类"),
    ]
    if require_actionformer:
        checks.insert(0, (args.actionformer_ckpt, "ActionFormer ckpt"))
    if getattr(args, "merge_adjacent_tear", False):
        checks.append((args.tear_model, "撕膜分类"))
    for p, lab in checks:
        if p is None or not Path(p).is_file():
            log(f"缺少{lab}: {p}")
            return False
    if args.merge_adjacent_tear:
        tmw = (args.tear_merge_weights or args.tear_model).resolve()
        if not tmw.is_file():
            log(f"撕膜门控需要权重文件: {tmw}")
            return False
    return True


def _filter_segments_by_min_length(
    segs: list[tuple[float, float, float]], min_seg_seconds: float
) -> list[tuple[float, float, float]]:
    if min_seg_seconds <= 0:
        return segs
    return [(s, e, sc) for s, e, sc in segs if (e - s) >= min_seg_seconds - 1e-9]


class PipelineManager:
    def __init__(self, args: Namespace) -> None:
        self.args = args

    def run(self) -> int:
        args = self.args
        video_path = args.video.resolve()
        if not video_path.is_file():
            log(f"找不到视频: {video_path}")
            return 1
        excel_path = args.excel.resolve()
        if not excel_path.is_file():
            log(f"找不到 Excel: {excel_path}")
            return 1

        allowed_names = _resolve_allowed_names(args, excel_path)
        if allowed_names is None:
            return 1
        if not _validate_phase2_weights(args, require_actionformer=True):
            return 1

        stem = video_path.stem
        tmp_ctx: tempfile.TemporaryDirectory | None = None
        if args.work_dir is not None:
            work = Path(args.work_dir).resolve()
            work.mkdir(parents=True, exist_ok=True)
        elif args.keep_work_dir:
            work = Path(tempfile.mkdtemp(prefix="main_pipeline_"))
            log(f"工作目录（保留）: {work}")
        else:
            tmp_ctx = tempfile.TemporaryDirectory(prefix="main_pipeline_")
            work = Path(tmp_ctx.name)

        try:
            product_map = e2e.load_product_code_map(excel_path)
            segs = ActionSegmenter.build_segments(
                video_path=video_path,
                stem=stem,
                work=work,
                actionformer_ckpt=args.actionformer_ckpt,
                af_min_score=args.af_min_score,
                af_min_seg_seconds=args.af_min_seg_seconds,
                python_exe=args.python,
                feat_batch_size=args.feat_batch_size,
                device=args.device,
            )
            return self._run_phase2_and_write(
                segs,
                video_path=video_path,
                excel_path=excel_path,
                allowed_names=allowed_names,
                product_map=product_map,
                work_dir_log=work if args.work_dir is not None or args.keep_work_dir else None,
            )
        finally:
            if tmp_ctx is not None:
                tmp_ctx.cleanup()

    def _run_phase2_and_write(
        self,
        segs: list[tuple[float, float, float]],
        *,
        video_path: Path,
        excel_path: Path,
        allowed_names: list[str],
        product_map: dict[str, str],
        work_dir_log: Path | None = None,
    ) -> int:
        args = self.args

        predict_kw: dict[str, Any] = {"device": args.device}
        if args.half:
            predict_kw["half"] = True

        log("Phase2：加载 YOLO（手 / 好坏帧 / 耗材）…")
        from hand_detector import create_hand_detector

        det = create_hand_detector(args)
        gb = YOLO(str(args.goodbad_model))
        cls_m = YOLO(str(args.haocai_model))

        cls_names = cls_m.names
        hc = HaocaiOnlyClassifier(
            cls_m,
            cls_names=cls_names,
            imgsz_cls=int(args.imgsz_cls),
            predict_kw=predict_kw,
            gb=gb,
            gb_names=gb.names,
        )
        infer_kw = _haocai_infer_kwargs(args, cls_names, None, predict_kw)

        try:
            allowed_idx = resolve_allowed_class_idx(args, excel_path, cls_names)
        except FileNotFoundError as exc:
            log(str(exc))
            return 1
        infer_kw["allowed_class_idx"] = allowed_idx
        if getattr(args, "use_whitelist", True):
            log(f"白名单启用，{len(allowed_idx or ())} 个类参与投票")
        else:
            log("白名单已关闭，使用全 41 类")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            log("无法打开视频")
            return 1

        sep = "\t"
        base_cols = [
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
        ext_cols = ["tear_top1_name", "tear_top2_name"]
        header = sep.join(base_cols if args.legacy_12_col_only else base_cols + ext_cols)
        lines_out = [header]
        span_to_cells: dict[tuple[float, float], list[str]] = {}
        span_to_pairs: dict[tuple[float, float], list[tuple[str, float]]] = {}

        def span_key(t0: float, t1: float) -> tuple[float, float]:
            return (round(float(t0), 6), round(float(t1), 6))

        def infer_one(rank: int, t0: float, t1: float) -> str:
            info = process_segment_haocai_from_cap_with_gate_retries(
                cap,
                det,
                hc,
                start_sec=t0,
                end_sec=t1,
                seek_margin_sec=float(args.seek_margin_sec),
                log_fn=log,
                log_prefix=f"段落 rank={rank}: ",
                **infer_kw,
            )
            if not info.get("ok"):
                reason = str(info.get("reason", ""))
                span_to_pairs[span_key(t0, t1)] = []
                row = [
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
                ]
                if not args.legacy_12_col_only:
                    row.extend(["", ""])
                span_to_cells[span_key(t0, t1)] = row[1:]
                return sep.join(row)

            n1, n2, n3 = info["top_names"]
            c1, c2, c3 = info["top_confs"]
            id1 = product_map.get(n1, "") if n1 else ""
            id2 = product_map.get(n2, "") if n2 else ""
            id3 = product_map.get(n3, "") if n3 else ""
            for nm, pid in ((n1, id1), (n2, id2), (n3, id3)):
                if nm and not pid:
                    log(f"警告: 商品表无名称「{nm}」，产品编码置空。")

            row = [
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
            if not args.legacy_12_col_only:
                row.extend(["", ""])
            span_to_cells[span_key(t0, t1)] = row[1:]
            span_to_pairs[span_key(t0, t1)] = list(info.get("pairs") or [])
            return sep.join(row)

        try:
            for rank, (t0, t1, af_sc) in enumerate(segs, start=1):
                log(f"段落 rank={rank} [{t0:.3f},{t1:.3f}] score={af_sc:.4f} …")
                lines_out.append(infer_one(rank, t0, t1))

            if args.merge_adjacent_tear:
                log("撕膜门控：合并相邻同 top1 成功段…")
                if args.tear_model is None or not Path(args.tear_model).is_file():
                    log(f"缺少撕膜分类权重，跳过 tear_merge: {args.tear_model}")
                else:
                    tw_path = (args.tear_merge_weights or args.tear_model).resolve()
                    tear_gate_m = YOLO(str(tw_path))
                    tidx = tear_class_index(tear_gate_m, args.tear_merge_class)
                    merge_cfg = HandMergeConfig(
                        merge_iou_gt=args.merge_iou_gt,
                        merge_center_dist_max_px=args.merge_center_dist_max_px,
                        merge_center_dist_max_frac_diag=args.merge_center_dist_max_frac_diag,
                    )
                    grouper = HandRoiGrouper(
                        merge_cfg, pad_box_fn=_pad_box, pad_ratio=args.pad_ratio
                    )
                    body_lines = lines_out[1:]
                    e2e_rows = parse_e2e_rows_from_body_lines(body_lines)
                    mg_det = det if not args.tear_merge_full_frame else None
                    mg_grouper = grouper if not args.tear_merge_full_frame else None
                    merged_rows = merge_all(
                        e2e_rows,
                        cap,
                        tear_gate_m,
                        tidx,
                        head_sec=float(args.tear_merge_head_sec),
                        tear_prob=float(args.tear_merge_prob),
                        tear_min_frames=int(args.tear_merge_min_frames),
                        imgsz=int(args.imgsz_cls),
                        predict_kw=predict_kw,
                        verbose=bool(args.tear_merge_verbose),
                        det=mg_det,
                        grouper=mg_grouper,
                        imgsz_det=int(args.imgsz_det),
                        det_conf=float(args.det_conf),
                    )
                    lines_out = [header]
                    for j, er in enumerate(merged_rows, start=1):
                        sk = span_key(er.start_sec, er.end_sec)
                        if sk in span_to_cells:
                            lines_out.append(sep.join([str(j)] + span_to_cells[sk]))
                        else:
                            log(
                                f"[tear_merge] 合并窗段全量重推理 rank={j} "
                                f"[{er.start_sec:.3f},{er.end_sec:.3f}]"
                            )
                            lines_out.append(infer_one(j, er.start_sec, er.end_sec))

            if getattr(args, "gap_merge_enabled", False):
                log("相邻 gap 合并…")
                body_lines = lines_out[1:]
                e2e_rows = parse_e2e_rows_from_body_lines(body_lines)
                gap_merged = merge_all_by_gap(
                    e2e_rows,
                    span_to_pairs,
                    product_map,
                    max_gap_sec=float(args.gap_merge_max_gap_sec),
                    log_fn=log,
                )
                lines_out = [header]
                for er in gap_merged:
                    lines_out.append(er.to_line12(er.rank))
        finally:
            cap.release()

        log("医生识别：开始执行…")
        doctor_text = infer_doctor_text_offline(args, video_path)
        log(f"医生识别：{doctor_text}")
        lines_out.append(f"医生信息：{doctor_text}")

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
        log(f"已写出: {args.out.resolve()}")
        if work_dir_log is not None:
            log(f"工作目录: {work_dir_log}")

        return 0


class DebugPipelineManager(PipelineManager):
    """跳过 ActionFormer，用 Excel 时间段列作为段列表。"""

    def run(self) -> int:
        args = self.args
        video_path = args.video.resolve()
        if not video_path.is_file():
            log(f"找不到视频: {video_path}")
            return 1
        excel_path = args.excel.resolve()
        if not excel_path.is_file():
            log(f"找不到 Excel: {excel_path}")
            return 1

        log("[debug] 使用 Excel 时间段，跳过 ActionFormer")
        args.merge_adjacent_tear = False
        log("[debug] 跳过撕膜相邻段合并（merge_adjacent_tear=false）")

        allowed_names = _resolve_allowed_names(args, excel_path)
        if allowed_names is None:
            return 1
        if not _validate_phase2_weights(args, require_actionformer=False):
            return 1

        col_index = int(getattr(args, "excel_time_col_index", 8))
        segs = load_segments_from_excel_column_i(
            excel_path,
            col_index=col_index,
            video_path=video_path,
        )
        if not segs:
            log("Excel 未解析到任何有效时间段")
            return 1

        min_seg = float(getattr(args, "af_min_seg_seconds", 0.0))
        segs = _filter_segments_by_min_length(segs, min_seg)
        if not segs:
            log(f"最短段过滤（>={min_seg:g}s）后无剩余段")
            return 1

        product_map = e2e.load_product_code_map(excel_path)
        return self._run_phase2_and_write(
            segs,
            video_path=video_path,
            excel_path=excel_path,
            allowed_names=allowed_names,
            product_map=product_map,
        )


class BasketPipelineManager(PipelineManager):
    """跳过 ActionFormer：OpenCV 框选篮子 + 手篮接触上升沿 → 固定窗口段。"""

    def run(self) -> int:
        args = self.args
        video_path = args.video.resolve()
        if not video_path.is_file():
            log(f"找不到视频: {video_path}")
            return 1
        excel_path = args.excel.resolve()
        if not excel_path.is_file():
            log(f"找不到 Excel: {excel_path}")
            return 1

        log("[basket] 使用篮子接触分段，跳过 ActionFormer")
        args.merge_adjacent_tear = False
        log("[basket] 跳过撕膜相邻段合并（merge_adjacent_tear=false）")

        allowed_names = _resolve_allowed_names(args, excel_path)
        if allowed_names is None:
            return 1
        if not _validate_phase2_weights(args, require_actionformer=False):
            return 1

        save_json = getattr(args, "basket_save_roi_json", None)

        load_json = getattr(args, "basket_load_roi_json", None)
        segs, _roi = build_segments_from_basket(
            video_path,
            Path(args.hand_model),
            basket_roi_json=Path(load_json) if load_json is not None else None,
            save_roi_json=Path(save_json) if save_json else None,
            skip_roi_select=bool(getattr(args, "basket_skip_roi_select", False)),
            roi_frame=getattr(args, "basket_roi_frame", "middle"),
            roi_backend=str(getattr(args, "basket_roi_backend", "tkinter")),
            contact_iou_threshold=float(getattr(args, "basket_contact_iou_threshold", 0.05)),
            contact_iou_on=float(getattr(args, "basket_contact_iou_on", 0.08)),
            contact_iou_off=float(getattr(args, "basket_contact_iou_off", 0.03)),
            confirm_seconds=float(getattr(args, "basket_confirm_seconds", 0.4)),
            cooldown_seconds=float(getattr(args, "basket_cooldown_seconds", 5.0)),
            segment_start_offset_sec=float(getattr(args, "basket_segment_start_offset_sec", 1.0)),
            segment_end_offset_sec=float(getattr(args, "basket_segment_end_offset_sec", 5.0)),
            min_segment_sec=float(getattr(args, "basket_min_segment_sec", 4.0)),
            scan_frame_stride=int(getattr(args, "basket_scan_frame_stride", 1)),
            det_conf=float(getattr(args, "basket_det_conf", args.det_conf)),
            imgsz_det=int(args.imgsz_det),
            device=str(args.device),
            half=bool(args.half),
            args=args,
            pack_root=getattr(args, "pack_root", None),
            log_fn=log,
        )
        if not segs:
            log("未检测到任何手篮接触上升沿，退出")
            return 1

        product_map = e2e.load_product_code_map(excel_path)
        return self._run_phase2_and_write(
            segs,
            video_path=video_path,
            excel_path=excel_path,
            allowed_names=allowed_names,
            product_map=product_map,
        )


def run_pipeline(args: Namespace) -> int:
    return PipelineManager(args).run()


def run_debug_pipeline(args: Namespace) -> int:
    return DebugPipelineManager(args).run()


def run_basket_pipeline(args: Namespace) -> int:
    return BasketPipelineManager(args).run()
