from __future__ import annotations

import time
from argparse import Namespace
from pathlib import Path


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def resolve_allowed_class_idx(
    args: Namespace,
    excel_path: Path,
    cls_names: dict,
) -> frozenset[int] | None:
    """None 表示不裁剪类别（全类参与投票）。"""
    if not getattr(args, "use_whitelist", True):
        return None
    import run_haocai_actionformer_consumables_e2e as e2e

    if args.whitelist_json is not None:
        wpath = Path(args.whitelist_json)
        if not wpath.is_file():
            raise FileNotFoundError(f"找不到白名单 JSON: {wpath}")
        allowed_names = e2e.load_whitelist_json(wpath.resolve())
    else:
        allowed_names = load_allowed_names_from_excel(excel_path)
    return e2e.allowed_indices_from_json_names(allowed_names, cls_names)


def load_allowed_names_from_excel(excel_path: Path) -> list[str]:
    import pandas as pd

    df = pd.read_excel(excel_path, sheet_name=0, header=0)
    if df.shape[1] < 3:
        raise ValueError(f"Excel 至少需要 C 列（第 3 列）: {excel_path}")
    col = df.iloc[:, 2]
    names: list[str] = []
    seen: set[str] = set()
    for raw in col:
        if pd.isna(raw):
            continue
        s = str(raw).strip()
        if not s or s == "商品名称":
            continue
        if s not in seen:
            seen.add(s)
            names.append(s)
    return names
