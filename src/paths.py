"""pack/5.11：将 vendor code 根目录加入 sys.path（顺序与 main_pipeline 一致）。"""
from __future__ import annotations

import sys
from pathlib import Path


def ensure_code_on_path(pack_root: Path) -> Path:
    """
    pack_root: pack/5.11 根目录。
    返回 CODE_ROOT（即 pack_root / 'code'）。
    """
    code = (pack_root / "code").resolve()
    if not (code / "repo_root.py").is_file():
        raise FileNotFoundError(f"缺少 vendor code 根: {code}")

    scripts = code / "video_clip_cls" / "scripts"
    infer = code / "video_clip_cls" / "infer_single_0506"
    for p in (infer, scripts, code):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    return code
