"""医生人脸识别服务：基于 InsightFace 的医生身份识别（推流段内）。

封装 doctor_identify 模块，提供推流友好的帧级接口：
  - 加载 InsightFace 模型 + 人脸图库（一次性初始化）
  - identify_from_frames() 对若干帧运行人脸识别 → KNN 投票 → 医生 ID
  - 返回格式兼容 vote_doctor_from_segment_results()
"""
from __future__ import annotations

import sys
from argparse import Namespace
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# ---- 将 doctor_identify 加入 sys.path，使其内部相对 import 正常工作 ----------
_DOCTOR_IDENTIFY_DIR = Path(__file__).resolve().parent.parent / "doctor_identify"
if str(_DOCTOR_IDENTIFY_DIR) not in sys.path:
    sys.path.insert(0, str(_DOCTOR_IDENTIFY_DIR))


class DoctorFaceIdentityService:
    """加载 InsightFace 模型 + 人脸图库，提供基于帧的医生识别。"""

    def __init__(self, args: Namespace) -> None:
        self.args = args

        # 延迟 import —— doctor_identify/config.py 在 import 时会预加载 CUDA 库
        from doctor_identify.config import (
            MATCH_THRESHOLD,
            UNKNOWN_THRESHOLD,
            TOP2_MARGIN,
            KNN_K,
            GALLERY_INDEX_FILE,
        )
        from doctor_identify.extract_faces import get_app, get_face_embedding
        from doctor_identify.identify import (
            load_gallery,
            _build_knn_index,
            _knn_vote,
        )

        self._match_threshold = MATCH_THRESHOLD
        self._unknown_threshold = UNKNOWN_THRESHOLD
        self._top2_margin = TOP2_MARGIN
        self._knn_k = KNN_K

        # 初始化 InsightFace（全局单例）
        self._app = get_app()

        # 加载图库
        if not Path(GALLERY_INDEX_FILE).is_file():
            raise FileNotFoundError(
                f"人脸图库索引不存在: {GALLERY_INDEX_FILE}。"
                f"请先运行 doctor_identify/build_gallery.py 构建图库。"
            )
        self._gallery = load_gallery()
        self._all_embs, self._knn_labels = _build_knn_index(self._gallery)
        if self._all_embs is None:
            raise RuntimeError("图库为空，请先构建图库。")

        # 缓存函数引用
        self._get_face_embedding = get_face_embedding
        self._knn_vote = _knn_vote
        self._gallery_index_file = GALLERY_INDEX_FILE

    # ------------------------------------------------------------------
    # 核心识别
    # ------------------------------------------------------------------

    def identify_from_frames(self, frames: list[np.ndarray]) -> dict[str, Any]:
        """对若干帧运行人脸识别，返回结果 dict。

        Args:
            frames: BGR 图像列表 (np.ndarray, HxWx3)

        Returns:
            {"ok": True/False, "doctor_id": ..., "doctor_name": ...,
             "doctor_conf": ..., "low_confidence": bool}
            失败时 ok=False，附带 reason。
        """
        if not frames:
            return {"ok": False, "reason": "医生人脸窗口无帧"}

        # 每帧提取人脸 embedding
        face_embeddings: list[np.ndarray] = []
        for frame in frames:
            emb = self._get_face_embedding(frame, self._app)
            if emb is not None:
                face_embeddings.append(emb)

        if not face_embeddings:
            return {"ok": False, "reason": "医生人脸窗口未检测到人脸"}

        # KNN 投票：每帧的 embedding 查询 K 个最近邻
        knn_votes: list[tuple[str, float]] = []
        knn_sim_sums: dict[str, float] = defaultdict(float)
        for emb in face_embeddings:
            neighbors = self._knn_vote(
                emb, self._all_embs, self._knn_labels, self._knn_k
            )
            knn_votes.extend(neighbors)
            for doc_id, sim in neighbors:
                knn_sim_sums[doc_id] += sim

        if not knn_votes:
            return {"ok": False, "reason": "医生人脸 KNN 投票无结果"}

        # 统计投票
        knn_counter = Counter(doc_id for doc_id, _ in knn_votes)

        # Mean-pooled embedding → KNN (辅助)
        all_embs_arr = np.stack(face_embeddings, axis=0)
        mean_emb = all_embs_arr.mean(axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
        mean_neighbors = self._knn_vote(
            mean_emb, self._all_embs, self._knn_labels, self._knn_k * 3
        )
        mean_counter = Counter(doc_id for doc_id, _ in mean_neighbors)

        # 综合得分：KNN per-frame (60%) + mean-pool KNN (40%)
        combined: dict[str, float] = defaultdict(float)
        knn_total = len(knn_votes)
        mean_total = len(mean_neighbors)
        all_ids = set(list(knn_counter.keys()) + list(mean_counter.keys()))
        for doc_id in all_ids:
            knn_score = knn_counter.get(doc_id, 0) / max(knn_total, 1)
            mean_score = mean_counter.get(doc_id, 0) / max(mean_total, 1)
            combined[doc_id] = knn_score * 0.6 + mean_score * 0.4

        # 最佳医生
        top_doc_id = max(combined, key=combined.get)
        top_name = self._gallery[top_doc_id]["name"]

        # 计算 top-KNN 平均相似度
        knn_mean_sim = (
            knn_sim_sums[top_doc_id] / knn_counter[top_doc_id]
            if knn_counter[top_doc_id] > 0
            else 0.0
        )

        # 置信度
        confidence = "high"
        if knn_mean_sim < self._unknown_threshold:
            confidence = "unknown"
        elif knn_mean_sim < self._match_threshold:
            confidence = "low"

        # Top-2 margin
        if len(combined) >= 2:
            sorted_docs = sorted(combined, key=combined.get, reverse=True)
            top2_margin = combined[top_doc_id] - combined[sorted_docs[1]]
        else:
            top2_margin = 1.0

        if top2_margin < self._top2_margin and confidence == "high":
            confidence = "low"

        low_confidence = confidence in ("low", "unknown")

        return {
            "ok": True,
            "doctor_id": str(top_doc_id),
            "doctor_name": top_name,
            "doctor_conf": round(float(knn_mean_sim), 4),
            "low_confidence": low_confidence,
        }

    # ------------------------------------------------------------------
    # 资源释放
    # ------------------------------------------------------------------

    def close(self) -> None:
        """释放资源（InsightFace 无显式 close，保留接口兼容）。"""
        self._app = None
        self._gallery = None
        self._all_embs = None
        self._knn_labels = None
