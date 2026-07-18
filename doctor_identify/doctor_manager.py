"""
医生库管理模块 - 医生身份注册、删除、查询。

功能:
  - build: 从 people_datasets 扁平目录构建医生库
  - add:   添加单个医生
  - remove: 删除医生（按姓名或ID）
  - list:  列出所有已注册医生

使用示例:
  python doctor_manager.py build --dataset data/people_datasets
  python doctor_manager.py list
  python doctor_manager.py add --name "新医生" --videos a.mp4 b.mp4
  python doctor_manager.py remove --name "新医生"
  python doctor_manager.py remove --id 25003
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

from config import (
    PEOPLE_DATASETS_DIR,
    GALLERY_DIR,
    GALLERY_INDEX_FILE,
    DOCTOR_REGISTRY_FILE,
    OUTPUT_DIR,
    NEXT_DOCTOR_ID,
    FACE_SAMPLES_PER_DOCTOR,
    SAVE_FACE_SAMPLES,
    FACES_DIR,
    MIN_QUALITY_SCORE,
    MIN_SHARPNESS,
    MIN_FRONTALITY,
)
from extract_faces import get_app, get_best_face_with_quality


# ============================================================
# 工具函数
# ============================================================

def _ensure_dirs():
    """创建必要的输出目录。"""
    os.makedirs(GALLERY_DIR, exist_ok=True)
    os.makedirs(FACES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _get_gallery_doctor_dir(doctor_id: str, name: str) -> str:
    """获取医生在图库中的文件夹路径。"""
    return os.path.join(GALLERY_DIR, f"{doctor_id}_{name}")


def _compute_prototype(embeddings: np.ndarray) -> np.ndarray:
    """计算 L2 归一化的平均嵌入向量（原型）。"""
    mean = embeddings.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm > 0:
        mean = mean / norm
    return mean


def _process_single_doctor_video(
    video_path: str,
    doctor_id: str,
    doctor_name: str,
    app,
) -> list[np.ndarray]:
    """
    处理单个医生的完整视频，提取所有人脸嵌入向量。

    对整段视频进行均匀采样（~5 fps），无需时间分段标注。
    """
    video_basename = os.path.basename(video_path)
    print(f"\n  [处理] {video_basename}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    [错误] 无法打开视频: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # ~5 fps 均匀采样
    sample_interval = max(1, int(fps / 5))
    num_samples = total_frames // sample_interval

    duration = total_frames / max(fps, 1)
    print(f"    时长: {duration:.0f}s, {total_frames} 帧 @ {fps:.1f} fps")
    print(f"    采样间隔 {sample_interval} 帧 -> 约 {num_samples} 帧")

    embeddings = []
    processed = 0
    faces_found = 0
    t_start = time.time()

    for frame_idx in range(0, total_frames, sample_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        processed += 1
        result = get_best_face_with_quality(frame, app)
        if result is None:
            continue

        embedding, bbox, quality = result

        # 质量过滤
        if (quality["overall"] >= MIN_QUALITY_SCORE
                and quality["sharpness"] >= MIN_SHARPNESS
                and quality["frontality"] >= MIN_FRONTALITY):
            embeddings.append(embedding)
            faces_found += 1

        # 每 5 秒输出进度
        if processed % 25 == 0 and processed > 0:
            progress = frame_idx / max(total_frames, 1) * 100
            print(f"    ... {progress:.0f}% ({faces_found} 张高质量人脸)")

    cap.release()

    elapsed = time.time() - t_start
    accept_rate = faces_found / max(processed, 1) * 100
    print(f"    [完成] {faces_found} 张高质量人脸 / {processed} 帧 "
          f"({accept_rate:.0f}%) 耗时 {elapsed:.1f}s")

    return embeddings


def _save_doctor_gallery(doctor_id: str, name: str, embeddings_list: list[np.ndarray]) -> dict | None:
    """
    保存单个医生的嵌入向量和元数据到图库文件夹。
    如果已有数据，会合并而不是覆盖。
    """
    doctor_dir = _get_gallery_doctor_dir(doctor_id, name)
    os.makedirs(doctor_dir, exist_ok=True)

    # 加载已有嵌入向量（如果存在）
    existing_path = os.path.join(doctor_dir, "embeddings.npy")
    if os.path.exists(existing_path):
        existing = np.load(existing_path)
        if len(embeddings_list) > 0:
            new_embs = np.stack(embeddings_list, axis=0)
            embeddings = np.concatenate([existing, new_embs], axis=0)
        else:
            embeddings = existing
    else:
        if len(embeddings_list) == 0:
            print(f"  [警告] 医生 {name} ({doctor_id}) 没有提取到任何嵌入向量且无已有数据！")
            return None
        embeddings = np.stack(embeddings_list, axis=0)

    prototype = _compute_prototype(embeddings)

    np.save(os.path.join(doctor_dir, "embeddings.npy"), embeddings)
    np.save(os.path.join(doctor_dir, "prototype.npy"), prototype)

    meta = {
        "id": doctor_id,
        "name": name,
        "count": int(len(embeddings)),
        "embedding_dim": int(embeddings.shape[1]),
    }
    with open(os.path.join(doctor_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "name": name,
        "folder": f"{doctor_id}_{name}",
        "embedding_count": int(len(embeddings)),
    }


# ============================================================
# DoctorManager 类
# ============================================================

class DoctorManager:
    """
    医生库管理器。

    管理医生的注册（姓名 → ID 映射）和图库（人脸嵌入向量）。
    注册表持久化到 doctor_registry.json，确保 ID 稳定。
    """

    def __init__(self):
        _ensure_dirs()
        self._app = None
        self.registry = self._load_registry()

    # ---- InsightFace 懒加载 ----

    def _get_app(self):
        """懒加载 InsightFace 模型。"""
        if self._app is None:
            self._app = get_app()
        return self._app

    # ---- 注册表管理 ----

    def _load_registry(self) -> dict:
        """
        加载医生注册表。
        返回: {name: doctor_id} 映射
        """
        if os.path.exists(DOCTOR_REGISTRY_FILE):
            with open(DOCTOR_REGISTRY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 注册表格式: {"25001": "伏林晗", ...}
            # 反转为 {name: id}
            return {name: doc_id for doc_id, name in data.items()}
        return {}

    def _save_registry(self):
        """保存医生注册表。"""
        # 反转为 {id: name} 格式存储
        data = {doc_id: name for name, doc_id in self.registry.items()}
        with open(DOCTOR_REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _get_next_id(self) -> str:
        """分配下一个可用的医生 ID。"""
        if not self.registry:
            return str(NEXT_DOCTOR_ID)
        existing_ids = [int(doc_id) for doc_id in self.registry.values()]
        next_id = max(max(existing_ids) + 1, NEXT_DOCTOR_ID)
        return str(next_id)

    def _get_or_create_id(self, name: str) -> str:
        """
        获取医生的已有 ID，或分配新 ID。

        如果医生已在注册表中，返回已有 ID；
        否则分配新 ID 并写入注册表。
        """
        if name in self.registry:
            return self.registry[name]

        doc_id = self._get_next_id()
        self.registry[name] = doc_id
        self._save_registry()
        return doc_id

    # ---- 数据集解析 ----

    def _parse_dataset_dir(self, dataset_dir: str) -> dict[str, list[str]]:
        """
        扫描扁平目录，按医生姓名分组视频文件。

        文件名格式: {姓名}({颜色}).mp4  如 付玉峰(蓝).mp4
        返回: {name: [video_path, ...]}
        """
        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(f"数据集目录不存在: {dataset_dir}")

        # 匹配 "姓名(任意字符).mp4"
        pattern = re.compile(r"^(.+?)\([^)]+\)\.mp4$")

        doctor_videos = defaultdict(list)
        for filename in sorted(os.listdir(dataset_dir)):
            if not filename.lower().endswith(".mp4"):
                continue
            match = pattern.match(filename)
            if match:
                name = match.group(1)
                full_path = os.path.join(dataset_dir, filename)
                doctor_videos[name].append(full_path)
            else:
                print(f"  [跳过] 无法解析文件名: {filename}")

        return dict(doctor_videos)

    # ---- 核心操作 ----

    def build_from_dataset(
        self,
        dataset_dir: str = None,
        clear_existing: bool = False,
    ) -> dict:
        """
        从扁平数据集目录构建医生图库。

        Args:
            dataset_dir: 数据集目录路径，默认使用 PEOPLE_DATASETS_DIR
            clear_existing: 是否清空已有图库（重新构建）

        Returns:
            gallery_index 字典
        """
        if dataset_dir is None:
            dataset_dir = PEOPLE_DATASETS_DIR

        print("=" * 60)
        print("  医生库构建 - 从数据集")
        print("=" * 60)
        print(f"\n数据集目录: {dataset_dir}")

        if clear_existing:
            self._clear_gallery()
            self.registry = {}
            self._save_registry()

        # 解析数据集
        doctor_videos = self._parse_dataset_dir(dataset_dir)

        if not doctor_videos:
            print("\n[错误] 数据集中未找到视频文件")
            return {}

        print(f"\n找到 {len(doctor_videos)} 位医生:")
        for name, videos in doctor_videos.items():
            doc_id = self._get_or_create_id(name)
            print(f"  {doc_id} {name}: {len(videos)} 个视频")
            for v in videos:
                print(f"    - {os.path.basename(v)}")

        # 加载模型
        app = self._get_app()

        # 处理每位医生
        gallery_index = {}
        total_start = time.time()

        for name, video_paths in sorted(doctor_videos.items()):
            doc_id = self._get_or_create_id(name)

            print(f"\n{'=' * 40}")
            print(f"  医生: {name} (ID: {doc_id})")
            print(f"{'=' * 40}")

            all_embeddings = []
            for video_path in sorted(video_paths):
                embs = _process_single_doctor_video(
                    video_path, doc_id, name, app
                )
                all_embeddings.extend(embs)

            result = _save_doctor_gallery(doc_id, name, all_embeddings)
            if result:
                gallery_index[doc_id] = result
                print(f"  [保存] {name} ({doc_id}): {result['embedding_count']} 个嵌入向量")
            else:
                print(f"  [警告] {name} ({doc_id}): 未提取到任何嵌入向量")

        # 保存全局索引
        with open(GALLERY_INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(gallery_index, f, ensure_ascii=False, indent=2)
        print(f"\n图库索引已保存: {GALLERY_INDEX_FILE}")

        elapsed = time.time() - total_start
        print(f"\n[DONE] 图库构建完成，共 {len(gallery_index)} 位医生，耗时 {elapsed:.1f}s")

        return gallery_index

    def add_doctor(self, name: str, video_paths: list[str]) -> dict | None:
        """
        添加一位新医生到图库。

        Args:
            name: 医生姓名
            video_paths: 视频文件路径列表（每个视频只包含该医生）

        Returns:
            医生信息字典，失败返回 None
        """
        print("=" * 60)
        print(f"  添加医生: {name}")
        print("=" * 60)

        # 检查是否已存在
        if name in self.registry:
            existing_id = self.registry[name]
            print(f"\n[注意] 医生 '{name}' 已存在 (ID: {existing_id})")
            print(f"  将合并新的嵌入向量到已有数据中...")

        # 验证视频文件
        valid_videos = []
        for vp in video_paths:
            if not os.path.exists(vp):
                print(f"  [警告] 视频文件不存在，跳过: {vp}")
            else:
                valid_videos.append(vp)

        if not valid_videos:
            print("[错误] 没有有效的视频文件")
            return None

        print(f"视频文件 ({len(valid_videos)} 个):")
        for vp in valid_videos:
            print(f"  - {os.path.basename(vp)}")

        # 分配/获取 ID
        doc_id = self._get_or_create_id(name)
        print(f"医生 ID: {doc_id}")

        # 提取人脸
        app = self._get_app()
        all_embeddings = []
        for vp in valid_videos:
            embs = _process_single_doctor_video(vp, doc_id, name, app)
            all_embeddings.extend(embs)

        # 保存
        result = _save_doctor_gallery(doc_id, name, all_embeddings)
        if result is None:
            print(f"\n[失败] 未能提取到任何人脸嵌入向量")
            return None

        # 更新全局索引
        gallery_index = self._load_gallery_index()
        gallery_index[doc_id] = result
        self._save_gallery_index(gallery_index)

        print(f"\n[成功] 医生 '{name}' (ID: {doc_id}) 已添加，共 {result['embedding_count']} 个嵌入向量")
        return result

    def remove_doctor(self, identifier: str) -> bool:
        """
        删除一位医生。

        Args:
            identifier: 医生姓名或 ID

        Returns:
            是否成功删除
        """
        # 解析 identifier -> doc_id
        doc_id, name = self._resolve_identifier(identifier)
        if doc_id is None:
            print(f"[错误] 未找到医生: {identifier}")
            return False

        print("=" * 60)
        print(f"  删除医生: {name} (ID: {doc_id})")
        print("=" * 60)

        # 删除图库文件夹
        doctor_dir = _get_gallery_doctor_dir(doc_id, name)
        if os.path.exists(doctor_dir):
            shutil.rmtree(doctor_dir)
            print(f"  已删除图库文件夹: {doctor_dir}")
        else:
            print(f"  [注意] 图库文件夹不存在: {doctor_dir}")

        # 从注册表中删除
        if name in self.registry:
            del self.registry[name]
            self._save_registry()
            print(f"  已从注册表中删除")

        # 从图库索引中删除
        gallery_index = self._load_gallery_index()
        if doc_id in gallery_index:
            del gallery_index[doc_id]
            self._save_gallery_index(gallery_index)
            print(f"  已从图库索引中删除")

        print(f"\n[成功] 医生 '{name}' 已删除")
        return True

    def list_doctors(self) -> list[dict]:
        """
        列出所有已注册医生及其统计信息。

        Returns:
            医生信息列表，每项包含 id, name, embedding_count
        """
        gallery_index = self._load_gallery_index()

        if not gallery_index:
            print("医生库为空。请先运行 'build' 命令。")
            return []

        print(f"\n{'=' * 60}")
        print(f"  医生库 - 共 {len(gallery_index)} 位医生")
        print(f"{'=' * 60}")
        print(f"\n{'ID':<8} {'姓名':<10} {'嵌入向量数':<12}")
        print("-" * 30)

        doctors = []
        for doc_id in sorted(gallery_index.keys()):
            info = gallery_index[doc_id]
            name = info.get("name", "?")
            count = info.get("embedding_count", 0)
            print(f"{doc_id:<8} {name:<10} {count:<12}")
            doctors.append({
                "id": doc_id,
                "name": name,
                "embedding_count": count,
            })

        print("-" * 30)
        total_embeddings = sum(d["embedding_count"] for d in doctors)
        print(f"合计: {len(doctors)} 位医生, {total_embeddings} 个嵌入向量\n")

        return doctors

    # ---- 辅助方法 ----

    def _clear_gallery(self):
        """清空整个图库（包括所有医生文件夹和索引）。"""
        if os.path.exists(GALLERY_DIR):
            for item in os.listdir(GALLERY_DIR):
                item_path = os.path.join(GALLERY_DIR, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"  已删除: {item_path}")
                elif item.endswith(".json"):
                    os.remove(item_path)
            print("图库已清空。")

    def _resolve_identifier(self, identifier: str) -> tuple[str | None, str | None]:
        """
        解析标识符（姓名或 ID），返回 (doc_id, name)。
        找不到时返回 (None, None)。
        """
        # 尝试作为 ID 查找
        gallery_index = self._load_gallery_index()
        if identifier in gallery_index:
            name = gallery_index[identifier].get("name", identifier)
            return identifier, name

        # 尝试作为姓名查找（先在 registry 找，再在 gallery_index 找）
        if identifier in self.registry:
            doc_id = self.registry[identifier]
            return doc_id, identifier

        # 在 gallery_index 中按姓名搜索
        for doc_id, info in gallery_index.items():
            if info.get("name") == identifier:
                return doc_id, identifier

        return None, None

    def _load_gallery_index(self) -> dict:
        """加载图库索引。"""
        if os.path.exists(GALLERY_INDEX_FILE):
            with open(GALLERY_INDEX_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_gallery_index(self, index: dict):
        """保存图库索引。"""
        with open(GALLERY_INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="医生库管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
命令示例:
  python doctor_manager.py build --dataset data/people_datasets
  python doctor_manager.py build --dataset data/people_datasets --clear
  python doctor_manager.py list
  python doctor_manager.py add --name "新医生" --videos a.mp4 b.mp4
  python doctor_manager.py remove --name "旧医生"
  python doctor_manager.py remove --id 25003
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="操作命令")

    # build 命令
    build_parser = subparsers.add_parser("build", help="从数据集构建医生库")
    build_parser.add_argument(
        "--dataset", type=str, default=None,
        help=f"数据集目录路径（默认: {PEOPLE_DATASETS_DIR}）",
    )
    build_parser.add_argument(
        "--clear", action="store_true",
        help="构建前清空已有图库",
    )

    # list 命令
    subparsers.add_parser("list", help="列出所有已注册医生")

    # add 命令
    add_parser = subparsers.add_parser("add", help="添加一位医生")
    add_parser.add_argument("--name", type=str, required=True, help="医生姓名")
    add_parser.add_argument("--videos", type=str, nargs="+", required=True,
                            help="视频文件路径（可多个）")

    # remove 命令
    remove_parser = subparsers.add_parser("remove", help="删除一位医生")
    remove_parser.add_argument("--name", type=str, default=None, help="医生姓名")
    remove_parser.add_argument("--id", type=str, default=None, help="医生 ID")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    manager = DoctorManager()

    if args.command == "build":
        manager.build_from_dataset(
            dataset_dir=args.dataset,
            clear_existing=args.clear,
        )
    elif args.command == "list":
        manager.list_doctors()
    elif args.command == "add":
        manager.add_doctor(args.name, args.videos)
    elif args.command == "remove":
        identifier = args.id or args.name
        if identifier is None:
            print("[错误] 请指定 --name 或 --id")
            sys.exit(1)
        manager.remove_doctor(identifier)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
