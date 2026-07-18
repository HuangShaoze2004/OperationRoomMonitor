"""
交叉验证评估 - 评估医生身份识别系统的准确性。

评估策略:
  1. Leave-One-Video-Out (LOVO): 每次留 1 个视频测试，其余建库
  2. Cross-Color: 蓝色手术服建库→绿色测试，反之亦然

使用示例:
  python evaluate.py --dataset data/people_datasets
  python evaluate.py --dataset data/people_datasets --method lovo
  python evaluate.py --dataset data/people_datasets --method cross-color
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np

from config import (
    PEOPLE_DATASETS_DIR,
    OUTPUT_DIR,
    MIN_QUALITY_SCORE,
    MIN_SHARPNESS,
    MIN_FRONTALITY,
)
from extract_faces import get_app, get_best_face_with_quality

# 复用 build_gallery 的工具函数
from build_gallery import (
    _compute_prototype,
    _parse_flat_dir,
)


# ============================================================
# 预提取：一次性提取所有视频的人脸嵌入
# ============================================================

def _extract_all_videos(doctor_videos: dict[str, list[str]], app) -> dict:
    """
    预提取所有视频的人脸嵌入向量。

    Args:
        doctor_videos: {name: [video_path, ...]}
        app: InsightFace app

    Returns:
        {
            name: {
                "videos": {
                    filename: {
                        "path": str,
                        "embeddings": np.ndarray (N, 512),
                        "color": "蓝" | "绿" | "unknown",
                    }
                },
                "all_embeddings": np.ndarray,
            }
        }
    """
    import cv2

    all_data = {}
    total_videos = sum(len(v) for v in doctor_videos.values())
    processed = 0

    print(f"\n[预提取] 共 {total_videos} 个视频...")

    for name, video_paths in sorted(doctor_videos.items()):
        print(f"\n  医生: {name} ({len(video_paths)} 个视频)")

        doctor_data = {"videos": {}, "all_embeddings": []}

        for vp in sorted(video_paths):
            filename = os.path.basename(vp)
            processed += 1

            # 解析颜色
            color_match = re.search(r"\(([^)]+)\)", filename)
            color = color_match.group(1) if color_match else "unknown"

            print(f"    [{processed}/{total_videos}] {filename} ...", end=" ", flush=True)

            cap = cv2.VideoCapture(vp)
            if not cap.isOpened():
                print("打开失败")
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            sample_interval = max(1, int(fps / 5))

            embeddings = []
            faces_found = 0
            for frame_idx in range(0, total_frames, sample_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                result = get_best_face_with_quality(frame, app)
                if result is None:
                    continue

                embedding, bbox, quality = result
                if (quality["overall"] >= MIN_QUALITY_SCORE
                        and quality["sharpness"] >= MIN_SHARPNESS
                        and quality["frontality"] >= MIN_FRONTALITY):
                    embeddings.append(embedding)
                    faces_found += 1

            cap.release()

            emb_array = np.stack(embeddings, axis=0) if embeddings else np.array([])
            doctor_data["videos"][filename] = {
                "path": vp,
                "embeddings": emb_array,
                "color": color,
            }

            if len(emb_array) > 0:
                doctor_data["all_embeddings"].append(emb_array)

            print(f"{faces_found} 张人脸")

        # 合并该医生所有视频的嵌入
        if doctor_data["all_embeddings"]:
            doctor_data["all_embeddings"] = np.concatenate(doctor_data["all_embeddings"], axis=0)
        else:
            doctor_data["all_embeddings"] = np.array([])

        all_data[name] = doctor_data

    return all_data


# ============================================================
# 临时图库 & 识别
# ============================================================

def _build_temp_gallery(all_data: dict, exclude_video: str = None,
                        exclude_doctor: str = None,
                        only_color: str = None) -> dict:
    """
    从预提取数据构建临时内存图库。

    Args:
        all_data: 预提取的所有数据
        exclude_video: 排除的视频文件名（LOVO 模式）
        exclude_doctor: 排除整个医生（通常不用）
        only_color: 仅使用指定颜色（cross-color 模式）

    Returns:
        gallery dict: {doctor_name: {"prototype": np.ndarray, "embeddings": np.ndarray, "name": str}}
    """
    gallery = {}

    for name, data in all_data.items():
        if exclude_doctor and name == exclude_doctor:
            continue

        emb_list = []
        for filename, vinfo in data["videos"].items():
            # 跳过被排除的视频
            if exclude_video and filename == exclude_video:
                continue
            # 颜色过滤
            if only_color and vinfo["color"] != only_color:
                continue
            # 添加嵌入向量
            if len(vinfo["embeddings"]) > 0:
                emb_list.append(vinfo["embeddings"])

        if not emb_list:
            continue

        all_embs = np.concatenate(emb_list, axis=0)
        prototype = _compute_prototype(all_embs)

        gallery[name] = {
            "name": name,
            "prototype": prototype,
            "embeddings": all_embs,
            "count": len(all_embs),
        }

    return gallery


def _identify_with_gallery(video_embeddings: np.ndarray, gallery: dict) -> dict:
    """
    使用内存图库识别一段视频中的人脸。

    Args:
        video_embeddings: (N, 512) 测试视频的人脸嵌入
        gallery: 临时图库

    Returns:
        识别结果 dict
    """
    if len(video_embeddings) == 0:
        return {
            "predicted_name": "no_face",
            "confidence": "no_face",
            "top_similarity": 0.0,
            "vote_percentages": {},
            "proto_similarities": {},
        }

    # 构建 KNN 索引
    all_embs_list = []
    labels = []
    for name, info in gallery.items():
        embs = info["embeddings"]
        if len(embs) > 0:
            all_embs_list.append(embs)
            labels.extend([name] * len(embs))

    if not all_embs_list:
        return {"predicted_name": "unknown", "confidence": "unknown",
                "top_similarity": 0.0, "vote_percentages": {}, "proto_similarities": {}}

    all_embs = np.concatenate(all_embs_list, axis=0)

    # KNN 投票
    K = 5
    knn_votes = []
    for emb in video_embeddings:
        sims = np.dot(all_embs, emb)
        top_k = np.argpartition(-sims, min(K, len(sims) - 1))[:K]
        for idx in top_k:
            knn_votes.append(labels[idx])

    from collections import Counter
    vote_counter = Counter(knn_votes)
    total_votes = len(knn_votes)

    # Mean pooled KNN
    mean_emb = video_embeddings.mean(axis=0)
    mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
    mean_sims = np.dot(all_embs, mean_emb)
    top_k_mean = np.argpartition(-mean_sims, min(K * 3, len(mean_sims) - 1))[:K * 3]
    mean_counter = Counter(labels[idx] for idx in top_k_mean)

    # 组合投票 (60% frame-level + 40% mean-pooled)
    combined = defaultdict(float)
    for name in set(list(vote_counter.keys()) + list(mean_counter.keys())):
        frame_score = vote_counter.get(name, 0) / max(total_votes, 1)
        mean_score = mean_counter.get(name, 0) / max(K * 3, 1)
        combined[name] = frame_score * 0.6 + mean_score * 0.4

    if not combined:
        return {"predicted_name": "unknown", "confidence": "unknown",
                "top_similarity": 0.0, "vote_percentages": {}, "proto_similarities": {}}

    top_name = max(combined, key=combined.get)

    # 置信度
    sorted_names = sorted(combined, key=combined.get, reverse=True)
    top2_margin = combined[top_name] - (combined[sorted_names[1]] if len(sorted_names) > 1 else 0)

    top_sim = float(np.dot(gallery[top_name]["prototype"], mean_emb))

    confidence = "high"
    if top_sim < 0.3:
        confidence = "unknown"
    elif top_sim < 0.4:
        confidence = "low"
    if top2_margin < 0.05 and confidence == "high":
        confidence = "low"

    vote_pcts = {
        name: round(c / total_votes * 100, 1)
        for name, c in vote_counter.most_common(8)
    }

    proto_sims = {
        name: round(float(np.dot(mean_emb, info["prototype"])), 4)
        for name, info in sorted(gallery.items(), key=lambda x: -np.dot(mean_emb, x[1]["prototype"]))
    }

    return {
        "predicted_name": top_name,
        "confidence": confidence,
        "top_similarity": round(top_sim, 4),
        "vote_percentages": vote_pcts,
        "proto_similarities": proto_sims,
        "top2_margin": round(top2_margin, 4),
    }


# ============================================================
# 评估方法
# ============================================================

def evaluate_lovo(all_data: dict) -> dict:
    """
    Leave-One-Video-Out 交叉验证。

    每个视频轮流作为测试集，其余视频构建图库。
    """
    print("\n" + "=" * 60)
    print("  Leave-One-Video-Out 交叉验证")
    print("=" * 60)

    results = []
    correct = 0
    total = 0

    # 收集所有视频
    all_videos = []
    for name, data in all_data.items():
        for filename in data["videos"]:
            all_videos.append((name, filename))

    print(f"\n共 {len(all_videos)} 个测试样本\n")

    for i, (true_name, test_filename) in enumerate(all_videos):
        print(f"[{i + 1}/{len(all_videos)}] 测试: {test_filename} (真实: {true_name})")

        # 构建图库（排除当前视频）
        gallery = _build_temp_gallery(all_data, exclude_video=test_filename)

        if true_name not in gallery:
            print(f"  [警告] 真实医生 '{true_name}' 不在图库中（可能只有1个视频？）")
            # 使用该医生其他视频的嵌入测试
            test_embs = all_data[true_name]["videos"][test_filename]["embeddings"]
        else:
            test_embs = all_data[true_name]["videos"][test_filename]["embeddings"]

        if len(test_embs) == 0:
            print(f"  [跳过] 视频中没有人脸")
            results.append({
                "test_video": test_filename,
                "true_name": true_name,
                "predicted_name": "no_face",
                "correct": False,
                "confidence": "no_face",
                "face_frames": 0,
            })
            total += 1
            continue

        # 识别
        pred = _identify_with_gallery(test_embs, gallery)
        is_correct = pred["predicted_name"] == true_name

        if is_correct:
            correct += 1
            print(f"  ✅ 预测: {pred['predicted_name']} (正确) 置信度: {pred['confidence']}"
                  f"  相似度: {pred['top_similarity']:.4f}")
        else:
            print(f"  ❌ 预测: {pred['predicted_name']} (错误) 置信度: {pred['confidence']}"
                  f"  相似度: {pred['top_similarity']:.4f}")

        total += 1
        results.append({
            "test_video": test_filename,
            "true_name": true_name,
            "predicted_name": pred["predicted_name"],
            "correct": is_correct,
            "confidence": pred["confidence"],
            "top_similarity": pred["top_similarity"],
            "vote_percentages": pred["vote_percentages"],
            "face_frames": len(test_embs),
        })

    accuracy = correct / max(total, 1) * 100

    return {
        "method": "lovo",
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 1),
        "results": results,
    }


def evaluate_cross_color(all_data: dict) -> dict:
    """
    跨颜色交叉验证。

    - 蓝色手术服建库 → 绿色测试
    - 绿色手术服建库 → 蓝色测试
    """
    print("\n" + "=" * 60)
    print("  跨颜色交叉验证 (Cross-Color)")
    print("=" * 60)

    colors = ["蓝", "绿"]
    all_results = []
    total_correct = 0
    total_samples = 0

    for train_color, test_color in [("蓝", "绿"), ("绿", "蓝")]:
        print(f"\n{'=' * 40}")
        print(f"  训练: {train_color}色手术服  →  测试: {test_color}色手术服")
        print(f"{'=' * 40}")

        # 构建图库
        gallery = _build_temp_gallery(all_data, only_color=train_color)
        print(f"图库: {len(gallery)} 位医生")

        correct = 0
        samples = 0
        fold_results = []

        for name, data in sorted(all_data.items()):
            # 找测试颜色的视频
            test_videos = {
                fn: vinfo for fn, vinfo in data["videos"].items()
                if vinfo["color"] == test_color
            }

            if not test_videos:
                # 如果没有指定颜色的视频，跳过
                continue

            if name not in gallery:
                print(f"  [跳过] '{name}' 不在图库中（训练集中没有 {train_color} 色视频）")
                continue

            for filename, vinfo in test_videos.items():
                test_embs = vinfo["embeddings"]
                if len(test_embs) == 0:
                    print(f"  [跳过] {filename}: 没有人脸")
                    continue

                pred = _identify_with_gallery(test_embs, gallery)
                is_correct = pred["predicted_name"] == name

                if is_correct:
                    correct += 1
                    print(f"  ✅ {filename}: 预测={pred['predicted_name']} "
                          f"({pred['confidence']})")
                else:
                    print(f"  ❌ {filename}: 预测={pred['predicted_name']} "
                          f"实际={name} ({pred['confidence']})")

                samples += 1
                fold_results.append({
                    "test_video": filename,
                    "true_name": name,
                    "predicted_name": pred["predicted_name"],
                    "correct": is_correct,
                    "confidence": pred["confidence"],
                    "top_similarity": pred["top_similarity"],
                    "vote_percentages": pred["vote_percentages"],
                    "face_frames": len(test_embs),
                })

        acc = correct / max(samples, 1) * 100
        print(f"\n  {train_color}→{test_color} 准确率: {correct}/{samples} = {acc:.1f}%")
        total_correct += correct
        total_samples += samples
        all_results.extend(fold_results)

    overall_acc = total_correct / max(total_samples, 1) * 100

    return {
        "method": "cross_color",
        "total": total_samples,
        "correct": total_correct,
        "accuracy": round(overall_acc, 1),
        "results": all_results,
    }


# ============================================================
# 输出
# ============================================================

def print_evaluation_report(eval_result: dict):
    """打印评估报告。"""
    print("\n" + "=" * 60)
    print("  评 估 报 告")
    print("=" * 60)

    method = eval_result["method"]
    total = eval_result["total"]
    correct = eval_result["correct"]
    accuracy = eval_result["accuracy"]
    results = eval_result["results"]

    print(f"\n方法: {method}")
    print(f"准确率: {correct}/{total} = {accuracy}%")

    # 每位医生的准确率
    print(f"\n{'=' * 40}")
    print("  按医生统计")
    print(f"{'=' * 40}")

    doctor_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        name = r["true_name"]
        doctor_stats[name]["total"] += 1
        if r["correct"]:
            doctor_stats[name]["correct"] += 1

    for name in sorted(doctor_stats.keys()):
        stats = doctor_stats[name]
        acc = stats["correct"] / max(stats["total"], 1) * 100
        emoji = "✅" if acc >= 80 else "⚠️" if acc >= 50 else "❌"
        print(f"  {emoji} {name}: {stats['correct']}/{stats['total']} = {acc:.0f}%")

    # 混淆矩阵
    if method == "lovo" and len(doctor_stats) > 1:
        print(f"\n{'=' * 40}")
        print("  混淆矩阵")
        print(f"{'=' * 40}")

        all_names = sorted(doctor_stats.keys())
        confusion = defaultdict(lambda: defaultdict(int))
        for r in results:
            if r["confidence"] != "no_face":
                confusion[r["true_name"]][r["predicted_name"]] += 1

        # 表头
        header = f"{'':>8}"
        for name in all_names:
            header += f"  {name:>6}"
        print(header)
        print("-" * len(header))

        for true_name in all_names:
            row = f"{true_name:>8}"
            for pred_name in all_names:
                count = confusion[true_name][pred_name]
                row += f"  {count:>6}"
            print(row)

    # 详细结果
    print(f"\n{'=' * 40}")
    print("  详细结果")
    print(f"{'=' * 40}")

    for i, r in enumerate(results):
        status = "✅" if r["correct"] else "❌"
        print(f"\n  [{i + 1}] {status} {r['test_video']}")
        print(f"      真实: {r['true_name']}  →  预测: {r['predicted_name']}")
        print(f"      置信度: {r['confidence']}  人脸帧数: {r.get('face_frames', '?')}")
        if r.get("vote_percentages"):
            top3 = list(r["vote_percentages"].items())[:3]
            print(f"      投票: {', '.join(f'{n}:{p}%' for n, p in top3)}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="医生身份识别 - 交叉验证评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python evaluate.py --dataset data/people_datasets
  python evaluate.py --dataset data/people_datasets --method lovo
  python evaluate.py --dataset data/people_datasets --method cross-color
  python evaluate.py --dataset data/people_datasets --method both
        """,
    )
    parser.add_argument(
        "--dataset", type=str, default=PEOPLE_DATASETS_DIR,
        help=f"数据集目录（默认: {PEOPLE_DATASETS_DIR}）",
    )
    parser.add_argument(
        "--method", type=str, default="lovo",
        choices=["lovo", "cross-color", "both"],
        help="评估方法: lovo (留一视频), cross-color (跨颜色), both (两种都做)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="保存详细结果的 JSON 文件路径",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dataset):
        print(f"[错误] 数据集目录不存在: {args.dataset}")
        sys.exit(1)

    # 解析数据集
    print("=" * 60)
    print("  医生身份识别 - 评估")
    print("=" * 60)
    print(f"\n数据集: {args.dataset}")

    doctor_videos = _parse_flat_dir(args.dataset)
    if not doctor_videos:
        print("[错误] 数据集中未找到有效视频")
        sys.exit(1)

    print(f"找到 {len(doctor_videos)} 位医生, "
          f"共 {sum(len(v) for v in doctor_videos.values())} 个视频")

    # 加载模型
    print("\n加载 InsightFace 模型...")
    app = get_app()

    # 预提取所有人脸嵌入
    t_start = time.time()
    all_data = _extract_all_videos(doctor_videos, app)
    elapsed = time.time() - t_start
    print(f"\n预提取完成，耗时 {elapsed:.1f}s")

    # 统计
    total_embeddings = sum(
        len(data["all_embeddings"]) for data in all_data.values()
    )
    print(f"共提取 {total_embeddings} 个人脸嵌入向量")

    # 运行评估
    all_eval_results = []

    if args.method in ("lovo", "both"):
        lovo_result = evaluate_lovo(all_data)
        all_eval_results.append(lovo_result)
        print_evaluation_report(lovo_result)

    if args.method in ("cross-color", "both"):
        cc_result = evaluate_cross_color(all_data)
        all_eval_results.append(cc_result)
        print_evaluation_report(cc_result)

    # 保存
    if args.output:
        os.makedirs(os.path.dirname(args.output) or OUTPUT_DIR, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_eval_results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {args.output}")

    # 汇总
    print(f"\n{'=' * 60}")
    print("  评 估 汇 总")
    print(f"{'=' * 60}")
    for er in all_eval_results:
        print(f"  {er['method']}: {er['accuracy']}% ({er['correct']}/{er['total']})")


if __name__ == "__main__":
    main()
