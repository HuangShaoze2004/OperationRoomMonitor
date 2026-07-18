#!/usr/bin/env python3
"""
医生人脸识别流水线 - 端到端自动化。

用法:
    python run_pipeline.py                         # 完整流程: 构建图库 + 识别
    python run_pipeline.py --skip-build            # 跳过构建，仅识别
    python run_pipeline.py --build-only            # 仅构建图库
    python run_pipeline.py --reindex               # 重建索引
    python run_pipeline.py --dataset data/people_datasets   # 指定数据集
    python run_pipeline.py --evaluate              # 构建后自动评估
"""

import argparse
import os
import sys
import time

import numpy as np

from config import OUTPUT_DIR, PEOPLE_DATASETS_DIR, TEST_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="医生人脸识别流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python run_pipeline.py                              # 完整流程
    python run_pipeline.py --skip-build                 # 仅识别（使用已有图库）
    python run_pipeline.py --build-only                 # 仅构建图库
    python run_pipeline.py --reindex                    # 重建索引
    python run_pipeline.py --dataset data/people_datasets --evaluate
        """,
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="跳过图库构建，使用已有图库",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="仅构建图库，跳过识别",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="从已有文件夹重建图库索引（不重新提取人脸）",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=f"数据集目录（默认: {PEOPLE_DATASETS_DIR}）",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="构建图库后运行交叉验证评估",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="构建前清空已有图库",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  医生人脸识别流水线")
    print("=" * 60)

    t_total_start = time.time()
    dataset_dir = args.dataset or PEOPLE_DATASETS_DIR

    # ================================================================
    # Step 1: 构建图库
    # ================================================================
    if args.reindex:
        print("\n[步骤 1] 从已有文件夹重建图库索引...")
        from build_gallery import reindex
        reindex()
        print("\n[DONE] 图库索引已重建。")
        return

    if not args.skip_build:
        print(f"\n[步骤 1] 从数据集构建医生人脸图库...")
        print(f"数据集: {dataset_dir}")
        print("-" * 40)

        from extract_faces import get_app
        from build_gallery import build_gallery_from_flat_dataset

        app = get_app()

        # 如果 dataset 目录不存在且是默认目录，尝试 fallback
        if not os.path.isdir(dataset_dir):
            if args.dataset:
                print(f"[ERROR] 数据集目录不存在: {dataset_dir}")
                sys.exit(1)
            else:
                print(f"[WARN] 数据集目录不存在: {dataset_dir}")
                print(f"      尝试使用传统模式构建图库...")
                from build_gallery import build_gallery_legacy
                build_gallery_legacy(app)
        else:
            t_start = time.time()
            build_gallery_from_flat_dataset(dataset_dir, app, clear_existing=args.clear)
            elapsed = time.time() - t_start
            print(f"\n[DONE] 图库构建完成，耗时 {elapsed:.1f}s ({elapsed / 60:.1f} min)")

        if args.build_only:
            print("\n[DONE] 仅构建模式，流水线完成。")
            return
    else:
        print("\n[步骤 1] 跳过图库构建 (--skip-build)")

    # ================================================================
    # Step 2: 图库质量检查
    # ================================================================
    print(f"\n[步骤 2] 图库质量检查...")
    print("-" * 40)

    from identify import load_gallery
    gallery = load_gallery()

    # 类内相似度检查
    print("\n  类内相似度 (每位医生内部):")
    for doctor_id, info in gallery.items():
        embeddings = info.get("embeddings")
        if embeddings is not None and len(embeddings) >= 2:
            n = min(len(embeddings), 100)
            sample = embeddings[:n]
            sample = sample / (np.linalg.norm(sample, axis=1, keepdims=True) + 1e-8)
            sim_matrix = np.dot(sample, sample.T)
            mask = ~np.eye(n, dtype=bool)
            pairwise_sims = sim_matrix[mask]
            min_sim = pairwise_sims.min()
            mean_sim = pairwise_sims.mean()
            status = "✅" if min_sim >= 0.3 else "⚠️" if min_sim >= 0.2 else "❌"
            print(f"    {status} {info['name']} ({doctor_id}): "
                  f"min={min_sim:.3f}, mean={mean_sim:.3f}, count={len(embeddings)}")
        else:
            print(f"    ⚠️  {info['name']} ({doctor_id}): "
                  f"仅 {len(embeddings) if embeddings is not None else 0} 个嵌入向量")

    # 类间分离度检查
    print("\n  类间分离度 (不同医生原型之间):")
    proto_ids = list(gallery.keys())
    for i in range(len(proto_ids)):
        for j in range(i + 1, len(proto_ids)):
            id_a, id_b = proto_ids[i], proto_ids[j]
            sim = float(np.dot(gallery[id_a]["prototype"], gallery[id_b]["prototype"]))
            status = "✅" if sim < 0.3 else "⚠️" if sim < 0.4 else "❌"
            print(f"    {status} {gallery[id_a]['name']} vs {gallery[id_b]['name']}: sim={sim:.4f}")

    # ================================================================
    # Step 3: 识别测试视频
    # ================================================================
    print(f"\n[步骤 3] 识别测试视频...")
    print("-" * 40)

    from extract_faces import get_app
    from identify import identify_test_videos, save_results, print_results

    app = get_app()

    # 识别 test_video 中的视频（如果存在）
    if os.path.isdir(TEST_DIR):
        t_start = time.time()
        results = identify_test_videos(gallery, app)
        elapsed = time.time() - t_start
        print(f"\n[DONE] 识别完成，耗时 {elapsed:.1f}s")
        save_results(results)
        print_results(results)
    else:
        print(f"\n[INFO] test_video 目录不存在，跳过识别。")
        print(f"  使用 'python identify.py --video <路径>' 进行单视频识别。")
        results = []

    # ================================================================
    # Step 4: 评估（可选）
    # ================================================================
    if args.evaluate:
        print(f"\n[步骤 4] 交叉验证评估...")
        print("-" * 40)

        if os.path.isdir(dataset_dir):
            from evaluate import main as evaluate_main
            import subprocess
            subprocess.run([
                sys.executable, "-m", "evaluate",
                "--dataset", dataset_dir,
                "--method", "lovo",
            ])
        else:
            print(f"[WARN] 数据集目录不存在，跳过评估: {dataset_dir}")

    # ================================================================
    # 汇总
    # ================================================================
    t_total = time.time() - t_total_start
    print(f"\n{'=' * 60}")
    print(f"  流水线完成，总耗时 {t_total:.1f}s ({t_total / 60:.1f} min)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
