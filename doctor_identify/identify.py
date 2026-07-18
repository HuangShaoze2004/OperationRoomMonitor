"""
Identify doctors in test videos by matching face embeddings against the gallery.

Matching strategy:
  1. Sample evenly-spaced frames across the entire test video
  2. Extract face embeddings from each frame
  3. Compute mean pooled embedding of all face frames
  4. For each face frame, find K nearest neighbors among ALL gallery embeddings
  5. Vote: each frame's K neighbors vote for their doctor
  6. Output identity with confidence level
"""

import json
import os
import time
from collections import Counter, defaultdict

import cv2
import numpy as np

from config import (
    TEST_DIR,
    GALLERY_DIR,
    GALLERY_INDEX_FILE,
    RESULTS_FILE,
    TEST_FRAME_COUNT,
    MATCH_THRESHOLD,
    UNKNOWN_THRESHOLD,
    TOP2_MARGIN,
    OUTPUT_DIR,
    KNN_K,
)
from extract_faces import get_app, get_face_embedding, cosine_similarity


def load_gallery() -> dict:
    """
    Load the gallery from disk.

    Returns:
        Dict mapping doctor_id -> {
            "name": str,
            "prototype": np.ndarray (512,),
            "embeddings": np.ndarray (N, 512),
            "count": int,
        }
    """
    if not os.path.exists(GALLERY_INDEX_FILE):
        raise FileNotFoundError(
            f"Gallery index not found at {GALLERY_INDEX_FILE}. "
            f"Run 'python build_gallery.py' first."
        )

    with open(GALLERY_INDEX_FILE, "r", encoding="utf-8") as f:
        index = json.load(f)

    gallery = {}
    for doctor_id, info in index.items():
        folder_path = os.path.join(GALLERY_DIR, info["folder"])
        prototype_path = os.path.join(folder_path, "prototype.npy")
        embeddings_path = os.path.join(folder_path, "embeddings.npy")

        if not os.path.exists(prototype_path):
            print(f"  [WARN] Missing prototype for {info['name']} ({doctor_id}), skipping")
            continue

        prototype = np.load(prototype_path)
        embeddings = np.load(embeddings_path) if os.path.exists(embeddings_path) else None

        gallery[doctor_id] = {
            "name": info["name"],
            "prototype": prototype,
            "embeddings": embeddings,
            "count": info["embedding_count"],
        }

    print(f"[INFO] Loaded gallery: {len(gallery)} doctors")
    for doctor_id, g in gallery.items():
        print(f"       {doctor_id} {g['name']}: {g['count']} embeddings")

    return gallery


def _build_knn_index(gallery: dict):
    """
    Build a flat embedding matrix and label array for fast KNN search.

    Returns:
        all_embeddings: np.ndarray of shape (total_N, 512)
        labels: list of (doctor_id, doctor_name) for each embedding
    """
    all_embs = []
    labels = []
    for doc_id, info in gallery.items():
        embs = info["embeddings"]
        if embs is not None and len(embs) > 0:
            all_embs.append(embs)
            labels.extend([(doc_id, info["name"])] * len(embs))

    if not all_embs:
        return None, []

    all_embs = np.concatenate(all_embs, axis=0)
    return all_embs, labels


def _knn_vote(query_emb: np.ndarray, all_embs: np.ndarray, labels: list, k: int = KNN_K) -> list:
    """
    Find K nearest neighbors for a query embedding and return their labels.

    Args:
        query_emb: (512,) L2-normalized query embedding
        all_embs: (N, 512) all gallery embeddings
        labels: list of (doctor_id, name) for each embedding
        k: number of neighbors

    Returns:
        list of (doctor_id, similarity) for the k nearest neighbors
    """
    # Cosine similarity = dot product for normalized vectors
    sims = np.dot(all_embs, query_emb)  # (N,)
    top_k_indices = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
    top_k_indices = top_k_indices[np.argsort(-sims[top_k_indices])]

    results = []
    for idx in top_k_indices:
        doc_id, _ = labels[idx]
        results.append((doc_id, float(sims[idx])))
    return results


def _identify_single_video(
    video_path: str,
    gallery: dict,
    app,
    num_frames: int = TEST_FRAME_COUNT,
) -> dict:
    """
    Identify the doctor in a single test video using KNN matching.

    Strategy:
      1. Sample frames, extract face embeddings
      2. Build KNN index from all gallery embeddings (once)
      3. Each test frame's embedding queries K nearest gallery embeddings
      4. Neighbors vote for their doctor (K votes per frame)
      5. Mean-pool all test face embeddings, also query KNN for final vote
    """
    video_name = os.path.basename(video_path)
    print(f"\n{'=' * 60}")
    print(f"[IDENTIFY] {video_name}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "video": video_name,
            "error": f"Failed to open video: {video_path}",
        }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Total frames: {total_frames}, FPS: {fps:.1f}")

    # Determine frame indices to sample (evenly spaced)
    if total_frames <= num_frames:
        sample_indices = list(range(total_frames))
    else:
        step = total_frames / num_frames
        sample_indices = [int(i * step) for i in range(num_frames)]
        sample_indices = sorted(set(sample_indices))

    print(f"  Sampling {len(sample_indices)} frames")

    # Build KNN index from ALL gallery embeddings
    all_embs, knn_labels = _build_knn_index(gallery)
    if all_embs is None:
        return {"video": video_name, "error": "Gallery is empty"}

    # Collect face embeddings and per-frame KNN votes
    face_embeddings = []
    knn_votes = []  # per-frame: list of (doctor_id, similarity) from KNN
    frames_with_faces = 0

    t_start = time.time()

    for sample_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, sample_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        embedding = get_face_embedding(frame, app)
        if embedding is None:
            continue

        frames_with_faces += 1
        face_embeddings.append(embedding)

        # KNN query for this frame
        neighbors = _knn_vote(embedding, all_embs, knn_labels, KNN_K)
        knn_votes.extend(neighbors)

    cap.release()

    elapsed = time.time() - t_start
    print(f"  Frames with faces: {frames_with_faces}/{len(sample_indices)}")
    print(f"  Time: {elapsed:.1f}s")

    if not face_embeddings:
        return {
            "video": video_name,
            "predicted_doctor_id": None,
            "predicted_doctor_name": "unknown",
            "mean_similarity": 0.0,
            "total_face_frames": 0,
            "vote_counts": {},
            "vote_percentages": {},
            "confidence": "no_face",
            "message": "No faces detected in any sampled frame",
        }

    # ================================================================
    # Method 1: KNN voting (each neighbor = 1 vote for its doctor)
    # ================================================================
    knn_counter = Counter(doc_id for doc_id, _ in knn_votes)
    knn_sim_sums = defaultdict(float)
    for doc_id, sim in knn_votes:
        knn_sim_sums[doc_id] += sim
    knn_total = len(knn_votes)

    # ================================================================
    # Method 2: Mean pooled embedding -> KNN
    # ================================================================
    all_face_embs = np.stack(face_embeddings, axis=0)
    mean_emb = all_face_embs.mean(axis=0)
    mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
    mean_neighbors = _knn_vote(mean_emb, all_embs, knn_labels, KNN_K * 3)
    mean_counter = Counter(doc_id for doc_id, _ in mean_neighbors)

    # ================================================================
    # Method 3: Classic prototype comparison (for reference)
    # ================================================================
    proto_sims = {}
    for doc_id, doc_info in gallery.items():
        proto_sims[doc_id] = float(np.dot(mean_emb, doc_info["prototype"]))

    # ================================================================
    # Combine KNN voting + mean-pool KNN for final decision
    # ================================================================
    # Weighted: KNN per-frame votes (60%) + mean-pooled KNN (40%)
    combined = defaultdict(float)
    for doc_id in set(list(knn_counter.keys()) + list(mean_counter.keys())):
        knn_score = knn_counter.get(doc_id, 0) / max(knn_total, 1)
        mean_score = mean_counter.get(doc_id, 0) / max(len(mean_neighbors), 1)
        combined[doc_id] = knn_score * 0.6 + mean_score * 0.4

    # Best doctor by combined score
    top_doc_id = max(combined, key=combined.get)
    top_name = gallery[top_doc_id]["name"]

    # Compute per-doctor mean KNN similarity
    knn_mean_sims = {}
    for doc_id in knn_counter:
        knn_mean_sims[doc_id] = round(knn_sim_sums[doc_id] / knn_counter[doc_id], 4)

    # Top-KNN-sim for the predicted doctor
    top_knn_sim = knn_mean_sims.get(top_doc_id, 0.0)

    # Overall prototype similarity
    overall_sim = proto_sims.get(top_doc_id, 0.0)

    # ================================================================
    # Confidence assessment
    # ================================================================
    if len(combined) >= 2:
        sorted_docs = sorted(combined, key=combined.get, reverse=True)
        second_id = sorted_docs[1]
        top2_margin = combined[top_doc_id] - combined[second_id]
    else:
        top2_margin = 1.0

    confidence = "high"
    if top_knn_sim < UNKNOWN_THRESHOLD:
        confidence = "unknown"
    elif top_knn_sim < MATCH_THRESHOLD:
        confidence = "low"

    if top2_margin < TOP2_MARGIN and confidence == "high":
        confidence = "low"

    # ================================================================
    # Build result
    # ================================================================
    knn_percentages = {
        gallery[k]["name"]: round(c / knn_total * 100, 1)
        for k, c in knn_counter.most_common(8)
    }

    result = {
        "video": video_name,
        "predicted_doctor_id": top_doc_id,
        "predicted_doctor_name": top_name,
        "knn_similarity": round(top_knn_sim, 4),
        "overall_similarity": round(overall_sim, 4),
        "total_face_frames": frames_with_faces,
        "total_knn_votes": knn_total,
        "vote_counts": {gallery[k]["name"]: v for k, v in knn_counter.most_common(8)},
        "vote_percentages": knn_percentages,
        "mean_similarities": {gallery[k]["name"]: knn_mean_sims[k] for k, _ in knn_counter.most_common()},
        "proto_similarities": {gallery[k]["name"]: round(v, 4) for k, v in sorted(proto_sims.items(), key=lambda x: -x[1])},
        "top2_margin": round(top2_margin, 4),
        "confidence": confidence,
    }

    return result


def identify_test_videos(gallery: dict = None, app=None, test_dir: str = None,
                         video_paths: list[str] = None) -> list[dict]:
    """
    Identify doctors in test videos.

    Args:
        gallery: Loaded gallery dict (auto-loaded if None)
        app: InsightFace app (auto-initialized if None)
        test_dir: Directory to scan for .mp4 files (default: TEST_DIR)
        video_paths: Explicit list of video file paths (overrides test_dir)

    Returns:
        List of result dicts, one per video
    """
    if gallery is None:
        gallery = load_gallery()
    if app is None:
        app = get_app()

    if video_paths is not None:
        # Use explicit video paths
        video_files = [os.path.abspath(p) for p in video_paths if os.path.exists(p)]
        if not video_files:
            print("[WARN] None of the specified video paths exist")
            return []
    elif test_dir is not None:
        video_files = [
            os.path.join(test_dir, f) for f in sorted(os.listdir(test_dir))
            if f.lower().endswith(".mp4")
        ]
    else:
        video_files = [
            os.path.join(TEST_DIR, f) for f in sorted(os.listdir(TEST_DIR))
            if f.lower().endswith(".mp4")
        ]

    if not video_files:
        print("[WARN] No .mp4 files found to identify")
        return []

    results = []
    for video_path in video_files:
        result = _identify_single_video(video_path, gallery, app)
        results.append(result)

    return results


def save_results(results: list[dict], output_path: str = None):
    """Save identification results to JSON."""
    if output_path is None:
        output_path = RESULTS_FILE
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] Results saved to {output_path}")


def print_results(results: list[dict]):
    """Print identification results in a readable format."""
    print(f"\n{'=' * 60}")
    print("IDENTIFICATION RESULTS")
    print(f"{'=' * 60}")

    for r in results:
        if "error" in r:
            print(f"\n  {r['video']}: ERROR - {r['error']}")
            continue

        conf_emoji = {"high": "✅", "low": "⚠️", "unknown": "❓", "no_face": "🚫"}
        emoji = conf_emoji.get(r["confidence"], "❓")

        print(f"\n  {emoji} {r['video']}")
        print(f"    Predicted: {r['predicted_doctor_name']} (ID: {r['predicted_doctor_id']})")
        print(f"    Confidence: {r['confidence']}")
        print(f"    KNN similarity: {r['knn_similarity']:.4f}")
        print(f"    Overall similarity: {r['overall_similarity']:.4f}")
        print(f"    Top-2 margin: {r['top2_margin']:.4f}")
        print(f"    Face frames: {r['total_face_frames']}")

        if r.get("vote_percentages"):
            print(f"    KNN vote distribution:")
            for name, pct in r["vote_percentages"].items():
                sim = r.get("mean_similarities", {}).get(name, 0)
                bar = "█" * int(pct / 5)
                print(f"      {name}: {pct:5.1f}% {bar} (sim={sim:.4f})")

        if r.get("proto_similarities"):
            print(f"    Prototype similarities:")
            for name, sim in list(r["proto_similarities"].items())[:5]:
                print(f"      {name}: {sim:.4f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Identify doctors in test videos using the face gallery",
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Identify a single video file (instead of scanning test directory)",
    )
    parser.add_argument(
        "--test-dir", type=str, default=None,
        help=f"Directory with test videos (default: {TEST_DIR})",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help=f"Output JSON file (default: {RESULTS_FILE})",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    gallery = load_gallery()
    app = get_app()

    if args.video:
        # Single video mode
        video_path = os.path.abspath(args.video)
        if not os.path.exists(video_path):
            print(f"[ERROR] Video not found: {video_path}")
            return
        print(f"[INFO] Identifying single video: {os.path.basename(video_path)}")
        results = identify_test_videos(gallery, app, video_paths=[video_path])
    else:
        results = identify_test_videos(gallery, app, test_dir=args.test_dir)

    output_path = args.output or RESULTS_FILE
    save_results(results, output_path)
    print_results(results)


if __name__ == "__main__":
    main()
