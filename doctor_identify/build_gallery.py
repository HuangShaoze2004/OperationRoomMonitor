"""
Build the doctor face gallery from training videos.

Two modes are supported:

Mode 1 — Excel + video (legacy):
  Uses the original April 27/28 training videos with Excel time-segment
  annotations. Videos are read sequentially due to IMKH container constraints.

Mode 2 — Single-doctor videos (NEW):
  Scans each doctor's folder under output/gallery/ for .mp4 files.
  Each video is treated as containing ONLY that doctor.
  The entire video is sampled to extract face embeddings.

  Expected directory structure:
      output/gallery/
      ├── 24503_付玉峰/
      │   ├── 付玉峰_no_mask.mp4     ← dropped here by operator
      │   ├── 付玉峰_mask.mp4
      │   ├── embeddings.npy
      │   └── meta.json
      ├── 24504_李树华/
      │   ├── 李树华_no_mask.mp4
      │   └── ...

Usage:
    python build_gallery.py                          # Legacy mode (Excel + training videos)
    python build_gallery.py --single-doctor-videos   # Scan gallery folders for per-doctor videos
    python build_gallery.py --reindex                # Rebuild index from existing embeddings only
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

from config import (
    VIDEO_0427,
    EXCEL_0427,
    VIDEO_0428,
    EXCEL_0428,
    GALLERY_DIR,
    GALLERY_INDEX_FILE,
    FACES_DIR,
    FRAME_SAMPLE_INTERVAL,
    VIDEO_FPS,
    SAVE_FACE_SAMPLES,
    FACE_SAMPLES_PER_DOCTOR,
    OUTPUT_DIR,
    MIN_QUALITY_SCORE,
    MIN_SHARPNESS,
    MIN_FRONTALITY,
    DET_SIZE,
)
from parse_excel import parse_excel
from extract_faces import get_app, get_best_face_with_quality, cosine_similarity


# ============================================================
# Shared utilities
# ============================================================

def _ensure_dirs():
    """Create output directories."""
    os.makedirs(GALLERY_DIR, exist_ok=True)
    os.makedirs(FACES_DIR, exist_ok=True)


def _get_gallery_doctor_dir(doctor_id: str, name: str) -> str:
    """Get the gallery folder path for a doctor."""
    return os.path.join(GALLERY_DIR, f"{doctor_id}_{name}")


def _compute_prototype(embeddings: np.ndarray) -> np.ndarray:
    """Compute the L2-normalized mean of a set of embeddings."""
    mean = embeddings.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm > 0:
        mean = mean / norm
    return mean


def _save_doctor_gallery(doctor_id: str, name: str, embeddings_list: list[np.ndarray]):
    """
    Save a single doctor's embeddings and metadata to their gallery folder.
    Merges with existing embeddings if the folder already has an embeddings.npy.
    """
    doctor_dir = _get_gallery_doctor_dir(doctor_id, name)
    os.makedirs(doctor_dir, exist_ok=True)

    # Load existing embeddings if present
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
            print(f"  [WARN] Doctor {name} ({doctor_id}) has zero embeddings and no existing data!")
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
# Mode 1: Legacy sequential video + Excel processing
# ============================================================

def _process_video_sequential(
    video_path: str,
    all_segments: list[dict],
    app,
    video_label: str,
) -> dict[str, list[np.ndarray]]:
    """
    Process a single training video in sequential (non-seeking) mode.
    """
    flat_segments = []
    for doc in all_segments:
        for seg_start, seg_end in doc["segments"]:
            flat_segments.append({
                "start_frame": seg_start,
                "end_frame": seg_end,
                "doctor_id": doc["doctor_id"],
                "name": doc["name"],
            })

    flat_segments.sort(key=lambda s: s["start_frame"])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"\n[INFO] Processing '{video_label}'")
    print(f"       Total frames: {total_frames}, FPS: {fps:.1f}")
    print(f"       Segments to extract: {len(flat_segments)}")

    gallery_data = defaultdict(list)
    face_samples_saved = defaultdict(int)

    seg_idx = 0
    frame_idx = 0
    processed_frames = 0
    faces_found = 0

    t_start = time.time()
    last_log_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        while seg_idx < len(flat_segments) and frame_idx > flat_segments[seg_idx]["end_frame"]:
            seg = flat_segments[seg_idx]
            doc_id = seg["doctor_id"]
            count = len(gallery_data.get(doc_id, []))
            print(f"  [OK] {seg['name']} ({doc_id}): segment ended, "
                  f"{count} faces extracted so far")
            seg_idx += 1

        if seg_idx < len(flat_segments):
            seg = flat_segments[seg_idx]
            if seg["start_frame"] <= frame_idx <= seg["end_frame"]:
                if frame_idx % FRAME_SAMPLE_INTERVAL == 0:
                    processed_frames += 1
                    result = get_best_face_with_quality(frame, app)
                    if result is not None:
                        embedding, bbox, quality = result
                        if (quality["overall"] >= MIN_QUALITY_SCORE
                                and quality["sharpness"] >= MIN_SHARPNESS
                                and quality["frontality"] >= MIN_FRONTALITY):
                            doc_id = seg["doctor_id"]
                            gallery_data[doc_id].append(embedding)
                            faces_found += 1

                            if (SAVE_FACE_SAMPLES
                                    and face_samples_saved[doc_id] < FACE_SAMPLES_PER_DOCTOR):
                                x1, y1, x2, y2 = bbox
                                margin = int(0.2 * (y2 - y1))
                                x1c = max(0, x1 - margin)
                                y1c = max(0, y1 - margin)
                                x2c = min(frame.shape[1], x2 + margin)
                                y2c = min(frame.shape[0], y2 + margin)
                                face_crop = frame[y1c:y2c, x1c:x2c]
                                doctor_face_dir = os.path.join(FACES_DIR, f"{doc_id}_{seg['name']}")
                                os.makedirs(doctor_face_dir, exist_ok=True)
                                cv2.imwrite(
                                    os.path.join(doctor_face_dir, f"frame_{frame_idx:06d}.jpg"),
                                    face_crop,
                                )
                                face_samples_saved[doc_id] += 1

        if frame_idx - last_log_frame >= 500:
            elapsed = time.time() - t_start
            fps_proc = frame_idx / max(elapsed, 0.001)
            print(f"  ... frame {frame_idx}/{total_frames} ({fps_proc:.0f} fps), "
                  f"{faces_found} faces extracted")
            last_log_frame = frame_idx

        frame_idx += 1

    cap.release()

    elapsed = time.time() - t_start
    print(f"  [DONE] {frame_idx} frames read in {elapsed:.1f}s "
          f"({frame_idx / max(elapsed, 0.001):.0f} fps)")
    print(f"         {processed_frames} frames processed, {faces_found} faces extracted")

    return dict(gallery_data)


def build_gallery_legacy(app=None) -> dict:
    """Build gallery from the original April 27/28 training videos."""
    if app is None:
        app = get_app()

    _ensure_dirs()

    all_doctors = {}
    for excel_path in [EXCEL_0427, EXCEL_0428]:
        doctors = parse_excel(excel_path)
        for doc in doctors:
            all_doctors[doc["doctor_id"]] = doc["name"]

    for doctor_id, name in all_doctors.items():
        doctor_dir = _get_gallery_doctor_dir(doctor_id, name)
        os.makedirs(doctor_dir, exist_ok=True)

    video_configs = [
        (VIDEO_0427, EXCEL_0427, "April 27"),
        (VIDEO_0428, EXCEL_0428, "April 28"),
    ]

    aggregated = defaultdict(list)

    for video_path, excel_path, label in video_configs:
        doctors = parse_excel(excel_path)
        video_embeddings = _process_video_sequential(video_path, doctors, app, label)
        for doc_id, embs in video_embeddings.items():
            aggregated[doc_id].extend(embs)

    index = {}
    for doctor_id, embeddings_list in sorted(aggregated.items()):
        name = all_doctors.get(doctor_id, f"Unknown_{doctor_id}")
        result = _save_doctor_gallery(doctor_id, name, embeddings_list)
        if result:
            index[doctor_id] = result
            print(f"  [SAVE] {name} ({doctor_id}): {result['embedding_count']} total embeddings")

    with open(GALLERY_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] Gallery index saved to {GALLERY_INDEX_FILE}")

    return index


# ============================================================
# Mode 2: Single-doctor video processing (NEW)
# ============================================================

def _process_single_doctor_video(
    video_path: str,
    doctor_id: str,
    doctor_name: str,
    app,
) -> list[np.ndarray]:
    """
    Process an entire video, extracting face embeddings for a single doctor.

    The ENTIRE video is sampled — no time segments needed.
    Uses EVENLY SPACED frames across the full duration.
    """
    video_basename = os.path.basename(video_path)
    print(f"\n  [PROCESS] {video_basename}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    [ERROR] Failed to open: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / max(fps, 1)

    # Sample at ~5 fps across the whole video
    sample_interval = max(1, int(fps / 5))
    num_samples = total_frames // sample_interval

    print(f"    Duration: {duration:.0f}s, {total_frames} frames @ {fps:.1f} fps")
    print(f"    Sampling every {sample_interval} frames -> ~{num_samples} frames")

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

        # Quality filtering
        if (quality["overall"] >= MIN_QUALITY_SCORE
                and quality["sharpness"] >= MIN_SHARPNESS
                and quality["frontality"] >= MIN_FRONTALITY):
            embeddings.append(embedding)
            faces_found += 1

        # Progress every 5 seconds
        if processed % 25 == 0 and processed > 0:
            elapsed = time.time() - t_start
            progress = frame_idx / max(total_frames, 1) * 100
            print(f"    ... {progress:.0f}% ({faces_found} quality faces)")

    cap.release()

    elapsed = time.time() - t_start
    accept_rate = faces_found / max(processed, 1) * 100
    print(f"    [DONE] {faces_found} quality faces from {processed} frames "
          f"({accept_rate:.0f}%) in {elapsed:.1f}s")

    return embeddings


def build_gallery_from_single_videos(app=None) -> dict:
    """
    Scan each doctor's folder under output/gallery/ for .mp4 files.

    Each .mp4 is treated as a single-doctor video — the entire video is
    sampled to extract face embeddings and assigned to that doctor.
    """
    if app is None:
        app = get_app()

    _ensure_dirs()

    # Scan for doctor folders (matching pattern: {doctor_id}_{name}/)
    doctor_dirs = []
    for item in sorted(os.listdir(GALLERY_DIR)):
        item_path = os.path.join(GALLERY_DIR, item)
        if not os.path.isdir(item_path):
            continue
        # Parse doctor_id and name from folder name
        parts = item.split("_", 1)
        if len(parts) != 2:
            continue
        doctor_id = parts[0]
        doctor_name = parts[1]

        # Find .mp4 files in this folder
        videos = sorted([
            f for f in os.listdir(item_path)
            if f.lower().endswith(".mp4") and not f.startswith(".")
        ])
        if videos:
            doctor_dirs.append((doctor_id, doctor_name, item_path, videos))

    if not doctor_dirs:
        print("[ERROR] No doctor folders with .mp4 videos found under output/gallery/")
        print("        Expected structure: output/gallery/{doctor_id}_{name}/*.mp4")
        return {}

    print(f"\n[INFO] Found {len(doctor_dirs)} doctor(s) with videos:")
    for doctor_id, doctor_name, folder, videos in doctor_dirs:
        print(f"       {doctor_id} {doctor_name}: {len(videos)} video(s) in {folder}")

    # Process each doctor's videos
    index = {}
    for doctor_id, doctor_name, folder_path, videos in doctor_dirs:
        all_embeddings = []
        for video_file in videos:
            video_path = os.path.join(folder_path, video_file)
            embs = _process_single_doctor_video(
                video_path, doctor_id, doctor_name, app
            )
            all_embeddings.extend(embs)

        result = _save_doctor_gallery(doctor_id, doctor_name, all_embeddings)
        if result:
            index[doctor_id] = result
            print(f"  [SAVE] {doctor_name} ({doctor_id}): {result['embedding_count']} total embeddings")
        else:
            print(f"  [WARN] {doctor_name} ({doctor_id}): NO embeddings extracted")

    # Save global index
    with open(GALLERY_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] Gallery index saved to {GALLERY_INDEX_FILE}")

    return index


# ============================================================
# Reindex
# ============================================================

def reindex() -> dict:
    """Rebuild gallery_index.json from existing gallery folders."""
    _ensure_dirs()

    index = {}
    for folder_name in sorted(os.listdir(GALLERY_DIR)):
        folder_path = os.path.join(GALLERY_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        meta_path = os.path.join(folder_path, "meta.json")
        if not os.path.exists(meta_path):
            print(f"  [SKIP] {folder_name}: no meta.json found")
            continue

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        doctor_id = meta["id"]
        name = meta["name"]
        embedding_path = os.path.join(folder_path, "embeddings.npy")

        count = 0
        if os.path.exists(embedding_path):
            embeddings = np.load(embedding_path)
            count = len(embeddings)
            prototype = _compute_prototype(embeddings)
            np.save(os.path.join(folder_path, "prototype.npy"), prototype)
            meta["count"] = count
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        index[doctor_id] = {
            "name": name,
            "folder": folder_name,
            "embedding_count": count,
        }
        print(f"  [INDEX] {name} ({doctor_id}): {count} embeddings")

    with open(GALLERY_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] Gallery index rebuilt: {len(index)} doctors")

    return index


# ============================================================
# Gateway function
# ============================================================

def build_gallery(mode: str = "legacy", app=None, **kwargs):
    """
    Gallery building gateway — dispatches to the correct mode.

    Modes:
      - "legacy": Excel + sequential video (April 27/28)
      - "single-doctor-videos": per-doctor folders under output/gallery/
      - "flat-dataset": flat directory, filename={Name}({Color}).mp4
    """
    if app is None:
        app = get_app()

    if mode == "legacy":
        return build_gallery_legacy(app)
    elif mode == "single-doctor-videos":
        return build_gallery_from_single_videos(app)
    elif mode == "flat-dataset":
        dataset_dir = kwargs.get("dataset_dir")
        if not dataset_dir:
            raise ValueError("flat-dataset mode requires --dataset-dir")
        return build_gallery_from_flat_dataset(dataset_dir, app)
    else:
        raise ValueError(f"Unknown build mode: {mode}")


# ============================================================
# Mode 3: Flat directory with filename-as-label (NEW)
# ============================================================

def _parse_flat_dir(directory: str) -> dict[str, list[str]]:
    """
    Scan a flat directory for {Name}({Color}).mp4 files, group by doctor name.

    Returns: {name: [video_path, ...]}
    """
    import re
    from collections import defaultdict

    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Dataset directory not found: {directory}")

    pattern = re.compile(r"^(.+?)\([^)]+\)\.mp4$")
    doctor_videos = defaultdict(list)

    for filename in sorted(os.listdir(directory)):
        if not filename.lower().endswith(".mp4"):
            continue
        match = pattern.match(filename)
        if match:
            name = match.group(1)
            full_path = os.path.join(directory, filename)
            doctor_videos[name].append(full_path)
        else:
            print(f"  [SKIP] Cannot parse filename: {filename}")

    return dict(doctor_videos)


def build_gallery_from_flat_dataset(
    dataset_dir: str,
    app=None,
    clear_existing: bool = False,
) -> dict:
    """
    Build gallery from a flat directory of single-doctor videos.

    Filename convention: {DoctorName}({Color}).mp4
    Example: "付玉峰(蓝).mp4" -> doctor="付玉峰", color="蓝"

    Each video is treated as containing ONLY the doctor named in the filename.
    All videos for the same doctor are merged into one gallery entry.

    Args:
        dataset_dir: Path to the directory containing .mp4 files
        app: InsightFace FaceAnalysis instance (auto-initialized if None)
        clear_existing: If True, delete all existing gallery data before building

    Returns:
        gallery_index dict
    """
    from config import NEXT_DOCTOR_ID

    if app is None:
        app = get_app()

    _ensure_dirs()

    # Optionally clear existing gallery
    if clear_existing:
        import shutil
        if os.path.exists(GALLERY_DIR):
            for item in os.listdir(GALLERY_DIR):
                item_path = os.path.join(GALLERY_DIR, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                elif item.endswith(".json"):
                    os.remove(item_path)
            print("[INFO] Existing gallery cleared.")

    # Load existing index for ID reuse
    existing_index = {}
    if os.path.exists(GALLERY_INDEX_FILE):
        with open(GALLERY_INDEX_FILE, "r", encoding="utf-8") as f:
            existing_index = json.load(f)

    # Build name->id mapping from existing index
    name_to_id = {}
    for doc_id, info in existing_index.items():
        name_to_id[info["name"]] = doc_id

    # Get max existing ID for new doctor assignment
    all_ids = [int(k) for k in existing_index.keys()]
    max_existing_id = max(all_ids) if all_ids else NEXT_DOCTOR_ID - 1

    # Parse dataset
    doctor_videos = _parse_flat_dir(dataset_dir)

    if not doctor_videos:
        print("[ERROR] No valid videos found in dataset directory")
        return {}

    print(f"\n[INFO] Found {len(doctor_videos)} doctor(s) in dataset:")
    for name, videos in doctor_videos.items():
        reuse = " (reuse)" if name in name_to_id else " (NEW)"
        print(f"       {name}: {len(videos)} video(s){reuse}")
        for v in videos:
            print(f"         - {os.path.basename(v)}")

    # Process each doctor
    total_start = time.time()
    gallery_index = dict(existing_index)  # Start with existing

    for name, video_paths in sorted(doctor_videos.items()):
        # Assign ID
        if name in name_to_id:
            doctor_id = name_to_id[name]
        else:
            max_existing_id += 1
            doctor_id = str(max_existing_id)
            name_to_id[name] = doctor_id

        print(f"\n{'=' * 40}")
        print(f"  Doctor: {name} (ID: {doctor_id})")
        print(f"{'=' * 40}")

        all_embeddings = []
        for video_path in sorted(video_paths):
            embs = _process_single_doctor_video(video_path, doctor_id, name, app)
            all_embeddings.extend(embs)

        result = _save_doctor_gallery(doctor_id, name, all_embeddings)
        if result:
            gallery_index[doctor_id] = result
            print(f"  [SAVE] {name} ({doctor_id}): {result['embedding_count']} embeddings")
        else:
            print(f"  [WARN] {name} ({doctor_id}): NO embeddings extracted")

    # Save global index
    with open(GALLERY_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(gallery_index, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] Gallery index saved to {GALLERY_INDEX_FILE}")

    elapsed = time.time() - total_start
    new_count = len(doctor_videos)
    print(f"\n[DONE] Gallery built: {len(gallery_index)} doctors total "
          f"({new_count} from dataset) in {elapsed:.1f}s")

    return gallery_index


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build doctor face gallery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  (default)                 Legacy mode: process April 27/28 training videos with Excel
  --single-doctor-videos    Scan output/gallery/{doctor_id}_{name}/ for .mp4 files
  --flat-dataset DIR        Build from flat directory (filename={Name}({Color}).mp4)
  --reindex                 Rebuild index from existing embeddings (no video processing)
        """,
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Rebuild index from existing gallery folders only",
    )
    parser.add_argument(
        "--single-doctor-videos",
        action="store_true",
        help="Scan gallery folders for per-doctor .mp4 files and extract faces",
    )
    parser.add_argument(
        "--flat-dataset",
        type=str,
        default=None,
        metavar="DIR",
        help="Build from flat directory of single-doctor videos (filename-as-label)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing gallery before building (only with --flat-dataset)",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.reindex:
        reindex()
    elif args.flat_dataset:
        app = get_app()
        build_gallery_from_flat_dataset(args.flat_dataset, app, clear_existing=args.clear)
    elif args.single_doctor_videos:
        app = get_app()
        build_gallery_from_single_videos(app)
    else:
        app = get_app()
        build_gallery_legacy(app)


if __name__ == "__main__":
    main()
