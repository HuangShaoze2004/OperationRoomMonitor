"""
Face detection and embedding extraction using InsightFace (buffalo_l).

CRITICAL: SCRFD detector requires SQUARE input -- non-square sizes trigger a
known shape-mismatch bug in insightface 1.0.1. We pad frames to square before
passing to the detector.

The buffalo_l model pack includes:
  - Detection: SCRFD det_10g (stronger detection)
  - Recognition: ArcFace w600k_r50 (512-d embeddings, L2-normalized)
"""

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from config import (
    INSIGHTFACE_MODEL,
    DET_SIZE,
    MIN_FACE_SIZE,
    DETECTION_CONFIDENCE,
    DEVICE,
)


# Global app instance (lazy init)
_app: FaceAnalysis | None = None


def get_app() -> FaceAnalysis:
    """Get or initialize the InsightFace FaceAnalysis app."""
    global _app
    if _app is None:
        providers = ["CUDAExecutionProvider"] if DEVICE == "cuda" else ["CPUExecutionProvider"]
        _app = FaceAnalysis(name=INSIGHTFACE_MODEL, providers=providers)
        ctx = 0 if DEVICE == "cuda" else -1
        _app.prepare(ctx_id=ctx, det_size=DET_SIZE)
        print(f"[INFO] InsightFace '{INSIGHTFACE_MODEL}' loaded (device={DEVICE}, det_size={DET_SIZE})")
    return _app


def _norm_embedding(emb: np.ndarray) -> np.ndarray:
    """L2-normalize embedding vector (insightface buffalo_sc does NOT auto-normalize)."""
    norm = np.linalg.norm(emb)
    if norm > 0:
        return emb / norm
    return emb


def _detect_faces(frame: np.ndarray, app: FaceAnalysis) -> list:
    """
    Detect faces in a frame. Pads to square to avoid SCRFD shape-mismatch bug.

    The pipeline:
      1. Pad original frame (H x W) to square (S x S)
      2. Resize square to DET_SIZE for the detector
      3. Run detection
      4. Map bbox coordinates back: DET_SIZE -> square -> original frame

    Returns list of Face objects with:
      - bbox mapped to original frame coordinates
      - embedding manually L2-normalized
    """
    h, w = frame.shape[:2]
    square_size = max(h, w)

    # Step 1: Pad to square
    square = np.zeros((square_size, square_size, 3), dtype=np.uint8)
    square[:h, :w] = frame

    # Step 2: Resize to detection size
    scale = square_size / DET_SIZE[0]  # same for both axes (DET_SIZE is square)
    square_resized = cv2.resize(square, DET_SIZE)

    # Step 3: Run detection
    faces = app.get(square_resized)

    if not faces:
        return []

    # Step 4: Map bbox from DET_SIZE coords back to original frame coords
    for face in faces:
        # face.bbox is in det_size (e.g. 640x640) coordinates
        # Convert to square (e.g. 3840x3840) coordinates
        bbox_square = face.bbox * scale
        # Clip to original frame boundaries (since padding adds black bars)
        x1, y1, x2, y2 = bbox_square
        x1 = min(x1, w)
        y1 = min(y1, h)
        x2 = min(x2, w)
        y2 = min(y2, h)
        face.bbox = np.array([x1, y1, x2, y2])

        # Manually L2-normalize the embedding (buffalo_sc does not auto-normalize)
        face.embedding = _norm_embedding(face.embedding)

    return faces


def get_face_embedding(frame: np.ndarray, app: FaceAnalysis = None) -> np.ndarray | None:
    """
    Detect faces in a frame and return the L2-normalized embedding of the most confident face.

    Args:
        frame: BGR image as numpy array (H, W, 3)
        app: FaceAnalysis instance (auto-initialized if None)

    Returns:
        512-d L2-normalized embedding vector, or None if no face detected
    """
    if app is None:
        app = get_app()

    faces = _detect_faces(frame, app)

    if not faces:
        return None

    # Filter by confidence and face size
    valid_faces = []
    for face in faces:
        if face.det_score < DETECTION_CONFIDENCE:
            continue
        x1, y1, x2, y2 = face.bbox.astype(int)
        w, h = x2 - x1, y2 - y1
        if w < MIN_FACE_SIZE[0] or h < MIN_FACE_SIZE[1]:
            continue
        valid_faces.append(face)

    if not valid_faces:
        return None

    best_face = max(valid_faces, key=lambda f: f.det_score)
    return best_face.embedding


def get_face_with_bbox(frame: np.ndarray, app: FaceAnalysis = None) -> tuple[np.ndarray, tuple] | None:
    """
    Like get_face_embedding but also returns the bounding box.

    Returns:
        (embedding, bbox) tuple where bbox = (x1, y1, x2, y2) in original frame coords,
        or None if no face detected
    """
    if app is None:
        app = get_app()

    faces = _detect_faces(frame, app)

    if not faces:
        return None

    valid_faces = []
    for face in faces:
        if face.det_score < DETECTION_CONFIDENCE:
            continue
        x1, y1, x2, y2 = face.bbox.astype(int)
        w_box, h_box = x2 - x1, y2 - y1
        if w_box < MIN_FACE_SIZE[0] or h_box < MIN_FACE_SIZE[1]:
            continue
        valid_faces.append(face)

    if not valid_faces:
        return None

    best_face = max(valid_faces, key=lambda f: f.det_score)
    bbox = tuple(best_face.bbox.astype(int))
    return best_face.embedding, bbox


def get_face_quality(face, frame: np.ndarray) -> dict:
    """
    Compute face quality metrics for filtering low-quality faces.

    Args:
        face: InsightFace Face object (must have .bbox, .kps, .det_score)
        frame: Original BGR frame (used for Laplacian sharpness check)

    Returns:
        dict with keys:
          - sharpness: Laplacian variance (higher = sharper)
          - frontality: 0-1 score based on facial landmark symmetry (higher = more frontal)
          - face_size: area of bbox in pixels
          - det_score: raw detection confidence
          - overall: weighted quality score (0-1)
    """
    x1, y1, x2, y2 = face.bbox.astype(int)
    w_box, h_box = x2 - x1, y2 - y1
    face_area = w_box * h_box

    # 1. Sharpness: Laplacian variance on face region
    x1c = max(0, x1)
    y1c = max(0, y1)
    x2c = min(frame.shape[1], x2)
    y2c = min(frame.shape[0], y2)
    if x2c > x1c and y2c > y1c:
        face_crop = frame[y1c:y2c, x1c:x2c]
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalize: good faces typically have laplacian_var > 100
        sharpness = min(laplacian_var / 500.0, 1.0)
    else:
        sharpness = 0.0

    # 2. Frontality: based on 5-point landmark symmetry
    # kps: [left_eye, right_eye, nose, left_mouth, right_mouth]
    if face.kps is not None and len(face.kps) >= 5:
        kps = face.kps
        left_eye = kps[0]
        right_eye = kps[1]
        nose = kps[2]

        # Eye-level horizontal alignment
        eye_dy = abs(left_eye[1] - right_eye[1])
        eye_dx = max(abs(left_eye[0] - right_eye[0]), 1)
        eye_tilt = eye_dy / eye_dx  # 0 = perfectly horizontal

        # Nose centeredness between eyes
        eye_center_x = (left_eye[0] + right_eye[0]) / 2
        nose_offset = abs(nose[0] - eye_center_x) / eye_dx

        # Combine: low tilt + centered nose = frontal face
        frontality = max(0.0, 1.0 - (eye_tilt * 2 + nose_offset * 3))
    else:
        frontality = 0.5  # Unknown, neutral

    # 3. Face size relative to frame
    frame_area = frame.shape[0] * frame.shape[1]
    size_ratio = face_area / frame_area
    # Good faces occupy >1% of frame for 4K video
    size_score = min(size_ratio * 50, 1.0)  # 2% = score 1.0

    # 4. Overall quality (weighted)
    overall = (
        sharpness * 0.35
        + frontality * 0.35
        + size_score * 0.15
        + face.det_score * 0.15
    )

    return {
        "sharpness": round(sharpness, 3),
        "frontality": round(frontality, 3),
        "face_size": face_area,
        "det_score": round(float(face.det_score), 3),
        "overall": round(overall, 3),
    }


def get_best_face_with_quality(frame: np.ndarray, app: FaceAnalysis = None):
    """
    Detect faces and return the best face with embedding, bbox, and quality score.

    Returns:
        (embedding, bbox, quality_dict) or None if no face meets criteria
    """
    if app is None:
        app = get_app()

    faces = _detect_faces(frame, app)

    if not faces:
        return None

    # Filter by minimum confidence and face size
    valid = []
    for face in faces:
        if face.det_score < DETECTION_CONFIDENCE:
            continue
        x1, y1, x2, y2 = face.bbox.astype(int)
        w_box, h_box = x2 - x1, y2 - y1
        if w_box < MIN_FACE_SIZE[0] or h_box < MIN_FACE_SIZE[1]:
            continue
        quality = get_face_quality(face, frame)
        valid.append((face, quality))

    if not valid:
        return None

    # Pick best by overall quality (not just det_score)
    best_face, best_quality = max(valid, key=lambda x: x[1]["overall"])
    bbox = tuple(best_face.bbox.astype(int))
    return best_face.embedding, bbox, best_quality


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two L2-normalized vectors.
    Both should already be unit vectors; result = dot product.
    """
    return float(np.dot(a, b))
