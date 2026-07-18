"""
Doctor Face Identification System - Configuration
集中管理所有路径、阈值、采样率等参数
"""

import os
import sys

# Fix CUDA/cuDNN library paths for onnxruntime-gpu (needed on this system)
# NOTE: os.environ['LD_LIBRARY_PATH'] changes in Python do NOT affect the
# dynamic linker (ld.so) which reads LD_LIBRARY_PATH at process startup.
# Use `bash run.sh <script>` to set LD_LIBRARY_PATH before Python starts,
# or preload the CUDA runtime library explicitly via ctypes below.

_site_packages = os.path.join(
    os.path.dirname(os.path.dirname(sys.executable)),
    "lib", "python3.12", "site-packages"
)

# Preload ALL CUDA/cuDNN shared libraries via ctypes BEFORE onnxruntime imports them.
# os.environ['LD_LIBRARY_PATH'] changes in Python do NOT affect the dynamic linker —
# we must load every .so explicitly with RTLD_GLOBAL so downstream libs can see them.
import ctypes
import glob

_cu13_lib_dir = os.path.join(_site_packages, "nvidia", "cu13", "lib")
_cudnn_lib_dir = os.path.join(_site_packages, "nvidia", "cudnn", "lib")

# Load ALL .so files from both directories (dependency order: cu13 first, then cudnn)
for _lib_dir in [_cu13_lib_dir, _cudnn_lib_dir]:
    if not os.path.isdir(_lib_dir):
        continue
    for _so_path in sorted(glob.glob(os.path.join(_lib_dir, "*.so*"))):
        # Skip static libs and symlinks we already loaded
        if _so_path.endswith(".a"):
            continue
        try:
            ctypes.CDLL(_so_path, mode=ctypes.RTLD_GLOBAL)
        except Exception:
            pass  # Already loaded, symlink alias, or incompatible

# Also set LD_LIBRARY_PATH as fallback (for subprocess / external tools)
for _d in [_cu13_lib_dir, _cudnn_lib_dir]:
    if os.path.isdir(_d):
        os.environ.setdefault("LD_LIBRARY_PATH", "")
        if _d not in os.environ["LD_LIBRARY_PATH"]:
            os.environ["LD_LIBRARY_PATH"] = f"{_d}:{os.environ['LD_LIBRARY_PATH']}".strip(":")


# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "doctor")
TEST_DIR = os.path.join(PROJECT_ROOT, "test_video")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
GALLERY_DIR = os.path.join(OUTPUT_DIR, "gallery")
FACES_DIR = os.path.join(OUTPUT_DIR, "faces")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "results.json")
GALLERY_INDEX_FILE = os.path.join(GALLERY_DIR, "gallery_index.json")

# New dataset: per-doctor single-person videos
PEOPLE_DATASETS_DIR = os.path.join(PROJECT_ROOT, "data", "people_datasets")
DOCTOR_REGISTRY_FILE = os.path.join(OUTPUT_DIR, "doctor_registry.json")
NEXT_DOCTOR_ID = 25001

# Training video sources
VIDEO_0427 = os.path.join(DATA_DIR, "4月27日医生视频", "1.mp4")
EXCEL_0427 = os.path.join(DATA_DIR, "4月27日医生视频", "doctor_info.xlsx")
VIDEO_0428 = os.path.join(DATA_DIR, "4月28日医生视频", "1.mp4")
EXCEL_0428 = os.path.join(DATA_DIR, "4月28日医生视频", "视频中的医生信息表.xlsx")

# ============================================================
# Video / Frame Processing
# ============================================================
VIDEO_FPS = 25                              # Both training and test videos are 25 fps
DETECTION_SIZE = (1920, 1080)               # Downsample 4K for face detection speed
FRAME_SAMPLE_INTERVAL = 5                   # Process every Nth frame within time segments
TEST_FRAME_COUNT = 100                      # Number of evenly-spaced frames to sample per test video

# ============================================================
# Face Detection
# ============================================================
MIN_FACE_SIZE = (40, 40)                    # Minimum detection size in pixels (relaxed for small faces)
DETECTION_CONFIDENCE = 0.25                  # Minimum face detection confidence (relaxed)

# Face Quality Filtering
MIN_QUALITY_SCORE = 0.3                     # Minimum overall quality score to keep a face embedding
MIN_SHARPNESS = 0.1                         # Minimum Laplacian sharpness score
MIN_FRONTALITY = 0.2                        # Minimum frontality score

# ============================================================
# Recognition
# ============================================================
INSIGHTFACE_MODEL = "buffalo_l"              # buffalo_l: det_10g + w600k_r50 (ARC-FACE, stronger recognition)
DEVICE = "cuda"                             # "cuda" or "cpu"
DET_SIZE = (1920, 1920)                     # Higher res for better face quality
MATCH_THRESHOLD = 0.4                       # Cosine similarity threshold for positive match
UNKNOWN_THRESHOLD = 0.3                     # Below this, report "unknown"
TOP2_MARGIN = 0.05                          # Min diff between top-1 and top-2 similarity for high confidence
KNN_K = 5                                   # K for KNN voting in identification

# ============================================================
# Output
# ============================================================
SAVE_FACE_SAMPLES = True                    # Save sample face crops for visual verification
FACE_SAMPLES_PER_DOCTOR = 5                 # Max face crops to save per doctor
