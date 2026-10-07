"""Shared paths and parameters for the writer-retrieval XAI project."""
from pathlib import Path

# --- Data -------------------------------------------------------------------
ROOT = Path("/media/Data/WRITER_IDENTIFICATION/ICDAR2017")
# All generated files (index, patches, clusters, models) go here.
WORK = Path("/media/Data/WRITER_IDENTIFICATION/work_icdar17")

SPLITS = {
    "train": {
        "bin": ROOT / "icdar17-historical-training-binarized",
        "color": ROOT / "icdar17-historical-training-color",
    },
    "test": {
        "bin": ROOT / "ScriptNet-HistoricalWI-2017-binarized",
        "color": ROOT / "ScriptNet-HistoricalWI-2017-color",
    },
}
# Expected sizes from the competition description (used as a sanity check).
EXPECTED = {"train": (394, 3), "test": (720, 5)}  # (writers, pages per writer)

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# Writer ID = leading number of the file name, e.g. "123-IMG_MAX_4567.jpg".
# build_index.py reports any file this fails on, so adjust here if needed.
WRITER_REGEX = r"^(\d+)[-_]"

# --- Patch extraction (following Peer et al., ICDAR 2023) -------------------
PATCH_SIZE = 32          # 32x32 patches centred on SIFT keypoints
MIN_INK_FRAC = 0.05      # drop patches that are almost empty
DEDUP_PX = 2             # merge keypoints closer than this (pixels)
MIN_KP_SIZE = 0.0        # optional filter on SIFT keypoint scale

# --- Surrogate labels (cluster-as-label) ------------------------------------
PCA_DIM = 32             # SIFT 128 -> 32 before clustering
N_CLUSTERS = 5000
RATIO_RHO = 0.9          # keep patch if d(1st centre) < rho * d(2nd centre)
DESC_SAMPLE_PER_PAGE = 2000  # descriptors sampled per page to fit PCA/k-means
