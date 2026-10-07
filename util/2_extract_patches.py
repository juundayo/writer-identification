"""Step 2: SIFT keypoints -> 32x32 patches, one compressed .npz per page.

Each .npz stores, for N kept keypoints:
  patches   (N,32,32) uint8   binarized patch (ink = 0, paper = 255)
  patches_c (N,32,32,3) uint8 colour patch at the same location (if --color)
  xy        (N,2) float32     keypoint centre in page pixel coordinates
  size, angle, response (N,) float32   SIFT keypoint attributes
  desc      (N,128) uint8     SIFT descriptor (for surrogate clustering)
Coordinates are kept so explanations can later be drawn back on the page.

Run:  python extract_patches.py --split train --workers 8 [--color]
"""
import argparse
import csv
from multiprocessing import Pool

import cv2
import numpy as np

cv2.setNumThreads(1)  # avoid oversubscription with multiprocessing
from tqdm import tqdm

import config as C

H = C.PATCH_SIZE // 2


def load_binary(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"cannot read {path}")
    img = np.where(img < 128, 0, 255).astype(np.uint8)
    if img.mean() < 127:  # ensure ink is dark on light background
        img = 255 - img
    return img


def pad(img):
    """Pad once per page with paper colour so border keypoints can be cropped."""
    val = 255 if img.ndim == 2 else (255, 255, 255)
    return cv2.copyMakeBorder(img, H, H, H, H, cv2.BORDER_CONSTANT, value=val)


def crop(padded, x, y):
    """Crop a PATCH_SIZE square centred on page coords (x, y) from a padded page."""
    xi, yi = int(round(x)), int(round(y))
    return padded[yi:yi + C.PATCH_SIZE, xi:xi + C.PATCH_SIZE]


def process(args):
    row, out_dir, use_color = args
    out = out_dir / f"{row['page_id']}.npz"
    if out.exists():
        return row["page_id"], -1  # already done (resume)

    img = load_binary(row["bin_path"])
    sift = cv2.SIFT_create()
    kps, desc = sift.detectAndCompute(img, None)
    if desc is None:
        kps, desc = [], np.zeros((0, 128), np.float32)

    color = cv2.imread(row["color_path"]) if use_color and row["color_path"] else None
    if color is not None and color.shape[:2] != img.shape:
        color = cv2.resize(color, (img.shape[1], img.shape[0]))

    img_p = pad(img)
    color_p = pad(color) if color is not None else None
    seen, keep = set(), []
    for i, kp in enumerate(kps):
        if kp.size < C.MIN_KP_SIZE:
            continue
        key = (round(kp.pt[0] / C.DEDUP_PX), round(kp.pt[1] / C.DEDUP_PX))
        if key in seen:
            continue
        p = crop(img_p, *kp.pt)
        if (p < 128).mean() < C.MIN_INK_FRAC:
            continue
        seen.add(key)
        keep.append((i, p))

    idx = np.array([i for i, _ in keep], dtype=np.int64)
    data = {
        "patches": np.stack([p for _, p in keep]) if keep
        else np.zeros((0, C.PATCH_SIZE, C.PATCH_SIZE), np.uint8),
        "xy": np.array([kps[i].pt for i in idx], np.float32).reshape(-1, 2),
        "size": np.array([kps[i].size for i in idx], np.float32),
        "angle": np.array([kps[i].angle for i in idx], np.float32),
        "response": np.array([kps[i].response for i in idx], np.float32),
        "desc": np.clip(desc[idx], 0, 255).astype(np.uint8) if len(idx)
        else np.zeros((0, 128), np.uint8),
        "writer": np.int64(row["writer"]),
    }
    if color is not None:
        data["patches_c"] = (np.stack([crop(color_p, *kps[i].pt) for i in idx]) if len(idx)
                             else np.zeros((0, C.PATCH_SIZE, C.PATCH_SIZE, 3), np.uint8))
    np.savez_compressed(out, **data)
    return row["page_id"], len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(C.SPLITS), required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--color", action="store_true", help="also store colour patches")
    ap.add_argument("--limit", type=int, default=0, help="process only N pages (debug)")
    a = ap.parse_args()

    with open(C.WORK / f"index_{a.split}.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["bin_path"]]
    if a.limit:
        rows = rows[:a.limit]
    out_dir = C.WORK / "patches" / a.split
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = []
    with Pool(a.workers) as pool:
        jobs = [(r, out_dir, a.color) for r in rows]
        for pid, n in tqdm(pool.imap_unordered(process, jobs), total=len(jobs)):
            if n >= 0:
                counts.append(n)
    if counts:
        c = np.array(counts)
        print(f"pages: {len(c)}  patches/page min/median/max: "
              f"{c.min()}/{int(np.median(c))}/{c.max()}  total: {c.sum()}")


if __name__ == "__main__":
    main()
