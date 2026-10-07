"""Step 1: index Historical-WI, parse writer IDs, pair binarized <-> color pages.

Writes WORK/index_<split>.csv with one row per page and prints sanity checks.
Run:  python build_index.py
"""
import csv
import re
from collections import Counter

from PIL import Image

import config as C

WRITER_RE = re.compile(C.WRITER_REGEX)


def scan(folder):
    """Map page stem -> path for all images below folder (handles nested dirs)."""
    files = {}
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() in C.IMG_EXT and p.is_file():
            if p.stem in files:
                print(f"  [warn] duplicate stem {p.stem} in {folder.name}")
            files[p.stem] = p
    return files


def index_split(split, dirs):
    print(f"\n=== {split} ===")
    bins = scan(dirs["bin"])
    cols = scan(dirs["color"])
    print(f"binarized images: {len(bins)}   color images: {len(cols)}")

    only_bin = sorted(set(bins) - set(cols))
    only_col = sorted(set(cols) - set(bins))
    if only_bin or only_col:
        print(f"  [warn] unpaired: {len(only_bin)} binarized-only, {len(only_col)} color-only")
        print("    examples bin-only:", only_bin[:5])
        print("    examples color-only:", only_col[:5])

    rows, bad = [], []
    for stem in sorted(set(bins) | set(cols)):
        m = WRITER_RE.match(stem)
        if not m:
            bad.append(stem)
            continue
        ref = bins.get(stem) or cols.get(stem)
        with Image.open(ref) as im:  # reads header only
            w, h = im.size
        rows.append({
            "page_id": stem,
            "writer": int(m.group(1)),
            "bin_path": str(bins.get(stem, "")),
            "color_path": str(cols.get(stem, "")),
            "width": w,
            "height": h,
        })
    if bad:
        print(f"  [warn] writer ID not parsed for {len(bad)} files, e.g. {bad[:5]}")
        print("         -> adjust WRITER_REGEX in config.py")

    # Sanity checks against the competition description
    per_writer = Counter(r["writer"] for r in rows)
    dist = Counter(per_writer.values())
    exp_w, exp_p = C.EXPECTED[split]
    print(f"pages: {len(rows)}   writers: {len(per_writer)}   (expected {exp_w} x {exp_p})")
    print("pages-per-writer distribution:", dict(sorted(dist.items())))
    if rows:
        ws = sorted(r["width"] for r in rows)
        hs = sorted(r["height"] for r in rows)
        print(f"width  min/median/max: {ws[0]}/{ws[len(ws)//2]}/{ws[-1]}")
        print(f"height min/median/max: {hs[0]}/{hs[len(hs)//2]}/{hs[-1]}")

    C.WORK.mkdir(parents=True, exist_ok=True)
    out = C.WORK / f"index_{split}.csv"
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    for split, dirs in C.SPLITS.items():
        index_split(split, dirs)
