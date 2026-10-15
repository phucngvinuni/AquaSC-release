#!/usr/bin/env python3
"""siteb_splits.py — video-disjoint split of FishDet-NB2 (Site B).

Filenames: YYYYMMDD_HHh_fNNNNN.jpg -> video_id = YYYYMMDD_HH (21 videos).
Creates a video-disjoint YOLO root (symlinks) + manifest, because the
shipped train/val split is video-INTERSECTING (all 21 videos in both).

Usage: __SIONNA_PY__ siteb_splits.py
"""
import json
import os
import re
from collections import Counter
from glob import glob

SRC = "__AQUASC_DATA__/yolo_fishdet_nb2"
DST = "__AQUASC_DATA__/yolo_fishdet_nb2_vd"
SEED = 42

PAT = re.compile(r"^(\d{8})_(\d{2})h_f\d+\.jpg$")


def video_id(fname):
    m = PAT.match(fname)
    if not m:
        raise ValueError(f"unparseable: {fname}")
    return f"{m.group(1)}_{m.group(2)}h"


def main():
    imgs = sorted(glob(os.path.join(SRC, "images", "train", "*.jpg")) +
                  glob(os.path.join(SRC, "images", "val", "*.jpg")))
    vids = sorted({video_id(os.path.basename(p)) for p in imgs})
    n = len(vids)
    n_tr, n_va = max(1, round(n * 0.80)), max(1, round(n * 0.10))
    tr, va, te = set(vids[:n_tr]), set(vids[n_tr:n_tr + n_va]), set(vids[n_tr + n_va:])
    assert len(tr | va | te) == n and not (tr & va) and not (tr & te) and not (va & te)

    split_imgs = Counter()
    records = []
    for p in imgs:
        base = os.path.basename(p)
        vid = video_id(base)
        split = "train" if vid in tr else ("valid" if vid in va else "test")
        split_imgs[split] += 1
        label = os.path.join(SRC, "labels", os.path.dirname(p).split("/")[-1],
                             os.path.splitext(base)[0] + ".txt")
        if not os.path.exists(label):
            raise FileNotFoundError(f"missing label for {base}")
        dst = os.path.join(DST, split)
        os.makedirs(os.path.join(dst, "images"), exist_ok=True)
        os.makedirs(os.path.join(dst, "labels"), exist_ok=True)
        os.symlink(p, os.path.join(dst, "images", base))
        os.symlink(label, os.path.join(dst, "labels", os.path.splitext(base)[0] + ".txt"))
        records.append({"file": base, "video_id": vid, "split": split})

    manifest = {
        "dataset": "FishDet-NB2 (Site B)", "seed": SEED,
        "videos": {v: {"split": "train" if v in tr else ("valid" if v in va else "test"),
                       "frames": sum(1 for r in records if r["video_id"] == v)}
                   for v in vids},
        "split_counts": dict(split_imgs),
        "note": "video-disjoint at YYYYMMDD_HH granularity; replaces the shipped video-intersecting split",
    }
    with open(os.path.join(DST, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps(manifest["split_counts"]))
    print(f"{n} videos: train {len(tr)} / valid {len(va)} / test {len(te)}")


if __name__ == "__main__":
    main()
