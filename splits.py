#!/usr/bin/env python3
"""splits.py -- P1 video-disjoint split of the VinFish dataset into YOLO roots.

Master revision plan SS6.3 P1: entire video groups go to exactly one of
train/valid/test. Optional --kfold N mode runs GroupKFold with group=video_id
and prints per-fold class distribution and image counts with mean +/- std.

Creates:
  __AQUASC_DATA__/yolo_fish_dataset_root/{train,valid,test}/{images,labels}
  __AQUASC_DATA__/yolo_fish_dataset_root_ir/{train,valid,test}/{images,labels}
  __AQUASC_ROOT__/splits/manifest.json

Images and labels are SYMLINKED (not copied) from the raw VinFish tree.

Filename convention: frame_<index>_<video_id>.png  (e.g. frame_000100_2kk.png).
Frames with no suffix (frame_NNNNNN.png) are grouped under the synthetic
video_id "nosuffix" (136 frames in the Color set, index 14672..29792 step 112).
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter, OrderedDict, defaultdict

import numpy as np
from PIL import Image

try:
    import cv2
    HAVE_CV2 = True
except Exception:  # pragma: no cover
    HAVE_CV2 = False

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MAIN_ROOT = "__AQUASC_DATA__/yolo_fish_dataset_root"
DEFAULT_IR_ROOT = "__AQUASC_DATA__/yolo_fish_dataset_root_ir"
MANIFEST_PATH = os.path.join(BASE, "splits", "manifest.json")

FRAME_RE = re.compile(r"^frame_(\d+)(?:_([^.]+))?\.(png|jpg|jpeg)$")
LABEL_EXT = {".png": ".txt", ".jpg": ".txt", ".jpeg": ".txt"}

MODALITIES = [
    {
        "name": "rgb",
        "images_dir": "__AQUASC_DATA__/VinFish/Color/images",
        "labels_dir": "__AQUASC_DATA__/VinFish/Color/labels",
        "root": None,  # set via --root
    },
    {
        "name": "ir",
        "images_dir": "__AQUASC_DATA__/VinFish/Gray data/images",
        "labels_dir": "__AQUASC_DATA__/VinFish/Gray data/labels",
        "root": None,  # set via --root-ir
    },
]


def parse_filename(fname):
    m = FRAME_RE.match(fname)
    if m is None:
        raise ValueError(f"unparseable filename: {fname}")
    frame_index = int(m.group(1))
    video_id = m.group(2)
    if video_id is None or video_id == "":
        video_id = "nosuffix"
    return video_id, frame_index


def collect(images_dir, labels_dir):
    """Return list of dicts {file, label_file, video_id, frame_index, path, label_path}."""
    entries = []
    for fname in sorted(os.listdir(images_dir)):
        video_id, frame_index = parse_filename(fname)
        base = os.path.splitext(fname)[0]
        label_name = base + ".txt"
        label_path = os.path.join(labels_dir, label_name)
        if not os.path.exists(label_path):
            print(f"  WARNING: {label_name} has no label file; skipped", file=sys.stderr)
            continue
        entries.append(
            {
                "file": fname,
                "label_file": label_name,
                "video_id": video_id,
                "frame_index": frame_index,
                "img_path": os.path.join(images_dir, fname),
                "label_path": label_path,
            }
        )
    return entries


def class_distribution(entries):
    dist = Counter()
    for e in entries:
        try:
            with open(e["label_path"]) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    dist[int(line.split()[0])] += 1
        except OSError as exc:
            print(f"  WARNING: cannot read {e['label_path']}: {exc}", file=sys.stderr)
    return dist


def summarize(entries):
    dist = class_distribution(entries)
    return {
        "images": len(entries),
        "videos": len({e["video_id"] for e in entries}),
        "class_distribution": dict(sorted(dist.items())),
    }


def assign_videos_to_splits(entries, ratios, seed):
    """Deterministic greedy assignment of whole videos to splits.

    Videos are shuffled with the given seed, then each video is assigned to the
    split whose frame-count shortfall vs. the target ratio is largest, so the
    resulting frame proportions approximate the requested ratios.

    Fallback: if the greedy pass would leave any split empty (e.g. the IR
    modality has only 3 videos), assign by size -- largest video to train, then
    alternate valid/test with the next largest, and all remaining videos to
    train -- so every split is non-empty.
    """
    videos = defaultdict(list)
    for e in entries:
        videos[e["video_id"]].append(e)
    video_ids = list(videos.keys())
    rng = random.Random(seed)
    rng.shuffle(video_ids)

    n_total = len(entries)
    counts = {"train": 0, "valid": 0, "test": 0}
    assignment = {}
    for vid in video_ids:
        vframes = len(videos[vid])
        target = {s: ratios[s] * n_total for s in ratios}
        shortfall = {s: target[s] - counts[s] for s in ratios}
        chosen = max(shortfall, key=shortfall.get)
        assignment[vid] = chosen
        counts[chosen] += vframes

    if all(counts[s] > 0 for s in counts):
        return assignment

    print("  WARNING: greedy by-frame assignment leaves an empty split; "
          "falling back to size-ordered 1-video-per-split assignment",
          file=sys.stderr)
    ordered = sorted(video_ids, key=lambda v: len(videos[v]), reverse=True)
    assignment = {v: "train" for v in ordered}
    assignment[ordered[0]] = "train"
    for i, vid in enumerate(ordered[1:], start=1):
        if i == 1:
            assignment[vid] = "valid"
        elif i == 2:
            assignment[vid] = "test"
        else:
            assignment[vid] = "train" if (i % 2 == 0) else "valid"
    return assignment


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.8, help="train fraction of frames (approx)")
    ap.add_argument("--valid", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--kfold", type=int, default=0,
                    help="run GroupKFold (group=video_id) evaluation and print fold stats "
                         "instead of the default 80/10/10 split")
    ap.add_argument("--root", default=DEFAULT_MAIN_ROOT, help="RGB YOLO root")
    ap.add_argument("--root-ir", default=DEFAULT_IR_ROOT, help="IR YOLO root")
    ap.add_argument("--no-symlink-check", action="store_true",
                    help="skip the PIL/cv2 read-through-symlink verification")
    args = ap.parse_args()

    ratios = OrderedDict([("train", args.train), ("valid", args.valid), ("test", args.test)])
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        ap.error(f"ratios must sum to 1, got {sum(ratios.values())}")
    for k, v in ratios.items():
        if v < 0:
            ap.error(f"ratio {k} must be >= 0")

    mod_roots = {MODALITIES[0]["name"]: args.root, MODALITIES[1]["name"]: args.root_ir}
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)

    all_records = []
    per_mod_summary = {}
    fold_stats = {}

    for mod in MODALITIES:
        name = mod["name"]
        root = mod_roots[name]
        print(f"\n=== Modality: {name} (root: {root}) ===")
        entries = collect(mod["images_dir"], mod["labels_dir"])
        if not entries:
            print(f"  ERROR: no images found in {mod['images_dir']}", file=sys.stderr)
            sys.exit(1)

        dist = class_distribution(entries)
        print(f"  total images: {len(entries)}  total videos: "
              f"{len({e['video_id'] for e in entries})}")
        print(f"  class distribution: {dict(sorted(dist.items()))}")
        vids = OrderedDict()
        for e in entries:
            vids.setdefault(e["video_id"], []).append(e)
        for vid, ves in sorted(vids.items()):
            idxs = sorted(e["frame_index"] for e in ves)
            print(f"    video {vid:>12s}: {len(ves):5d} frames, index {idxs[0]:>6d}..{idxs[-1]:>6d}")

        if args.kfold and args.kfold > 1:
            stats = run_groupkfold(entries, args.kfold)
            fold_stats[name] = stats
            per_mod_summary[name] = {"mode": f"groupkfold-{args.kfold}", "root": root}
            all_records.extend(
                {"file": e["file"], "modality": name, "split": None,
                 "video_id": e["video_id"], "frame_index": e["frame_index"]}
                for e in entries
            )
            continue

        assignment = assign_videos_to_splits(entries, ratios, args.seed)
        split_entries = defaultdict(list)
        for e in entries:
            split_entries[assignment[e["video_id"]]].append(e)

        print(f"  seed={args.seed}  requested ratios: {dict(ratios)}")
        make_yolo_root(root, split_entries, name, check=not args.no_symlink_check)

        per_split = {}
        for split in ("train", "valid", "test"):
            se = split_entries[split]
            per_split[split] = summarize(se)
            print(f"    {split:>5s}: {len(se):5d} images, "
                  f"{per_split[split]['videos']:3d} videos, "
                  f"classes {per_split[split]['class_distribution']}")
        actual = {s: len(split_entries[s]) for s in split_entries}
        ntot = float(sum(actual.values()))
        print("  actual frame fractions: " + ", ".join(
            f"{s}={actual[s]/ntot:.3f}" for s in ("train", "valid", "test")))

        per_mod_summary[name] = {
            "mode": "video_disjoint_80_10_10",
            "seed": args.seed,
            "root": root,
            "split": per_split,
            "actual_fractions": {s: actual[s] / ntot for s in actual},
        }
        for e in entries:
            all_records.append(
                {"file": e["file"], "modality": name,
                 "split": assignment[e["video_id"]],
                 "video_id": e["video_id"], "frame_index": e["frame_index"]}
            )

    manifest = {
        "description": "VinFish video-disjoint split manifest (plan SS6.3 P1)",
        "seed": args.seed,
        "ratios": dict(ratios),
        "kfold": args.kfold or None,
        "modalities": per_mod_summary,
        "records": all_records,
    }
    if fold_stats:
        manifest["fold_stats"] = fold_stats
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {MANIFEST_PATH}")


def run_groupkfold(entries, k):
    """GroupKFold with group=video_id (deterministic, sklearn-equivalent).

    Each video is one sample; groups are the sorted video ids. With one sample
    per group this is plain deterministic KFold over the sorted video ids,
    identical to sklearn's GroupKFold in this setting (consecutive folds,
    remainder spread over the first n_samples % k folds). Implemented locally
    to avoid a scikit-learn dependency.
    """
    videos = OrderedDict()
    for e in entries:
        videos.setdefault(e["video_id"], []).append(e)
    video_ids = sorted(videos.keys())
    n_vids = len(video_ids)

    n_splits = min(k, n_vids)
    if n_splits < k:
        print(f"  WARNING: requested {k} folds but only {n_vids} videos; "
              f"using {n_splits} folds", file=sys.stderr)
    if n_splits < 2:
        print("  WARNING: cannot GroupKFold with fewer than 2 videos",
              file=sys.stderr)
        return {"k": k, "n_videos": n_vids, "folds": []}

    fold_sizes = np.full(n_splits, n_vids // n_splits, dtype=int)
    fold_sizes[: n_vids % n_splits] += 1
    boundaries = np.concatenate([[0], np.cumsum(fold_sizes)])

    folds = []
    for fold in range(n_splits):
        te_idx = set(range(boundaries[fold], boundaries[fold + 1]))
        tr_vids = [video_ids[i] for i in range(n_vids) if i not in te_idx]
        te_vids = [video_ids[i] for i in sorted(te_idx)]
        tr_entries = [e for v in tr_vids for e in videos[v]]
        te_entries = [e for v in te_vids for e in videos[v]]
        tr_dist = class_distribution(tr_entries)
        te_dist = class_distribution(te_entries)
        stats = {
            "fold": fold + 1,
            "train_videos": tr_vids,
            "test_videos": te_vids,
            "train_images": len(tr_entries),
            "test_images": len(te_entries),
            "train_classes": dict(sorted(tr_dist.items())),
            "test_classes": dict(sorted(te_dist.items())),
        }
        folds.append(stats)
        print(f"  fold {fold+1}/{k}: train={len(tr_entries)} images "
              f"({len(tr_vids)} vids: {tr_vids})  test={len(te_entries)} images "
              f"({len(te_vids)} vids: {te_vids})")
        print(f"      train classes {dict(sorted(tr_dist.items()))}  "
              f"test classes {dict(sorted(te_dist.items()))}")

    n_tr = np.array([s["train_images"] for s in folds], dtype=float)
    n_te = np.array([s["test_images"] for s in folds], dtype=float)
    print(f"  GroupKFold({k}) over {len(video_ids)} videos: "
          f"train images mean={n_tr.mean():.1f} +/- {n_tr.std():.1f}; "
          f"test images mean={n_te.mean():.1f} +/- {n_te.std():.1f}")
    return {"k": k, "n_videos": len(video_ids), "folds": folds,
            "train_images_mean": float(n_tr.mean()), "train_images_std": float(n_tr.std()),
            "test_images_mean": float(n_te.mean()), "test_images_std": float(n_te.std())}


def make_yolo_root(root, split_entries, modality, check=True):
    for split, entries in split_entries.items():
        img_dir = os.path.join(root, split, "images")
        lbl_dir = os.path.join(root, split, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        made = 0
        for e in entries:
            img_link = os.path.join(img_dir, e["file"])
            lbl_link = os.path.join(lbl_dir, e["label_file"])
            if not os.path.lexists(img_link):
                os.symlink(e["img_path"], img_link)
            if not os.path.lexists(lbl_link):
                os.symlink(e["label_path"], lbl_link)
            made += 1
        print(f"  symlinked {split}: {made} image+label pairs -> {img_dir}")

        if check:
            verify_symlinks(img_dir, lbl_dir, modality, split)

    missing_pairs = 0
    for split in ("train", "valid", "test"):
        img_dir = os.path.join(root, split, "images")
        lbl_dir = os.path.join(root, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        for fn in os.listdir(img_dir):
            base = os.path.splitext(fn)[0]
            if not os.path.exists(os.path.join(lbl_dir, base + ".txt")):
                missing_pairs += 1
    if missing_pairs:
        print(f"  WARNING: {missing_pairs} images without labels in {root}",
              file=sys.stderr)


def verify_symlinks(img_dir, lbl_dir, modality, split):
    try:
        sample = sorted(os.listdir(img_dir))[0]
    except (IndexError, OSError):
        print(f"  WARNING: nothing to verify in {img_dir}", file=sys.stderr)
        return
    img_link = os.path.join(img_dir, sample)
    lbl_link = os.path.join(lbl_dir, os.path.splitext(sample)[0] + ".txt")
    ok = []

    try:
        im = Image.open(img_link)
        im.load()
        ok.append(f"PIL read {img_link} -> {im.size} {im.mode}")
    except Exception as exc:
        ok.append(f"PIL FAILED on {img_link}: {exc}")

    if HAVE_CV2:
        try:
            mat = cv2.imread(img_link, cv2.IMREAD_COLOR)
            ok.append(f"cv2 read {img_link} -> shape {None if mat is None else mat.shape}")
            if mat is None:
                ok[-1] = "cv2 FAILED on %s -> None" % img_link
        except Exception as exc:
            ok.append(f"cv2 FAILED on {img_link}: {exc}")
    else:
        ok.append("cv2 not installed; skipped cv2 check")

    try:
        with open(lbl_link) as f:
            n = sum(1 for line in f if line.strip())
        ok.append(f"label {lbl_link} has {n} objects")
    except OSError as exc:
        ok.append(f"label read FAILED: {exc}")

    for msg in ok:
        print(f"    verify[{split}]: {msg}")


if __name__ == "__main__":
    main()
