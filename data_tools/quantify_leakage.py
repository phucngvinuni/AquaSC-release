#!/usr/bin/env python3
"""data_tools/quantify_leakage.py -- plan SS6.2 leakage quantification.

For each TEST image, finds its nearest TRAIN image two ways:
  * pHash (hash_size=8) minimum Hamming distance to any train image
  * ResNet50 (ImageNet-1K V2, fc=Identity) maximum cosine similarity to any
    train image

Reports medians and near-duplicate fractions (phash Hamming <= 5, cosine >= 0.95)
and saves an npz with full arrays plus histograms.

Two modes:
  (a) random frame-level 80/10/10 split (the OLD defective protocol):
        --random-split --images-dir <dir> --labels-dir <dir>
      Materializes the split as a symlinked YOLO root in a temp dir, quantifies,
      then removes the temp root (nothing is left behind).
  (b) video-disjoint split produced by data_tools/splits.py:
        --root __AQUASC_DATA__/yolo_fish_dataset_root
        (or any YOLO root with train/ and test/ subdirs)

Run on CPU; do not train anything.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repo root (core modules)
import argparse
import json
import os
import shutil
import sys
import tempfile

import imagehash
import numpy as np
from PIL import Image

PHASH_NEAR_DUP_BITS = 5
COS_NEAR_DUP = 0.95
PHASH_SIZE = 8

try:
    import torch
    import torchvision
    from torchvision import transforms
    HAVE_TORCHVISION = True
except Exception as exc:  # pragma: no cover
    HAVE_TORCHVISION = False
    TORCH_IMPORT_ERR = exc


def list_images(d):
    return sorted(
        os.path.join(d, f)
        for f in os.listdir(d)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )


def compute_phash(paths):
    hashes = []
    for p in paths:
        with Image.open(p) as im:
            hashes.append(imagehash.phash(im.convert("RGB"), hash_size=PHASH_SIZE))
    return hashes


def make_embedder():
    if not HAVE_TORCHVISION:
        raise RuntimeError("torchvision unavailable: %s" % TORCH_IMPORT_ERR)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bb = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    bb.fc = torch.nn.Identity()
    bb.eval().to(dev)
    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return dev, bb, tf


@torch.no_grad()
def compute_embeddings(paths, dev, bb, tf, batch_size=32):
    bb.eval()
    E = []
    for i in range(0, len(paths), batch_size):
        batch = []
        for p in paths[i : i + batch_size]:
            with Image.open(p) as im:
                batch.append(tf(im.convert("RGB")))
        x = torch.stack(batch).to(dev)
        e = bb(x).cpu().numpy()
        E.append(e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9))
        print(f"    embedded {min(i + batch_size, len(paths))}/{len(paths)}", end="\r")
    print()
    return np.concatenate(E, axis=0)


def make_random_root(images_dir, labels_dir, seed, train=0.8, test=0.1):
    """Old defective protocol: random frame-level split, symlinked YOLO root."""
    rng = np.random.RandomState(seed)
    files = sorted(os.listdir(images_dir))
    files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    files = [f for f in files if os.path.exists(
        os.path.join(labels_dir, os.path.splitext(f)[0] + ".txt"))]
    perm = rng.permutation(len(files))
    n_train = int(round(train * len(files)))
    n_test = int(round(test * len(files)))
    # valid is the remainder; mirror the data_tools/splits.py convention order
    idx = {"train": perm[:n_train],
           "test": perm[n_train:n_train + n_test],
           "valid": perm[n_train + n_test:]}
    root = tempfile.mkdtemp(prefix="leakage_random_split_")
    for split, ii in idx.items():
        img_dir = os.path.join(root, split, "images")
        lbl_dir = os.path.join(root, split, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for i in ii:
            fn = files[i]
            os.symlink(os.path.join(images_dir, fn), os.path.join(img_dir, fn))
            os.symlink(os.path.join(labels_dir, os.path.splitext(fn)[0] + ".txt"),
                       os.path.join(lbl_dir, os.path.splitext(fn)[0] + ".txt"))
    print(f"  random frame split (seed {seed}): train={len(idx['train'])} "
          f"valid={len(idx['valid'])} test={len(idx['test'])} at {root}")
    return root


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True, help="label for this split (e.g. video_disjoint)")
    ap.add_argument("--out", required=True, help="npz output path")
    ap.add_argument("--root", default=None,
                    help="YOLO root with train/ and test/ subdirs (video-disjoint mode)")
    ap.add_argument("--random-split", action="store_true",
                    help="random frame-level 80/10/10 split (old protocol) on --images-dir")
    ap.add_argument("--images-dir", default="__AQUASC_DATA__/VinFish/Color/images")
    ap.add_argument("--labels-dir", default="__AQUASC_DATA__/VinFish/Color/labels")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-train", type=int, default=0,
                    help="cap train set size for nearest-neighbour search (0 = no cap)")
    ap.add_argument("--no-resnet", action="store_true",
                    help="phash only (fallback when torchvision weights cannot be fetched)")
    args = ap.parse_args()

    if not (args.root or args.random_split):
        ap.error("provide --root or --random-split")
    if args.random_split and args.root:
        ap.error("--random-split and --root are mutually exclusive")

    tmp_root = None
    if args.random_split:
        tmp_root = make_random_root(args.images_dir, args.labels_dir, args.seed)
        args.root = tmp_root

    train_dir = os.path.join(args.root, "train", "images")
    test_dir = os.path.join(args.root, "test", "images")
    for d in (train_dir, test_dir):
        if not os.path.isdir(d):
            ap.error(f"missing directory: {d}")

    train_paths = list_images(train_dir)
    test_paths = list_images(test_dir)
    if args.max_train:
        train_paths = train_paths[: args.max_train]
    print(f"[{args.name}] train images: {len(train_paths)}  test images: {len(test_paths)}")

    print("  computing pHash ...")
    train_ph = compute_phash(train_paths)
    test_ph = compute_phash(test_paths)

    ph = np.array([min(t - r for r in train_ph) for t in test_ph])
    print(f"  pHash Hamming to nearest train: median={np.median(ph):.1f} "
          f"mean={ph.mean():.1f} min={ph.min()} max={ph.max()}")
    ph_frac = float((ph <= PHASH_NEAR_DUP_BITS).mean())
    print(f"  pHash near-dup fraction (<= {PHASH_NEAR_DUP_BITS} bits): {ph_frac:.4f}")

    cos = None
    hist_phash = np.histogram(ph, bins=np.arange(0, 66))[0]
    if not args.no_resnet:
        try:
            dev, bb, tf = make_embedder()
            print(f"  embedding on {dev} (batch {args.batch_size}) ...")
            tr_e = compute_embeddings(train_paths, dev, bb, tf, args.batch_size)
            te_e = compute_embeddings(test_paths, dev, bb, tf, args.batch_size)
            sims = te_e @ tr_e.T
            cos = sims.max(axis=1)
            cos_frac = float((cos >= COS_NEAR_DUP).mean())
            print(f"  cosine to nearest train: median={np.median(cos):.4f} "
                  f"mean={cos.mean():.4f} min={cos.min():.4f} max={cos.max():.4f}")
            print(f"  cosine near-dup fraction (>= {COS_NEAR_DUP}): {cos_frac:.4f}")
            hist_cos = np.histogram(cos, bins=np.linspace(0.0, 1.0, 101))[0]
        except Exception as exc:
            print(f"  WARNING: ResNet50 embedding failed ({exc}); "
                  f"phash-only analysis", file=sys.stderr)
            cos = None
            hist_cos = None
    else:
        hist_cos = None

    out = {
        "name": args.name,
        "root": args.root,
        "n_train": len(train_paths),
        "n_test": len(test_paths),
        "phash_min_hamming": ph,
        "phash_near_dup_bits": PHASH_NEAR_DUP_BITS,
        "phash_near_dup_frac": ph_frac,
        "phash_hist": hist_phash,
        "cos_near_dup": COS_NEAR_DUP,
        "cos_hist": hist_cos,
    }
    if cos is not None:
        out["cosine_max_sim"] = cos
        out["cos_near_dup_frac"] = cos_frac

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, **out)
    print(f"  saved {args.out}")

    print("  writing JSON sidecar ...")
    side = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in out.items()}
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(side, f, indent=2)

    if tmp_root:
        shutil.rmtree(tmp_root, ignore_errors=True)
        print(f"  removed temp random-split root {tmp_root}")


if __name__ == "__main__":
    main()
