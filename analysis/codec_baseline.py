import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repo root (core modules)
# analysis/codec_baseline.py — the ONE feasible codec baseline (plan §9.2, R2 #1).
#
# Replaces the infeasible JPEG@0.08 BPP operating point. JPEG is infeasible
# below ~0.17 BPP because a standards-compliant baseline JFIF file needs ~625 B
# of headers before any image data at 224x224 (0.08 BPP = 502 B total budget).
# WebP is the chosen feasible codec: encode the test split at 224x224 with PIL
# WebP, sweep quality to hit ~0.20 BPP, then score the decoded images with the
# AquaSC detector (best.pt) at conf 0.001 through metrics.py.
#
# Usage (run from AquaSC-D2DB/):
#   # 1) find the WebP quality whose mean BPP ~ 0.20
#   python3 analysis/codec_baseline.py --sweep --sample 64 --target_bpp 0.20
#   # 2) full encode + detector score at that quality
#   python3 analysis/codec_baseline.py --quality Q --out_csv results/codec/codec_webp_eval.csv
#
# CPU is sufficient (PIL encode + YOLOv8s @224 on 432 images).

import argparse
import csv
import io
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics as M  # noqa: E402

IMG = 224
DATA_ROOT = "__AQUASC_DATA__/yolo_fish_dataset_root/test"
CANONICAL_CEILING = 0.7292  # plan §0.2: best.pt conf 0.001 on clean test split

# Site B support (2026-09-17): additive CLI overrides, defaults unchanged so
# every existing Site A call (and results/codec/codec_webp_eval.csv) is untouched. Site B's
# own pooled ceiling is 0.6044 (RESULTS_OF_RECORD.md section 6b) -- never
# Site A's 0.7292 -- pass --canonical_ceiling 0.6044 alongside --data_root/
# --yolo/--out_csv for a Site B run.


def list_images(data_root):
    d = os.path.join(data_root, "images")
    return sorted(f for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".jpeg")))


def load_boxes(label_path):
    boxes, labels = [], []
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                p = line.strip().split()
                if len(p) != 5:
                    continue
                try:
                    cls = int(p[0])
                    cx, cy, w, h = map(float, p[1:])
                    x1, y1 = np.clip(cx - w / 2, 0, 1), np.clip(cy - h / 2, 0, 1)
                    x2, y2 = np.clip(cx + w / 2, 0, 1), np.clip(cy + h / 2, 0, 1)
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2]); labels.append(cls)
                except ValueError:
                    continue
    if boxes:
        b = torch.as_tensor(boxes, dtype=torch.float32)
        b[:, [0, 2]] *= IMG; b[:, [1, 3]] *= IMG
        return b, torch.as_tensor(labels, dtype=torch.int64)
    return torch.empty((0, 4)), torch.empty(0, dtype=torch.int64)


def load_resized(path):
    """Load + bicubic-resize to 224x224 (the AquaSC pipeline; test video '19' is
    natively 224x224 so this is a no-op there). Returns PIL RGB image."""
    im = Image.open(path).convert("RGB")
    if im.size != (IMG, IMG):
        im = im.resize((IMG, IMG), Image.BICUBIC)
    return im


def encode_webp(pil_img, quality):
    buf = io.BytesIO()
    pil_img.save(buf, "WEBP", quality=quality, method=4)
    return buf.getvalue()


def bpp_of(data, n_pix=IMG * IMG):
    return len(data) * 8.0 / n_pix


def sweep(data_root=DATA_ROOT, target_bpp=0.20, sample=64,
          qualities=(1, 2, 3, 4, 5, 8, 10, 12, 15, 20, 30, 50)):
    names = list_images(data_root)
    if sample and sample < len(names):
        rng = np.random.RandomState(42)
        names = list(rng.choice(names, sample, replace=False))
    print(f"sweep over {len(names)} images")
    rows = []
    for q in qualities:
        bpps = []
        for n in names:
            im = load_resized(os.path.join(data_root, "images", n))
            bpps.append(bpp_of(encode_webp(im, q)))
        mean_bpp = float(np.mean(bpps))
        rows.append((q, mean_bpp))
        print(f"  quality={q:3d}  mean_BPP={mean_bpp:.4f}")
    best = min(rows, key=lambda r: abs(r[1] - target_bpp))
    print(f"closest quality to {target_bpp:.3f} BPP: q={best[0]} -> {best[1]:.4f} BPP")
    return best[0]


def score(quality, out_csv, data_root=DATA_ROOT, yolo_weights="best.pt",
          canonical_ceiling=CANONICAL_CEILING):
    names = list_images(data_root)
    to_tensor = transforms.ToTensor()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    yolo = YOLO(yolo_weights)
    yolo.model.eval()
    acc = M.MetricsAccumulator(device=str(device), conf_thres=0.001, iou_thres=0.5)

    bpps = []
    with torch.no_grad():
        for n in names:
            stem = os.path.splitext(n)[0]
            im = load_resized(os.path.join(data_root, "images", n))
            data = encode_webp(im, quality)
            bpps.append(bpp_of(data))
            recon = Image.open(io.BytesIO(data)).convert("RGB")
            recon_t = to_tensor(recon).unsqueeze(0)          # [1,3,224,224] in [0,1]
            orig_t = to_tensor(im).unsqueeze(0)
            boxes, labels = load_boxes(os.path.join(data_root, "labels", stem + ".txt"))
            gt = [{"boxes": boxes, "labels": labels}]
            orig_d = orig_t.to(device); recon_d = recon_t.to(device)
            preds_orig = M.yolo_predict(yolo, orig_d, conf=0.001, iou=0.5, device=device)
            preds_recon = M.yolo_predict(yolo, recon_d, conf=0.001, iou=0.5, device=device)
            acc.accumulate(orig_d, recon_d, gt, preds_orig, preds_recon)

    res = acc.compute()
    mean_bpp = float(np.mean(bpps))
    res["codec"] = "webp"
    res["quality"] = quality
    res["mean_bpp"] = mean_bpp
    res["n_images"] = len(names)
    res["retention_vs_canonical_07292"] = res["map_50_recon"] / canonical_ceiling
    print(f"WebP q={quality}  mean_BPP={mean_bpp:.4f}  "
          f"mAP@0.5_recon={res['map_50_recon']:.4f}  "
          f"mAP@0.5_orig={res['map_50_orig']:.4f}  "
          f"retention_vs_{canonical_ceiling}={res['retention_vs_canonical_07292']:.4f}  "
          f"PSNR={res['psnr']:.3f} dB")

    keep = ["codec", "quality", "mean_bpp", "n_images", "n_gt_boxes",
            "map_50_orig", "map_50_recon", "retention_vs_canonical_07292",
            "map_orig", "map_recon", "precision_orig", "precision_recon",
            "recall_orig", "recall_recon", "f1_orig", "f1_recon",
            "psnr", "ssim", "in_box_psnr", "in_box_ssim"]
    row = {k: (round(float(res[k]), 6) if isinstance(res.get(k), (int, float)) else res.get(k))
           for k in keep if k in res}
    new = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keep)
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"WROTE {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--quality", type=int, default=None)
    ap.add_argument("--target_bpp", type=float, default=0.20)
    ap.add_argument("--sample", type=int, default=64)
    ap.add_argument("--out_csv", default="results/codec/codec_webp_eval.csv")
    ap.add_argument("--data_root", default=DATA_ROOT,
                    help="Site B (2026-09-17): pass the Site B test split root "
                         "(.../yolo_fishdet_nb2_vd/test) to evaluate there instead.")
    ap.add_argument("--yolo", default="best.pt",
                    help="Site B: pass best_siteB.pt.")
    ap.add_argument("--canonical_ceiling", type=float, default=CANONICAL_CEILING,
                    help="Site B: pass 0.6044 (pooled ceiling, section 6b) -- never Site A's 0.7292.")
    args = ap.parse_args()

    if args.sweep:
        sweep(data_root=args.data_root, target_bpp=args.target_bpp, sample=args.sample)
        return
    if args.quality is None:
        raise SystemExit("provide --quality Q (or run --sweep first)")
    score(args.quality, args.out_csv, data_root=args.data_root,
          yolo_weights=args.yolo, canonical_ceiling=args.canonical_ceiling)


if __name__ == "__main__":
    main()
