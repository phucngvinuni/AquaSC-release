#!/usr/bin/env python3
"""
eval_roijscc_fish.py — Wave 9b ROIJSCC agent: evaluate the trained ROI-JSCC
checkpoint on the honest test split (432) through the SAME seeded channel bank
(channel_bank.npz, Wave 8), scoring mAP@0.5 (best.pt, conf 0.001, metrics.py)
+ PSNR/SSIM + retention vs 0.7292.

Run from AquaSC-D2DB/ so `import metrics` / `import channel_bank` resolve to the
AquaSC harness. This script lives in competitors/ROI-JSCC/.

Channel: BlockRayleighChannel driven by bank.get_h() / bank.get_noise_real()
(block-fading Rayleigh + equalization, matching analysis/stats_eval_jscc.py / AquaSC).

Image order = sorted test filenames = bank.image_names (asserted).

Usage (from AquaSC-D2DB/):
  python3 competitors/ROI-JSCC/eval_roijscc_fish.py \
      --ckpt competitors/ROI-JSCC/saved_models/ROIIT_FISH_....pt \
      --rcpp 32 --roi_mode largest_box --snr 0 5 10 15 20 --m 0..9 \
      --out_csv competitors/ROI-JSCC/roi_jscc_eval.csv
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

_here = os.path.dirname(os.path.abspath(__file__))
_aq = os.path.dirname(os.path.dirname(_here))   # AquaSC-D2DB

# Import the AquaSC harness FIRST (metrics.py does `import utils`), so that the
# AquaSC-D2DB `utils`/`model` names resolve to the AquaSC repo (not ROI-JSCC's
# utils.py, and not AquaSC-D2DB/model.py).  ROI-JSCC's package is added to
# sys.path only afterwards, for the ROI-JSCC model import.
if _aq not in sys.path:
    sys.path.insert(0, _aq)
import metrics as M  # noqa: E402
import channel_bank as CB  # noqa: E402
from ultralytics import YOLO  # noqa: E402

if _here not in sys.path:
    sys.path.insert(0, _here)
from roijscc_fish_common import (  # noqa: E402
    build_model, BlockRayleighChannel, rate_constants,
    compute_roi_index_from_norm, load_boxes_norm, ROI_RES,
)

IMG = 224
CEILING = 0.7292
TEST_IMG_DIR = "__AQUASC_DATA__/yolo_fish_dataset_root/test/images"
TEST_LBL_DIR = "__AQUASC_DATA__/yolo_fish_dataset_root/test/labels"


class TestDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.names = sorted(f for f in os.listdir(img_dir)
                            if f.lower().endswith((".png", ".jpg", ".jpeg")))
        self.t = transforms.ToTensor()

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        return self.t(Image.open(os.path.join(self.img_dir, self.names[idx])).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rcpp", type=int, default=32, choices=[12, 16, 24, 32, 64, 128])
    ap.add_argument("--roi_mode", type=str, default="largest_box",
                    choices=["largest_box", "shoal_bbox"])
    ap.add_argument("--snr", type=float, nargs="+", default=[0, 5, 10, 15, 20])
    ap.add_argument("--m", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--yolo", default=os.path.join(_aq, "best.pt"))
    ap.add_argument("--bank", default=os.path.join(_aq, "channel_bank.npz"))
    ap.add_argument("--out_csv", default=os.path.join(_here, "roi_jscc_eval.csv"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(42)
    np.random.seed(42)

    rc = rate_constants(args.rcpp)
    L_real, N = rc["L_real"], rc["N_complex"]
    print(f"[eval] rcpp={args.rcpp} L_real={L_real} N_complex={N} "
          f"uses/px={rc['C']/256.0:.4f} roi_mode={args.roi_mode}")

    model = build_model(rcpp=args.rcpp, chan_type="Rayleigh", device=device)
    model.channel = BlockRayleighChannel()
    sd = torch.load(args.ckpt, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[eval] load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
    if missing or unexpected:
        print(f"  missing={missing[:5]} unexpected={unexpected[:5]}")
    model.eval()

    bank = CB.ChannelBank(args.bank)
    ds = TestDataset(TEST_IMG_DIR)
    ds_names = ds.names
    assert len(ds) == 432 and ds_names == bank.image_names, \
        "test image order must match bank.image_names"

    # GT boxes + ROI indices (channel-independent)
    gt_cache = []
    roi_cache = []
    for idx in range(len(ds)):
        stem = os.path.splitext(ds_names[idx])[0]
        boxes_norm = load_boxes_norm(os.path.join(TEST_LBL_DIR, stem + ".txt"))
        roi_cache.append(compute_roi_index_from_norm(boxes_norm, mode=args.roi_mode))
        boxes, labels = [], []
        for b in boxes_norm:
            boxes.append([b[0] * IMG, b[1] * IMG, b[2] * IMG, b[3] * IMG])
            labels.append(0)
        if boxes:
            gt_cache.append({"boxes": torch.as_tensor(boxes, dtype=torch.float32),
                             "labels": torch.as_tensor(labels, dtype=torch.int64)})
        else:
            gt_cache.append({"boxes": torch.empty((0, 4)),
                             "labels": torch.empty(0, dtype=torch.int64)})

    yolo = YOLO(args.yolo)
    yolo.to(device)
    if hasattr(yolo, "fuse") and callable(yolo.fuse):
        yolo.fuse()
    yolo.eval()

    # orig predictions once (channel-independent) -> sanity mAP should be 0.7292
    print("[eval] running YOLO on original images (once)...")
    preds_orig_cache = []
    with torch.no_grad():
        for idx in range(len(ds)):
            orig = ds[idx].unsqueeze(0).to(device)
            preds_orig_cache.append(M.yolo_predict(yolo, orig, conf=0.001, iou=0.5, device=device)[0])

    rows = []
    with torch.no_grad():
        for snr in args.snr:
            for m in args.m:
                acc = M.MetricsAccumulator(device=str(device), conf_thres=0.001,
                                           iou_thres=0.5, compute_lpips=False)
                for idx in range(len(ds)):
                    orig = ds[idx].unsqueeze(0).to(device)
                    # ROI-JSCC forward with bank channel
                    h = bank.get_h(idx, snr, m)
                    noise_real = bank.get_noise_real(idx, snr, m, L_real)
                    model.channel.h = torch.tensor(h, dtype=torch.complex64, device=device).view(1, 1)
                    model.channel.noise_real = torch.from_numpy(
                        np.asarray(noise_real, dtype=np.float32)).to(device).view(1, L_real)
                    roi = roi_cache[idx]
                    img256 = torch.nn.functional.interpolate(
                        orig, size=(ROI_RES, ROI_RES), mode="bicubic", align_corners=False)
                    img_norm = img256 * 2.0 - 1.0
                    out = model(img_norm, ROI_Index=roi, SNR_info=snr)
                    recon256 = torch.clamp((out + 1.0) / 2.0, 0.0, 1.0)
                    recon = torch.nn.functional.interpolate(
                        recon256, size=(IMG, IMG), mode="bicubic", align_corners=False)
                    recon = torch.clamp(recon, 0.0, 1.0)
                    preds_recon = M.yolo_predict(yolo, recon, conf=0.001, iou=0.5, device=device)
                    gt = [{"boxes": gt_cache[idx]["boxes"], "labels": gt_cache[idx]["labels"]}]
                    acc.accumulate(orig, recon, gt, [preds_orig_cache[idx]], preds_recon)
                res = acc.compute()
                row = {
                    "snr_db": snr, "m": m, "roi_mode": args.roi_mode,
                    "map_50_recon": res["map_50_recon"],
                    "map_50_orig": res["map_50_orig"],
                    "retention_vs_07292": res["map_50_recon"] / CEILING,
                    "psnr": res["psnr"], "ssim": res["ssim"],
                    "in_box_psnr": res["in_box_psnr"], "in_box_ssim": res["in_box_ssim"],
                }
                rows.append(row)
                print(f"[eval] snr={snr:5.1f} m={m} map50={row['map_50_recon']:.4f} "
                      f"psnr={row['psnr']:.3f} ssim={row['ssim']:.4f} "
                      f"ret={row['retention_vs_07292']:.4f}", flush=True)

    with open(args.out_csv, "w") as f:
        keys = list(rows[0].keys())
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"[eval] wrote {args.out_csv} ({len(rows)} rows)")

    # aggregate (mean over M) per SNR
    agg = {}
    for snr in args.snr:
        sub = [r for r in rows if r["snr_db"] == snr]
        agg[str(snr)] = {
            "map_50_recon_mean": float(np.mean([r["map_50_recon"] for r in sub])),
            "psnr_mean": float(np.mean([r["psnr"] for r in sub])),
            "ssim_mean": float(np.mean([r["ssim"] for r in sub])),
            "retention_mean": float(np.mean([r["retention_vs_07292"] for r in sub])),
        }
        print(f"[eval] SNR {snr:5.1f} dB: map50={agg[str(snr)]['map_50_recon_mean']:.4f} "
              f"psnr={agg[str(snr)]['psnr_mean']:.3f} ssim={agg[str(snr)]['ssim_mean']:.4f} "
              f"retention={agg[str(snr)]['retention_mean']:.4f}")
    with open(args.out_csv.replace(".csv", "_agg.json"), "w") as f:
        json.dump(agg, f, indent=2)
    print("[eval] DONE")


if __name__ == "__main__":
    main()
