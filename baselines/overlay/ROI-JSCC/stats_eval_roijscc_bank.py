#!/usr/bin/env python3
"""
stats_eval_roijscc_bank.py -- Track F1/F2 (plan 16.9): per-image AP@0.5 for
ROI-JSCC through the seeded channel bank, for paired bootstrap/Wilcoxon
against AquaSC's BPSK r1/3 and QPSK r1/2 rows (F1), and ROI-JSCC's own
behavior at -5/-3/-1 dB (F2).

F1 SNRs {-5,0,5,10,15,20} use channel_bank.npz verbatim (SHA 369d33f6...).
F2 SNRs {-3,-1} are NOT in the committed bank -- they are generated on the fly
with the SAME documented derivation (channel_bank.py module docstring):
    subseed(i, snr_idx, m) = 42_000_000 + snr_idx*1_000_000 + m*1_000 + i
    rng = np.random.default_rng(subseed); h = (rng.standard_normal()+1j*rng.standard_normal())/sqrt(2)
    noise = rng.standard_normal(L)
using new, disjoint snr_idx values (6 for -3dB, 7 for -1dB) so they never
collide with the committed bank's indices 0-5. This is disclosed explicitly:
these two SNRs are NOT paired against an AquaSC bank cell (no such cell
exists), they are ROI-JSCC's own measurement at that SNR.

Saves, per SNR: per-image AP@0.5 averaged over M realizations (npy, 432,)
and the pooled/aggregate mAP (for cross-check against roi_jscc_eval.csv's
existing bank-based numbers at 0/5/10/15/20 dB).

Run from AquaSC-D2DB/ (so `import stats` / `import channel_bank` resolve).
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
_aq = os.path.dirname(os.path.dirname(_here))

if _aq not in sys.path:
    sys.path.insert(0, _aq)
import metrics as M  # noqa: E402
import stats  # noqa: E402
import channel_bank as CB  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from torchmetrics.detection import MeanAveragePrecision  # noqa: E402

if _here not in sys.path:
    sys.path.insert(0, _here)
from roijscc_fish_common import (  # noqa: E402
    build_model, BlockRayleighChannel, rate_constants,
    compute_roi_index_from_norm, load_boxes_norm, ROI_RES,
)

IMG = 224
# G1 denominator rule: this script's own metric is PER-IMAGE mean AP, so its
# retention field must divide by the PER-IMAGE ceiling (0.7743), not the
# pooled/micro-averaged ceiling (0.7292) -- the exact bug the standing rule
# ("pooled / pooled, per-image / per-image, never mixed") exists to prevent.
CEILING = 0.7743
TEST_IMG_DIR = "__AQUASC_DATA__/yolo_fish_dataset_root/test/images"
TEST_LBL_DIR = "__AQUASC_DATA__/yolo_fish_dataset_root/test/labels"

# F2 extension: SNRs not in the committed bank, generated ad hoc with new
# disjoint snr_idx values (disclosed in the module docstring above).
_EXT_SNR_IDX = {-3.0: 6, -1.0: 7}
_SUBSEED_BASE = 42_000_000
_SNR_STRIDE = 1_000_000
_M_STRIDE = 1_000


def ext_h_and_noise(image_idx: int, snr_db: float, m: int, L: int):
    s_idx = _EXT_SNR_IDX[float(snr_db)]
    ss = _SUBSEED_BASE + s_idx * _SNR_STRIDE + int(m) * _M_STRIDE + int(image_idx)
    rng = np.random.default_rng(ss)
    hr, hi = rng.standard_normal(), rng.standard_normal()
    h = (hr + 1j * hi) / np.sqrt(2.0)
    noise = rng.standard_normal(L)
    return h, noise


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
    ap.add_argument("--rcpp", type=int, default=32)
    ap.add_argument("--roi_mode", type=str, default="largest_box")
    ap.add_argument("--snr", type=float, nargs="+", default=[-5, 0, 5, 10, 15, 20])
    ap.add_argument("--m", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--yolo", default=os.path.join(_aq, "best.pt"))
    ap.add_argument("--bank", default=os.path.join(_aq, "channel_bank.npz"))
    ap.add_argument("--out_dir", default=os.path.join(_aq, "f1_perimage"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(42)
    np.random.seed(42)

    rc = rate_constants(args.rcpp)
    L_real = rc["L_real"]
    print(f"[eval] rcpp={args.rcpp} L_real={L_real} uses/px={rc['C']/256.0:.4f}")

    model = build_model(rcpp=args.rcpp, chan_type="Rayleigh", device=device)
    model.channel = BlockRayleighChannel()
    sd = torch.load(args.ckpt, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)
    model.eval()

    bank = CB.ChannelBank(args.bank)
    ds = TestDataset(TEST_IMG_DIR)
    ds_names = ds.names
    assert len(ds) == 432 and ds_names == bank.image_names, \
        "test image order must match bank.image_names"

    gt_cache, roi_cache = [], []
    for idx in range(len(ds)):
        stem = os.path.splitext(ds_names[idx])[0]
        boxes_norm = load_boxes_norm(os.path.join(TEST_LBL_DIR, stem + ".txt"))
        roi_cache.append(compute_roi_index_from_norm(boxes_norm, mode=args.roi_mode))
        gt_cache.append(boxes_norm * IMG)  # xyxy in pixel units, (K,4)

    yolo = YOLO(args.yolo)
    yolo.to(device)
    if hasattr(yolo, "fuse") and callable(yolo.fuse):
        yolo.fuse()
    yolo.eval()

    agg_summary = {}
    with torch.no_grad():
        for snr in args.snr:
            in_bank = float(snr) in bank.snr_db
            per_image_sum = np.zeros(len(ds), dtype=np.float64)
            pooled_map50s = []
            for m in args.m:
                ap_list = np.zeros(len(ds), dtype=np.float32)
                pooled_metric = MeanAveragePrecision(iou_type="bbox").to(device)
                for idx in range(len(ds)):
                    orig = ds[idx].unsqueeze(0).to(device)
                    if in_bank:
                        h = bank.get_h(idx, snr, m)
                        noise_real = bank.get_noise_real(idx, snr, m, L_real)
                    else:
                        h, noise_real = ext_h_and_noise(idx, snr, m, L_real)
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
                    preds = M.yolo_predict(yolo, recon, conf=0.001, iou=0.5, device=device)[0]
                    ap_list[idx] = stats.per_image_ap50(
                        preds["boxes"].detach().cpu().numpy(),
                        preds["scores"].detach().cpu().numpy(),
                        gt_cache[idx], iou_thr=0.5)
                    gt_boxes_t = torch.as_tensor(gt_cache[idx], dtype=torch.float32, device=device)
                    pooled_metric.update(
                        [{"boxes": preds["boxes"], "scores": preds["scores"],
                          "labels": torch.zeros(preds["boxes"].shape[0], dtype=torch.int64, device=device)}],
                        [{"boxes": gt_boxes_t,
                          "labels": torch.zeros(gt_boxes_t.shape[0], dtype=torch.int64, device=device)}])
                per_image_sum += ap_list
                pooled_map50_m = float(pooled_metric.compute()["map_50"])
                pooled_map50s.append(pooled_map50_m)
                print(f"[eval] snr={snr:5.1f} m={m} mean_per_image_ap50={ap_list.mean():.4f} "
                      f"pooled_map50={pooled_map50_m:.4f}", flush=True)
            per_image_mean = (per_image_sum / len(args.m)).astype(np.float32)
            pooled_map50_mean = float(np.mean(pooled_map50s))
            tag = f"snr{snr:g}".replace("-", "m").replace(".", "p")
            out_path = os.path.join(args.out_dir, f"roijscc_rcpp{args.rcpp}_{tag}.npy")
            np.save(out_path, per_image_mean)
            agg_summary[str(snr)] = dict(
                mean_per_image_ap50=float(per_image_mean.mean()),
                retention_per_image=float(per_image_mean.mean() / CEILING),
                pooled_map50=pooled_map50_mean,
                pooled_map50_ceiling=0.7292211,
                retention_pooled=pooled_map50_mean / 0.7292211,
                in_bank=in_bank, M=len(args.m), out_path=out_path,
            )
            print(f"[eval] SNR {snr:5.1f} dB DONE: mean_per_image_ap50={per_image_mean.mean():.4f} "
                  f"retention_per_image={per_image_mean.mean()/CEILING:.4f} "
                  f"pooled_map50={pooled_map50_mean:.4f} in_bank={in_bank} -> {out_path}")

    out = {
        "provenance": {
            "checkpoint": args.ckpt,
            "rcpp": args.rcpp,
            "uses_per_px": rc["C"] / 256.0,
            "training_epochs_roijscc": 100,
            "training_epochs_aquasc": 300,
            "epoch_budget_note": ("ROI-JSCC trained for 100 epochs vs AquaSC's 300 "
                                  "-- this asymmetry must be stated wherever this number "
                                  "is used in a matched-rate comparison (plan v5.9, J1)."),
            "channel_bank_sha256": "369d33f6ceeddc0190030d790077f16ca2284fa8aa9fc1dc5c1a58d219c9cddb",
            "conf_thres": 0.001,
            "per_image_ceiling": CEILING,
            "pooled_ceiling": 0.7292211,
        },
        "by_snr": agg_summary,
    }
    out_json = os.path.join(args.out_dir, f"roijscc_summary_rcpp{args.rcpp}.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[eval] DONE -> {out_json}")


if __name__ == "__main__":
    main()
