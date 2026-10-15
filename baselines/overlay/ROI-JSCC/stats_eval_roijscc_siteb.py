#!/usr/bin/env python3
"""
stats_eval_roijscc_siteb.py -- Wave 18 dispatch B1: ROI-JSCC zero-shot on
Site B (no retraining -- same Site-A-trained checkpoints, evaluated on Site
B's own 455 video-disjoint test images through Site B's own seeded channel
bank, scored against Site B's own detector/ceiling per RESULTS_OF_RECORD.md
§6b).

Deliberate near-duplicate of stats_eval_roijscc_bank.py (Site A), NOT a
parameterized reuse of it -- same reason channel_bank_siteb.py is a
near-duplicate of channel_bank.py: keeping the two sites' scripts/outputs
textually separate makes a Rule-11 filename collision structurally harder,
not just disciplined-by-convention.

Per dispatch: do NOT divide by AquaSC's prior floor (0.2711/Site-B floor
0.0002/0.0012) -- that floor is our decoder emitting its own learned prior
with no image information at all; an analog JSCC scheme has no equivalent
"prior-only" mode, so there is nothing to floor-normalize against. Report
raw per-image AP (with bootstrap CI) against Site B's own ceiling only
(0.6451 per-image / 0.6044 pooled, RESULTS_OF_RECORD.md §6b) -- never
Site A's 0.7743/0.7292.

Run from AquaSC-D2DB/ :
  <siso-env>/bin/python competitors/ROI-JSCC/stats_eval_roijscc_siteb.py --ckpt <path> --rcpp {32,64,128}
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

_here = os.path.dirname(os.path.abspath(__file__))
_aq = os.path.dirname(os.path.dirname(_here))

if _aq not in sys.path:
    sys.path.insert(0, _aq)
import metrics as M  # noqa: E402
import stats  # noqa: E402
import channel_bank_siteb as CBB  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from torchmetrics.detection import MeanAveragePrecision  # noqa: E402

if _here not in sys.path:
    sys.path.insert(0, _here)
from roijscc_fish_common import (  # noqa: E402
    build_model, BlockRayleighChannel, rate_constants,
    compute_roi_index_from_norm, load_boxes_norm, ROI_RES,
)

IMG = 224
# Site B's own denominators (RESULTS_OF_RECORD.md §6b) -- never Site A's.
PER_IMAGE_CEILING_SITEB = 0.6451
POOLED_CEILING_SITEB = 0.6044
TEST_IMG_DIR = "__AQUASC_DATA__/yolo_fishdet_nb2_vd/test/images"
TEST_LBL_DIR = "__AQUASC_DATA__/yolo_fishdet_nb2_vd/test/labels"
DETECTOR = os.path.join(_aq, "best_siteB.pt")
BANK_PATH = os.path.join(_aq, "channel_bank_siteb.npz")


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
    ap.add_argument("--snr", type=float, nargs="+", default=[0, 10, 20])
    ap.add_argument("--m", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--yolo", default=DETECTOR)
    ap.add_argument("--bank", default=BANK_PATH)
    ap.add_argument("--out_dir", default=os.path.join(_aq, "f1_perimage"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(42)
    np.random.seed(42)

    rc = rate_constants(args.rcpp)
    L_real = rc["L_real"]
    print(f"[eval-siteb] rcpp={args.rcpp} L_real={L_real} uses/px={rc['C']/256.0:.4f}")

    model = build_model(rcpp=args.rcpp, chan_type="Rayleigh", device=device)
    model.channel = BlockRayleighChannel()
    sd = torch.load(args.ckpt, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)
    model.eval()

    bank = CBB.ChannelBankSiteB(args.bank)
    ds = TestDataset(TEST_IMG_DIR)
    ds_names = ds.names
    assert len(ds) == 455 and ds_names == bank.image_names, \
        "Site B test image order must match channel_bank_siteb.image_names"
    assert L_real <= bank.L_real, f"L_real={L_real} exceeds bank L_real={bank.L_real}"

    gt_cache, roi_cache = [], []
    for idx in range(len(ds)):
        stem = os.path.splitext(ds_names[idx])[0]
        boxes_norm = load_boxes_norm(os.path.join(TEST_LBL_DIR, stem + ".txt"))
        roi_cache.append(compute_roi_index_from_norm(boxes_norm, mode=args.roi_mode))
        gt_cache.append(boxes_norm * IMG)

    yolo = YOLO(args.yolo)
    yolo.to(device)
    if hasattr(yolo, "fuse") and callable(yolo.fuse):
        yolo.fuse()
    yolo.eval()

    agg_summary = {}
    with torch.no_grad():
        for snr in args.snr:
            in_bank = float(snr) in bank.snr_db
            assert in_bank, f"snr={snr} not in Site B bank grid {bank.snr_db}"
            per_image_sum = np.zeros(len(ds), dtype=np.float64)
            pooled_map50s = []
            for m in args.m:
                ap_list = np.zeros(len(ds), dtype=np.float32)
                pooled_metric = MeanAveragePrecision(iou_type="bbox").to(device)
                for idx in range(len(ds)):
                    orig = ds[idx].unsqueeze(0).to(device)
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
                print(f"[eval-siteb] snr={snr:5.1f} m={m} mean_per_image_ap50={ap_list.mean():.4f} "
                      f"pooled_map50={pooled_map50_m:.4f}", flush=True)
            per_image_mean = (per_image_sum / len(args.m)).astype(np.float32)
            pooled_map50_mean = float(np.mean(pooled_map50s))
            ci = stats.bootstrap_ci(per_image_mean)
            tag = f"snr{snr:g}".replace("-", "m").replace(".", "p")
            # Rule 11: "siteb" in both the filename stem and the tag keeps this
            # disjoint from Site A's roijscc_rcpp{rcpp}_{tag}.npy pattern.
            out_path = os.path.join(args.out_dir, f"roijscc_siteb_rcpp{args.rcpp}_{tag}.npy")
            np.save(out_path, per_image_mean)
            agg_summary[str(snr)] = dict(
                mean_per_image_ap50=float(per_image_mean.mean()),
                ci_low=ci["ci_low"], ci_high=ci["ci_high"],
                retention_per_image_siteb=float(per_image_mean.mean() / PER_IMAGE_CEILING_SITEB),
                pooled_map50=pooled_map50_mean,
                retention_pooled_siteb=pooled_map50_mean / POOLED_CEILING_SITEB,
                per_image_ceiling_siteb=PER_IMAGE_CEILING_SITEB,
                pooled_ceiling_siteb=POOLED_CEILING_SITEB,
                in_bank=in_bank, M=len(args.m), out_path=out_path,
            )
            print(f"[eval-siteb] SNR {snr:5.1f} dB DONE: mean_per_image_ap50={per_image_mean.mean():.4f} "
                  f"[{ci['ci_low']:.4f},{ci['ci_high']:.4f}] "
                  f"retention_per_image(Site B)={per_image_mean.mean()/PER_IMAGE_CEILING_SITEB:.4f} "
                  f"pooled_map50={pooled_map50_mean:.4f} -> {out_path}")

    out = {
        "provenance": {
            "checkpoint": args.ckpt,
            "rcpp": args.rcpp,
            "uses_per_px": rc["C"] / 256.0,
            "site": "B (zero-shot, no retraining -- Site-A-trained checkpoint)",
            "training_epochs_roijscc": 100,
            "training_epochs_aquasc": 300,
            "epoch_budget_note": ("ROI-JSCC trained for 100 epochs vs AquaSC's 300 "
                                  "-- this asymmetry must be stated wherever this number "
                                  "is used in a matched-rate comparison (plan v5.9, J1); "
                                  "carried into every cell of this Site B table too."),
            "channel_bank_sha256_note": "channel_bank_siteb.npz, subseed base 55,000,000 (disjoint from Site A's 42,000,000)",
            "conf_thres": 0.001,
            "per_image_ceiling_siteb": PER_IMAGE_CEILING_SITEB,
            "pooled_ceiling_siteb": POOLED_CEILING_SITEB,
            "note_no_prior_floor_comparison": (
                "Do not divide by AquaSC's prior floor (Site A 0.2711 or Site B "
                "0.0002/0.0012) -- that floor is AquaSC's decoder emitting its "
                "learned prior with zero image information; ROI-JSCC (analog "
                "JSCC) has no equivalent no-information mode to floor-normalize "
                "against. Report raw AP against Site B's own ceiling only."),
        },
        "by_snr": agg_summary,
    }
    out_json = os.path.join(args.out_dir, f"roijscc_siteb_summary_rcpp{args.rcpp}.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[eval-siteb] DONE -> {out_json}")


if __name__ == "__main__":
    main()
