#!/usr/bin/env python3
"""
stats_eval_jscc_bank.py -- Wave 18 dispatch B2: bring JSCC-CNN to the same
standard as every other baseline. Previously: one point (10dB, Site A,
pooled, no per-image arrays). This script adds: full 6-point SNR sweep
(-5,0,5,10,15,20dB) on Site A through the SAME seeded channel bank every
other method replays (channel_bank.py, per-image, M=10, per-image AP + CI),
plus a zero-shot pass on Site B (channel_bank_siteb.py, its own ceiling).

JSCC-CNN's own channel.py generates its Rayleigh h/noise internally (no bank
hook), so this script bypasses model.forward()/channel_layer entirely: it
calls encoder(x) and decoder(.) directly and reproduces channel.py's exact
Rayleigh-fading + AWGN + equalization math (verified by direct code read,
2026-09-13) with the bank's h and standard-normal noise draws substituted
for the internal torch.randn calls. Per channel_bank.py's own module
docstring: "JSCC-CNN c=2 needs 2*c*H*W = 2*2*56*56 = 12544 real draws
(exactly L)" -- c=1 uses the first half of the same per-cell array.

Run from AquaSC-D2DB/competitors/jscc/:
  <siso-env>/bin/python stats_eval_jscc_bank.py --site A --variant std --c 1
  <siso-env>/bin/python stats_eval_jscc_bank.py --site B --variant bbw --c 2
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from collections import OrderedDict

_here = os.path.dirname(os.path.abspath(__file__))
_aq = os.path.dirname(os.path.dirname(_here))
if _aq not in sys.path:
    sys.path.insert(0, _aq)
import metrics as M  # noqa: E402
import stats  # noqa: E402
import channel_bank as CBA  # noqa: E402
import channel_bank_siteb as CBB  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from torchmetrics.detection import MeanAveragePrecision  # noqa: E402

# Unconditional (not "if _here not in sys.path"): AquaSC-D2DB/model.py and
# this dir's model.py share a name, and _aq was already inserted at index 0
# above -- a guarded insert here is a no-op when _here == sys.path[0] (the
# script's own auto-added directory), silently importing the WRONG "model".
sys.path.insert(0, _here)
from model import DeepJSCC  # noqa: E402

IMG = 224

SITE_CFG = {
    "A": dict(
        img_dir="__AQUASC_DATA__/yolo_fish_dataset_root/test/images",
        lbl_dir="__AQUASC_DATA__/yolo_fish_dataset_root/test/labels",
        yolo=os.path.join(_aq, "best.pt"),
        bank_path=os.path.join(_aq, "channel_bank.npz"),
        bank_mod=CBA, bank_cls_kw=dict(),
        n_images=432,
        per_image_ceiling=0.7743, pooled_ceiling=0.7292211,
        default_snrs=[-5, 0, 5, 10, 15, 20],
    ),
    "B": dict(
        img_dir="__AQUASC_DATA__/yolo_fishdet_nb2_vd/test/images",
        lbl_dir="__AQUASC_DATA__/yolo_fishdet_nb2_vd/test/labels",
        yolo=os.path.join(_aq, "best_siteB.pt"),
        bank_path=os.path.join(_aq, "channel_bank_siteb.npz"),
        bank_mod=CBB, bank_cls_kw=dict(),
        n_images=455,
        per_image_ceiling=0.6451, pooled_ceiling=0.6044,
        default_snrs=[0, 10, 20],
    ),
}

CKPT_RATIO = {1: "0.0417", 2: "0.0833"}


def load_boxes_norm(label_path):
    if not os.path.exists(label_path):
        return np.zeros((0, 4), dtype=np.float32)
    rows = []
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            _, xc, yc, w, h = parts[:5]
            xc, yc, w, h = float(xc), float(yc), float(w), float(h)
            rows.append([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])
    return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 4), dtype=np.float32)


class ImageOnly(Dataset):
    def __init__(self, image_dir):
        self.names = sorted(f for f in os.listdir(image_dir)
                            if f.lower().endswith((".png", ".jpg", ".jpeg")))
        self.image_dir = image_dir
        self.t = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.ToTensor()])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        return self.t(Image.open(os.path.join(self.image_dir, self.names[idx])).convert("RGB"))


def bank_rayleigh_forward(z_hat, h_complex, noise_std_normal_flat, snr_db):
    """Exact reproduction of competitors/jscc/channel.py's Channel.forward
    Rayleigh branch (verified by direct read, 2026-09-13), with h and the
    AWGN draw substituted for the bank's seeded values instead of internal
    torch.randn calls. z_hat: [1, 2c, H, W]. h_complex: python complex.
    noise_std_normal_flat: flat standard-normal array of length 2*c*H*W.
    """
    B, C2, H, W = z_hat.shape
    c = C2 // 2
    device = z_hat.device
    noise_std = math.sqrt(1.0 / (10 ** (float(snr_db) / 10.0)))
    noise = torch.from_numpy(noise_std_normal_flat.astype(np.float32)).to(device).reshape(1, C2, H, W) * noise_std

    h_t = torch.tensor(h_complex, dtype=torch.complex64, device=device)

    z_real = z_hat[:, :c]
    z_imag = z_hat[:, c:]
    z_complex = torch.complex(z_real, z_imag)
    z_faded = z_complex * h_t
    z_processed = torch.cat([z_faded.real, z_faded.imag], dim=1)

    z_noisy = z_processed + noise
    zn_real = z_noisy[:, :c]
    zn_imag = z_noisy[:, c:]
    zn_complex = torch.complex(zn_real, zn_imag)
    z_eq = zn_complex / (h_t + 1e-9)
    return torch.cat([z_eq.real, z_eq.imag], dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", choices=["A", "B"], required=True)
    ap.add_argument("--variant", choices=["std", "bbw"], required=True)
    ap.add_argument("--c", type=int, choices=[1, 2], required=True)
    ap.add_argument("--ckpt_name", default="best_psnr")
    ap.add_argument("--ckpt_root", default=None,
                    help="Site B fine-tune (2026-09-16): override the checkpoint root "
                         "(default: output_jscc_baselinea/, the Site-A-trained zero-shot "
                         "checkpoints). Pass the fine-tune output dir, e.g. "
                         "output_jscc_siteb_finetune/, to evaluate a fine-tuned checkpoint "
                         "-- combine with a distinct --out_dir to avoid overwriting the "
                         "zero-shot per-image .npy/.json files (Rule 11).")
    ap.add_argument("--snr", type=float, nargs="+", default=None)
    ap.add_argument("--m", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--out_dir", default=os.path.join(_aq, "f1_perimage"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = SITE_CFG[args.site]
    snrs = args.snr if args.snr is not None else cfg["default_snrs"]
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(42)
    np.random.seed(42)

    ratio = CKPT_RATIO[args.c]
    ckpt_root = args.ckpt_root or os.path.join(_here, "output_jscc_baselinea")
    ckpt_path = os.path.join(ckpt_root,
                             f"{args.variant}_c{args.c}_snr10_ratio{ratio}",
                             "checkpoints", f"{args.ckpt_name}.pkl")
    print(f"[eval-jscc] site={args.site} variant={args.variant} c={args.c} ckpt={ckpt_path}")

    model = DeepJSCC(c=args.c, channel_type="Rayleigh", snr=10, P_encoder=1.0).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    if next(iter(sd)).startswith("module."):
        sd = OrderedDict((k[7:], v) for k, v in sd.items())
    model.load_state_dict(sd)
    model.eval()
    encoder, decoder = model.encoder, model.decoder

    L_real = 2 * args.c * 56 * 56

    bank = (CBA.ChannelBank(cfg["bank_path"]) if args.site == "A"
            else CBB.ChannelBankSiteB(cfg["bank_path"]))
    ds = ImageOnly(cfg["img_dir"])
    assert len(ds) == cfg["n_images"] and ds.names == bank.image_names, \
        f"Site {args.site} test image order must match its channel bank's image_names"
    assert L_real <= bank.L_real, f"L_real={L_real} exceeds bank L_real={bank.L_real}"

    gt_cache = []
    for idx in range(len(ds)):
        stem = os.path.splitext(ds.names[idx])[0]
        boxes_norm = load_boxes_norm(os.path.join(cfg["lbl_dir"], stem + ".txt"))
        gt_cache.append(boxes_norm * IMG)

    yolo = YOLO(cfg["yolo"])
    yolo.to(device)
    if hasattr(yolo, "fuse") and callable(yolo.fuse):
        yolo.fuse()
    yolo.eval()

    tag = f"{args.variant}_c{args.c}"
    agg_summary = {}
    with torch.no_grad():
        # cache clean-input encoder features per image (channel realization
        # varies per (snr,m), the encoder output does not)
        z_cache = []
        for idx in range(len(ds)):
            x = ds[idx].unsqueeze(0).to(device)
            z_cache.append(encoder(x))

        for snr in snrs:
            in_bank = float(snr) in bank.snr_db
            assert in_bank, f"snr={snr} not in {args.site} bank grid {bank.snr_db}"
            per_image_sum = np.zeros(len(ds), dtype=np.float64)
            pooled_map50s = []
            for m in args.m:
                ap_list = np.zeros(len(ds), dtype=np.float32)
                pooled_metric = MeanAveragePrecision(iou_type="bbox").to(device)
                for idx in range(len(ds)):
                    h = bank.get_h(idx, snr, m)
                    noise_flat = bank.get_noise_real(idx, snr, m, L_real)
                    z_after = bank_rayleigh_forward(z_cache[idx], h, noise_flat, snr)
                    recon = decoder(z_after).clamp(0.0, 1.0)
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
                print(f"[eval-jscc] site={args.site} {tag} snr={snr:5.1f} m={m} "
                      f"mean_ap50={ap_list.mean():.4f} pooled={pooled_map50_m:.4f}", flush=True)
            per_image_mean = (per_image_sum / len(args.m)).astype(np.float32)
            pooled_map50_mean = float(np.mean(pooled_map50s))
            ci = stats.bootstrap_ci(per_image_mean)
            snr_tag = f"snr{snr:g}".replace("-", "m").replace(".", "p")
            # Rule 11: site letter is in both the directory-independent stem
            # and every filename, so Site A and Site B outputs for the same
            # (variant,c) never share a filename pattern.
            site_infix = "" if args.site == "A" else "_siteb"
            out_path = os.path.join(args.out_dir, f"jsccnn{site_infix}_{tag}_{snr_tag}.npy")
            np.save(out_path, per_image_mean)
            agg_summary[str(snr)] = dict(
                mean_per_image_ap50=float(per_image_mean.mean()),
                ci_low=ci["ci_low"], ci_high=ci["ci_high"],
                retention_per_image=float(per_image_mean.mean() / cfg["per_image_ceiling"]),
                pooled_map50=pooled_map50_mean,
                retention_pooled=pooled_map50_mean / cfg["pooled_ceiling"],
                in_bank=in_bank, M=len(args.m), out_path=out_path,
            )
            print(f"[eval-jscc] site={args.site} {tag} SNR {snr:5.1f} DONE: "
                  f"mean_ap50={per_image_mean.mean():.4f} [{ci['ci_low']:.4f},{ci['ci_high']:.4f}] "
                  f"retention={per_image_mean.mean()/cfg['per_image_ceiling']:.4f} "
                  f"pooled={pooled_map50_mean:.4f} -> {out_path}")

    out = {
        "provenance": {
            "site": args.site, "variant": args.variant, "c": args.c,
            "checkpoint": ckpt_path,
            "uses_per_px": args.c * 56 * 56 / (224.0 * 224.0),
            "per_image_ceiling": cfg["per_image_ceiling"],
            "pooled_ceiling": cfg["pooled_ceiling"],
            "conf_thres": 0.001,
            "channel_replay_method": ("direct encoder()/decoder() calls + a standalone "
                                      "reimplementation of channel.py's Rayleigh math "
                                      "(h, AWGN from the seeded bank instead of internal "
                                      "torch.randn) -- model.forward()/channel_layer are "
                                      "not used for this measurement."),
        },
        "by_snr": agg_summary,
    }
    site_infix = "" if args.site == "A" else "_siteb"
    out_json = os.path.join(args.out_dir, f"jsccnn{site_infix}_{tag}_summary.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[eval-jscc] DONE -> {out_json}")


if __name__ == "__main__":
    main()
