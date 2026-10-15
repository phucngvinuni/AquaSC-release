import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repo root (core modules)
# analysis/stats_eval_jscc.py — Wave 8 STATS step 6: run the JSCC-CNN BBW (c=2) baseline
# through the SAME seeded channel bank at 10 dB, and save per-image AP@0.5.
#
# Run with the jscc dir FIRST on sys.path so `model`/`utils`/`channel` resolve to
# the JSCC repo; AquaSC-D2DB is second so `stats`/`channel_bank` resolve there.
# No files in competitors/jscc are modified: the bank is injected by replacing
# model.channel_layer with a bank-driven Rayleigh module (documented below).
#
# Image order = sorted test filenames = bank image_names (verified identical).

import argparse
import json
import os
import sys

import numpy as np
import torch

_AQ = "__AQUASC_ROOT__"
_JSCC = os.path.join(_AQ, "competitors", "jscc")
sys.path.insert(0, _JSCC)
sys.path.insert(1, _AQ)

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from model import DeepJSCC  # jscc model
from utils import set_seed   # jscc utils

import stats
import channel_bank as CB
from ultralytics import YOLO

try:
    from torchmetrics.detection import MeanAveragePrecision
    _TM = True
except ImportError:
    _TM = False

IMG = 224
CEILING = 0.7292


class ImageOnly(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.paths = sorted(os.path.join(image_dir, f) for f in os.listdir(image_dir)
                            if f.lower().endswith((".png", ".jpg", ".jpeg")))
        self.t = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.ToTensor()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.t(Image.open(self.paths[idx]).convert("RGB"))


class BankRayleigh(torch.nn.Module):
    """JSCC Rayleigh channel that replays the bank {h, n} instead of torch.randn.

    Mirrors competitors/jscc/channel.py Channel.forward (Rayleigh branch) exactly:
    complex pair split, multiply h, add noise, equalize by h.  For batch=1 the
    caller sets .h (scalar complex) and .noise_real (N(0,1) array of length 2c*H*W)
    before each forward.
    """

    def __init__(self, snr_db):
        super().__init__()
        self.snr_db = float(snr_db)
        self.h = None
        self.noise_real = None

    def forward(self, z_hat):
        N, C, H, W = z_hat.shape
        c = C // 2
        snr_linear = 10 ** (self.snr_db / 10.0)
        noise_std = (1.0 / snr_linear) ** 0.5
        noise = torch.from_numpy(np.asarray(self.noise_real, dtype=np.float32)) \
                    .to(z_hat.device, z_hat.dtype).view(1, C, H, W) * noise_std
        hc = complex(self.h)
        h_real = torch.tensor(hc.real, device=z_hat.device, dtype=z_hat.dtype).view(1, 1, 1, 1)
        h_imag = torch.tensor(hc.imag, device=z_hat.device, dtype=z_hat.dtype).view(1, 1, 1, 1)
        h_complex = torch.complex(h_real, h_imag)
        z_complex = torch.complex(z_hat[:, :c], z_hat[:, c:])
        z_faded = z_complex * h_complex
        z_proc = torch.cat((z_faded.real, z_faded.imag), dim=1)
        z_noisy = z_proc + noise
        zc = torch.complex(z_noisy[:, :c], z_noisy[:, c:])
        zeq = zc / (h_complex + 1e-9)
        return torch.cat((zeq.real, zeq.imag), dim=1)


def yolo_predict(yolo, images, conf, iou, device):
    out = []
    results = yolo(images.detach(), verbose=False, conf=conf, iou=iou)
    for r in results:
        if r.boxes is not None and len(r.boxes.xyxy) > 0:
            out.append({"boxes": r.boxes.xyxy.to(device), "scores": r.boxes.conf.to(device),
                        "labels": r.boxes.cls.to(torch.int64).to(device)})
        else:
            out.append({"boxes": torch.empty((0, 4), dtype=torch.float32, device=device),
                        "scores": torch.empty(0, dtype=torch.float32, device=device),
                        "labels": torch.empty(0, dtype=torch.int64, device=device)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--c", type=int, default=2)
    ap.add_argument("--snr", type=float, default=10.0)
    ap.add_argument("--yolo", default=os.path.join(_AQ, "best.pt"))
    ap.add_argument("--input_dir", default="__AQUASC_DATA__/yolo_fish_dataset_root/test/images")
    ap.add_argument("--label_dir", default="__AQUASC_DATA__/yolo_fish_dataset_root/test/labels")
    ap.add_argument("--bank", default=os.path.join(_AQ, "channel_bank.npz"))
    ap.add_argument("--m", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--out_prefix", default=os.path.join(_AQ, "stats_bank_bbw_c2"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    set_seed(42)

    model = DeepJSCC(c=args.c, channel_type="Rayleigh", snr=args.snr, P_encoder=1.0).to(device)
    sd = torch.load(args.checkpoint, map_location=device)
    if next(iter(sd)).startswith("module."):
        sd = {k[7:]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.channel_layer = BankRayleigh(args.snr).to(device)
    model.eval()
    print(f"loaded {args.checkpoint} (c={args.c}), channel_layer replaced with BankRayleigh")

    bank = CB.ChannelBank(args.bank)
    ds = ImageOnly(args.input_dir)
    ds_names = [os.path.basename(p) for p in ds.paths]
    assert len(ds) == 432 and ds_names == bank.image_names, \
        "JSCC image order must match bank.image_names"

    yolo = YOLO(args.yolo)
    yolo.to(device)
    if hasattr(yolo, "fuse") and callable(yolo.fuse):
        yolo.fuse()
    yolo.eval()

    # GT boxes (same parser as analysis/baselinea_score.py / metrics.py)
    def load_boxes(label_path):
        boxes, labels = [], []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    p = line.strip().split()
                    if len(p) != 5:
                        continue
                    try:
                        cx, cy, w, h = map(float, p[1:])
                        x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
                        x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2]); labels.append(int(p[0]))
                    except ValueError:
                        continue
        if boxes:
            b = torch.as_tensor(boxes, dtype=torch.float32) * IMG
            return b, torch.as_tensor(labels, dtype=torch.int64)
        return torch.empty((0, 4)), torch.empty(0, dtype=torch.int64)

    gt_cache = []
    for idx in range(len(ds)):
        stem = os.path.splitext(os.path.basename(ds.paths[idx]))[0]
        b, l = load_boxes(os.path.join(args.label_dir, stem + ".txt"))
        gt_cache.append({"boxes": b, "labels": l})

    n = len(ds)
    M = args.m
    per_image = np.full((len(M), n), np.nan, dtype=np.float32)
    agg = {}
    with torch.no_grad():
        for mi, m in enumerate(M):
            map_calc = MeanAveragePrecision(iou_type="bbox", class_metrics=True).to(device) if _TM else None
            ap_list = np.zeros(n, dtype=np.float32)
            for idx in range(n):
                x = ds[idx].unsqueeze(0).to(device)
                model.channel_layer.h = bank.get_h(idx, args.snr, m)
                model.channel_layer.noise_real = bank.get_noise_real(idx, args.snr, m, 4 * 56 * 56)
                recon = model(x).clamp(0, 1)
                preds = yolo_predict(yolo, recon, conf=0.001, iou=0.5, device=device)
                if map_calc is not None:
                    map_calc.update(preds, [{"boxes": gt_cache[idx]["boxes"].to(device),
                                             "labels": gt_cache[idx]["labels"].to(device)}])
                ap_list[idx] = stats.per_image_ap50(
                    preds[0]["boxes"].cpu().numpy(), preds[0]["scores"].cpu().numpy(),
                    gt_cache[idx]["boxes"].cpu().numpy(), iou_thr=0.5)
            if map_calc is not None:
                r = map_calc.compute()
                map50 = float(r["map_50"]) if "map_50" in r and r["map_50"].numel() == 1 else float("nan")
            else:
                map50 = float("nan")
            agg[m] = map50
            per_image[mi] = ap_list
            print(f"BBW c=2 @{args.snr}dB m={m}: mAP@0.5={map50:.4f} mean per-image AP={ap_list.mean():.4f}")
    np.savez(f"{args.out_prefix}_perimage_ap.npz", per_image_ap=per_image, m=np.array(M))
    with open(f"{args.out_prefix}_agg.json", "w") as f:
        json.dump({str(m): v for m, v in agg.items()}, f, indent=2)
    print("analysis/stats_eval_jscc.py DONE")


if __name__ == "__main__":
    main()
