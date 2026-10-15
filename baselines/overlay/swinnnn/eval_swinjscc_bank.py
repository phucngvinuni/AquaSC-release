# eval_swinjscc_bank.py — Wave 9b SWINJSCC agent: evaluate a trained SwinJSCC
# checkpoint through the SAME seeded channel bank as AquaSC (Wave 8 STATS),
# and score PSNR/SSIM (224 grid) + mAP@0.5 (best.pt conf 0.001) + per-image AP.
#
# Mirrors analysis/stats_eval_jscc.py protocol (batch=1, bank {h,n} replay, torchmetrics
# MeanAveragePrecision, stats.per_image_ap50).  The bank's block-fading h is the
# scalar per-image gain; the bank's N(0,1) noise stream supplies the complex
# noise (first F/2 samples = real part, next F/2 = imag part, scaled by
# sqrt(1/(2*10^(SNR/10)))), exactly AquaSC's noise convention.
#
# Noise length: feature numel F = 256*C reals (256x256, 16x16 tokens).
#   C=20 -> F=5120 ; C=33 -> F=8448 ; C=64 -> F=16384.
# The bank stores L_real=12544, so C=64 needs 16384 > 12544: the noise is
# extended deterministically by continuing the same per-cell RNG stream
# (np.random.default_rng(subseed)); the first 12544 samples are byte-identical
# to the bank.  This is flagged in output as `noise_extended`.

import argparse
import json
import os
import sys
import hashlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
_AQ = "__AQUASC_ROOT__"
sys.path.insert(0, os.path.join(HERE, "swinjscc_code"))
sys.path.insert(1, _AQ)

from net.network import SwinJSCC  # noqa: E402
import channel_bank as CB  # noqa: E402
import stats as STATS  # noqa: E402
import utils as AQUTILS  # noqa: E402
from ultralytics import YOLO  # noqa: E402

try:
    from torchmetrics.detection import MeanAveragePrecision
    _TM = True
except ImportError:
    _TM = False

IMG = 256
IMG_DET = 224
CEILING = 0.7292
SNRS = (-5, 0, 5, 10, 15, 20)
M_RANGE = list(range(10))


def build_config(args, device):
    class Config:
        pass
    c = Config()
    c.logger = None
    c.pass_channel = True
    c.CUDA = True
    c.device = device
    c.norm = False
    c.downsample = 4
    c.encoder_kwargs = dict(
        model=args.model, img_size=(IMG, IMG), patch_size=2, in_chans=3,
        embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10],
        C=args.C, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, patch_norm=True,
    )
    c.decoder_kwargs = dict(
        model=args.model, img_size=(IMG, IMG),
        embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4],
        C=args.C, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, patch_norm=True,
    )
    return c


def forward_block(net, x, snr_db, C, h, noise_real):
    B = x.shape[0]
    feature = net.encoder(x, snr_db, C, net.model)
    pwr = (feature ** 2).mean() * 2.0
    tx = feature / pwr.sqrt()
    flat = tx.reshape(B, -1)
    half = flat.shape[1] // 2
    cin = torch.complex(flat[:, :half], flat[:, half:])
    sigma = (1.0 / (2.0 * 10 ** (float(snr_db) / 10.0))) ** 0.5
    h = torch.as_tensor(h, device=x.device).view(-1)
    nr = noise_real[:, :half]
    ni = noise_real[:, half:]
    noise = (nr + 1j * ni) * sigma
    y = cin * h[:, None] + noise
    yflat = torch.cat([y.real, y.imag], dim=1) * pwr.sqrt()
    noisy_feature = yflat.reshape_as(feature)
    recon = net.decoder(noisy_feature, snr_db, net.model)
    return recon.clamp(0.0, 1.0)


class FishTest(Dataset):
    def __init__(self, image_dir, label_dir):
        self.paths = sorted(os.path.join(image_dir, f) for f in os.listdir(image_dir)
                            if f.lower().endswith((".png", ".jpg", ".jpeg")))
        self.label_dir = label_dir
        self.t256 = transforms.Compose([
            transforms.Resize((IMG, IMG), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
        self.t224 = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        im = Image.open(p).convert("RGB")
        x256 = self.t256(im)
        x224 = self.t224(im)
        stem = os.path.splitext(os.path.basename(p))[0]
        boxes, labels = [], []
        lp = os.path.join(self.label_dir, stem + ".txt")
        if os.path.exists(lp):
            with open(lp) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    try:
                        cx, cy, w, h = map(float, parts[1:])
                        x1 = max(0.0, cx - w / 2); y1 = max(0.0, cy - h / 2)
                        x2 = min(1.0, cx + w / 2); y2 = min(1.0, cy + h / 2)
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2]); labels.append(int(parts[0]))
                    except ValueError:
                        continue
        if boxes:
            b = torch.as_tensor(boxes, dtype=torch.float32) * IMG_DET
            l = torch.as_tensor(labels, dtype=torch.int64)
        else:
            b = torch.empty((0, 4)); l = torch.empty(0, dtype=torch.int64)
        return x256, x224, {"boxes": b, "labels": l}


def yolo_predict(yolo, images, device):
    out = []
    results = yolo(images.detach(), verbose=False, conf=0.001, iou=0.5)
    for r in results:
        if r.boxes is not None and len(r.boxes.xyxy) > 0:
            out.append({"boxes": r.boxes.xyxy.to(device), "scores": r.boxes.conf.to(device),
                        "labels": r.boxes.cls.to(torch.int64).to(device)})
        else:
            out.append({"boxes": torch.empty((0, 4), dtype=torch.float32, device=device),
                        "scores": torch.empty(0, dtype=torch.float32, device=device),
                        "labels": torch.empty(0, dtype=torch.int64, device=device)})
    return out


def get_noise_extended(bank, image_idx, snr_db, m, F):
    """Return first F standard-normal real samples; if F > bank L_real, extend
    deterministically via the same per-cell RNG (first L_real == bank)."""
    if F <= bank.L_real:
        return bank.get_noise_real(image_idx, snr_db, m, F), False
    sub = bank.get_subseed(image_idx, snr_db, m)
    rng = np.random.default_rng(sub)
    rng.standard_normal(); rng.standard_normal()  # h draws (hr, hi)
    n = rng.standard_normal(F)
    assert np.allclose(n[:bank.L_real],
                       bank.get_noise_real(image_idx, snr_db, m, bank.L_real)), \
        "extended stream prefix must equal bank n"
    return n.astype(np.float32), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--C", type=int, required=True)
    ap.add_argument("--yolo", default=os.path.join(_AQ, "best.pt"))
    ap.add_argument("--input_dir", default="__AQUASC_DATA__/yolo_fish_dataset_root/test/images")
    ap.add_argument("--label_dir", default="__AQUASC_DATA__/yolo_fish_dataset_root/test/labels")
    ap.add_argument("--bank", default=os.path.join(_AQ, "channel_bank.npz"))
    ap.add_argument("--snrs", nargs="+", type=float, default=list(SNRS))
    ap.add_argument("--m", type=int, nargs="+", default=M_RANGE)
    ap.add_argument("--out_prefix", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="SwinJSCC_w/o_SAandRA")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(42)

    c = build_config(args, device)
    net = SwinJSCC(
        argparse.Namespace(model=args.model, channel_type="rayleigh", C=str(args.C),
                           multiple_snr="10", distortion_metric="MSE", trainset="DIV2K"),
        c,
    ).to(device)
    sd = torch.load(args.checkpoint, map_location=device)
    sd = sd["model_state_dict"] if "model_state_dict" in sd else sd
    net.load_state_dict(sd)
    net.eval()
    print(f"[eval] loaded {args.checkpoint} C={args.C}", flush=True)

    bank = CB.ChannelBank(args.bank)
    ds = FishTest(args.input_dir, args.label_dir)
    ds_names = [os.path.basename(p) for p in ds.paths]
    assert len(ds) == 432 and ds_names == bank.image_names, \
        "test image order must match bank.image_names"
    # encoder output = (IMG/16)^2 tokens x C  (patch2 -> 4 Swin stages -> /16)
    F = (IMG // 16) ** 2 * args.C  # feature numel reals (256 tokens * C)

    yolo = YOLO(args.yolo)
    yolo.to(device)
    if hasattr(yolo, "fuse") and callable(yolo.fuse):
        yolo.fuse()
    yolo.eval()

    out_prefix = args.out_prefix or os.path.join(
        HERE, "eval", f"swinjscc_C{args.C}")
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    n = len(ds)
    M = args.m
    snrs = args.snrs
    agg = {}
    per_image = {}  # (snr, m) -> [n] per-image AP
    psnr_agg = {}
    ssim_agg = {}
    extended_any = False

    with torch.no_grad():
        for snr in snrs:
            for m in M:
                map_calc = MeanAveragePrecision(iou_type="bbox", class_metrics=True).to(device) if _TM else None
                ap_list = np.zeros(n, dtype=np.float32)
                ps_list = np.zeros(n, dtype=np.float32)
                ss_list = np.zeros(n, dtype=np.float32)
                for idx in range(n):
                    x256, x224, gt = ds[idx]
                    x256 = x256.unsqueeze(0).to(device)
                    h = bank.get_h(idx, snr, m)
                    noise, ext = get_noise_extended(bank, idx, snr, m, F)
                    if ext:
                        extended_any = True
                    noise = torch.from_numpy(noise).to(device, torch.float32).view(1, F)
                    recon256 = forward_block(net, x256, snr, args.C, h, noise)
                    recon224 = F.interpolate(recon256, size=(IMG_DET, IMG_DET),
                                             mode="bicubic", align_corners=False)
                    ps_list[idx] = AQUTILS.calc_psnr(recon224, x224)[0]
                    ss_list[idx] = AQUTILS.calc_ssim(recon224, x224)[0]
                    preds = yolo_predict(yolo, recon224, device)
                    if map_calc is not None:
                        map_calc.update(preds, [{"boxes": gt["boxes"].to(device),
                                                 "labels": gt["labels"].to(device)}])
                    ap_list[idx] = STATS.per_image_ap50(
                        preds[0]["boxes"].cpu().numpy(), preds[0]["scores"].cpu().numpy(),
                        gt["boxes"].cpu().numpy(), iou_thr=0.5)
                if map_calc is not None:
                    r = map_calc.compute()
                    map50 = float(r["map_50"]) if "map_50" in r and r["map_50"].numel() == 1 else float("nan")
                else:
                    map50 = float("nan")
                key = f"{snr:g}_{m}"
                agg[key] = {"map50": map50, "psnr": float(ps_list.mean()),
                            "ssim": float(ss_list.mean()),
                            "retention": map50 / CEILING if map50 == map50 else float("nan")}
                per_image[key] = ap_list.tolist()
                psnr_agg[key] = float(ps_list.mean())
                ssim_agg[key] = float(ss_list.mean())
                print(f"[eval] C={args.C} SNR={snr:g} m={m}: mAP@0.5={map50:.4f} "
                      f"PSNR={ps_list.mean():.2f} SSIM={ss_list.mean():.4f} "
                      f"per-imgAP={ap_list.mean():.4f}", flush=True)

    out = {"C": args.C, "F": F, "noise_extended": bool(extended_any),
           "checkpoint": args.checkpoint, "per_snr_m": agg}
    with open(f"{out_prefix}_agg.json", "w") as f:
        json.dump(out, f, indent=2)
    np.savez(f"{out_prefix}_perimage.npz", per_image={k: np.array(v) for k, v in per_image.items()})
    print(f"[eval] DONE C={args.C} -> {out_prefix}_agg.json", flush=True)


if __name__ == "__main__":
    main()
