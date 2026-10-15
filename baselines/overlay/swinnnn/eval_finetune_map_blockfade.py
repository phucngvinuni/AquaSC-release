# eval_finetune_map_blockfade.py -- corrected re-measurement of the fine-tuned
# SwinJSCC_w/_SAandRA checkpoint under the project's SHARED channel convention
# (one Rayleigh fading draw per image + explicit equalization -- §8.1,
# ROI-JSCC's BlockRayleighChannel, AquaSC's digital_pipeline.py), not the
# official SwinJSCC channel (net/channel.py's rayleigh_noise_layer draws an
# INDEPENDENT fading gain per element with NO equalization -- confirmed by
# direct read, 2026-09-13). The first mAP measurement (eval_finetune_map.py)
# used the official per-element channel and scored 0.7292 per-image AP at
# 10dB/rate64 -- suspiciously far above every other baseline (flagship 0.5568)
# at a CHEAPER rate, which is the signature of an easier channel (per-symbol
# diversity averaging), not a better codec. This script re-measures on the
# channel every other baseline actually shares, to see what survives.
#
# Adaptation for the RA-family's variable rate: network.py computes
# avg_pwr = sum(feature**2)/mask.sum() (power-normalizes over only the
# `rate`-many active channels per token, since inactive positions are exactly
# zero) -- reproduced exactly here, then the tx/rx pairing and equalization
# use ONE shared complex h for the whole image (matching BlockRayleighChannel)
# instead of the official code's per-element h.
#
# Run: __TORCH_PY__ eval_finetune_map_blockfade.py --rate 64

import os
import sys
import math

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "swinjscc_code"))
_aq = os.path.dirname(os.path.dirname(HERE))
if _aq not in sys.path:
    sys.path.insert(0, _aq)

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from net.network import SwinJSCC
import metrics as M  # noqa: E402
import stats  # noqa: E402
import utils as aq_utils  # noqa: E402 -- for calc_psnr/calc_ssim, same convention every other system in RESULTS_SUMMARY_TABLE.md uses
from ultralytics import YOLO  # noqa: E402
from torchmetrics.detection import MeanAveragePrecision  # noqa: E402

IMG_SWIN = 256
IMG_YOLO = 224
CKPT = os.path.join(HERE, "history", "SwinJSCC_SAandRA_finetune_rayleigh", "model_best_psnr.pth")
TEST_IMG_DIR = "__AQUASC_DATA__/yolo_fish_dataset_root/test/images"
TEST_LBL_DIR = "__AQUASC_DATA__/yolo_fish_dataset_root/test/labels"
YOLO_WEIGHTS = os.path.join(_aq, "best.pt")
PER_IMAGE_CEILING = 0.7743
POOLED_CEILING = 0.7292211
# 2026-09-20 (Site B): Site B's own test root/detector/ceilings, from
# RESULTS_OF_RECORD.md section 6b -- overridable via --site b, default
# (--site a) is byte-for-byte the original Site A behavior above.
TEST_IMG_DIR_SITEB = "__AQUASC_DATA__/yolo_fishdet_nb2_vd/test/images"
TEST_LBL_DIR_SITEB = "__AQUASC_DATA__/yolo_fishdet_nb2_vd/test/labels"
YOLO_WEIGHTS_SITEB = os.path.join(_aq, "best_siteB.pt")
PER_IMAGE_CEILING_SITEB = 0.6451
POOLED_CEILING_SITEB = 0.6044


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


class TestDataset(Dataset):
    def __init__(self, img_dir):
        self.names = sorted(f for f in os.listdir(img_dir)
                            if f.lower().endswith((".png", ".jpg", ".jpeg")))
        self.img_dir = img_dir
        self.t_swin = transforms.Compose([transforms.Resize((IMG_SWIN, IMG_SWIN)), transforms.ToTensor()])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, self.names[idx])).convert("RGB")
        return self.t_swin(img)


def build_net(device, ckpt_path):
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
        model="SwinJSCC_w/_SAandRA", img_size=(IMG_SWIN, IMG_SWIN), patch_size=2, in_chans=3,
        embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10],
        C=None, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, patch_norm=True,
    )
    c.decoder_kwargs = dict(
        model="SwinJSCC_w/_SAandRA", img_size=(IMG_SWIN, IMG_SWIN),
        embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4],
        C=None, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, patch_norm=True,
    )
    net_args = argparse.Namespace(model="SwinJSCC_w/_SAandRA", channel_type="rayleigh",
                                  C="96", multiple_snr="10", distortion_metric="MSE",
                                  trainset="DIV2K")
    net = SwinJSCC(net_args, c).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    net.eval()
    return net


def blockfade_forward(net, x, snr_db, rate, model):
    """One shared complex h per image (matching BlockRayleighChannel / §8.1),
    with explicit equalization -- the two deliberate deviations from
    net/channel.py's per-element, non-equalized Rayleigh model. Power
    normalization reproduces network.py's avg_pwr = sum(feat**2)/mask.sum()
    exactly (active-channels-only), matching the RA family's masked payload.
    """
    feat, mask = net.encoder(x, float(snr_db), rate, model)
    B, L, C = feat.shape
    assert B == 1
    active = mask.sum()
    avg_pwr = (feat ** 2).sum() / active
    channel_tx = feat / torch.sqrt(avg_pwr * 2)

    flat = channel_tx.reshape(-1)
    half = flat.shape[0] // 2
    device = feat.device
    h = (torch.randn((), device=device) + 1j * torch.randn((), device=device)) / math.sqrt(2)
    sigma = math.sqrt(1.0 / (2 * 10 ** (float(snr_db) / 10.0)))
    noise = sigma * (torch.randn(half, device=device) + 1j * torch.randn(half, device=device))
    z = torch.complex(flat[:half], flat[half:])
    y = z * h + noise
    z_hat = y / (h + 1e-9)
    out_flat = torch.cat([z_hat.real, z_hat.imag])
    feat_hat = (out_flat.reshape(B, L, C) * torch.sqrt(avg_pwr * 2)).float()

    recon = net.decoder(feat_hat, float(snr_db), model)
    return recon.clamp(0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=int, default=64)
    ap.add_argument("--snr", type=float, nargs="+", default=[-5, 0, 5, 10, 15, 20])
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--tag", default="finetune", help="distinguishes output filenames, e.g. 'scratch'")
    ap.add_argument("--out_dir", default=os.path.join(_aq, "f1_perimage"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--site", choices=("a", "b"), default="a",
                    help="2026-09-20: Site B uses its own test root/detector/ceilings "
                         "(RESULTS_OF_RECORD.md section 6b); default 'a' is unchanged.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    if args.site == "b":
        img_dir, lbl_dir, yolo_weights = TEST_IMG_DIR_SITEB, TEST_LBL_DIR_SITEB, YOLO_WEIGHTS_SITEB
        ceiling_perimage, ceiling_pooled = PER_IMAGE_CEILING_SITEB, POOLED_CEILING_SITEB
    else:
        img_dir, lbl_dir, yolo_weights = TEST_IMG_DIR, TEST_LBL_DIR, YOLO_WEIGHTS
        ceiling_perimage, ceiling_pooled = PER_IMAGE_CEILING, POOLED_CEILING

    net = build_net(device, args.ckpt)
    ds = TestDataset(img_dir)
    print(f"[eval-bf] site={args.site} tag={args.tag} rate={args.rate} n={len(ds)}")

    gt_cache = []
    for idx in range(len(ds)):
        stem = os.path.splitext(ds.names[idx])[0]
        boxes_norm = load_boxes_norm(os.path.join(lbl_dir, stem + ".txt"))
        gt_cache.append(boxes_norm * IMG_YOLO)

    yolo = YOLO(yolo_weights)
    yolo.to(device)
    if hasattr(yolo, "fuse") and callable(yolo.fuse):
        yolo.fuse()
    yolo.eval()

    results = {}
    with torch.no_grad():
        for snr in args.snr:
            ap_list = np.zeros(len(ds), dtype=np.float32)
            psnr_list = np.zeros(len(ds), dtype=np.float32)
            ssim_list = np.zeros(len(ds), dtype=np.float32)
            pooled_metric = MeanAveragePrecision(iou_type="bbox").to(device)
            for idx in range(len(ds)):
                x = ds[idx].unsqueeze(0).to(device)
                recon256 = blockfade_forward(net, x, snr, args.rate, net.model)
                recon256_clamped = torch.clamp(recon256, 0.0, 1.0)
                # 2026-09-18: PSNR/SSIM on the SAME test-set images the mAP
                # numbers above use, at SwinJSCC's own native 256x256 decode
                # resolution (x is also 256x256 -- see TestDataset) -- this
                # was previously missing entirely (only validation-set,
                # training-time PSNR existed for this system). Same
                # utils.calc_psnr/calc_ssim convention as every other system.
                psnr_list[idx] = aq_utils.calc_psnr(recon256_clamped[0], x[0])[0]
                ssim_list[idx] = aq_utils.calc_ssim(recon256_clamped[0], x[0])[0]
                recon = torch.nn.functional.interpolate(
                    recon256, size=(IMG_YOLO, IMG_YOLO), mode="bicubic", align_corners=False)
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
            pooled_map50 = float(pooled_metric.compute()["map_50"])
            snr_tag = f"snr{snr:g}".replace("-", "m").replace(".", "p")
            out_path = os.path.join(args.out_dir, f"swinjscc_{args.tag}_bf_rate{args.rate}_{snr_tag}.npy")
            np.save(out_path, ap_list)
            psnr_ssim_path = os.path.join(args.out_dir, f"swinjscc_{args.tag}_bf_rate{args.rate}_{snr_tag}_psnrssim.npz")
            np.savez_compressed(psnr_ssim_path, psnr=psnr_list, ssim=ssim_list)
            results[snr] = dict(mean_ap50=float(ap_list.mean()), pooled_map50=pooled_map50,
                                retention_per_image=float(ap_list.mean() / ceiling_perimage),
                                retention_pooled=pooled_map50 / ceiling_pooled,
                                psnr_mean=float(psnr_list.mean()), ssim_mean=float(ssim_list.mean()))
            print(f"[eval-bf] snr={snr:5.1f} mean_ap50={ap_list.mean():.4f} "
                  f"retention={ap_list.mean()/ceiling_perimage:.4f} pooled={pooled_map50:.4f} "
                  f"psnr={psnr_list.mean():.3f} ssim={ssim_list.mean():.4f} -> {out_path}")

    print("[eval-bf] DONE", results)


if __name__ == "__main__":
    main()
