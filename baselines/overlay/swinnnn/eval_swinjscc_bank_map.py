# eval_swinjscc_bank_map.py -- Wave 19 Part 1 fix (Check 3): SwinJSCC mAP
# through the SAME channel_bank.npz / M=10-realization protocol ROI-JSCC and
# JSCC-CNN use, replacing eval_finetune_map_blockfade.py's single-seed draw.
# One complex Rayleigh gain per image per realization (matching
# BlockRayleighChannel / digital_pipeline.py's block-fading convention),
# explicit equalization, masked power normalization reproducing
# network.py's avg_pwr = sum(feature**2)/mask.sum() exactly for the RA family.
#
# Run: __TORCH_PY__ eval_swinjscc_bank_map.py --rate 32 --ckpt <path> --tag scratch_r32

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
import channel_bank as CB  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from torchmetrics.detection import MeanAveragePrecision  # noqa: E402

IMG_SWIN = 256
IMG_YOLO = 224
TEST_IMG_DIR = "__AQUASC_DATA__/yolo_fish_dataset_root/test/images"
TEST_LBL_DIR = "__AQUASC_DATA__/yolo_fish_dataset_root/test/labels"
YOLO_WEIGHTS = os.path.join(_aq, "best.pt")
BANK_PATH = os.path.join(_aq, "channel_bank.npz")
PER_IMAGE_CEILING = 0.7743
POOLED_CEILING = 0.7292211


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


def bank_forward(net, x, snr_db, rate, model, h_complex, noise_flat):
    """Same masked-power-normalization + block-fading + equalization as
    eval_finetune_map_blockfade.py's blockfade_forward, but h/noise come from
    the seeded bank instead of torch.randn -- the only change needed to match
    ROI-JSCC/JSCC-CNN's replay protocol.
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
    assert noise_flat.shape[0] >= 2 * half, "bank L_real too short for this rate"
    h = torch.tensor(h_complex, dtype=torch.complex64, device=device)
    nr = torch.from_numpy(noise_flat[:half].astype(np.float32)).to(device)
    ni = torch.from_numpy(noise_flat[half:2 * half].astype(np.float32)).to(device)
    sigma = math.sqrt(1.0 / (2 * 10 ** (float(snr_db) / 10.0)))
    noise = sigma * torch.complex(nr, ni)
    z = torch.complex(flat[:half], flat[half:])
    y = z * h + noise
    z_hat = y / (h + 1e-9)
    out_flat = torch.cat([z_hat.real, z_hat.imag])
    feat_hat = (out_flat.reshape(B, L, C) * torch.sqrt(avg_pwr * 2)).float()

    recon = net.decoder(feat_hat, float(snr_db), model)
    return recon.clamp(0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=int, default=32)
    ap.add_argument("--snr", type=float, nargs="+", default=[-5, 0, 5, 10, 15, 20])
    ap.add_argument("--m", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True, help="distinguishes output filenames, e.g. 'scratch_r32'")
    ap.add_argument("--out_dir", default=os.path.join(_aq, "f1_perimage"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(42)

    net = build_net(device, args.ckpt)
    bank = CB.ChannelBank(BANK_PATH)
    ds = TestDataset(TEST_IMG_DIR)
    assert len(ds) == 432 and ds.names == bank.image_names, \
        "test image order must match bank.image_names"

    L_real_needed = 256 * args.rate  # active real elements at this rate (L=256 tokens)
    assert L_real_needed <= bank.L_real, f"rate={args.rate} needs {L_real_needed} > bank L_real={bank.L_real}"
    print(f"[eval-swin-bank] tag={args.tag} rate={args.rate} L_real_needed={L_real_needed} n={len(ds)}")

    gt_cache = []
    for idx in range(len(ds)):
        stem = os.path.splitext(ds.names[idx])[0]
        boxes_norm = load_boxes_norm(os.path.join(TEST_LBL_DIR, stem + ".txt"))
        gt_cache.append(boxes_norm * IMG_YOLO)

    yolo = YOLO(YOLO_WEIGHTS)
    yolo.to(device)
    if hasattr(yolo, "fuse") and callable(yolo.fuse):
        yolo.fuse()
    yolo.eval()

    results = {}
    with torch.no_grad():
        for snr in args.snr:
            in_bank = float(snr) in bank.snr_db
            assert in_bank, f"snr={snr} not in bank grid {bank.snr_db}"
            per_image_sum = np.zeros(len(ds), dtype=np.float64)
            pooled_map50s = []
            for m in args.m:
                ap_list = np.zeros(len(ds), dtype=np.float32)
                pooled_metric = MeanAveragePrecision(iou_type="bbox").to(device)
                for idx in range(len(ds)):
                    x = ds[idx].unsqueeze(0).to(device)
                    h = bank.get_h(idx, snr, m)
                    noise_flat = bank.get_noise_real(idx, snr, m, L_real_needed)
                    recon256 = bank_forward(net, x, snr, args.rate, net.model, h, noise_flat)
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
                per_image_sum += ap_list
                pooled_map50_m = float(pooled_metric.compute()["map_50"])
                pooled_map50s.append(pooled_map50_m)
                print(f"[eval-swin-bank] snr={snr:5.1f} m={m} mean_ap50={ap_list.mean():.4f} "
                      f"pooled={pooled_map50_m:.4f}", flush=True)
            per_image_mean = (per_image_sum / len(args.m)).astype(np.float32)
            pooled_map50_mean = float(np.mean(pooled_map50s))
            snr_tag = f"snr{snr:g}".replace("-", "m").replace(".", "p")
            out_path = os.path.join(args.out_dir, f"swinjscc_{args.tag}_bank_rate{args.rate}_{snr_tag}.npy")
            np.save(out_path, per_image_mean)
            results[snr] = dict(mean_ap50=float(per_image_mean.mean()), pooled_map50=pooled_map50_mean,
                                retention_per_image=float(per_image_mean.mean() / PER_IMAGE_CEILING),
                                retention_pooled=pooled_map50_mean / POOLED_CEILING,
                                M=len(args.m), out_path=out_path)
            print(f"[eval-swin-bank] SNR {snr:5.1f} DONE mean_ap50={per_image_mean.mean():.4f} "
                  f"retention={per_image_mean.mean()/PER_IMAGE_CEILING:.4f} pooled={pooled_map50_mean:.4f} -> {out_path}")

    print("[eval-swin-bank] DONE", results)


if __name__ == "__main__":
    main()
