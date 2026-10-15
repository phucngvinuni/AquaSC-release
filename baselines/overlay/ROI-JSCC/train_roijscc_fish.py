#!/usr/bin/env python3
"""
train_roijscc_fish.py — Wave 9b ROIJSCC agent: train ONE ROI-JSCC checkpoint on
the honest (video-disjoint) AquaSC train split.

Setup (documented in report):
  * dataset   : 4066 train images (1920x1080 PNG) -> bicubic Resize 256x256 ->
                Normalize(0.5,0.5) to [-1,1].
  * ROI       : largest GT box centre -> single cell of the 4x4 mask grid
                (compute_roi_index_from_norm, mode="largest_box"). The boxes come
                from the YOLO labels (AquaSC data root train/labels).
  * channel   : BlockRayleighChannel (block-fading Rayleigh + equalization by h,
                matching AquaSC / the seeded channel bank), SNR fixed 10 dB.
  * loss      : ROIIT_MSE (ROI-JSCC's own ROI-weighted MSE, copied verbatim).
  * optimizer : Adam lr=1e-4, no scheduler (upstream train.yaml defaults).
  * rcpp      : 32  -> 0.1875 uses/px (ROI-JSCC's LOWEST supported rate; the only
                one whose L_real=12288 fits the 12544-sample channel bank).
  * seed      : 42 (torch + numpy) for determinism.

Usage:
  python3 train_roijscc_fish.py --epochs 200 --batch_size 16 --device cuda \
      --out_dir __AQUASC_ROOT__/competitors/ROI-JSCC/saved_models
"""

import argparse
import hashlib
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from roijscc_fish_common import (
    build_model, BlockRayleighChannel, rate_constants, compute_roi_index_from_norm,
    load_boxes_norm, ROIIT_MSE, ROI_RES,
)

TRAIN_IMG_DIR = "__AQUASC_DATA__/yolo_fish_dataset_root/train/images"
TRAIN_LBL_DIR = "__AQUASC_DATA__/yolo_fish_dataset_root/train/labels"
VAL_IMG_DIR = "__AQUASC_DATA__/yolo_fish_dataset_root/valid/images"
VAL_LBL_DIR = "__AQUASC_DATA__/yolo_fish_dataset_root/valid/labels"


class FishROIDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, roi_mode="largest_box"):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.roi_mode = roi_mode
        self.names = sorted(f for f in os.listdir(img_dir)
                            if f.lower().endswith((".png", ".jpg", ".jpeg")))
        self.t = transforms.Compose([
            transforms.Resize((ROI_RES, ROI_RES), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        stem = os.path.splitext(name)[0]
        img = self.t(Image.open(os.path.join(self.img_dir, name)).convert("RGB"))
        boxes = load_boxes_norm(os.path.join(self.lbl_dir, stem + ".txt"))
        roi = compute_roi_index_from_norm(boxes, mode=self.roi_mode)
        return img, roi[0], roi[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--snr_db", type=float, default=10.0)
    ap.add_argument("--rcpp", type=int, default=32, choices=[12, 16, 24, 32, 64, 128])
    ap.add_argument("--roi_mode", type=str, default="largest_box",
                    choices=["largest_box", "shoal_bbox"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_dir", type=str,
                    default="__AQUASC_ROOT__/competitors/ROI-JSCC/saved_models")
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--val_subset", type=int, default=100,
                    help="number of valid images used for the best-PSNR selection")
    ap.add_argument("--smoke", action="store_true",
                     help="one training step then exit (memory/throughput probe)")
    ap.add_argument("--resume", type=str, default=None,
                    help="Site B fine-tune (2026-09-16): load a pretrained checkpoint's "
                         "state_dict before training (fresh Adam optimizer regardless, "
                         "matching AquaSC's S-3 --finetune_reset convention). Nothing is "
                         "frozen -- this is a full fine-tune, not a partial one.")
    ap.add_argument("--train_img_dir", type=str, default=None)
    ap.add_argument("--train_lbl_dir", type=str, default=None)
    ap.add_argument("--val_img_dir", type=str, default=None)
    ap.add_argument("--val_lbl_dir", type=str, default=None)
    args = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device(args.device)
    rc = rate_constants(args.rcpp)
    L_real = rc["L_real"]
    N = rc["N_complex"]
    print(f"[train] rcpp={args.rcpp} C={rc['C']} RONI_C={rc['RONI_C']} ROI_C={rc['ROI_C']} "
          f"L_real={L_real} N_complex={N} (uses/px={rc['C']/256.0:.4f})")

    train_img_dir = args.train_img_dir or TRAIN_IMG_DIR
    train_lbl_dir = args.train_lbl_dir or TRAIN_LBL_DIR
    val_img_dir = args.val_img_dir or VAL_IMG_DIR
    val_lbl_dir = args.val_lbl_dir or VAL_LBL_DIR

    ds = FishROIDataset(train_img_dir, train_lbl_dir, roi_mode=args.roi_mode)
    print(f"[train] dataset size = {len(ds)} (img_dir={train_img_dir})")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=True)

    # validation subset (best-PSNR selection), deterministic first `val_subset` images
    val_ds = FishROIDataset(val_img_dir, val_lbl_dir, roi_mode=args.roi_mode)
    val_names = val_ds.names[:args.val_subset]
    val_indices = [val_ds.names.index(n) for n in val_names]

    model = build_model(rcpp=args.rcpp, chan_type="Rayleigh", device=device)
    model.channel = BlockRayleighChannel()
    if args.resume:
        sd = torch.load(args.resume, map_location=device)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=True)
        print(f"[train] resumed full state_dict from {args.resume}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params = {n_params/1e6:.2f}M")

    criterion = ROIIT_MSE(d=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    save_name = (f"ROIIT_FISH_Rayleigh_SNR010_rcpp{args.rcpp:03d}_PSNR_ROIJSCC_"
                 f"roi-{args.roi_mode}.pt")
    save_path = os.path.join(args.out_dir, save_name)
    latest_path = os.path.join(args.out_dir, save_name.replace(".pt", "_latest.pt"))
    best_path = os.path.join(args.out_dir, save_name.replace(".pt", "_best.pt"))

    def make_h_noise(B):
        hr = torch.randn(B, 1, device=device)
        hi = torch.randn(B, 1, device=device)
        h = (hr + 1j * hi) / np.sqrt(2.0)
        noise_real = torch.randn(B, 2 * N, device=device)
        return h, noise_real

    @torch.no_grad()
    def val_psnr():
        model.eval()
        psnrs = []
        for idx in val_indices:
            img, ih, iw = val_ds[idx]
            img = img.unsqueeze(0).to(device)
            h, noise_real = make_h_noise(1)
            model.channel.h = h
            model.channel.noise_real = noise_real
            out = model(img, ROI_Index=(int(ih), int(iw)), SNR_info=args.snr_db)
            mse = torch.mean((out - img) ** 2).item()
            psnrs.append(10 * np.log10(4.0 / mse))  # [-1,1] -> peak 2.0 -> 4.0
        model.train()
        return float(np.mean(psnrs))

    if args.smoke:
        img, ih, iw = next(iter(loader))
        img = img.to(device)
        t0 = time.time()
        h, noise_real = make_h_noise(img.shape[0])
        model.channel.h = h
        model.channel.noise_real = noise_real
        out = model(img, ROI_Index=(int(ih[0]), int(iw[0])), SNR_info=args.snr_db)
        total_loss, psnr = criterion(out, img, (int(ih[0]), int(iw[0])))
        total_loss.backward()
        torch.cuda.synchronize()
        dt = time.time() - t0
        mem = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"[smoke] batch={img.shape[0]} fwd+bwd {dt:.3f}s loss={total_loss.item():.4f} "
              f"psnr={psnr.item():.2f} peak_alloc={mem:.2f}GB out.shape={tuple(out.shape)}")
        return

    since = time.time()
    best_val = -1.0
    for epoch in range(args.epochs):
        model.train()
        ep_loss = 0.0
        ep_psnr = 0.0
        count = 0
        for img, ih, iw in loader:
            img = img.to(device)
            B = img.shape[0]
            optimizer.zero_grad()
            h, noise_real = make_h_noise(B)
            model.channel.h = h
            model.channel.noise_real = noise_real
            roi = (int(ih[0]), int(iw[0]))
            out = model(img, ROI_Index=roi, SNR_info=args.snr_db)
            total_loss, psnr = criterion(out, img, roi)
            total_loss.backward()
            optimizer.step()
            ep_loss += total_loss.item() * B
            ep_psnr += psnr.item() * B
            count += B
        ep_loss /= count
        ep_psnr /= count
        if (epoch + 1) % args.save_every == 0 or epoch == 0:
            el = time.time() - since
            print(f"[train] epoch {epoch+1}/{args.epochs} loss={ep_loss:.6f} "
                  f"psnr={ep_psnr:.3f} dB  elapsed={el/60:.1f}m  "
                  f"~{el/(epoch+1)*(args.epochs-epoch-1)/60:.1f}m left", flush=True)
            torch.save(model.state_dict(), latest_path)
            vp = val_psnr()
            print(f"[train]   val_psnr (subset {len(val_indices)}) = {vp:.3f} dB", flush=True)
            if vp > best_val:
                best_val = vp
                torch.save(model.state_dict(), best_path)
                print(f"[train]   new best val_psnr -> saved {best_path}", flush=True)

    torch.save(model.state_dict(), save_path)
    for p in (save_path, best_path):
        sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
        print(f"[train] saved {p}\n[train] SHA-256 {sha}")
    print(f"[train] best_val_psnr={best_val:.3f} dB")
    print(f"[train] total wall { (time.time()-since)/3600:.2f} h")


if __name__ == "__main__":
    main()
