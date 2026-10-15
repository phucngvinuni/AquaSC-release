# train_swinjscc_scratch_sandra.py -- controlled from-scratch counterpart to
# finetune_swinjscc_pretrained.py: identical architecture (SwinJSCC_w/_SAandRA,
# base size), identical LR (1e-5), identical official-channel/batch_size=1
# training path, identical random rate/SNR sampling per step -- the ONLY
# difference is that this one starts from random initialization instead of
# the paper's released pretrained checkpoint. Isolates whether the pretrained
# init (vs. every earlier from-scratch attempt's different architecture/LR/
# channel-mode choices) is really what made the difference.
#
# Run: __TORCH_PY__ train_swinjscc_scratch_sandra.py

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, os.path.join(HERE, "swinjscc_code"))

from net.network import SwinJSCC  # noqa: E402

IMG = 256
RATE_CANDIDATES = (32, 64, 96)
SNR_CANDIDATES = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ImageOnly(Dataset):
    def __init__(self, image_dir, size=IMG, limit=None):
        paths = sorted(os.path.join(image_dir, f) for f in os.listdir(image_dir)
                       if f.lower().endswith((".png", ".jpg", ".jpeg")))
        if limit:
            paths = paths[:limit]
        self.paths = paths
        self.t = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.t(Image.open(self.paths[idx]).convert("RGB"))


def build_net(device):
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
        model="SwinJSCC_w/_SAandRA", img_size=(IMG, IMG), patch_size=2, in_chans=3,
        embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10],
        C=None, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, patch_norm=True,
    )
    c.decoder_kwargs = dict(
        model="SwinJSCC_w/_SAandRA", img_size=(IMG, IMG),
        embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4],
        C=None, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, patch_norm=True,
    )
    net_args = argparse.Namespace(model="SwinJSCC_w/_SAandRA", channel_type="rayleigh",
                                  C="96", multiple_snr="10", distortion_metric="MSE",
                                  trainset="DIV2K")
    net = SwinJSCC(net_args, c).to(device)
    # NO checkpoint load -- random init, everything else identical to the
    # fine-tune script.
    return net


@torch.no_grad()
def val_psnr(net, loader, device, rates=RATE_CANDIDATES, snrs=(0, 10, 20)):
    net.eval()
    per = {(r, s): 0.0 for r in rates for s in snrs}
    n = 0
    for x in loader:
        x = x.to(device)
        for r in rates:
            for s in snrs:
                recon, CBR, chan_param, mse, loss_G = net(x, given_SNR=float(s), given_rate=r)
                mse_v = mse.item()
                ps = 10 * math.log10(65025.0 / mse_v) if mse_v > 0 else 100.0
                per[(r, s)] += ps
        n += 1
    net.train()
    return {k: v / n for k, v in per.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="__AQUASC_DATA__/yolo_fish_dataset_root")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--val_every", type=int, default=5)
    ap.add_argument("--limit_train", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default=None,
                    help="Wave 21 A-6 (2026-09-15): resume from a state_dict checkpoint, continuing "
                         "the epoch counter from --resume_epoch. LR is constant (Adam, no scheduler) "
                         "so this is a low-risk resume -- only the optimizer's own momentum/variance "
                         "state is not restored (reset to Adam's default init), everything else "
                         "(model weights, RNG-driven rate/snr sampling continuing forward) is exact.")
    ap.add_argument("--resume_epoch", type=int, default=0,
                    help="Epoch index to resume AT (i.e. training continues from range(resume_epoch, epochs)).")
    ap.add_argument("--resume_best", type=float, default=-1.0,
                    help="The historical best val_mean PSNR at --resume_epoch (read from the old "
                         "train_log.csv's 'best' column) -- without this, the resumed run's own "
                         "'best' tracker restarts at -1.0 and can silently overwrite model_best_psnr.pth "
                         "with a worse checkpoint on the very first post-resume validation.")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    out_dir = args.out_dir or os.path.join(HERE, "history", "SwinJSCC_SAandRA_scratch_rayleigh")
    os.makedirs(out_dir, exist_ok=True)

    net = build_net(device)
    nparams = sum(p.numel() for p in net.parameters())
    init_desc = "random"
    if args.resume:
        sd = torch.load(args.resume, map_location=device)
        net.load_state_dict(sd, strict=True)
        init_desc = f"resumed from {args.resume} at epoch {args.resume_epoch}"
    print(f"[scratch] params={nparams/1e6:.2f}M out={out_dir} init={init_desc}", flush=True)

    train_ds = ImageOnly(os.path.join(args.data_path, "train", "images"), limit=args.limit_train)
    val_ds = ImageOnly(os.path.join(args.data_path, "valid", "images"))
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    print(f"[scratch] train={len(train_ds)} valid={len(val_ds)} batch_size=1 lr={args.lr}", flush=True)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    best = args.resume_best
    log_path = os.path.join(out_dir, "train_log.csv")
    val_keys = [(r, s) for r in RATE_CANDIDATES for s in (0, 10, 20)]
    with open(log_path, "w") as f:
        f.write("epoch,loss," + ",".join(f"val_r{r}_s{s}" for r, s in val_keys) + ",val_mean,best,sec_per_epoch\n")
    if args.resume:
        with open(os.path.join(out_dir, "resume_info.json"), "w") as f:
            json.dump(dict(resumed_from=args.resume, resume_epoch=args.resume_epoch,
                           resume_best=args.resume_best), f, indent=2)

    rng = random.Random(args.seed + args.resume_epoch)
    for ep in range(args.resume_epoch, args.epochs):
        t0 = time.time()
        net.train()
        tot, nb = 0.0, 0
        for x in train_loader:
            x = x.to(device)
            rate = rng.choice(RATE_CANDIDATES)
            snr = rng.choice(SNR_CANDIDATES)
            recon, CBR, chan_param, mse, loss_G = net(x, given_SNR=float(snr), given_rate=rate)
            opt.zero_grad()
            loss_G.backward()
            opt.step()
            tot += float(loss_G.item())
            nb += 1
        dt = time.time() - t0
        avg_loss = tot / max(nb, 1)

        row = [str(ep), f"{avg_loss:.6f}"]
        if ep % args.val_every == 0 or ep == args.epochs - 1:
            vp = val_psnr(net, val_loader, device)
            for r, s in val_keys:
                row.append(f"{vp[(r, s)]:.3f}")
            val_mean = sum(vp.values()) / len(vp)
            row.append(f"{val_mean:.4f}")
            is_best = val_mean > best
            if is_best:
                best = val_mean
            row.append(f"{best:.4f}")
            row.append(f"{dt:.1f}")
            with open(log_path, "a") as f:
                f.write(",".join(row) + "\n")
            tag = " <-best" if is_best else ""
            print(f"[scratch] ep={ep} loss={avg_loss:.5f} val_mean_psnr={val_mean:.2f} best={best:.2f} ({dt:.0f}s){tag}", flush=True)
            if is_best:
                torch.save(net.state_dict(), os.path.join(out_dir, "model_best_psnr.pth"))
        else:
            row += [""] * (len(val_keys) + 3)
            with open(log_path, "a") as f:
                f.write(",".join(row) + "\n")
            print(f"[scratch] ep={ep} loss={avg_loss:.5f} ({dt:.0f}s)", flush=True)

    # Save the true final-epoch checkpoint regardless of PSNR-best -- the
    # paper's own thesis is that PSNR does not track task performance, so
    # stopping/selecting by PSNR risks under-reporting a competitor's real
    # mAP (2026-09-14 correction). Evaluate this file, not model_best_psnr.pth,
    # for any comparison that must not be PSNR-selected.
    torch.save(net.state_dict(), os.path.join(out_dir, "model_final.pth"))
    print(f"[scratch] DONE best_val_mean_psnr={best:.4f} (model_final.pth saved at epoch {args.epochs - 1})", flush=True)


if __name__ == "__main__":
    main()
