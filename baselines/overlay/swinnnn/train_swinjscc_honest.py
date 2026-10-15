# train_swinjscc_honest.py — Wave 9b SWINJSCC agent.
#
# Trains the official SwinJSCC (arXiv 2308.09361) "SwinJSCC_w/o_SAandRA" model
# on the HONEST video-disjoint fish split (yolo_fish_dataset_root) at a single
# rate point (bottleneck dim C), over a Rayleigh channel sampled uniformly in
# [0,20] dB to match AquaSC's training channel.
#
# ADAPTATIONS vs the official repo (documented for the report):
#   1. Official SwinJSCC requires python 3.8 / torch 1.9; we run under
#      siso-project-env (python 3.10 / torch 2.1.2 / timm 0.9.12).  Verified
#      working: timm DropPath/to_2tuple/trunc_normal_ exist; loss.distortion
#      @torch.jit.script functions import cleanly.
#   2. Official create_encoder()/create_decoder() fail to forward the `model`
#      positional arg to SwinJSCC_Encoder/SwinJSCC_Decoder (a repo bug).  We
#      inject model=<name> into encoder_kwargs/decoder_kwargs.
#   3. Official Channel uses PER-ELEMENT Rayleigh fading.  To replay the same
#      seeded channel bank as AquaSC (block fading: one scalar h per image) we
#      replace the channel with block-fading Rayleigh (scalar h, CSIR-free,
#      noise variance 1/snr_linear) — implemented inline in forward_block().
#   4. Operating resolution 256x256 (window_size=8 needs input % 128 == 0;
#      224 fails at the 14x14 deepest layer).  Test images are 224x224 and are
#      bicubic-upscaled to 256 for the encoder; reconstructions are
#      bicubic-downscaled back to 224 for PSNR/SSIM and detector scoring.

import argparse
import math
import os
import sys
import time
import hashlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "swinjscc_code"))

from net.network import SwinJSCC  # noqa: E402

IMG = 256


def seed_everything(seed=42):
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ImageOnly(Dataset):
    def __init__(self, image_dir, size=IMG):
        self.paths = sorted(os.path.join(image_dir, f) for f in os.listdir(image_dir)
                            if f.lower().endswith((".png", ".jpg", ".jpeg")))
        self.t = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.t(Image.open(self.paths[idx]).convert("RGB"))


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


def forward_block(net, x, snr_db, C, h=None, noise_real=None):
    """SwinJSCC_w/o_SAandRA encoder -> block-fading Rayleigh -> decoder.

    Mirrors the official net/channel.py Channel.forward Rayleigh branch (power
    normalization -> complex pair -> *h -> +noise -> un-complex -> rescale) but
    with BLOCK fading (one scalar h per image) and an externally supplied
    standard-normal noise stream (used for seeded-bank replay).  CSIR-free (no
    ZF equalization), matching the official SwinJSCC decoder contract.
    """
    B = x.shape[0]
    feature = net.encoder(x, snr_db, C, net.model)          # [B, 256, C]
    pwr = (feature ** 2).mean() * 2.0                        # complex power norm
    tx = feature / pwr.sqrt()
    flat = tx.reshape(B, -1)
    half = flat.shape[1] // 2
    cin = torch.complex(flat[:, :half], flat[:, half:])      # [B, half]
    sigma = (1.0 / (2.0 * 10 ** (float(snr_db) / 10.0))) ** 0.5

    if h is None:
        hr = torch.randn(B, device=x.device, dtype=x.dtype)
        hi = torch.randn(B, device=x.device, dtype=x.dtype)
        h = torch.complex(hr, hi) / np.sqrt(2.0)
    else:
        h = torch.as_tensor(h, device=x.device)

    if noise_real is None:
        nr = torch.randn(B, half, device=x.device, dtype=x.dtype)
        ni = torch.randn(B, half, device=x.device, dtype=x.dtype)
        noise = (nr + 1j * ni) * sigma
    else:
        nr = noise_real[:, :half]
        ni = noise_real[:, half:]
        noise = (nr + 1j * ni) * sigma

    y = cin * h[:, None] + noise
    yflat = torch.cat([y.real, y.imag], dim=1) * pwr.sqrt()
    noisy_feature = yflat.reshape_as(feature)
    recon = net.decoder(noisy_feature, snr_db, net.model)
    return recon.clamp(0.0, 1.0)


@torch.no_grad()
def val_psnr(net, loader, device, C, snrs=(0, 10, 20)):
    net.eval()
    per = {s: 0.0 for s in snrs}
    n = 0
    for x in loader:
        x = x.to(device)
        B = x.shape[0]
        for s in snrs:
            recon = forward_block(net, x, s, C)
            mse = ((recon - x) ** 2).mean().item()
            ps = -10 * np.log10(mse) if mse > 0 else 100.0
            per[s] += ps * B
        n += B
    net.train()
    return {s: per[s] / n for s in snrs}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lr_at(epoch, base_lr, epochs, warmup=5):
    if epoch < warmup:
        return base_lr * (epoch + 1) / max(warmup, 1)
    progress = (epoch - warmup) / max(epochs - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=int, required=True)
    ap.add_argument("--data_path", default="__AQUASC_DATA__/yolo_fish_dataset_root")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_every", type=int, default=5)
    ap.add_argument("--snr_min", type=float, default=0.0)
    ap.add_argument("--snr_max", type=float, default=20.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="SwinJSCC_w/o_SAandRA")
    args = ap.parse_args()

    seed_everything(42)
    device = torch.device(args.device)
    out_dir = args.out_dir or os.path.join(
        HERE, "history", f"SwinJSCC_honest_C{args.C}")
    os.makedirs(out_dir, exist_ok=True)

    c = build_config(args, device)
    net = SwinJSCC(
        argparse.Namespace(model=args.model, channel_type="rayleigh", C=str(args.C),
                           multiple_snr="10", distortion_metric="MSE",
                           trainset="DIV2K"),
        c,
    ).to(device)
    nparams = sum(p.numel() for p in net.parameters())
    print(f"[train] C={args.C} params={nparams/1e6:.2f}M out={out_dir}", flush=True)

    train_ds = ImageOnly(os.path.join(args.data_path, "train", "images"))
    val_ds = ImageOnly(os.path.join(args.data_path, "valid", "images"))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    print(f"[train] train={len(train_ds)} valid={len(val_ds)} bs={args.batch_size}", flush=True)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    best = -1.0
    log_path = os.path.join(out_dir, "train_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,loss,val_psnr0,val_psnr10,val_psnr20,val_mean,best,sec_per_epoch\n")

    for ep in range(args.epochs):
        for g in opt.param_groups:
            g["lr"] = lr_at(ep, args.lr, args.epochs)
        t0 = time.time()
        net.train()
        tot = 0.0
        nb = 0
        for x in train_loader:
            x = x.to(device)
            snr = float(np.random.uniform(args.snr_min, args.snr_max))
            recon = forward_block(net, x, snr, args.C)
            loss = F.mse_loss(recon, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        dt = time.time() - t0
        if ep % args.val_every == 0 or ep == args.epochs - 1:
            vp = val_psnr(net, val_loader, device, args.C, snrs=(0, 10, 20))
            vmean = float(np.mean(list(vp.values())))
            is_best = vmean > best
            if is_best:
                best = vmean
                ck = os.path.join(out_dir, "model_best_psnr.pth")
                torch.save({"model_state_dict": net.state_dict(), "epoch": ep, "C": args.C,
                            "best_val_psnr_mean": best}, ck)
            with open(log_path, "a") as f:
                f.write(f"{ep},{tot/max(nb,1):.6f},{vp[0]:.4f},{vp[10]:.4f},{vp[20]:.4f},"
                        f"{vmean:.4f},{best:.4f},{dt:.1f}\n")
            print(f"[train] C={args.C} ep={ep} loss={tot/max(nb,1):.6f} "
                  f"valPSNR0={vp[0]:.2f} 10={vp[10]:.2f} 20={vp[20]:.2f} "
                  f"mean={vmean:.2f} best={best:.2f} lr={opt.param_groups[0]['lr']:.2e} ({dt:.1f}s)", flush=True)
        else:
            with open(log_path, "a") as f:
                f.write(f"{ep},{tot/max(nb,1):.6f},,,,{best:.4f},{dt:.1f}\n")
            print(f"[train] C={args.C} ep={ep} loss={tot/max(nb,1):.6f} "
                  f"lr={opt.param_groups[0]['lr']:.2e} ({dt:.1f}s)", flush=True)

    ck = os.path.join(out_dir, "model_best_psnr.pth")
    if os.path.exists(ck):
        print(f"[train] DONE C={args.C} best_val_psnr_mean={best:.4f} "
              f"sha256={sha256_file(ck)}", flush=True)
    else:
        print(f"[train] DONE C={args.C} (no checkpoint saved)", flush=True)


if __name__ == "__main__":
    main()
