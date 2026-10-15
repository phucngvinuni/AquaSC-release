# SwinJSCC honest-split trainer, V2 (fixes the V1 collapse).
#
# V1 bug diagnosis (2026-08-21): trained checkpoints ignore their latent.
#   decoder(zeros) ~= decoder(features) (rel-diff 0.0034); recon worse than a
#   mean-image baseline; zero SNR sensitivity even through the OFFICIAL forward
#   path -> the checkpoint itself collapsed during V1 training, not just eval.
#   Root cause: V1 bypassed the official forward() and trained a hand-rolled
#   channel wrapper with plain [0,1]-scale F.mse_loss, whereas the known-good
#   fish script (train_swinjscc_on_fish_dataset.py:140) trains via the OFFICIAL
#   net(x) path whose Distortion loss operates at x255 scale. V2 restores the
#   official forward + loss exactly, keeping the honest split + seeds.
import argparse
import hashlib
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "swinjscc_code"))

from net.network import SwinJSCC  # noqa: E402

IMG = 256


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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


def lr_at(epoch, base_lr, epochs, warmup=5):
    if epoch < warmup:
        return base_lr * (epoch + 1) / max(warmup, 1)
    progress = (epoch - warmup) / max(epochs - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def blockfade_channel(flat_tx_norm, snr_db, h=None, noise_real=None):
    """Block-fading Rayleigh + equalization, mirroring the Wave-8 bank /
    roijscc_fish_common.BlockRayleighChannel so all methods share §8.1 channel.

    flat_tx_norm: [B, 2N] per-image power-normalized real latent.
    h: optional [B] complex tensor (bank replay); drawn CN(0,1) if None.
    noise_real: optional [B, 2N] standard-normal (bank replay); drawn if None.
    Returns real latent [B, 2N] AFTER equalization (z_hat scale).
    """
    B, L2 = flat_tx_norm.shape
    N = L2 // 2
    z = torch.complex(flat_tx_norm[:, :N], flat_tx_norm[:, N:])
    sigma = math.sqrt(1.0 / (2.0 * 10 ** (float(snr_db) / 10.0)))
    dev = flat_tx_norm.device
    if h is None:
        h = (torch.randn(B, device=dev) + 1j * torch.randn(B, device=dev)) / np.sqrt(2.0)
    else:
        h = torch.as_tensor(h, device=dev).reshape(-1).to(torch.complex64)
    if noise_real is None:
        noise_real = torch.randn(B, L2, device=dev)
    else:
        noise_real = torch.as_tensor(noise_real, device=dev).reshape(B, L2)
    noise = sigma * (noise_real[:, :N] + 1j * noise_real[:, N:])
    y = z * h.reshape(-1, 1) + noise
    z_hat = y / (h.reshape(-1, 1) + 1e-9)
    return torch.cat([z_hat.real, z_hat.imag], dim=1)


def forward_block(net, x, snr_db, C, mode="official", h=None, noise_real=None):
    """mode 'official': SwinJSCC net() channel semantics are NOT used here;
    see forward_official(). mode 'blockfade': per-image pairing + block fading
    + equalization (the §8.1 convention shared with AquaSC / ROI-JSCC / bank).
    """
    assert mode == "blockfade", "use net(x, given_SNR=..) for the official path"
    feat = net.encoder(x, float(snr_db), C, net.model)
    B = feat.shape[0]
    flat = feat.reshape(B, -1)
    s = flat.pow(2).mean(dim=1, keepdim=True).sqrt()
    tx = flat / s
    z_hat = blockfade_channel(tx, snr_db, h=h, noise_real=noise_real)
    feat_hat = (z_hat * s).reshape_as(feat)
    recon = net.decoder(feat_hat, float(snr_db), net.model)
    return recon.clamp(0.0, 1.0)


def loss_official(net, recon, x):
    """Official Distortion loss (x255-scale MSE variant), as returned by net()."""
    return net.distortion_loss_wrapper(recon.clamp(0., 1.), x)


@torch.no_grad()
def val_psnr(net, loader, device, C, snrs=(0, 10, 20), channel_mode="official"):
    net.eval()
    per = {s: 0.0 for s in snrs}
    n = 0
    for x in loader:
        x = x.to(device)
        B = x.shape[0]
        for s in snrs:
            if channel_mode == "blockfade":
                recon = forward_block(net, x, s, C, mode="blockfade")
            else:
                recon = net(x, given_SNR=float(s))[0].clamp(0., 1.)
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
    ap.add_argument("--limit_train", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="SwinJSCC_w/o_SAandRA")
    ap.add_argument("--channel_mode", choices=["official", "blockfade"],
                    default="official",
                    help="official = net() internal per-element Rayleigh with "
                         "cross-batch complex pairing (collapses for B>1); "
                         "blockfade = per-image pairing + block fading + "
                         "equalization, the §8.1 bank convention")
    args = ap.parse_args()

    seed_everything(42)
    device = torch.device(args.device)
    suffix = "_v2" if args.limit_train is None else f"_smoke{args.limit_train}"
    if args.channel_mode == "blockfade":
        suffix += "_bf"
    out_dir = args.out_dir or os.path.join(
        HERE, "history", f"SwinJSCC_honest_C{args.C}{suffix}")
    os.makedirs(out_dir, exist_ok=True)

    c = build_config(args, device)
    net = SwinJSCC(
        argparse.Namespace(model=args.model, channel_type="rayleigh", C=str(args.C),
                           multiple_snr="10", distortion_metric="MSE",
                           trainset="DIV2K"),
        c,
    ).to(device)
    nparams = sum(p.numel() for p in net.parameters())
    print(f"[train-v2] C={args.C} params={nparams/1e6:.2f}M out={out_dir}", flush=True)

    train_ds = ImageOnly(os.path.join(args.data_path, "train", "images"),
                         limit=args.limit_train)
    val_ds = ImageOnly(os.path.join(args.data_path, "valid", "images"))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    print(f"[train-v2] train={len(train_ds)} valid={len(val_ds)} bs={args.batch_size} "
          f"channel={args.channel_mode}", flush=True)

    if args.channel_mode == "blockfade":
        # one official-path call so update_resolution runs for enc+dec
        with torch.no_grad():
            net(next(iter(val_loader)).to(device), given_SNR=10.0)
        net.train()

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
            if args.channel_mode == "blockfade":
                recon = forward_block(net, x, snr, args.C, mode="blockfade")
                loss_dist = loss_official(net, recon, x)
            else:
                # OFFICIAL forward: update_resolution + internal Rayleigh Channel
                # + Distortion loss at x255 scale (the piece V1 replaced and broke).
                _, _, _, _, loss_dist = net(x, given_SNR=snr)
            opt.zero_grad()
            loss_dist.backward()
            opt.step()
            tot += float(loss_dist.item())
            nb += 1
        dt = time.time() - t0
        if ep % args.val_every == 0 or ep == args.epochs - 1:
            vp = val_psnr(net, val_loader, device, args.C, snrs=(0, 10, 20),
                          channel_mode=args.channel_mode)
            vmean = float(np.mean(list(vp.values())))
            tag = ""
            if vmean > best:
                best = vmean
                torch.save({"model_state_dict": net.state_dict(),
                            "epoch": ep, "C": args.C,
                            "best_val_psnr_mean": best},
                           os.path.join(out_dir, "model_best_psnr.pth"))
                tag = " <-best"
            print(f"[train-v2] C={args.C} ep={ep} loss={tot/max(nb,1):.5f} "
                  f"valPSNR0={vp[0]:.2f} 10={vp[10]:.2f} 20={vp[20]:.2f} "
                  f"mean={vmean:.2f} best={best:.2f} ({dt:.0f}s){tag}", flush=True)
        with open(log_path, "a") as f:
            f.write(f"{ep},{tot/max(nb,1):.6f},,,,,,,{dt:.1f}\n")

    vp = val_psnr(net, val_loader, device, args.C, snrs=(0, 10, 20),
                  channel_mode=args.channel_mode)
    vmean = float(np.mean(list(vp.values())))
    if vmean > best:
        best = vmean
        torch.save({"model_state_dict": net.state_dict(), "epoch": args.epochs - 1,
                    "C": args.C, "best_val_psnr_mean": best},
                   os.path.join(out_dir, "model_best_psnr.pth"))
    ck = os.path.join(out_dir, "model_best_psnr.pth")
    print(f"[train-v2] DONE C={args.C} best_val_psnr_mean={best:.4f} "
          f"sha256={sha256_file(ck)}", flush=True)


if __name__ == "__main__":
    main()
