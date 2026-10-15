#!/usr/bin/env python
"""Qualitative reconstruction examples: original vs. AquaSC joint-frame
(0.122449 uses/px) reconstruction at several SNRs, using the exact decoded
tokens/mask/frame_ok already computed for section 6s (realization m=0 of
the channel_bank.npz M=10 sweep) -- no new channel simulation, just decoding
and rendering already-verified receiver data.

Run: __TORCH_PY__ plotting/plot_qualitative_recon.py
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repo root (core modules)
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/matched5"))
import evaluate as ev

from plot_style import apply_style, plt, COL_DOUBLE

apply_style()

ROOT = Path(__file__).resolve().parents[1]
SNR_POINTS = [-5, 0, 10, 20]
snr_tags = {-5: "m5", 0: "0", 10: "10", 20: "20"}
IMAGE_INDICES = [3, 40, 120, 200, 300]  # spread across the 432-image test set

net = ev.load_aquasc("test")
ds = ev.dataset("test")

receivers = {}
for s in SNR_POINTS:
    tag = snr_tags[s]
    d = "phy_m5" if s == -5 else f"phy_{tag}"
    rx = np.load(ROOT / "experiments/matched5/runs/sweep" / d / "receiver.npz")
    receivers[s] = rx

n_rows = len(IMAGE_INDICES)
n_cols = 1 + len(SNR_POINTS)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(COL_DOUBLE, 1.7 * n_rows))

for row, idx in enumerate(IMAGE_INDICES):
    img_tensor = ds[idx][0][0]  # (image, mask) tuple -> image
    orig = img_tensor.permute(1, 2, 0).numpy()
    axes[row, 0].imshow(np.clip(orig, 0, 1))
    axes[row, 0].set_ylabel(f"#{idx}", fontsize=7)
    if row == 0:
        axes[row, 0].set_title("original", fontsize=8)
    axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])

    for col, s in enumerate(SNR_POINTS, start=1):
        rx = receivers[s]
        m = 0
        with torch.no_grad():
            recon = ev.decode_frame(
                net,
                rx["tokens"][idx:idx+1, m],
                rx["mask"][idx:idx+1, m],
                rx["frame_ok"][idx:idx+1, m],
            )
        recon_img = recon[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        ok = bool(rx["frame_ok"][idx, m])
        axes[row, col].imshow(recon_img)
        if row == 0:
            axes[row, col].set_title(f"{s:+d} dB", fontsize=8)
        tag = "ok" if ok else "LOST"
        color = "black" if ok else "#CC3311"
        axes[row, col].text(0.03, 0.94, tag, transform=axes[row, col].transAxes,
                             fontsize=6.5, color=color, va="top",
                             bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
        axes[row, col].set_xticks([]); axes[row, col].set_yticks([])

fig.tight_layout()
OUT_DIR = ROOT.parent / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "figure_qualitative_recon.png", dpi=200)
fig.savefig(OUT_DIR / "figure_qualitative_recon.pdf")
print("Saved figs/figure_qualitative_recon.{png,pdf}")
print("Rows (image index):", IMAGE_INDICES)
print("Columns (SNR):", SNR_POINTS, "(realization m=0 of the channel_bank.npz M=10 sweep)")
