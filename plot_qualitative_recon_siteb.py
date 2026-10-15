#!/usr/bin/env python
"""Qualitative reconstruction examples, Site B, zero-shot: original Site B
image vs. AquaSC joint-frame reconstruction (SAME flagship checkpoint, no
retraining) at several SNRs -- visualizes the section 6x domain-transfer
failure directly: the decoder emits a plausible-looking but Site-A-shaped
prior, not a reconstruction of the actual Site B scene/species.

Run: __TORCH_PY__ plot_qualitative_recon_siteb.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments/matched5"))
import evaluate as ev

from plot_style import apply_style, plt, COL_DOUBLE

apply_style()

ROOT = Path(__file__).resolve().parent
SNR_POINTS = [-5, 0, 10, 20]
snr_tags = {-5: "-5", 0: "0", 10: "10", 20: "20"}
IMAGE_INDICES = [3, 40, 120, 200, 300]

net = ev.load_aquasc("test", "b")
ds = ev.dataset("test", "b")

receivers = {}
for s in SNR_POINTS:
    rx = np.load(ROOT / "experiments/matched5/runs/sweep_siteb" / f"phy_{snr_tags[s]}" / "receiver.npz")
    receivers[s] = rx

n_rows = len(IMAGE_INDICES)
n_cols = 1 + len(SNR_POINTS)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(COL_DOUBLE, 1.7 * n_rows))

for row, idx in enumerate(IMAGE_INDICES):
    img_tensor = ds[idx][0][0]
    orig = img_tensor.permute(1, 2, 0).numpy()
    axes[row, 0].imshow(np.clip(orig, 0, 1))
    axes[row, 0].set_ylabel(f"#{idx}", fontsize=7)
    if row == 0:
        axes[row, 0].set_title("original (Site B)", fontsize=8)
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
fig.savefig(OUT_DIR / "figure_qualitative_recon_siteb.png", dpi=200)
fig.savefig(OUT_DIR / "figure_qualitative_recon_siteb.pdf")
print("Saved figs/figure_qualitative_recon_siteb.{png,pdf}")
print("Rows (image index):", IMAGE_INDICES)
print("Columns (SNR):", SNR_POINTS, "(realization m=0 of the channel_bank_siteb.npz M=10 sweep)")
