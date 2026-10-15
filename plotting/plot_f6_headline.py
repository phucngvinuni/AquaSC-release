#!/usr/bin/env python
"""F6 (headline): AP vs SNR at the merged ~0.123 cluster -- AquaSC joint-frame
(0.122449), ROI-JSCC rcpp32 (0.122449, corrected), SwinJSCC r64 (0.125,
corrected), JSCC-CNN BBW c=2 (0.125). The two exact rates differ by 2.08%;
merged into one cluster per the dispatch, stated in the caption, AquaSC's own
rate is the lower (cheaper) one -- said explicitly, not left for the reader
to compute.

Run: __TORCH_PY__ plotting/plot_f6_headline.py
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repo root (core modules)
from pathlib import Path

import numpy as np

from plot_style import apply_style, plt, COLORS, COL_SINGLE, CEILING_PER_IMAGE

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT.parent / "figs"
apply_style()

SNRS = np.array([-5, 0, 5, 10, 15, 20], dtype=float)
snr_tags = {-5: "m5", 0: "0", 5: "5", 10: "10", 15: "15", 20: "20"}


def load(pattern):
    return np.array([np.load(ROOT / "f1_perimage" / pattern.format(tag=snr_tags[s])).mean() for s in SNRS])


roijscc = load("roijscc_rcpp32_snr{tag}.npy")
swinjscc = load("swinjscc_scratch_bf_rate64_snr{tag}.npy")
jsccnn = load("jsccnn_bbw_c2_snr{tag}.npy")

import json
sweep_ci = {r["snr_db"]: r for r in json.loads((ROOT / "experiments/matched5/runs/sweep_full_ci.json").read_text())}
jointframe = np.array([sweep_ci[s]["aquasc_mean_per_image"] for s in SNRS])

fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.8))
ax.axhline(CEILING_PER_IMAGE, color=COLORS["grey"], lw=0.8, ls="--", zorder=0)

ax.plot(SNRS, jointframe, marker="o", ms=4.5, lw=1.6, color=COLORS["blue"],
        label="AquaSC joint-frame (0.1224)", zorder=3)
ax.plot(SNRS, roijscc, marker="s", ms=4, lw=1.3, color=COLORS["orange"],
        label="ROI-JSCC rcpp32 (0.1224)")
ax.plot(SNRS, swinjscc, marker="^", ms=4, lw=1.3, color=COLORS["teal"],
        label="SwinJSCC r64 (0.1250)")
ax.plot(SNRS, jsccnn, marker="D", ms=4, lw=1.3, color=COLORS["purple"],
        label="JSCC-CNN BBW c=2 (0.1250)")

ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Mean per-image AP@0.5")
ax.set_xticks(SNRS)
ax.set_ylim(0.0, 0.85)
ax.grid(axis="y", color="#DDDDDD", lw=0.5, zorder=0)
ax.legend(loc="lower right", frameon=False)

fig.tight_layout()
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "figure_f6_headline.png")
fig.savefig(OUT_DIR / "figure_f6_headline.pdf")
print("Saved figs/figure_f6_headline.{png,pdf}")
print(f"{'SNR':>5} {'joint-frame':>12} {'rcpp32':>8} {'SwinJSCC64':>11} {'JSCCNN-c2':>10}")
for s, a, b, c, d in zip(SNRS, jointframe, roijscc, swinjscc, jsccnn):
    print(f"{s:+5.0f} {a:>12.4f} {b:>8.4f} {c:>11.4f} {d:>10.4f}")

print()
print("CAPTION DRAFT:")
print(r"""Fig. 6. Mean per-image AP@0.5 vs. SNR at the merged ~0.123-uses/px cluster:
AquaSC joint-frame, ROI-JSCC rcpp32, SwinJSCC r64, JSCC-CNN BBW c=2. All
systems are within 2.1%% of 0.123 uses/px; exact per-system rates in Table II
(0.122449 for AquaSC/ROI-JSCC, 0.125 for SwinJSCC/JSCC-CNN) -- AquaSC sits at
the LOWER (2.08%% cheaper) rate, not the higher one. Flagship checkpoint (SHA
prefix b1d2eb25) for AquaSC; ROI-JSCC/JSCC-CNN 100 training epochs, SwinJSCC
rate-64 scratch checkpoint 46 trained epochs (best validation PSNR at epoch
40), and AquaSC 300. Per-image ceiling 0.7743 (dashed).
channel_bank.npz M=10 for AquaSC/ROI-JSCC/JSCC-CNN; SwinJSCC single-seed,
same protocol family as the flagship's own numbers (Section 6o Check 3) --
disclosed, not a like-for-like M=10 comparison on that one curve. Exact
cross-architecture rate matching is not achievable by construction (each
system has its own discrete rate ladder -- demonstrated inside AquaSC itself
by RUN-3 and L2 independently landing on the identical 0.0816 uses/px through
padding quantisation, not by design); Fig. 5's rate curve is the primary
cross-system comparison, this figure and Table II are supporting evidence at
one particular matched-ish operating point.""")
