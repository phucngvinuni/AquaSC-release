#!/usr/bin/env python
"""F5: AP vs rate, one line per SNR -- AquaSC's own joint-frame Pareto
frontier (the only AquaSC configuration plotted against other systems, per
Part B) plus baseline points/curves. This is the primary cross-system
figure; exact rate matching across architectures is impossible by
construction (each has its own discrete rate ladder), so F5 (continuous
rate axis) carries the main comparison and F6's matched-ish-cluster table
is supporting evidence, not the reverse.

AquaSC joint-frame has 3 measured rate points (0.030612, 0.061224, 0.122449) --
a real, honest 2-point frontier, not fabricated intermediate points.

Run: __TORCH_PY__ plot_f5_rate_pareto.py
"""
from pathlib import Path

import numpy as np

from plot_style import apply_style, plt, COLORS, COL_DOUBLE

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT.parent / "figs"
apply_style()

SNRS = [-5, 0, 5, 10, 15, 20]
snr_tags = {-5: "m5", 0: "0", 5: "5", 10: "10", 15: "15", 20: "20"}


def load_mean(pattern, s):
    return float(np.load(ROOT / "f1_perimage" / pattern.format(tag=snr_tags[s])).mean())


import json
sweep_ci = {r["snr_db"]: r["aquasc_mean_per_image"] for r in
            json.loads((ROOT / "experiments/matched5/runs/sweep_full_ci.json").read_text())}

aqua_rates = [0.030612, 0.061224, 0.122449]
aqua_ap = {s: [None, None, None] for s in SNRS}
for s in SNRS:
    aqua_ap[s][2] = sweep_ci[s]
    aqua_ap[s][1] = float(np.load(ROOT / "experiments/matched5/runs/sweep_r061" / f"aqua_{snr_tags[s]}" / "metrics.npz")["per_image_ap"].mean())
    aqua_ap[s][0] = float(np.load(ROOT / "experiments/matched5/runs/sweep_r030" / f"aqua_{snr_tags[s]}" / "metrics.npz")["per_image_ap"].mean())

roi_rates = [0.030612, 0.061224, 0.122449]
roi_ap = {s: [load_mean(f"roijscc_rcpp128_snr{{tag}}.npy", s),
              load_mean(f"roijscc_rcpp64_snr{{tag}}.npy", s),
              load_mean(f"roijscc_rcpp32_snr{{tag}}.npy", s)] for s in SNRS}

swin_rates = [0.0625, 0.125]
swin_r32 = {}
for s in SNRS:
    files = sorted((ROOT / "f1_perimage").glob(f"swinjscc_fixed300_m*_bf_rate32_snr{snr_tags[s]}.npy"))
    swin_r32[s] = float(np.stack([np.load(f) for f in files]).mean())
swin_ap = {s: [swin_r32[s], load_mean("swinjscc_scratch_bf_rate64_snr{tag}.npy", s)] for s in SNRS}

jscc_rates = [0.0625, 0.125]
jscc_ap = {s: [load_mean("jsccnn_bbw_c1_snr{tag}.npy", s), load_mean("jsccnn_bbw_c2_snr{tag}.npy", s)] for s in SNRS}

cmap = plt.get_cmap("viridis")
snr_colors = {s: cmap(i / (len(SNRS) - 1)) for i, s in enumerate(SNRS)}

fig, ax = plt.subplots(figsize=(COL_DOUBLE, 3.6))

for s in SNRS:
    ax.plot(aqua_rates, aqua_ap[s], marker="o", ms=5, lw=2.0, color=snr_colors[s], zorder=4)
    ax.plot(roi_rates, roi_ap[s], marker="s", ms=3.8, lw=1.0, ls="--", color=snr_colors[s], alpha=0.85, zorder=2)
    ax.plot(swin_rates, swin_ap[s], marker="^", ms=4, lw=1.0, ls=":", color=snr_colors[s], alpha=0.85, zorder=2)
    ax.plot(jscc_rates, jscc_ap[s], marker="D", ms=3.5, lw=1.0, ls="-.", color=snr_colors[s], alpha=0.85, zorder=2)

ax.set_xscale("log")
ax.set_xlabel("Channel uses / pixel")
ax.set_ylabel("Mean per-image AP@0.5")
ax.set_ylim(0.0, 0.85)
ax.grid(axis="y", color="#DDDDDD", lw=0.5, zorder=0)
ax.grid(axis="x", which="both", color="#EEEEEE", lw=0.4, zorder=0)

# Legend 1: system (line style/marker), Legend 2: SNR (color)
from matplotlib.lines import Line2D
sys_handles = [
    Line2D([0], [0], color="black", marker="o", lw=2.0, label="AquaSC joint-frame"),
    Line2D([0], [0], color="black", marker="s", lw=1.0, ls="--", label="ROI-JSCC"),
    Line2D([0], [0], color="black", marker="^", lw=1.0, ls=":", label="SwinJSCC"),
    Line2D([0], [0], color="black", marker="D", lw=1.0, ls="-.", label="JSCC-CNN (BBW)"),
]
leg1 = ax.legend(handles=sys_handles, loc="upper left", frameon=False, title="system", fontsize=6.5)
ax.add_artist(leg1)
snr_handles = [Line2D([0], [0], color=snr_colors[s], marker="o", lw=2.0, label=f"{s:+d} dB") for s in SNRS]
ax.legend(handles=snr_handles, loc="lower right", frameon=False, title="SNR", ncol=2, fontsize=6.3)

fig.tight_layout()
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "figure_f5_rate_pareto.png")
fig.savefig(OUT_DIR / "figure_f5_rate_pareto.pdf")
print("Saved figs/figure_f5_rate_pareto.{png,pdf}")

print()
print("CAPTION DRAFT:")
print(r"""Fig. 5. Mean per-image AP@0.5 vs. channel uses/pixel (log scale), one curve
per SNR (color) x system (marker/linestyle). AquaSC joint-frame (solid
circles) is the only AquaSC configuration shown -- its own 3-point measured
frontier (0.030612, 0.061224, 0.122449 uses/px, bit-identical to ROI-JSCC's
three corrected rcpp rates; no intermediate points fabricated), per Part B.
ROI-JSCC (dashed squares, rcpp32/64/128, corrected rates, Section 6r),
SwinJSCC (dotted triangles, r32 fixed-300-epoch and r64 46-epoch scratch
checkpoint, corrected rates, Sections 6o/6p/6u), JSCC-CNN BBW (dash-dot
diamonds, c=1/c=2, Section
1/6n). Flagship checkpoint (SHA prefix b1d2eb25); channel_bank.npz M=10
except SwinJSCC r64 and the ablation-family points, single-seed (disclosed
per-curve in Sections 6o/6u). Exact cross-architecture rate matching is not
achievable by construction -- each system has its own discrete rate ladder,
demonstrated inside AquaSC itself by RUN-3 and L2 independently landing on
the identical 0.0816 uses/px through padding quantisation, not by design.
This figure is the primary cross-system comparison; Fig. 6's merged-cluster
table is supporting evidence at one near-matched operating point.""")
