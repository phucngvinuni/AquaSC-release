#!/usr/bin/env python
"""F3: AquaSC joint-frame through the real coded chain, self only, with the
clean-image ceiling and prior floor as horizontal references.

Data: RESULTS_OF_RECORD.md section 6s (joint-frame @0.122449 uses/px,
channel_bank.npz M=10, n=432, checkpoint SHA b1d2eb25...), section 0
(ceiling/floor).

Run: __TORCH_PY__ plotting/plot_f3_jointframe_ceiling.py
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repo root (core modules)
import json
from pathlib import Path

import numpy as np

from plot_style import apply_style, plt, COLORS, COL_SINGLE, CEILING_PER_IMAGE, PRIOR_FLOOR

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT.parent / "figs"
apply_style()

SNRS = np.array([-5, 0, 5, 10, 15, 20], dtype=float)
sweep_ci = {r["snr_db"]: r for r in json.loads((ROOT / "experiments/matched5/runs/sweep_full_ci.json").read_text())}
per_image = np.array([sweep_ci[s]["aquasc_mean_per_image"] for s in SNRS])
pooled = np.array([sweep_ci[s]["aquasc_pooled"] for s in SNRS])

fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.6))
ax.axhline(CEILING_PER_IMAGE, color=COLORS["grey"], lw=1.0, ls="--", zorder=1)
ax.text(SNRS[0], CEILING_PER_IMAGE + 0.012, "clean-image ceiling", fontsize=6.5,
        color=COLORS["grey"], va="bottom")
ax.axhline(PRIOR_FLOOR, color=COLORS["grey"], lw=1.0, ls=":", zorder=1)
ax.text(SNRS[0], PRIOR_FLOOR + 0.012, "prior floor", fontsize=6.5,
        color=COLORS["grey"], va="bottom")

ax.plot(SNRS, per_image, marker="o", ms=4.5, lw=1.5, color=COLORS["blue"],
        label="per-image mean AP@0.5", zorder=3)
ax.plot(SNRS, pooled, marker="s", ms=4, lw=1.3, color=COLORS["orange"],
        label="pooled mAP@0.5", ls="--", zorder=3)

ax.set_xlabel("SNR (dB)")
ax.set_ylabel("AP@0.5")
ax.set_xticks(SNRS)
ax.set_ylim(0.0, 0.85)
ax.grid(axis="y", color="#DDDDDD", lw=0.5, zorder=0)
ax.legend(loc="lower right", frameon=False)

fig.tight_layout()
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "figure_f3_jointframe_ceiling.png")
fig.savefig(OUT_DIR / "figure_f3_jointframe_ceiling.pdf")
print("Saved figs/figure_f3_jointframe_ceiling.{png,pdf}")
for s, pi, po in zip(SNRS, per_image, pooled):
    print(f"  {s:+3.0f}dB: per-image={pi:.4f} pooled={po:.4f}")

print()
print("CAPTION DRAFT:")
print(r"""Fig. 3. AquaSC joint-frame (0.122449 uses/px, one BG2 codeword/image) through
the real coded channel, flagship checkpoint (SHA prefix b1d2eb25), perfect CSI,
n=432, M=10 (channel_bank.npz). Both per-image mean AP@0.5 (against the
per-image ceiling of 0.7743, dashed grey) and pooled mAP@0.5 are shown; the
dotted grey line is the Site A prior floor (0.2711 -- decoder fed random
codebook indices, no image content, Section 0/6f of RESULTS_OF_RECORD.md;
the floor is the mean of the 432 per-image AP values in
f1_perimage/prior_floor_per_image.npy; the separately measured pooled floor
is 0.2198 and is not drawn).""")
