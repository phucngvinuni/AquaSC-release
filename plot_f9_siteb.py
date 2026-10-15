#!/usr/bin/env python
"""F9: AquaSC joint-frame (0.122449 uses/px), zero-shot on Site B (SAME
flagship checkpoint, no retraining -- matches the S-1 protocol,
RESULTS_OF_RECORD.md section 6c), with Site B's OWN ceiling and floor as
horizontal references (section 6b) -- never Site A's.

Run: __TORCH_PY__ plot_f9_siteb.py
"""
from pathlib import Path

import numpy as np

from plot_style import apply_style, plt, COLORS, COL_SINGLE

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT.parent / "figs"
apply_style()

CEILING_SITEB_PER_IMAGE = 0.6451
FLOOR_SITEB_PER_IMAGE = 0.0012

SNRS = np.array([-5, 0, 5, 10, 15, 20], dtype=float)
snr_tags = {-5: "-5", 0: "0", 5: "5", 10: "10", 15: "15", 20: "20"}

per_image = []
pooled = []
for s in SNRS:
    import json
    summ = json.loads((ROOT / "experiments/matched5/runs/sweep_siteb" / f"aqua_{snr_tags[s]}" / "summary.json").read_text())
    assert summ["site"] == "b" and summ["n"] == 455
    per_image.append(summ["mean_per_image_ap50"])
    pooled.append(summ["pooled_map50"])
per_image = np.array(per_image)
pooled = np.array(pooled)

fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.6))
ax.axhline(CEILING_SITEB_PER_IMAGE, color=COLORS["grey"], lw=1.0, ls="--", zorder=1)
ax.text(SNRS[0], CEILING_SITEB_PER_IMAGE - 0.045, "Site B ceiling", fontsize=6.5,
        color=COLORS["grey"], va="top")
ax.axhline(FLOOR_SITEB_PER_IMAGE, color=COLORS["grey"], lw=1.0, ls=":", zorder=1)
ax.text(SNRS[0], FLOOR_SITEB_PER_IMAGE + 0.025, "Site B prior floor", fontsize=6.5,
        color=COLORS["grey"], va="bottom")

ax.plot(SNRS, per_image, marker="o", ms=4.5, lw=1.5, color=COLORS["blue"],
        label="per-image mean AP@0.5 (n=455)", zorder=3)
ax.plot(SNRS, pooled, marker="s", ms=4, lw=1.3, color=COLORS["orange"],
        label="pooled mAP@0.5", ls="--", zorder=3)

ax.set_xlabel("SNR (dB)")
ax.set_ylabel("AP@0.5, Site B")
ax.set_xticks(SNRS)
ax.set_ylim(-0.02, 0.72)
ax.grid(axis="y", color="#DDDDDD", lw=0.5, zorder=0)
ax.legend(loc="center right", frameon=False)

fig.tight_layout()
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "figure_f9_siteb.png")
fig.savefig(OUT_DIR / "figure_f9_siteb.pdf")
print("Saved figs/figure_f9_siteb.{png,pdf}")
for s, pi, po in zip(SNRS, per_image, pooled):
    print(f"  {s:+3.0f}dB: per-image={pi:.4f} ({pi/CEILING_SITEB_PER_IMAGE*100:.2f}% of ceiling) pooled={po:.4f}")

print()
print("CAPTION DRAFT:")
print(r"""Fig. 9. AquaSC joint-frame (0.122449 uses/px), zero-shot on Site B -- the
SAME flagship checkpoint (SHA prefix b1d2eb25), no retraining, matching the
S-1 protocol (Section 6c). Site B's own detector (best_siteB.pt, SHA prefix
1d85f04d), own channel bank (channel_bank_siteb.npz, n=455, M=10), own
ceiling (0.6451 per-image, dashed) and prior floor (0.0012 per-image,
dotted) -- never Site A's 0.7743/0.2711. Reconstruction carries essentially
no task-relevant information at any SNR (per-image AP sits at or near the
Site B prior floor throughout), consistent with the domain-transfer failure
already established for the multi-block chain (Sections 6c/6d) and for
ROI-JSCC/JSCC-CNN (Sections 6m/6n) -- the transport architecture (joint-frame
vs. multi-block) does not change this outcome, confirming the failure is
located in the encoder/decoder's Site-A-learned representation, not in the
digital transport.""")
