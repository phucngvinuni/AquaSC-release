#!/usr/bin/env python
"""F4: Ablation, four arms, at 10 and 20 dB, per-image AP with 95% CI. Self
comparison only (Part B rule).

Data: RESULTS_OF_RECORD.md section 3 (E-2 re-run, J2). Channel: uncoded
digital (16-QAM, no LDPC), NOT the real coded chain -- stated explicitly in
the caption per the record's own naming-correction note.

Run: __TORCH_PY__ plot_f4_ablation.py
"""
from pathlib import Path

import numpy as np

from plot_style import apply_style, plt, COLORS, COL_SINGLE, CEILING_PER_IMAGE

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT.parent / "figs"
apply_style()

ARMS = [
    ("Base\n(neither)", "ablation_BASE", COLORS["grey"]),
    ("FIM-NoBB", "ablation_FIM_NOBB", COLORS["teal"]),
    ("BB-NoFIM", "ablation_BB_NOFIM_V2", COLORS["orange"]),
    ("Full\n(flagship)", "ablation_FULL", COLORS["blue"]),
]
SNRS = [10, 20]

data = {}
for label, tag, color in ARMS:
    for snr in SNRS:
        arr = np.load(ROOT / "f1_perimage" / f"{tag}_snr{snr}.npy")
        n = len(arr)
        mean = arr.mean()
        se = arr.std(ddof=1) / np.sqrt(n)
        ci = 1.96 * se
        data[(label, snr)] = (mean, ci, n)

fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.6))
x = np.arange(len(ARMS))
width = 0.35
for i, snr in enumerate(SNRS):
    means = [data[(label, snr)][0] for label, _, _ in ARMS]
    cis = [data[(label, snr)][1] for label, _, _ in ARMS]
    offset = (i - 0.5) * width
    color = COLORS["cyan"] if snr == 10 else COLORS["red"]
    ax.bar(x + offset, means, width, yerr=cis, capsize=2.5, color=color,
           alpha=0.85, label=f"{snr} dB", error_kw=dict(lw=0.9, capthick=0.9))

ax.axhline(CEILING_PER_IMAGE, color=COLORS["grey"], lw=0.9, ls="--", zorder=0)
ax.text(2.5, CEILING_PER_IMAGE + 0.015, "ceiling", fontsize=6.3,
        color=COLORS["grey"], ha="center", va="bottom")

ax.set_xticks(x)
ax.set_xticklabels([a[0] for a in ARMS])
ax.set_ylabel("Mean per-image AP@0.5")
ax.set_ylim(0, 0.85)
ax.grid(axis="y", color="#DDDDDD", lw=0.5, zorder=0)
ax.set_axisbelow(True)
ax.legend(loc="upper left", frameon=False, title="uncoded-digital channel")

fig.tight_layout()
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "figure_f4_ablation.png")
fig.savefig(OUT_DIR / "figure_f4_ablation.pdf")
print("Saved figs/figure_f4_ablation.{png,pdf}")
for label, _, _ in ARMS:
    for snr in SNRS:
        m, ci, n = data[(label, snr)]
        print(f"  {label.replace(chr(10),' ')} @{snr}dB: {m:.4f} +/- {ci:.4f} (95% CI, n={n})")

print()
print("CAPTION DRAFT:")
print(r"""Fig. 4. Four-arm ablation (FIM routing, BBW loss, both, neither), per-image
mean AP@0.5 at 10 and 20 dB, 95%% CI (n=432 per bar). Uncoded-digital channel
(16-QAM, no LDPC coding -- NOT the real coded chain used elsewhere; see
RESULTS_OF_RECORD.md Section 3's naming-correction note), flagship-family
checkpoints (SHA prefixes in Table [ablation]), per-image ceiling 0.7743
(dashed). Contributions are approximately additive: +0.085 (FIM) and +0.091
(BBW) individually vs. +0.184 jointly at 10 dB; +0.101 and +0.312 vs. +0.395
at 20 dB.""")
