#!/usr/bin/env python
"""Site B @ 10dB, best achieved result per model, mean +/- std over M=10
independent channel realizations -- one representative checkpoint per
system (not a sweep): AquaSC (S-4 epoch 4, best of the S-2/S-2b/S-3/S-4 full
fine-tune series, RESULTS_OF_RECORD.md section 6aa), ROI-JSCC (fine-tuned
rcpp32, section 6z), JSCC-CNN (fine-tuned BBW c=1, section 6z). All three
share the SAME Site B ceiling (same detector best_siteB.pt, same 455-image
test set) -- 0.6451 per-image / 0.6044 pooled -- so results are plotted as
% of that shared ceiling.

M=10 SOURCE (matched across all three, not approximated): ROI-JSCC/JSCC-CNN
already draw M=10 realizations from channel_bank_siteb.npz per their own
protocol (section 6z); their per-draw pooled/per-image values were read
back from the existing eval logs (logs/v104_{roijscc,jsccnn}_siteb_
finetune_eval_best.log) rather than re-run. AquaSC's own protocol
(S-2/S-2b/S-3) only ever ran a single draw, so a matched M=10 was produced
here by re-running S-4 epoch 4's --eval 10 times at 10dB with --seed
42..51 (run_class_main.py seeds both numpy and torch globally --
run_class_main.py:40-48 -- which drives the per-batch channel noise draw in
eval mode too, confirmed by reading the code before relying on it).
Per-draw logs: logs/v109_aquasc_s4_ep4_snr10_m{0..9}.log, arrays
f1_perimage/s4_ema0p9_m10/m{0..9}.npy.

Run: __TORCH_PY__ plot_siteb_best_per_model.py
"""
from pathlib import Path

import numpy as np

from plot_style import apply_style, plt, COLORS, COL_SINGLE

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT.parent / "figs"
apply_style()

CEILING_PER_IMAGE = 0.6451
CEILING_POOLED = 0.6044

# name -> (per_image_10_draws, pooled_10_draws)
DRAWS = {
    "AquaSC": (
        [0.1363185, 0.1337151, 0.13486043, 0.13521351, 0.1327252,
         0.13439426, 0.13320293, 0.13015343, 0.13061498, 0.13412741],
        [0.0859, 0.0825, 0.0849, 0.0861, 0.0823, 0.0834, 0.0839, 0.0828, 0.0819, 0.0803],
    ),
    "ROI-JSCC": (
        [0.1421, 0.1401, 0.1422, 0.1416, 0.1409, 0.1449, 0.1440, 0.1419, 0.1399, 0.1389],
        [0.1035, 0.1024, 0.1036, 0.1034, 0.1017, 0.1049, 0.1054, 0.1024, 0.1028, 0.1022],
    ),
    "JSCC-CNN": (
        [0.1238, 0.1203, 0.1225, 0.1236, 0.1240, 0.1217, 0.1248, 0.1238, 0.1233, 0.1203],
        [0.0808, 0.0767, 0.0785, 0.0791, 0.0809, 0.0815, 0.0834, 0.0829, 0.0815, 0.0764],
    ),
}

names = list(DRAWS.keys())
pi_mean = np.array([np.mean(DRAWS[n][0]) for n in names]) / CEILING_PER_IMAGE * 100
pi_std = np.array([np.std(DRAWS[n][0], ddof=1) for n in names]) / CEILING_PER_IMAGE * 100
po_mean = np.array([np.mean(DRAWS[n][1]) for n in names]) / CEILING_POOLED * 100
po_std = np.array([np.std(DRAWS[n][1], ddof=1) for n in names]) / CEILING_POOLED * 100

x = np.arange(len(names))
w = 0.34

fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.6))

b1 = ax.bar(x - w / 2, pi_mean, width=w, yerr=pi_std, capsize=2.5,
            color=COLORS["blue"], label="per-image mean AP@0.5", zorder=3,
            error_kw=dict(lw=1.0, ecolor="#333333"))
b2 = ax.bar(x + w / 2, po_mean, width=w, yerr=po_std, capsize=2.5,
            color=COLORS["orange"], label="pooled mAP@0.5", zorder=3,
            error_kw=dict(lw=1.0, ecolor="#333333"))

for rects, means, stds in ((b1, pi_mean, pi_std), (b2, po_mean, po_std)):
    for rect, m, s in zip(rects, means, stds):
        ax.text(rect.get_x() + rect.get_width() / 2, m + s + 0.6,
                f"{m:.1f}", ha="center", va="bottom", fontsize=6.3)

ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel("% of Site B ceiling")
ax.set_ylim(0, 26)
ax.grid(axis="y", color="#DDDDDD", lw=0.5, zorder=0)
ax.legend(loc="upper left", frameon=False)

fig.tight_layout()
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "figure_siteb_best_per_model.png")
fig.savefig(OUT_DIR / "figure_siteb_best_per_model.pdf")
print("Saved figs/figure_siteb_best_per_model.{png,pdf}")
for n, pm, ps, om, os_ in zip(names, pi_mean, pi_std, po_mean, po_std):
    print(f"  {n:10s} per-image = {pm:.2f} +/- {ps:.2f} %   pooled = {om:.2f} +/- {os_:.2f} %  (M=10)")

print()
print("CAPTION DRAFT:")
print(r"""Fig. X. Best achieved Site B task performance per system at 10dB,
mean +/- std over M=10 independent channel realizations, one representative
checkpoint each (not a sweep): AquaSC S-4 epoch 4 (full fine-tune, faster
codebook EMA, Section 6aa), ROI-JSCC fine-tuned rcpp32 (Section 6z),
JSCC-CNN fine-tuned BBW c=1 (Section 6z). All three share the same Site B
detector (best_siteB.pt) and 455-image test set, so results are shown as %
of the same ceiling (0.6451 per-image / 0.6044 pooled) rather than raw AP.
ROI-JSCC/JSCC-CNN's M=10 draws are from their existing bank-based protocol
(channel_bank_siteb.npz); AquaSC's M=10 was produced by re-running its
single-draw analog-proxy protocol 10 times with different seeds to match --
the two protocols are not the same channel-simulation method, only the same
M and the same SNR (see Section 6z caveat 1). Error bars show +/-1 std
across the 10 draws, not a bootstrap CI. All three systems land within
about 3 percentage points of each other at this SNR.""")
