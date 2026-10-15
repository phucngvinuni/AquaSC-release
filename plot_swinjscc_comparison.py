#!/usr/bin/env python
"""Wave 18 exploratory figure: AquaSC flagship vs. every measured baseline
(SwinJSCC from-scratch and fine-tuned, ROI-JSCC, JSCC-CNN), per-image mean
AP@0.5 across the shared 6-point SNR grid. Not a cleared manuscript figure --
the fine-tuned SwinJSCC line is plotted for context only, per the standing
note in RESULTS_OF_RECORD.md section 6o that it is measured but not yet
written up pending a framing decision on the pretraining-corpus disclosure.

Run: __TORCH_PY__ plot_swinjscc_comparison.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "figs"

COLORS = {
    "blue": "#0077BB",
    "teal": "#009988",
    "orange": "#EE7733",
    "purple": "#AA3377",
    "cyan": "#33BBEE",
    "grey": "#777777",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

SNRS = np.array([-5, 0, 5, 10, 15, 20], dtype=float)

FLAGSHIP = np.array([0.2721, 0.2874, 0.3815, 0.5568, 0.6960, 0.7421])       # 0.0714 uses/px
# NOTE (2026-09-14, Wave 19 Part 1): SwinJSCC's rate was mislabeled 0.0417 --
# net/network.py's own CBR formula divides by 3*H*W, not H*W like this
# project's convention; corrected rate is 0.125 (3x). See RESULTS_OF_RECORD.md
# section 6p. The mAP numbers below are unaffected -- only the rate label was wrong.
SWIN_SCRATCH = np.array([0.2517, 0.4640, 0.5898, 0.6492, 0.6764, 0.6818])   # 0.125 uses/px (corrected), random init, 46 effective epochs (300 allocated)
SWIN_FINETUNE = np.array([0.3850, 0.5823, 0.6749, 0.7149, 0.7371, 0.7449])  # 0.125 uses/px (corrected), fine-tuned from released pretrained ckpt
ROIJSCC_128 = np.array([0.3896, 0.4655, 0.5073, 0.5302, 0.5431, 0.5482])    # 0.0469 uses/px, from f1_perimage/roijscc_rcpp128_snr*.npy
JSCCNN_BBW_C1 = np.array([0.3030, 0.4042, 0.4585, 0.4870, 0.5038, 0.5086])  # 0.0625 uses/px, from f1_perimage/jsccnn_bbw_c1_snr*.npy

CEILING = 0.7743  # per-image clean-image ceiling, RESULTS_OF_RECORD.md section 0


def main():
    fig, ax = plt.subplots(figsize=(5.6, 4.1))

    ax.axhline(CEILING, color=COLORS["grey"], linestyle=":", linewidth=1.1, zorder=1)
    ax.text(SNRS[0], CEILING + 0.012, "clean-image ceiling (0.7743)",
            fontsize=7.5, color=COLORS["grey"], ha="left", va="bottom")

    ax.plot(SNRS, FLAGSHIP, marker="o", color=COLORS["blue"], linewidth=2.2,
            markersize=5.5, label="AquaSC flagship (0.0714 uses/px)", zorder=5)
    ax.plot(SNRS, SWIN_SCRATCH, marker="^", color=COLORS["orange"], linewidth=2.0,
            markersize=5.5, label="SwinJSCC, from scratch (0.125 uses/px, corrected)", zorder=4)
    ax.plot(SNRS, SWIN_FINETUNE, marker="s", color=COLORS["teal"], linewidth=1.6,
            markersize=5, linestyle="--",
            label="SwinJSCC, fine-tuned from pretrained (dagger) (0.125 uses/px, corrected)")
    ax.plot(SNRS, ROIJSCC_128, marker="v", color=COLORS["purple"], linewidth=1.4,
            markersize=5, linestyle="-.",
            label="ROI-JSCC, rcpp128 (0.0469 uses/px)")
    ax.plot(SNRS, JSCCNN_BBW_C1, marker="D", color=COLORS["cyan"], linewidth=1.4,
            markersize=4.5, linestyle=":",
            label="JSCC-CNN, BBW c=1 (0.0625 uses/px)")

    # shade the region where SwinJSCC-scratch beats flagship (NOT a "cheaper
    # rate wins" region anymore -- SwinJSCC here is 1.75x flagship's rate;
    # kept only to mark the SNR band where the crossover happens)
    better = SWIN_SCRATCH > FLAGSHIP
    ax.fill_between(SNRS, FLAGSHIP, SWIN_SCRATCH, where=better,
                    color=COLORS["orange"], alpha=0.10, zorder=0)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Per-image mean AP@0.5")
    ax.set_xlim(SNRS.min() - 1, SNRS.max() + 1)
    ax.set_ylim(0, 0.85)
    ax.set_xticks(SNRS)
    ax.legend(loc="lower right", frameon=False, fontsize=7.3)
    ax.set_title("AquaSC vs. measured baselines (SwinJSCC rate corrected 3x, 2026-09-14)",
                 fontsize=9.0)

    fig.text(0.01, -0.02,
             "(dagger) fine-tuned from the paper's released large-corpus-pretrained checkpoint (context only). "
             "SwinJSCC's rate was mislabeled 0.0417 in earlier versions of this figure; corrected to 0.125 "
             "(net/network.py divides by 3xHxW, not HxW -- RESULTS_OF_RECORD.md section 6p). "
             "SwinJSCC here costs MORE rate than flagship, not less.",
             fontsize=6.3, color=COLORS["grey"], ha="left", va="top")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = OUT_DIR / f"figure_swinjscc_comparison_wave18.{suffix}"
        fig.savefig(path, format=suffix, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
