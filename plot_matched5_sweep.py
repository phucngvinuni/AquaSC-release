#!/usr/bin/env python
"""Wave 20 matched-rate figure: AquaSC joint-frame QPSK vs. ROI-JSCC rcpp32,
both at exactly 6144 complex symbols/image = 0.122449 uses/px, full 6-point
SNR sweep (-5/0/5/10/15/20 dB), Site A test (432 images, M=10 realizations).

AquaSC joint-frame beats ROI-JSCC from 0dB up (tie at 0dB, clear wins
5-20dB) but LOSES at -5dB -- the single-frame/single-CRC design has no
partial-failure recovery (the old UEP multi-block chain did), so a lost
frame drops the WHOLE image to the all-zero-latent fallback. Frame-loss
rate is plotted on a secondary axis to make this visible, not just asserted.

Not a cleared manuscript figure. Source data: experiments/matched5/runs/
sweep_summary.json (5dB point uses the originally locked comparison_test_v1
result; -5/0/10/15/20dB are a same-day extension of the same protocol,
reading real per-SNR h/noise from channel_bank.npz -- see phy.py's
transmit_cache --snr-db path, added 2026-09-15).

Run: __TORCH_PY__ plot_matched5_sweep.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT.parent / "figs"

COLORS = {
    "blue": "#0077BB",
    "orange": "#EE7733",
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

data = json.loads((ROOT / "experiments/matched5/runs/sweep_summary.json").read_text())
ci_data = {r["snr_db"]: r for r in json.loads((ROOT / "experiments/matched5/runs/sweep_full_ci.json").read_text())}
rows = sorted(data["rows"], key=lambda r: r["snr_db"])
snrs = np.array([r["snr_db"] for r in rows], dtype=float)
aqua = np.array([r["aquasc_joint_qpsk"] for r in rows])
roi = np.array([r["roijscc_rcpp32"] for r in rows])
frame_loss = np.array([r["aquasc_frame_loss_rate"] if r["aquasc_frame_loss_rate"] is not None
                        else np.nan for r in rows])
rate = data["rate_uses_px"]
holm_sig = np.array([ci_data[s]["holm_significant"] for s in snrs])
op_win = np.array([ci_data[s]["operational_win"] for s in snrs])

fig, ax = plt.subplots(figsize=(5.8, 4.2))

ax.plot(snrs, aqua, marker="o", ms=5.5, lw=1.8, color=COLORS["blue"],
        label=f"AquaSC joint-frame QPSK ({rate:.4f} uses/px)")
ax.plot(snrs, roi, marker="s", ms=5.5, lw=1.8, color=COLORS["orange"],
        label=f"ROI-JSCC rcpp32 ({rate:.4f} uses/px, corrected)")

# Mark each point: filled/bold = Holm-significant paired difference (n=432,
# 10000-resample bootstrap, seed 42, Holm-Bonferroni across all 6 SNRs as one
# family); a hollow ring on the AquaSC marker flags a point that is NOT
# significant (0dB only -- CI [-0.0113,+0.0162] crosses zero, a genuine tie,
# not a win) so the figure cannot be misread as "wins everywhere >=0dB".
for s, y in zip(snrs, aqua):
    if not ci_data[s]["holm_significant"]:
        ax.scatter([s], [y], s=130, facecolors="none", edgecolors=COLORS["blue"],
                   linewidths=1.6, zorder=5)
        ax.annotate("tie\n(not significant)", xy=(s, y), xytext=(s + 0.6, y - 0.10),
                    fontsize=7.3, color=COLORS["blue"], ha="left")

# Shade the loss region (-5dB) so the win/loss split is visible at a glance,
# not just implied by the crossing point.
cross_lo, cross_hi = -5.5, -2.5
ax.axvspan(cross_lo, cross_hi, color=COLORS["grey"], alpha=0.12, lw=0)
ax.annotate("AquaSC loses\n(78% frame loss)", xy=(-5, aqua[0]), xytext=(-4.6, 0.30),
            fontsize=7.7, color=COLORS["grey"], ha="left",
            arrowprops=dict(arrowstyle="-", color=COLORS["grey"], lw=0.7))

ax2 = ax.twinx()
ax2.plot(snrs, frame_loss * 100, marker="^", ms=4.5, lw=1.1, ls="--",
         color=COLORS["grey"], alpha=0.75, label="AquaSC frame-loss rate (right axis)")
ax2.set_ylabel("Frame-loss rate (%)", color=COLORS["grey"], fontsize=9)
ax2.tick_params(axis="y", colors=COLORS["grey"])
ax2.set_ylim(-4, 100)
ax2.spines["top"].set_visible(False)

ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Mean per-image AP@0.5 (n=432, M=10)")
ax.set_xticks(snrs)
ax.set_ylim(0.25, 0.82)
ax.grid(axis="y", color="#DDDDDD", lw=0.6, zorder=0)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right", frameon=False)

ax.set_title("Matched-rate digital transport: AquaSC vs. ROI-JSCC at 0.1224 uses/px\n"
             "(both at 6144 complex symbols/image, Site A test)", fontsize=9.5)
fig.text(0.01, 0.005,
         "Mean per-image AP@0.5 (n=432, M=10), NOT pooled mAP -- pooled values differ "
         "(e.g. 5dB: pooled 0.6405/0.5764 vs per-image 0.6956/0.6348, see sweep_full_ci.json).",
         fontsize=6.3, color=COLORS["grey"])

fig.tight_layout(rect=(0, 0.025, 1, 1))
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "wave20_matched5_sweep.png"
fig.savefig(out_path)
out_path_pdf = OUT_DIR / "wave20_matched5_sweep.pdf"
fig.savefig(out_path_pdf)
print(f"Saved: {out_path}")
print(f"Saved: {out_path_pdf}")

print()
print(f"{'SNR':>5}  {'AquaSC':>8}  {'ROI-JSCC':>9}  {'Delta':>8}  {'AquaSC frame-loss':>18}")
for r in rows:
    fl = f"{r['aquasc_frame_loss_rate']*100:.1f}%" if r["aquasc_frame_loss_rate"] is not None else "n/a (locked run)"
    print(f"{r['snr_db']:>5.0f}  {r['aquasc_joint_qpsk']:>8.4f}  {r['roijscc_rcpp32']:>9.4f}  "
          f"{r['delta']:>+8.4f}  {fl:>18}")
