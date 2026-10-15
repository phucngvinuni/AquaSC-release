#!/usr/bin/env python
"""F8: Failure mechanism, joint-frame, self only. (a) Frame-loss rate vs SNR.
(b) Per-image AP conditioned on frame-lost vs frame-alive, vs SNR, with the
prior floor as a reference -- frame-lost images sit AT the floor at every
SNR, frame-alive images sit near ceiling at every SNR; SNR only changes the
MIX, not what either group scores. This is the mechanistic explanation for
joint-frame's low-SNR losses in sections 6s/6t/6w.

Data: experiments/matched5/runs/{sweep,phy_test_qpsk_locked_v1,aqua_test_qpsk_locked_v1}
receiver.npz (frame_ok) paired with aqua_*/metrics.npz (per_image_ap), same
432 images x 10 (or locked single) realizations, joint-frame @0.122449 uses/px.

Run: __TORCH_PY__ plot_f8_failure_mechanism.py
"""
import json
from pathlib import Path

import numpy as np

from plot_style import apply_style, plt, COLORS, COL_DOUBLE, PRIOR_FLOOR, CEILING_PER_IMAGE, panel_label

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT.parent / "figs"
apply_style()

data = json.loads((ROOT / "experiments/matched5/runs/frame_loss_conditional_ap.json").read_text())
data = sorted(data, key=lambda r: r["snr"])
snrs = np.array([r["snr"] for r in data], dtype=float)
frame_loss = np.array([r["frame_loss_rate"] for r in data]) * 100
lost_ap = np.array([r["lost_mean"] for r in data])
alive_ap = np.array([r["alive_mean"] for r in data])
lost_n = np.array([r["lost_n"] for r in data])
alive_n = np.array([r["alive_n"] for r in data])

fig, (axa, axb) = plt.subplots(1, 2, figsize=(COL_DOUBLE, 2.7))

# --- Panel (a): frame-loss rate vs SNR ---
axa.plot(snrs, frame_loss, marker="o", ms=4.5, lw=1.5, color=COLORS["red"])
axa.set_xlabel("SNR (dB)")
axa.set_ylabel("Frame-loss rate (%)")
axa.set_xticks(snrs)
axa.set_ylim(-3, 100)
axa.grid(axis="y", color="#DDDDDD", lw=0.5, zorder=0)
panel_label(axa, "(a)")

# --- Panel (b): conditional AP, frame-lost vs frame-alive ---
axb.axhline(PRIOR_FLOOR, color=COLORS["grey"], lw=0.9, ls=":", zorder=0)
axb.text(snrs[0], PRIOR_FLOOR + 0.02, "prior floor", fontsize=6.3, color=COLORS["grey"])
axb.axhline(CEILING_PER_IMAGE, color=COLORS["grey"], lw=0.9, ls="--", zorder=0)
axb.text(snrs[0], CEILING_PER_IMAGE - 0.05, "ceiling", fontsize=6.3, color=COLORS["grey"])

axb.plot(snrs, alive_ap, marker="o", ms=4.5, lw=1.5, color=COLORS["blue"], label="frame-alive")
axb.plot(snrs, lost_ap, marker="s", ms=4.5, lw=1.5, color=COLORS["red"], label="frame-lost")
axb.set_xlabel("SNR (dB)")
axb.set_ylabel("Mean per-image AP@0.5")
axb.set_xticks(snrs)
axb.set_ylim(0.0, 0.85)
axb.grid(axis="y", color="#DDDDDD", lw=0.5, zorder=0)
axb.legend(loc="center right", frameon=False)
panel_label(axb, "(b)")

fig.tight_layout()
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "figure_f8_failure_mechanism.png")
fig.savefig(OUT_DIR / "figure_f8_failure_mechanism.pdf")
print("Saved figs/figure_f8_failure_mechanism.{png,pdf}")
for r in data:
    print(f"  {r['snr']:+3d}dB: frame_loss={r['frame_loss_rate']*100:.1f}% "
          f"lost_AP={r['lost_mean']:.4f}(n={r['lost_n']}) alive_AP={r['alive_mean']:.4f}(n={r['alive_n']})")

print()
print("CAPTION DRAFT:")
print(r"""Fig. 8. Failure mechanism, AquaSC joint-frame (0.122449 uses/px), flagship
checkpoint (SHA prefix b1d2eb25), perfect CSI, n=432, M=10 (channel_bank.npz,
432x10=4320 (image,realization) cells per SNR, pooled across realizations for
this figure). (a) Frame-loss rate (single CRC per image; any decode failure
drops the whole frame to the all-zero-latent fallback) vs. SNR. (b) Per-image
AP@0.5 conditioned on that same image/realization's frame outcome:
frame-alive cells cluster near the per-image ceiling (0.7743, dashed) at
every SNR; frame-lost cells cluster at the Site A prior floor (0.2711,
dotted) at every SNR. SNR changes the MIX (panel a), not what either group
scores (panel b) -- the low-SNR AP losses in Sections 6s/6t/6w are a direct,
mechanistic consequence of frame-loss rate, not a gradual quality
degradation.""")
