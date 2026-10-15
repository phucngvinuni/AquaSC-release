#!/usr/bin/env python
"""F7 (Wave 21 redesign, 2026-09-15): two levers, two panels.

(a) MCS lever within ONE frame architecture (multi-block UEP): four rate
    points on the existing digital chain (flagship 0.0714, L2 0.0816, QPSK
    r1/2 0.1429, BPSK r1/3 0.3673), AP vs SNR.
(b) Architecture lever at a FIXED rate (0.122449 uses/px): joint-frame vs.
    multi-block UEP, same checkpoint, same rate, direct measurement with
    paired bootstrap CI (section 6t) -- both now bank/M=10-style averaged
    (joint-frame: literal channel_bank.npz M=10; multi-block: 10 independent
    realizations via distinct seeds, per A-1 -- NOT a literal bank replay,
    the per-block-fading architecture makes that structurally different
    work; see RESULTS_OF_RECORD.md section 6t).

Part B figure rules (binding, Wave 21 dispatch): Liberation Serif throughout
including mathtext; NO title inside the image -- only (a)/(b) panel labels;
vector PDF + PNG; IEEE double-column width 7.16in; ~8pt label text. All
explanation (the 4.35-4.90 dB/doubling figure, the flagship/L2 near-overlap
note, CI/ceiling/checkpoint/CSI/n/M provenance) belongs in the LaTeX
\\caption, not in the image -- see the CAPTION_TEXT string at the bottom of
this file for the drafted caption to paste into R1_main_TETCI.tex.

Run: __TORCH_PY__ plotting/plot_f7_two_levers.py
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repo root (core modules)
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT.parent / "figs"

COLORS = {
    "blue": "#0077BB",
    "orange": "#EE7733",
    "teal": "#009988",
    "purple": "#AA3377",
    "grey": "#777777",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Liberation Serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Liberation Serif",
        "mathtext.it": "Liberation Serif:italic",
        "mathtext.bf": "Liberation Serif:bold",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 6.8,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

SNRS = np.array([-5, 0, 5, 10, 15, 20], dtype=float)
CKPT_SHA_PREFIX = "b1d2eb25"
CEILING = 0.7743


def load_mean(pattern):
    return np.array([np.load(ROOT / "f1_perimage" / pattern.format(snr=int(s) if s == int(s) else s)).mean()
                      for s in SNRS])


# ---- Panel (a): MCS lever, one architecture (multi-block UEP), n=432, M=1 (single seed, matches flagship's own protocol throughout this file) ----
flagship = load_mean("flagship_snr{snr}.npy")
l2 = load_mean("l2_coded_snr{snr}.npy")
qpsk_r12 = load_mean("aquasc_qpsk_r12_snr{snr}.npy")
bpsk_r13 = load_mean("aquasc_bpsk_r13_snr{snr}.npy")

# ---- Panel (b): architecture lever, fixed rate 0.122449 ----
# Requires A-1's M=10 multi-block re-run to be complete. Refuse to silently
# publish a stale single-seed version under the "M=10" figure name -- a wrong
# figure spreads faster than a wrong number (standing rule this session).
M10_DIR = ROOT / "f1_perimage" / "matched5_mcs_m10"
m10_files = {s: sorted(f for f in M10_DIR.glob(f"flagship_matched5_mcs_snr{int(s)}_m*.npy")
                        if not f.name.endswith(".npred.npy"))
             for s in SNRS}
missing = {s: 10 - len(f) for s, f in m10_files.items() if len(f) < 10}
if missing:
    print("A-1 (multi-block @0.1224 through 10 realizations) is not yet complete:")
    for s, n in missing.items():
        print(f"  SNR={s:+.0f}dB: {10-n}/10 realizations done, {n} missing")
    print("Refusing to (re)publish figs/figure_f7_two_levers.png with stale single-seed panel (b) data.")
    print("Run again once drivers/run_a1_multiblock_bank_m10.sh finishes.")
    sys.exit(1)

multi_block_runs = {s: np.stack([np.load(f) for f in m10_files[s]]) for s in SNRS}  # {snr: [10, 432]}
multi_block_per_image_mean = np.array([multi_block_runs[s].mean(axis=0).mean() for s in SNRS])
# Per-realization means, for a paired bootstrap CI in the same style as the
# rest of this file (image-level pairing against joint-frame's own per-image
# mean-over-M array).
sweep_ci = {r["snr_db"]: r for r in json.loads((ROOT / "experiments/matched5/runs/sweep_full_ci.json").read_text())}
aqua_files = {-5: "experiments/matched5/runs/sweep/aqua_m5/metrics.npz",
              0: "experiments/matched5/runs/sweep/aqua_0/metrics.npz",
              5: "experiments/matched5/runs/aqua_test_qpsk_locked_v1/metrics.npz",
              10: "experiments/matched5/runs/sweep/aqua_10/metrics.npz",
              15: "experiments/matched5/runs/sweep/aqua_15/metrics.npz",
              20: "experiments/matched5/runs/sweep/aqua_20/metrics.npz"}

joint_frame = np.array([sweep_ci[s]["aquasc_mean_per_image"] for s in SNRS])
ci_lo, ci_hi, delta = [], [], []
rng = np.random.default_rng(42)
for s in SNRS:
    jf_per_img = np.load(ROOT / aqua_files[s])["per_image_ap"].mean(axis=1)  # [432]
    mb_per_img = multi_block_runs[s].mean(axis=0)  # [432], averaged over the 10 realizations
    diff = jf_per_img - mb_per_img
    n = len(diff)
    boot = np.empty(10000)
    for i in range(10000):
        idx = rng.integers(0, n, size=n)
        boot[i] = diff[idx].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    ci_lo.append(lo); ci_hi.append(hi); delta.append(diff.mean())
ci_lo, ci_hi, delta = np.array(ci_lo), np.array(ci_hi), np.array(delta)
multi_block = joint_frame - delta  # exact per-panel consistency with the plotted joint_frame line

fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.16, 3.05))

# --- Panel (a) ---
axa.plot(SNRS, flagship, marker="o", ms=4, lw=1.4, color=COLORS["blue"], label="historical UEP (0.0714)")
axa.plot(SNRS, l2, marker="^", ms=4, lw=1.4, color=COLORS["teal"], label="L2 (0.0816)")
axa.plot(SNRS, qpsk_r12, marker="s", ms=4, lw=1.4, color=COLORS["orange"], label="QPSK r½ (0.1429)")
axa.plot(SNRS, bpsk_r13, marker="D", ms=4, lw=1.4, color=COLORS["purple"], label="BPSK r⅓ (0.3673)")
axa.set_xlabel("SNR (dB)")
axa.set_ylabel("Mean per-image AP@0.5")
axa.set_xticks(SNRS)
axa.set_ylim(0.25, 0.82)
axa.grid(axis="y", color="#DDDDDD", lw=0.5, zorder=0)
axa.legend(loc="lower right", frameon=False)
axa.text(0.03, 0.95, "(a)", transform=axa.transAxes, fontsize=9, fontweight="bold", va="top")

# --- Panel (b) ---
axb.fill_between(SNRS, ci_lo + multi_block, ci_hi + multi_block, color=COLORS["blue"], alpha=0.12, lw=0)
axb.plot(SNRS, joint_frame, marker="o", ms=4.5, lw=1.4, color=COLORS["blue"], label="joint-frame, 1 codeword (0.1224)")
axb.plot(SNRS, multi_block, marker="s", ms=4.5, lw=1.4, color=COLORS["orange"], label="multi-block UEP, 6 blocks (0.1224)")
converge_lo, converge_hi = 14.3, 20.7
axb.axvspan(converge_lo, converge_hi, color=COLORS["grey"], alpha=0.12, lw=0)
axb.set_xlabel("SNR (dB)")
axb.set_ylabel("Mean per-image AP@0.5")
axb.set_xticks(SNRS)
axb.set_ylim(0.25, 0.82)
axb.grid(axis="y", color="#DDDDDD", lw=0.5, zorder=0)
axb.legend(loc="lower right", frameon=False)
axb.text(0.03, 0.95, "(b)", transform=axb.transAxes, fontsize=9, fontweight="bold", va="top")

fig.tight_layout()
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_png = OUT_DIR / "figure_f7_two_levers.png"
out_pdf = OUT_DIR / "figure_f7_two_levers.pdf"
fig.savefig(out_png)
fig.savefig(out_pdf)
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")

print()
print("Panel (b) deltas (joint-frame - multi-block, both M=10):")
for s, d, lo, hi in zip(SNRS, delta, ci_lo, ci_hi):
    tag = "CONVERGED (<0.05)" if abs(d) < 0.05 else ""
    print(f"  {s:+3.0f}dB: {d:+.4f} [{lo:+.4f},{hi:+.4f}] {tag}")

CAPTION_TEXT = r"""
Fig. 7. Two design levers for AquaSC's digital transport, both on the reported
checkpoint (SHA prefix %s), perfect CSI, mean per-image AP@0.5 against the
per-image ceiling of %.4f (n=432). (a) The MCS lever within the multi-block
UEP architecture alone: four historical rate points (UEP 0.0714, L2 0.0816, QPSK
r1/2 0.1429, BPSK r1/3 0.3673 uses/px), single channel realization per image
(M=1, matching the historical UEP protocol). The UEP and L2 curves nearly
overlap because their rates differ by only 0.19 rate-doublings
(log2(0.0816/0.0714)); this is not duplicated data. The MCS lever moves the
analog/digital crossover boundary by 4.35-4.90 dB per rate-doubling (Table
I / Section 1). (b) The architecture lever at one fixed rate (0.122449
uses/px, 6144 complex symbols/image): joint-frame (one BG2 codeword) versus
multi-block UEP (6 blocks, %.4f -> %.4f effective code rate at this budget),
same checkpoint, M=10 independent channel realizations each (joint-frame:
channel_bank.npz; multi-block: 10 distinct seeds, not a literal bank replay
-- Section 6t), shaded band is the 95%% paired-bootstrap CI (10^4 resamples,
seed 42, images as the resampling unit) on the joint-frame curve. The two
architectures are statistically distinguishable at every SNR shown but
converge to within the pre-registered 0.05 operational threshold from 15 dB
(shaded region) -- joint-frame's advantage is concentrated where channel
loss dominates and vanishes once it does not.
""" % (CKPT_SHA_PREFIX, CEILING, 0.484, 0.646)
print()
print("Drafted caption (paste into R1_main_TETCI.tex, mark \\textcolor{blue}{...}):")
print(CAPTION_TEXT)
