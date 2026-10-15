import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repo root (core modules)
# analysis/stats_analyze.py — canonical post-processing for Wave 8 STATS (CPU, re-runnable).
#
# Reads the per-image AP / aggregate outputs written by the GPU drivers and
# produces every number the PAPER agent should use.
#
# TWO distinct AP scales are reported and clearly separated (they are NOT the
# same quantity and must never be conflated in the paper):
#   * POOLED mAP@0.5 (torchmetrics, detections pooled across all 432 images) —
#     the headline metric, identical to every prior wave's protocol.  Retention
#     = pooled mAP / ceiling 0.7292, reported as mean +/- t-CI over M=10 bank
#     realizations.
#   * MACRO per-image AP@0.5 (stats.per_image_ap50, 101-pt per image, then
#     mean over images) — the UNIT of the paired bootstrap / Wilcoxon tests.
#     It is a different (macro) quantity and must be labelled "per-image AP",
#     not "mAP", and must not be divided by the ceiling.

import argparse
import json
import os

import numpy as np

import stats
import channel_bank as CB

CEILING = 0.7292


def load_ap(npz_path):
    z = np.load(npz_path)
    per_image = z["per_image_ap"]  # [n_cells, n_images]
    keys = [(float(s), int(m)) for s, m in z["snr_keys"]] if "snr_keys" in z else None
    return per_image, keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aquasc_ap", default="results/stats/stats_bank_run1_perimage_ap.npz")
    ap.add_argument("--aquasc_agg", default="results/stats/stats_bank_run1_agg.json")
    ap.add_argument("--bbw_ap", default="results/stats/stats_bank_bbw_c2_perimage_ap.npz")
    ap.add_argument("--bbw_agg", default="results/stats/stats_bank_bbw_c2_agg.json")
    ap.add_argument("--paired_snr", type=float, default=10.0)
    args = ap.parse_args()

    print("=" * 80)
    print("AQUASC (RUN-1 flagship 18/2) — POOLED mAP@0.5 retention, mean +/- t-CI (M=10)")
    print("=" * 80)
    per_image, snr_keys = load_ap(args.aquasc_ap)
    agg = json.load(open(args.aquasc_agg)) if os.path.isfile(args.aquasc_agg) else {}
    snrs = sorted({s for s, m in snr_keys})
    for snr in snrs:
        idxs = [i for i, (s, m) in enumerate(snr_keys) if s == snr]
        ms = [snr_keys[i][1] for i in idxs]
        agg_m = [agg[f"{snr}_{m}"] for m in ms]
        mAP_mean = float(np.mean(agg_m))
        ret = np.array([v / CEILING for v in agg_m])
        tci = stats.t_confidence_interval(ret)
        ap_img = per_image[idxs].mean(axis=0)
        bs = stats.bootstrap_ci(ap_img, n_boot=10000, seed=42)   # macro AP, NOT /ceiling
        print(f"\nSNR {snr} dB  (M={len(ret)})")
        print(f"  pooled mAP@0.5 = {mAP_mean:.4f}   (per-realization: "
              f"{[f'{v:.4f}' for v in agg_m]})")
        print(f"  POOLED retention mean = {tci['mean']:.4f}   "
              f"(95% t-CI [{tci['ci_low']:.4f}, {tci['ci_high']:.4f}], "
              f"half-width {tci['half_width']:.4f}, t_crit={tci['t_crit']:.3f})")
        print(f"  MACRO per-image AP@0.5 = {ap_img.mean():.4f}   "
              f"(bootstrap 95% CI [{bs['ci_low']:.4f}, {bs['ci_high']:.4f}])  [paired-test unit]")

    print("\n" + "=" * 80)
    print(f"PAIRED AquaSC-vs-BBW(c=2) @ {args.paired_snr} dB  (same images, same bank)")
    print("unit = MACRO per-image AP@0.5 (mean over M realizations)")
    print("=" * 80)
    if os.path.isfile(args.bbw_ap):
        bbw_per, _ = load_ap(args.bbw_ap)
        aqua_idxs = [i for i, (s, m) in enumerate(snr_keys) if s == args.paired_snr]
        aqua_img = per_image[aqua_idxs].mean(axis=0)
        bbw_img = bbw_per.mean(axis=0)
        w = stats.wilcoxon_signed_rank(aqua_img, bbw_img)
        pb = stats.paired_bootstrap(aqua_img, bbw_img, n_boot=10000, seed=42)
        print(f"  AquaSC per-image AP@0.5 = {aqua_img.mean():.4f} (rate 0.0256 uses/px, digital)")
        print(f"  BBW c=2 per-image AP@0.5 = {bbw_img.mean():.4f} (rate 0.25 uses/px, analog)")
        print(f"  mean diff (AquaSC - BBW) = {pb['mean_diff']:.4f}")
        print(f"  paired bootstrap 95% CI on mean diff: [{pb['ci_low']:.4f}, {pb['ci_high']:.4f}]")
        print(f"  Wilcoxon signed-rank: W={w['statistic']:.1f}, p={w['p_value']:.3g} "
              f"(n={w['n']}, nonzero={w['n_nonzero']}, ties={w['n_zero']})")
        holm = stats.holm_bonferroni([w["p_value"]])
        print(f"  Holm-Bonferroni (1 test): p_holm={holm[0]['p_holm']:.3g} reject={holm[0]['reject']}")
        print("  NOTE: NOT rate-matched (BBW ~10x channel uses/px). Paired = shared images+bank only.")
    else:
        print(f"  {args.bbw_ap} not available — run analysis/stats_eval_jscc.py first.")

    print("\nstats_analyze.py DONE")


if __name__ == "__main__":
    main()
