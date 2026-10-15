import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repo root (core modules)
# analysis/noise_floor.py — recompute the plan §9.3 noise floor at the HONEST level
# (Wave 8 STATS step 3). Reads the EVAL-FINAL proxy CSVs (mAP@0.5, conf 0.001,
# n = 432 test images) and reports the binomial SE / 95% CI half-width per SNR,
# plus retention vs the 0.7292 ceiling, and restates which comparisons survive.

import csv
import os
import sys

import stats

CEILING = 0.7292          # detector2 clean ceiling @ conf 0.001 (wave-4 DETECTOR2)
N_TEST = 432              # video-disjoint test split size
Z = 1.96

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSVS = {
    "run1_18_2": os.path.join(ROOT, "results/eval/evalfinal_run1_proxy.csv"),
    "run3_18_4": os.path.join(ROOT, "results/eval/evalfinal_run3_proxy.csv"),
    "run2_op16": os.path.join(ROOT, "results/eval/evalfinal_run2_proxy.csv"),
}


def load_map50(csv_path):
    rows = {}
    if not os.path.isfile(csv_path):
        return None
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            try:
                snr = float(r["SNR_dB"])
                rows[snr] = float(r["mAP_50"])
            except (KeyError, ValueError):
                continue
    return rows


def main():
    data = {k: load_map50(v) for k, v in CSVS.items()}
    print("=" * 78)
    print("NOISE FLOOR — binomial SE / 95% CI half-width, n =", N_TEST,
          "ceiling =", CEILING)
    print("=" * 78)
    for name, rows in data.items():
        if not rows:
            print(f"\n[{name}] CSV not yet available -> {CSVS[name]}")
            continue
        print(f"\n[{name}]  ({CSVS[name]})")
        print(f"{'SNR':>6} {'mAP@0.5':>9} {'SE':>8} {'half-width':>11} "
              f"{'95% CI':>20} {'retention':>10} {'ret ±':>8}")
        for snr in sorted(rows):
            p = rows[snr]
            bf = stats.binomial_se_and_ci(p, N_TEST, Z)
            ret = p / CEILING
            # delta-method SE for the retention ratio r = p/c (c fixed)
            ret_se = bf["se"] / CEILING
            print(f"{snr:>6.1f} {p:>9.4f} {bf['se']:>8.5f} {bf['half_width']:>11.5f} "
                  f"[{bf['ci_low']:>7.4f}, {bf['ci_high']:>7.4f}] "
                  f"{ret:>10.4f} {Z * ret_se:>8.4f}")
    print("\n" + "=" * 78)
    print("CROSS-CONFIG GAP CHECK (18/2 vs 18/4) — does the rate gap exceed noise?")
    print("=" * 78)
    r1 = data["run1_18_2"]
    r3 = data["run3_18_4"]
    if r1 and r3:
        print(f"{'SNR':>6} {'18/2':>9} {'18/4':>9} {'gap':>8} "
              f"{'2xHW(18/2)':>12} {'survives?':>10}")
        for snr in sorted(set(r1) & set(r3)):
            gap = abs(r1[snr] - r3[snr])
            hw = 2 * stats.binomial_se_and_ci(r1[snr], N_TEST, Z)["half_width"]
            print(f"{snr:>6.1f} {r1[snr]:>9.4f} {r3[snr]:>9.4f} {gap:>8.4f} "
                  f"{hw:>12.5f} {'YES' if gap > hw else 'no'}")
    else:
        print("run3_18_4 not yet available; will fill once EVAL-FINAL completes.")
    print("\n" + "=" * 78)
    print("PLAN §9.3 original (for the rebuttal) — submitted split:")
    for p, n in ((0.90, 494), (0.20, 432)):
        bf = stats.binomial_se_and_ci(p, n, Z)
        print(f"  n={n}, p={p:.2f}: SE={bf['se']:.5f}, half-width={bf['half_width']:.5f}")


if __name__ == "__main__":
    main()
