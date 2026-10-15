"""paired_ci_ablation_routing_coded.py -- 2026-09-25. Paired per-image CIs
(Full - each ablation/router variant) for §4b's 8-condition real-coded-pipeline
table, from the per-image AP arrays already saved by
run_ablation_routing_coded.sh (no re-eval needed -- every condition already
has metrics.npz with per_image_ap [432,10]).

Per-image statistic: mean AP over the M=10 channel draws (this project's
established per-image-AP convention, e.g. RESULTS_OF_RECORD.md's other
per-image CI tables). Paired bootstrap (stats.paired_bootstrap, n_boot=10000,
seed=42, matching project default) over image resamples, keeping Full and
the comparator paired by image identity (verified: names arrays match
exactly across all conditions, confirmed before trusting any pairing).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "__AQUASC_ROOT__")
import stats  # noqa: E402

ROOT = Path("__AQUASC_ROOT__/experiments/matched5")
RUNS = ROOT / "runs" / "ablation_routing_coded"
SWEEP = ROOT / "runs" / "sweep"
SNRS = [10, 20]

CONDITIONS = [
    ("fim_nobb", "FIM-NoBB"),
    ("base", "Base"),
    ("bb_nofim", "BB-NoFIM_V2"),
    ("full_oracle", "Full + oracle_box"),
    ("full_random", "Full + random"),
    ("full_uniform", "Full + uniform"),
    ("full_detheat", "Full + detector_heatmap"),
]


def load_per_image_mean(path):
    d = np.load(path, allow_pickle=True)
    names = d["names"]
    per_image_mean = d["per_image_ap"].mean(axis=1)  # [432], mean over M=10
    return names, per_image_mean


def main():
    results = {}
    for snr in SNRS:
        full_names, full_vals = load_per_image_mean(SWEEP / f"aqua_{snr}" / "metrics.npz")
        results[snr] = {}
        for key, label in CONDITIONS:
            other_names, other_vals = load_per_image_mean(RUNS / f"aqua_{key}_snr{snr}" / "metrics.npz")
            assert (full_names == other_names).all(), f"name mismatch for {key} @ {snr}dB"
            r = stats.paired_bootstrap(full_vals, other_vals, n_boot=10000, seed=42)
            results[snr][label] = r
            sig = "significant (CI excludes 0)" if (r["ci_low"] > 0 or r["ci_high"] < 0) else "NOT significant (CI includes 0)"
            print(f"SNR={snr}dB  Full vs {label:28s}  mean_diff={r['mean_diff']:+.4f}  "
                  f"95% CI=[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  p={r['p_value']:.4f}  {sig}")

    out = {snr: {label: {k: v for k, v in r.items() if k != "boot_means"}
                 for label, r in results[snr].items()}
           for snr in SNRS}
    out_path = RUNS / "paired_ci_full_vs_ablations.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
