"""S3 audit: alignment, M=10 shape, and independent recompute of paired CIs."""
import json, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, "__AQUASC_ROOT__")
import stats
R = Path("__AQUASC_ROOT__/experiments/matched5/runs")
OUT = Path("__AQUASC_ROOT__/verification/s3_s7")
log = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)

def load(p):
    d = np.load(p, allow_pickle=True)
    return d["names"], d["per_image_ap"], d["pooled_map50"]

def indep_boot(a, b, n_boot=10000, seed=7):
    """Independent re-implementation (different seed, own code): resample image idx."""
    diff = np.asarray(a) - np.asarray(b); n = diff.size
    rng = np.random.default_rng(seed)
    bm = diff[rng.integers(0, n, (n_boot, n))].mean(1)
    return diff.mean(), *np.percentile(bm, [2.5, 97.5])

# ---- 3.1 shape / names / M for all runs used
P("== 3.1 shape & name alignment ==")
ref_names, _, _ = load(R/"sweep/aqua_10/metrics.npz")
bad = 0
import glob
files = sorted(glob.glob(str(R/"ablation_routing_coded/aqua_*/metrics.npz")) + glob.glob(str(R/"sweep/aqua_*/metrics.npz")) + glob.glob(str(R/"sweep/roi_*/metrics.npz")))
for f in files:
    n, a, p = load(f)
    ok = (a.shape == (432, 10)) and (n.tolist() == ref_names.tolist()) and not np.isnan(a).any()
    if not ok: bad += 1
    P(("OK  " if ok else "BAD ") + str(Path(f).parent.relative_to(R)), a.shape, "pooled_len", p.shape)
P("n files", len(files), "bad", bad)

# ---- 3.4a  ablation / router paired CIs vs stored json
P("\n== 3.4a Full vs ablations: stored vs recomputed ==")
stored = json.load(open(R/"ablation_routing_coded/paired_ci_full_vs_ablations.json"))
keys = {"FIM-NoBB": "fim_nobb", "Base": "base", "BB-NoFIM_V2": "bb_nofim", "Full + oracle_box": "full_oracle",
        "Full + random": "full_random", "Full + uniform": "full_uniform", "Full + detector_heatmap": "full_detheat"}
maxdiff = 0
rows = []
for snr in (10, 20):
    _, fa, _ = load(R/f"sweep/aqua_{snr}/metrics.npz"); fv = fa.mean(1)
    for lab, k in keys.items():
        _, oa, _ = load(R/f"ablation_routing_coded/aqua_{k}_snr{snr}/metrics.npz"); ov = oa.mean(1)
        r = stats.paired_bootstrap(fv, ov, n_boot=10000, seed=42)
        s = stored[str(snr)][lab]
        d = max(abs(r["mean_diff"]-s["mean_diff"]), abs(r["ci_low"]-s["ci_low"]), abs(r["ci_high"]-s["ci_high"]), abs(r["p_value"]-s["p_value"]))
        maxdiff = max(maxdiff, d)
        im, il, ih = indep_boot(fv, ov)
        # alt unit: resample all 4320 (image,draw) cells independently (NOT paired by image) for sensitivity
        rows.append((snr, lab, r["mean_diff"], r["ci_low"], r["ci_high"], r["p_value"], d, il, ih))
        P(f"{snr}dB {lab:26s} diff={r['mean_diff']:+.4f} CI[{r['ci_low']:+.4f},{r['ci_high']:+.4f}] p={r['p_value']:.4f} | stored-vs-recomp maxabs={d:.2e} | indep(seed7) CI[{il:+.4f},{ih:+.4f}]")
P("max |stored - recomputed| =", maxdiff)

# ---- 3.4b sweep_full_ci.json vs recompute (AquaSC joint-frame vs ROI-JSCC)
P("\n== 3.4b sweep_full_ci.json (AquaSC vs ROI-JSCC rcpp32) ==")
sw = {r["snr_db"]: r for r in json.load(open(R/"sweep_full_ci.json"))}
for snr in (-5, 0, 5, 10, 15, 20):
    ad = R/f"sweep/aqua_{'m5' if snr==-5 else snr}"
    rd = R/f"sweep/roi_{snr}"
    if snr == 5: rd = R/"roi_test_single_v1"  # fallback candidates below
    cand = [rd, R/"roi_5_verify", R/"sweep/roi_5_verify"] if snr == 5 else [rd]
    rp = next((c for c in cand if (c/"metrics.npz").exists()), None)
    if rp is None:
        # find any roi metrics for 5
        alt = sorted(glob.glob(str(R/"**/roi*5*/metrics.npz"), recursive=True)); rp = Path(alt[0]).parent if alt else None
    if rp is None or not (ad/"metrics.npz").exists():
        P(snr, "missing", ad, rp); continue
    na, aa, ap_ = load(ad/"metrics.npz"); nb, ba, bp = load(rp/"metrics.npz")
    al = na.tolist() == nb.tolist() and aa.shape == ba.shape == (432, 10)
    r = stats.paired_bootstrap(aa.mean(1), ba.mean(1), n_boot=10000, seed=42)
    s = sw[snr]
    P(f"{snr:>3}dB roi_dir={rp.relative_to(R)} aligned={al} diff {r['mean_diff']:+.4f} (stored {s['paired_mean_diff']:+.4f}) CI[{r['ci_low']:+.4f},{r['ci_high']:+.4f}] (stored [{s['ci_low']:+.4f},{s['ci_high']:+.4f}]) pooledA {ap_.mean():.4f}/{s['aquasc_pooled']:.4f} pooledR {bp.mean():.4f}/{s['roijscc_pooled']:.4f}")

# ---- 3.4c comparison_test_v1
P("\n== 3.4c comparison_test_v1 ==")
c = json.load(open(R/"comparison_test_v1/comparison.json"))
for sub in ("aqua_test_qpsk_locked_v1", ):
    pass
na, aa, ap_ = load(R/"aqua_test_qpsk_locked_v1/metrics.npz")
nb, ba, bp = load(R/"roi_test_single_v1/metrics.npz") if (R/"roi_test_single_v1/metrics.npz").exists() else (None,None,None)
if nb is not None:
    r = stats.paired_bootstrap(aa.mean(1), ba.mean(1), n_boot=10000, seed=42)
    P("recomp", {k: round(v,6) for k,v in r.items() if k in ("mean_diff","ci_low","ci_high","p_value")}, "stored", {k: round(c["paired_per_image_bootstrap"][k],6) for k in ("mean_diff","ci_low","ci_high","p_value")})
else:
    P("roi_test_single_v1 metrics missing")
json.dump(rows, open(OUT/"s3_ablation_recompute.json", "w"), indent=1)
open(OUT/"s3_recompute.log", "w").write("\n".join(log))
