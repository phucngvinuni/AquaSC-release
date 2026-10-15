"""S7: precision/recall of routing controls (oracle_box, uniform, random seeds 42-46) offline,
same protocol as analysis/routing_pr_topk.py (oracle_binary positives, PS=8, NP=784, 224x224, K=63,
precision=hit/K over all images, recall=hit/n_pos over images with n_pos>0, t-CI via stats.t_confidence_interval).
Also cross-checks computed masks against the masks ACTUALLY used in the coded runs (cache_*/source.npz high_mask)."""
import sys, os, json
import numpy as np, torch
sys.path.insert(0, "__AQUASC_ROOT__")
import stats, routing_baselines as rb
R = "__AQUASC_ROOT__/experiments/matched5/runs/ablation_routing_coded/"
IMG = "__AQUASC_DATA__/yolo_fish_dataset_root/test/images"
LBL = "__AQUASC_DATA__/yolo_fish_dataset_root/test/labels"
OUT = "__AQUASC_ROOT__/verification/s3_s7/"
H = W = 224; PS = 8; NP = 784; K = 63
names = sorted(f for f in os.listdir(IMG) if f.lower().endswith((".png", ".jpg", ".jpeg")))
N = len(names)
boxes = [rb.parse_yolo_label(os.path.join(LBL, os.path.splitext(n)[0] + ".txt"), H=H, W=W) for n in names]
gt = torch.cat([rb.oracle_binary(b, PS, NP, H, W) for b in boxes]) > 0.5   # [N,NP]
npos = gt.sum(1).numpy()
log = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)
P(f"N={N} images; n_pos patches: min={npos.min()} median={np.median(npos):.0f} mean={npos.mean():.2f} max={npos.max()}; "
  f"frac n_pos<63: {(npos<63).mean():.3f}; n_pos>63: {(npos>63).mean():.3f}; ==63: {(npos==63).mean():.3f}; zero-positive imgs: {(npos==0).sum()}")

def pr(mask):
    hit = (mask & gt).sum(1).numpy().astype(float)
    prec = hit / K
    ok = npos > 0
    rec = hit[ok] / npos[ok]
    return prec, rec, int((~ok).sum())

def row(label, mask, src):
    prec, rec, nz = pr(mask)
    pc, rc = stats.t_confidence_interval(prec), stats.t_confidence_interval(rec)
    P(f"{label:28s} prec {pc['mean']:.4f} [{pc['ci_low']:.4f},{pc['ci_high']:.4f}]  rec {rc['mean']:.4f} [{rc['ci_low']:.4f},{rc['ci_high']:.4f}]  n={N} (recall n={len(rec)}, skipped {nz})  {src}")
    return pc['mean'], rc['mean'], pc, rc

def check_saved(mask, cache):
    d = np.load(R + cache + "/source.npz", allow_pickle=True)
    sm = torch.from_numpy(d["high_mask"]).bool()
    same_names = [str(x) for x in d["names"]] == names
    return same_names, bool((sm == mask).all()), int((sm != mask).sum())

res = {}
# oracle_box
sc = rb.oracle_box_scores(boxes, PS, NP, H, W)
m_or, _ = rb.apply_topk(sc, K)
res["oracle_box"] = row("oracle_box", m_or, "routing_baselines.oracle_box_scores+apply_topk")
P("  matches saved coded-run mask:", check_saved(m_or, "cache_full_oracle"))
# uniform
m_un, idx_un = rb.apply_topk(rb.uniform_scores(NP, batch_size=N), K)
res["uniform"] = row("uniform", m_un, "routing_baselines.uniform_scores+apply_topk")
P("  matches saved coded-run mask:", check_saved(m_un, "cache_full_uniform"))
sel = idx_un.numpy()
P("  uniform selected patch indices (image 0):", sorted(sel[0].tolist())[:12], "... identical across images:", bool((np.sort(sel,1)==np.sort(sel[0])).all()),
  "| set == first 63 indices?", sorted(sel[0].tolist()) == list(range(63)), "| unique sets across images:", len({tuple(sorted(r)) for r in sel.tolist()}))
# random seeds
caches = {42: "cache_full_random", 43: "cache_full_random_seed43", 44: "cache_full_random_seed44", 45: "cache_full_random_seed45", 46: "cache_full_random_seed46"}
rp, rr = [], []
for s in (42, 43, 44, 45, 46):
    m_r, _ = rb.apply_topk(rb.random_scores(NP, seed=s, batch_size=N), K)
    r = row(f"random seed={s}", m_r, "routing_baselines.random_scores+apply_topk")
    P("  matches saved coded-run mask:", check_saved(m_r, caches[s]))
    rp.append(r[0]); rr.append(r[1]); res[f"random_{s}"] = r
rp, rr = np.array(rp), np.array(rr)
P(f"random 5 seeds: precision mean={rp.mean():.4f} std={rp.std(ddof=1):.4f}; recall mean={rr.mean():.4f} std={rr.std(ddof=1):.4f}")
# reference rows from saved npz
for f, lab in (("results/routing/routing_pr_topk63_result.npz", "FIM (saved npz)"), ("results/routing/routing_pr_segdet_topk63_result.npz", "detector_heatmap (saved npz)")):
    d = np.load("__AQUASC_ROOT__/" + f, allow_pickle=True)
    pc, rc = stats.t_confidence_interval(d["precisions"]), stats.t_confidence_interval(d["recalls"])
    P(f"{lab:28s} prec {pc['mean']:.4f} [{pc['ci_low']:.4f},{pc['ci_high']:.4f}]  rec {rc['mean']:.4f} [{rc['ci_low']:.4f},{rc['ci_high']:.4f}]  n={len(d['precisions'])} recall n={len(d['recalls'])} n_no_positive={int(d['n_no_positive'])}")
# bootstrap-unit CI (item-3 unit: images, paired-style percentile bootstrap, 10000, seed 42) for the controls
P("\nPercentile-bootstrap (images, n_boot=10000, seed=42) CIs:")
for lab, mask in (("oracle_box", m_or), ("uniform", m_un), ("random42", rb.apply_topk(rb.random_scores(NP, seed=42, batch_size=N), K)[0])):
    prec, rec, _ = pr(mask)
    bp, br = stats.bootstrap_ci(prec, n_boot=10000, seed=42), stats.bootstrap_ci(rec, n_boot=10000, seed=42)
    P(f"  {lab:12s} prec [{bp['ci_low']:.4f},{bp['ci_high']:.4f}] rec [{br['ci_low']:.4f},{br['ci_high']:.4f}]")
open(OUT + "s7_routing_pr.log", "w").write("\n".join(log))
