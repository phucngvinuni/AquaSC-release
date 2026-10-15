# stats.py — statistical protocol for AquaSC resubmission (plan §9.3, Wave 8 STATS).
#
# Implements, on per-image metric arrays (n = #test images = 432):
#   * Student-t confidence interval            xbar +- t_{1-alpha/2, n-1} * s / sqrt(n)
#   * paired bootstrap                         >= 10,000 image resamples, 2.5/97.5 percentiles
#   * Wilcoxon signed-rank                     on per-image AP differences
#   * Holm-Bonferroni                          across a family of comparisons
#   * binomial SE / 95% CI                     (plan §9.3 noise-floor protocol)
#   * per-image AP@0.5 (101-point interpolated PR) — the unit for the paired tests
#
# No torch / repo imports: this module is pure numpy + scipy so it also runs in the
# CPU venv and is trivially unit-testable.  Every function returns plain Python
# dicts / floats so results serialize to the report without ambiguity.
#
# Conventions (must match the harness):
#   * arrays are 1-D per-image quantities, aligned across methods by test-image
#     index (sorted filename order of yolo_fish_dataset_root/test/images).
#   * all CIs are two-sided; alpha defaults to 0.05 (95%).
#   * bootstrap resamples *image indices* (paired), preserving per-image pairing.

from __future__ import annotations

import numpy as np

try:
    from scipy import stats as _sps
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _sps = None
    _HAS_SCIPY = False


# --------------------------------------------------------------------------- #
# 1. Student-t confidence interval
# --------------------------------------------------------------------------- #
def t_confidence_interval(values, alpha: float = 0.05, ddof: int = 1) -> dict:
    """t-CI for the mean of ``values``: xbar +- t_{1-alpha/2, n-1} * s / sqrt(n).

    Returns {mean, se, std, half_width, ci_low, ci_high, n, t_crit, alpha}.
    NaNs are dropped; n < 2 returns nan for the interval components.
    """
    v = np.asarray(values, dtype=float).ravel()
    v = v[~np.isnan(v)]
    n = int(v.size)
    out = dict(alpha=float(alpha), n=n)
    if n == 0:
        out.update(mean=float("nan"), se=float("nan"), std=float("nan"),
                   half_width=float("nan"), ci_low=float("nan"),
                   ci_high=float("nan"), t_crit=float("nan"))
        return out
    mean = float(v.mean())
    if n < 2:
        out.update(mean=mean, se=float("nan"), std=float("nan"),
                   half_width=float("nan"), ci_low=mean, ci_high=mean,
                   t_crit=float("nan"))
        return out
    std = float(v.std(ddof=ddof))
    se = std / np.sqrt(n)
    t_crit = float(_sps.t.ppf(1 - alpha / 2.0, n - 1)) if _HAS_SCIPY else _t_crit_fallback(n, alpha)
    half = t_crit * se
    out.update(mean=mean, se=se, std=std, half_width=half,
               ci_low=mean - half, ci_high=mean + half, t_crit=t_crit)
    return out


def _t_crit_fallback(n: int, alpha: float) -> float:
    """Normal approximation if scipy is unavailable (documented, not used here)."""
    return float(_sps.norm.ppf(1 - alpha / 2.0))


# --------------------------------------------------------------------------- #
# 2. Paired bootstrap
# --------------------------------------------------------------------------- #
def paired_bootstrap(a, b, n_boot: int = 10000, seed: int = 42,
                     alpha: float = 0.05, statistic=np.mean) -> dict:
    """Paired bootstrap over image resamples for the mean of (a - b).

    Resamples image indices (so a and b stay paired), recomputes the mean
    difference per replicate, and reports the percentile CI (2.5/97.5) plus a
    two-sided bootstrap p-value for H0: mean(a-b) = 0.

    ``statistic`` must accept ``axis=`` (np.mean does).
    Returns {mean_diff, mean_a, mean_b, ci_low, ci_high, n_boot, p_value, alpha,
             seed, boot_means (array)}.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"a and b must have equal length ({a.shape} vs {b.shape})")
    n = int(a.size)
    if n == 0:
        raise ValueError("empty arrays")
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    try:
        boot = statistic(diff[idx], axis=1)
    except TypeError:
        boot = np.array([statistic(diff[i]) for i in idx])
    boot = np.asarray(boot, dtype=float)
    lo, hi = np.percentile(boot, [100 * alpha / 2.0, 100 * (1 - alpha / 2.0)])
    # two-sided bootstrap p-value (shift-free variant)
    p = 2.0 * min(float(np.mean(boot <= 0.0)), float(np.mean(boot >= 0.0)))
    p = float(min(1.0, p))
    return dict(mean_diff=float(np.mean(diff)), mean_a=float(a.mean()),
                mean_b=float(b.mean()), ci_low=float(lo), ci_high=float(hi),
                n_boot=int(n_boot), p_value=p, alpha=float(alpha), seed=int(seed),
                boot_means=boot)


def bootstrap_ci(values, n_boot: int = 10000, seed: int = 42, alpha: float = 0.05,
                 statistic=np.mean) -> dict:
    """One-sample bootstrap (image resampling) CI for a statistic of ``values``."""
    v = np.asarray(values, dtype=float).ravel()
    v = v[~np.isnan(v)]
    n = int(v.size)
    if n == 0:
        raise ValueError("empty array after NaN removal")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    try:
        boot = statistic(v[idx], axis=1)
    except TypeError:
        boot = np.array([statistic(v[i]) for i in idx])
    boot = np.asarray(boot, dtype=float)
    lo, hi = np.percentile(boot, [100 * alpha / 2.0, 100 * (1 - alpha / 2.0)])
    return dict(mean=float(statistic(v)), ci_low=float(lo), ci_high=float(hi),
                n_boot=int(n_boot), alpha=float(alpha), seed=int(seed),
                boot=boot)


# --------------------------------------------------------------------------- #
# 3. Wilcoxon signed-rank
# --------------------------------------------------------------------------- #
def wilcoxon_signed_rank(a, b, alternative: str = "two-sided") -> dict:
    """Wilcoxon signed-rank test on per-image (a - b).

    Uses scipy.stats.wilcoxon (zero differences dropped per the default
    zero_method='wilcox').  Reports n, n_nonzero and n_zero so the report can
    flag ties (common when both methods score AP = 0 on many images).
    """
    if not _HAS_SCIPY:
        raise ImportError("scipy is required for wilcoxon_signed_rank")
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError("a and b must have equal length")
    d = a - b
    n = int(d.size)
    n_zero = int(np.sum(d == 0.0))
    n_nonzero = n - n_zero
    out = dict(n=n, n_nonzero=n_nonzero, n_zero=n_zero,
               statistic=float("nan"), p_value=float("nan"), alternative=alternative)
    if n_nonzero == 0:
        return out
    res = _sps.wilcoxon(a, b, alternative=alternative)
    out["statistic"] = float(res.statistic)
    out["p_value"] = float(res.pvalue)
    return out


# --------------------------------------------------------------------------- #
# 4. Holm-Bonferroni
# --------------------------------------------------------------------------- #
def holm_bonferroni(pvalues, alpha: float = 0.05) -> list:
    """Holm-Bonferroni step-down across a family of p-values.

    Returns a list (in the input order) of {index, p, rank, alpha_i, p_holm, reject}.
    """
    p = np.asarray(pvalues, dtype=float).ravel()
    m = int(p.size)
    order = np.argsort(p)
    sorted_p = p[order]
    # Holm adjusted p_i = p_(i) * (m - i + 1), monotone via cumulative max.
    adj = sorted_p * np.arange(m, 0, -1)
    adj = np.minimum(1.0, np.maximum.accumulate(adj))
    reject = adj <= alpha
    alpha_i = alpha / np.arange(m, 0, -1)
    out = []
    for i in range(m):
        rank_pos = int(np.where(order == i)[0][0])
        out.append(dict(index=int(i), p=float(p[i]), rank=int(rank_pos + 1),
                        alpha_i=float(alpha_i[rank_pos]),
                        p_holm=float(adj[rank_pos]),
                        reject=bool(reject[rank_pos])))
    return out


# --------------------------------------------------------------------------- #
# 5. Binomial SE / CI (plan §9.3 noise-floor protocol)
# --------------------------------------------------------------------------- #
def binomial_se_and_ci(p: float, n: int, z: float = 1.96) -> dict:
    """Binomial SE and 95% CI half-width for a proportion-like estimate.

    SE = sqrt(p(1-p)/n); half_width = z * SE.  This is the *noise-floor* model
    from plan §9.3 (a deliberate simplification: mAP is treated as a proportion;
    the t-CI / bootstrap over per-image AP is the more precise companion).
    """
    p = float(p)
    n = int(n)
    se = np.sqrt(p * (1.0 - p) / n) if 0 < p < 1 else 0.0
    half = z * se
    return dict(p=p, n=n, se=float(se), half_width=float(half),
                ci_low=max(0.0, p - half), ci_high=min(1.0, p + half), z=z)


# --------------------------------------------------------------------------- #
# 6. Per-image AP@IoU (101-point interpolated PR) — the paired-test unit
# --------------------------------------------------------------------------- #
def _box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes (returns [N1, N2])."""
    b1 = np.asarray(boxes1, dtype=float)
    b2 = np.asarray(boxes2, dtype=float)
    if b1.size == 0 or b2.size == 0:
        return np.zeros((b1.shape[0], b2.shape[0]), dtype=float)
    b1 = b1.reshape(-1, 4)
    b2 = b2.reshape(-1, 4)
    x1 = np.maximum(b1[:, None, 0], b2[None, :, 0])
    y1 = np.maximum(b1[:, None, 1], b2[None, :, 1])
    x2 = np.minimum(b1[:, None, 2], b2[None, :, 2])
    y2 = np.minimum(b1[:, None, 3], b2[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    a2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    union = a1[:, None] + a2[None, :] - inter
    return inter / np.where(union > 0, union, 1e-9)


def per_image_ap50(pred_boxes, pred_scores, gt_boxes, iou_thr: float = 0.5,
                   n_points: int = 101) -> float:
    """COCO-style interpolated AP at IoU=iou_thr for a SINGLE image.

    ``pred_boxes``: [N,4] xyxy; ``pred_scores``: [N]; ``gt_boxes``: [M,4] xyxy.
    Greedy matching by descending score, then 101-point interpolated PR.
    Images with zero GT boxes -> AP 0.0 (documented convention).
    """
    pred_boxes = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)
    pred_scores = np.asarray(pred_scores, dtype=float).ravel()
    gt_boxes = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    ngt = gt_boxes.shape[0]
    npred = pred_boxes.shape[0]
    if ngt == 0:
        return 0.0
    if npred == 0:
        return 0.0
    if len(pred_scores) != npred:
        raise ValueError("pred_scores length must match pred_boxes")
    order = np.argsort(-pred_scores)
    tp = np.zeros(npred, dtype=float)
    fp = np.zeros(npred, dtype=float)
    matched = set()
    for k, idx in enumerate(order):
        ious = _box_iou(pred_boxes[idx:idx + 1], gt_boxes)[0]
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr and j not in matched:
            tp[k] = 1.0
            matched.add(j)
        else:
            fp[k] = 1.0
    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    recall = tp_c / ngt
    prec = tp_c / np.maximum(tp_c + fp_c, 1e-12)
    rec_thresh = np.linspace(0.0, 1.0, n_points)
    prec_interp = np.array([
        prec[recall >= t].max() if np.any(recall >= t) else 0.0
        for t in rec_thresh
    ])
    return float(prec_interp.mean())


# --------------------------------------------------------------------------- #
# Self-test (synthetic data) — kept here so `python stats.py` runs it directly.
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    rng = np.random.default_rng(0)
    n = 432
    # method A strictly dominates B by a small margin on the same images
    base = rng.random(n) * 0.4
    a = base + 0.02 + rng.normal(0, 0.005, n)
    b = base + rng.normal(0, 0.005, n)
    ci = t_confidence_interval(a - b)
    assert ci["ci_low"] > 0.0, f"t-CI should exclude 0, got {ci}"
    pb = paired_bootstrap(a, b, n_boot=10000, seed=42)
    assert pb["ci_low"] > 0.0 and pb["ci_high"] > 0.0
    assert 0.0 <= pb["p_value"] <= 0.001, pb["p_value"]
    w = wilcoxon_signed_rank(a, b)
    assert w["p_value"] < 0.001, f"wilcoxon p={w['p_value']}"
    # Holm across a family where only the first is truly significant
    hb = holm_bonferroni([1e-6, 0.5, 0.5, 0.5])
    assert hb[0]["reject"] and not hb[1]["reject"]
    # binomial noise floor matches plan §9.3 hand-calc (p=0.2, n=432)
    bf = binomial_se_and_ci(0.2, 432)
    assert abs(bf["se"] - 0.019245) < 1e-4, bf
    assert abs(bf["half_width"] - 0.037720) < 1e-4, bf
    # per-image AP: perfect prediction == 1.0
    assert per_image_ap50(np.array([[0, 0, 10, 10]]), np.array([0.9]),
                          np.array([[0, 0, 10, 10]])) == 1.0
    # one FP that outranks the TP -> interpolated AP below 1.0
    ap = per_image_ap50(np.array([[0, 0, 10, 10], [50, 50, 60, 60]]),
                        np.array([0.9, 0.99]), np.array([[0, 0, 10, 10]]))
    assert 0.0 < ap < 1.0
    print("stats.py self-test PASSED")


if __name__ == "__main__":
    _self_test()
