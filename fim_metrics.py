#!/usr/bin/env python
# fim_metrics.py - Wave-7 FIMEVAL (plan 8.2): FIM classification metrics.
#
# The FIM (FeatureImportanceTransformer) is trained with BCE against
# t_FIM,i = 1[patch i overlaps any GT box], but the paper reported zero
# classification numbers. This module produces them.
#
# Import-safe: nothing heavy is created at import time; model + dataset are
# constructed only inside main().
#
# Reuses the routing path in model.py (fim_scores -> sigmoid -> top-K OR
# threshold). Classification metrics are computed in THRESHOLD mode (the tau
# sweep), which is the mode plan 8.2 asks about; top-K=63 is the training-time
# default and is reported once as a cross-check.
#
# Usage:
#   python fim_metrics.py --checkpoint <ckpt> --data_path <root> \
#       --detector best.pt --device cpu --snr 10 \
#       --out_csv fim_classification.csv --sweep_csv fim_tau_sweep.csv \
#       --hist_png __AQUASC_DATA__/figs/fim_histograms.png
import argparse
import csv
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import datasets
import metrics
import utils
import model as _model  # noqa: F401  (registers ViT_Reconstruction_Model_Default with timm)
from digital_channel import transmit_and_receive_indices_batch as transmit_digital_indices
from torchmetrics.detection import MeanAveragePrecision

DEFAULT_CHECKPOINT = os.path.join(
    HERE, "ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS_FINAL", "checkpoint-best_psnr.pth")
DEFAULT_DATA = "__AQUASC_DATA__/yolo_fish_dataset_root/"
DEFAULT_DETECTOR = os.path.join(HERE, "best.pt")
TAUS = [0.1, 0.3, 0.5, 0.7, 0.9]
IMG_PIX = 224
PATCH = 8
NPATCH = (IMG_PIX // PATCH) ** 2  # 784


def build_model(checkpoint_path, device):
    """Build the RVQ model from the checkpoint's own `args` and load weights."""
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    a.data_path = DEFAULT_DATA
    a.eval = True
    a.device = str(device)
    if not hasattr(a, "no_augment"):
        a.no_augment = False
    m = utils.get_model(a)
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in ck["model"].items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    # Version-skew: the checkpoint predates the RVQFIX EMA buffers; those two
    # are inference-irrelevant. Everything else must load exactly.
    allowed_missing = {"quantizer_high.ema_cluster_size", "quantizer_high.ema_sum"}
    real_missing = [k for k in missing if k not in allowed_missing]
    if real_missing or unexpected:
        raise RuntimeError(
            f"checkpoint<->config mismatch: missing={real_missing} unexpected={unexpected}")
    m.to(device)
    m.eval()
    return m, a


def _pr(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else float("nan")
    r = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * p * r / (p + r) if (p + r) else float("nan")
    return p, r, f1


def average_precision(scores, labels):
    """sklearn.metrics.average_precision_score (interpolated AP), no sklearn dep."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    desc = np.argsort(-scores, kind="mergesort")
    y = labels[desc]
    distinct = np.where(np.diff(-scores[desc]))[0]
    thr = np.r_[distinct, y.size - 1]
    tps = np.cumsum(y)[thr]
    fps = 1 + thr - tps
    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    npos = tps[-1]
    if npos == 0:
        return float("nan")
    recall = tps / npos
    last = int(tps.searchsorted(tps[-1]))
    sl = slice(last, None, -1)
    precision = np.r_[precision[sl], 1]
    recall = np.r_[recall[sl], 0]
    return float(-np.sum(np.diff(recall) * precision[:-1]))


def roc_auc(scores, labels):
    """Trapezoidal ROC-AUC from the sorted score curve (handles ties)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    npos = tp[-1]
    nneg = fp[-1]
    if npos == 0 or nneg == 0:
        return float("nan")
    tpr = np.concatenate(([0.0], tp / npos, [1.0]))
    fpr = np.concatenate(([0.0], fp / nneg, [1.0]))
    return float(np.trapz(tpr, fpr))


def classify_at(scores, labels, tau):
    """Confusion matrix + P/R/F1 at a single threshold."""
    pred = scores > tau
    pos = labels > 0.5
    tp = int((pred & pos).sum())
    fp = int((pred & ~pos).sum())
    fn = int((~pred & pos).sum())
    tn = int((~pred & ~pos).sum())
    p, r, f1 = _pr(tp, fp, fn)
    return {"tau": tau, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": p, "recall": r, "f1": f1}


def reconstruct(model, tokens, fim_probs, tau, snr_db):
    """Eval-path reconstruction with THRESHOLD routing (mirrors model.forward
    eval branch, digital_ldpc=False), but reuses pre-computed encoder tokens so
    the encoder runs once across the whole tau sweep."""
    B, Np, _ = tokens.shape
    route = fim_probs > tau                       # [B, Np]
    x_proj = model.norm_before_quantizer(model.encoder_to_channel_proj(tokens))
    bits_high = int(np.log2(model.quantizer_high.num_embeddings))
    bits_low = int(np.log2(model.quantizer_low.num_embeddings))
    all_indices = torch.zeros((B, Np), dtype=torch.long, device=tokens.device)
    bits_per_index = torch.zeros((B, Np), dtype=torch.long, device=tokens.device)
    if route.any():
        _, _, _, idx_high = model.quantizer_high(x_proj[route], return_per_token_loss=True)
        all_indices[route] = idx_high
        bits_per_index[route] = bits_high
    low = ~route
    if low.any():
        _, _, _, idx_low = model.quantizer_low(x_proj[low], return_per_token_loss=True)
        all_indices[low] = idx_low
        bits_per_index[low] = bits_low
    recovered = transmit_digital_indices(all_indices, bits_per_index, snr_db=snr_db)
    tokens_after = torch.zeros_like(x_proj)
    for i in range(B):
        rh = route[i]
        if model.quantizer_type == "rvq":
            tokens_after[i, rh] = model.quantizer_high.decode(recovered[i][rh])
        else:
            tokens_after[i, rh] = model.quantizer_high.embedding(recovered[i][rh])
        tokens_after[i, ~rh] = model.quantizer_low.embedding(recovered[i][~rh])
    x_dec = model.channel_to_decoder_proj(tokens_after)
    bm = torch.zeros(B, Np, dtype=torch.bool, device=tokens.device)
    return model.img_decoder(x_vis_tokens=x_dec, encoder_mask_boolean=bm,
                             full_image_num_patches_h=model.full_image_num_patches_h,
                             full_image_num_patches_w=model.full_image_num_patches_w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--data_path", default=DEFAULT_DATA)
    ap.add_argument("--detector", default=DEFAULT_DETECTOR)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--snr", type=float, default=10.0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--taus", nargs="+", type=float, default=TAUS)
    ap.add_argument("--tau_main", type=float, default=0.7)
    ap.add_argument("--out_csv", default=None,
                   help="default: fim_classification_<tag>.csv")
    ap.add_argument("--sweep_csv", default=None,
                   help="default: fim_tau_sweep_<tag>.csv")
    ap.add_argument("--hist_png", default=None,
                   help="default: figs/fim_histograms_<tag>.png")
    ap.add_argument("--tag", default="siteA",
                    help="namespace for output files (per-site runs)")
    ap.add_argument("--tau_fine", type=int, default=37,
                    help="grid size for the fine tau sweep used to locate the "
                         "F1-optimal and recall-optimal thresholds")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_mAP", action="store_true",
                    help="skip the detector pass (fast classification-only run)")
    args = ap.parse_args()
    if args.out_csv is None:
        args.out_csv = os.path.join(HERE, f"fim_classification_{args.tag}.csv")
    if args.sweep_csv is None:
        args.sweep_csv = os.path.join(HERE, f"fim_tau_sweep_{args.tag}.csv")
    if args.hist_png is None:
        args.hist_png = f"__AQUASC_DATA__/figs/fim_histograms_{args.tag}.png"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print(f"[fim_metrics] checkpoint = {args.checkpoint}")
    print(f"[fim_metrics] data_path  = {args.data_path}")
    print(f"[fim_metrics] device      = {device}  snr = {args.snr} dB")

    model, margs = build_model(args.checkpoint, device)
    margs.data_path = args.data_path
    print(f"[fim_metrics] bits_high = {int(np.log2(model.quantizer_high.num_embeddings))} "
          f"(rvq {model.quantizer_high.num_stages}x{model.quantizer_high.bits_per_stage}) "
          f"bits_low = {int(np.log2(model.quantizer_low.num_embeddings))}")

    ds = datasets.build_dataset(is_train=False, args=margs)
    print(f"[fim_metrics] split size = {len(ds)}")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
        drop_last=False, collate_fn=datasets.yolo_collate_fn)

    # ---- Pass A: encoder + FIM once over the split -----------------------
    all_scores = []     # [N, 784] sigmoid(FIM logits)
    all_labels = []     # [N, 784] GT patch labels
    all_tokens = []     # list of [784, 512] encoded tokens (for tau sweep)
    all_orig = []       # list of [3,224,224] originals (for detector)
    all_gt = []         # list of per-image GT box dicts (for detector)
    for bi, (semcom_in, targets) in enumerate(loader):
        (orig, bm_pos) = semcom_in
        (recon_target, yolo_gt_list, fim_target_map) = targets
        orig = orig.to(device, non_blocking=True)
        if bm_pos is not None:
            bm_pos = bm_pos.to(device, non_blocking=True)
        with torch.no_grad():
            x_tok = model.img_encoder(orig, bm_pos if bm_pos is not None else None)
            fim_logits = model.fim_module(x_tok)
        scores = torch.sigmoid(fim_logits).squeeze(-1).cpu().float()      # [B, 784]
        labels = fim_target_map.squeeze(-1).cpu().float()                 # [B, 784]
        all_scores.append(scores)
        all_labels.append(labels)
        for i in range(orig.shape[0]):
            all_tokens.append(x_tok[i].cpu())
            all_orig.append(orig[i].cpu())
            all_gt.append({"boxes": yolo_gt_list[i]["boxes"].cpu(),
                           "labels": yolo_gt_list[i]["labels"].cpu()})
        if (bi + 1) % 25 == 0:
            print(f"  [pass A] {bi + 1}/{len(loader)}", flush=True)

    scores = torch.cat(all_scores, dim=0).numpy()     # [N, 784]
    labels = torch.cat(all_labels, dim=0).numpy()     # [N, 784]
    N = scores.shape[0]
    n_total = N * NPATCH
    print(f"[fim_metrics] pass A done: {N} images x {NPATCH} patches = {n_total} decisions")

    s_flat = scores.ravel()
    l_flat = labels.ravel()
    npos = int((l_flat > 0.5).sum())
    nneg = n_total - npos
    pos_rate = npos / n_total
    print(f"[fim_metrics] positive-patch rate = {pos_rate:.4f} ({npos}/{n_total})")

    # ---- Classification metrics at tau_main ------------------------------
    row = classify_at(s_flat, l_flat, args.tau_main)
    row.update({
        "n_images": N, "n_patches": n_total, "n_pos": npos, "n_neg": nneg,
        "positive_rate": pos_rate,
        "ap": average_precision(s_flat, l_flat),
        "roc_auc": roc_auc(s_flat, l_flat),
    })
    print(f"[fim_metrics] tau={args.tau_main}: P={row['precision']:.4f} "
          f"R={row['recall']:.4f} F1={row['f1']:.4f} AP={row['ap']:.4f} "
          f"AUC={row['roc_auc']:.4f}")

    # ---- Per-image P_high (fraction routed high) at tau_main -------------
    per_img_phigh = (scores > args.tau_main).mean(axis=1)
    row["p_high_mean"] = float(per_img_phigh.mean())
    row["p_high_std"] = float(per_img_phigh.std(ddof=1)) if N > 1 else 0.0
    row["p_high_min"] = float(per_img_phigh.min())
    row["p_high_max"] = float(per_img_phigh.max())
    # implied max recall at a hard P_high budget of 8% (top-K=63)
    p_high_budget = 63 / NPATCH
    row["implied_max_recall_p08"] = min(1.0, p_high_budget / pos_rate) if pos_rate > 0 else float("nan")

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print(f"[fim_metrics] WROTE {args.out_csv}")

    # ---- Fine tau sweep: F1-optimal and recall-optimal thresholds ---------
    # Plan 8.2: the loss is asymmetric (a missed fish patch is unrecoverable, a
    # falsely promoted background patch only wastes bits), so the threshold
    # should favour recall; tau=0.7 favours precision. Report the F1-optimal
    # threshold AND the best-recall operating point at precision >= 0.50.
    fine = np.linspace(0.05, 0.95, args.tau_fine)
    fine_rows = []
    for t in fine:
        pred = s_flat > t
        tp = int((pred & (l_flat > 0.5)).sum())
        fp = int((pred & (l_flat <= 0.5)).sum())
        fn = int((~pred & (l_flat > 0.5)).sum())
        p, r, f1 = _pr(tp, fp, fn)
        fine_rows.append({"tau": float(t), "precision": p, "recall": r, "f1": f1})
    valid_f1 = [r for r in fine_rows if not np.isnan(r["f1"])]
    f1_best = max(valid_f1, key=lambda r: r["f1"]) if valid_f1 else None
    prec_ok = [r for r in fine_rows if not np.isnan(r["precision"]) and r["precision"] >= 0.5]
    rec_best = max(prec_ok, key=lambda r: (r["recall"], r["tau"])) if prec_ok else None
    row["tau_f1_opt"] = f1_best["tau"] if f1_best else float("nan")
    row["f1_at_opt"] = f1_best["f1"] if f1_best else float("nan")
    row["recall_at_f1_opt"] = f1_best["recall"] if f1_best else float("nan")
    row["tau_rec_opt_prec50"] = rec_best["tau"] if rec_best else float("nan")
    row["recall_at_rec_opt"] = rec_best["recall"] if rec_best else float("nan")
    row["precision_at_rec_opt"] = rec_best["precision"] if rec_best else float("nan")
    with open(args.out_csv.replace(".csv", "_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print(f"[fim_metrics] tau_F1opt={row['tau_f1_opt']:.3f} "
          f"(P={f1_best['precision']:.3f} R={row['recall_at_f1_opt']:.3f} "
          f"F1={row['f1_at_opt']:.3f}) | tau_recopt(p>=.5)="
          f"{row['tau_rec_opt_prec50']:.3f} R={row['recall_at_rec_opt']:.3f}")
    with open(args.sweep_csv.replace(".csv", "_fine.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fine_rows[0].keys()))
        w.writeheader()
        w.writerows(fine_rows)
    print(f"[fim_metrics] WROTE {args.sweep_csv.replace('.csv', '_fine.csv')}")

    # ---- Per-image P_high variance dump (per-site deliverable) ------------
    np.save(args.out_csv.replace(".csv", "_per_image_phigh.npy"), per_img_phigh)

    # ---- Score histograms (PNG) ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.hist(s_flat[l_flat > 0.5], bins=60, range=(0, 1), density=True,
            alpha=0.6, color="tab:red", label=f"positive patches (n={npos})")
    ax.hist(s_flat[l_flat <= 0.5], bins=60, range=(0, 1), density=True,
            alpha=0.5, color="tab:blue", label=f"negative patches (n={nneg})")
    ax.axvline(args.tau_main, color="k", ls="--", lw=1, label=f"tau={args.tau_main}")
    ax.set_xlabel("FIM sigmoid score P(high)")
    ax.set_ylabel("density")
    ax.set_title("FIM score distributions (pos vs neg patches)")
    ax.legend()
    ax = axes[1]
    ax.hist(per_img_phigh, bins=40, color="tab:green", alpha=0.7)
    ax.axvline(63 / NPATCH, color="k", ls="--", lw=1, label="P_high = 8% (top-K=63)")
    ax.axvline(per_img_phigh.mean(), color="tab:red", lw=1.5,
               label=f"mean = {per_img_phigh.mean():.4f}")
    ax.set_xlabel(f"per-image P_high at tau={args.tau_main}")
    ax.set_ylabel("count")
    ax.set_title(f"Per-image P_high (mean {per_img_phigh.mean():.4f}, "
                 f"std {per_img_phigh.std(ddof=1):.4f})")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.hist_png), exist_ok=True)
    fig.savefig(args.hist_png, dpi=130)
    print(f"[fim_metrics] WROTE {args.hist_png}")

    # ---- Tau sweep (classification + K/BPP, and mAP if requested) --------
    sweep_rows = []
    if not args.skip_mAP:
        from ultralytics import YOLO
        yolo = YOLO(args.detector)
        yolo.model.eval()
    for tau in args.taus:
        pred = s_flat > tau
        tp = int((pred & (l_flat > 0.5)).sum())
        fp = int((pred & (l_flat <= 0.5)).sum())
        fn = int((~pred & (l_flat > 0.5)).sum())
        p, r, f1 = _pr(tp, fp, fn)
        K = int(pred.sum())
        p_high = K / n_total
        bits_high = int(np.log2(model.quantizer_high.num_embeddings))
        bits_low = int(np.log2(model.quantizer_low.num_embeddings))
        bpp = (p_high * bits_high + (1.0 - p_high) * bits_low) / (IMG_PIX * IMG_PIX) * NPATCH
        srow = {"tau": tau, "p_high": p_high, "K": K,
                "bpp": bpp, "precision": p, "recall": r, "f1": f1}
        if not args.skip_mAP:
            map_acc = MeanAveragePrecision(iou_type="bbox", class_metrics=True).to(device)
            with torch.no_grad():
                for i in range(N):
                    tok = all_tokens[i].unsqueeze(0).to(device)
                    prb = torch.from_numpy(scores[i]).unsqueeze(0).to(device)
                    recon = reconstruct(model, tok, prb, tau, args.snr)
                    po = metrics.yolo_predict(yolo, recon.detach(), conf=0.001,
                                              iou=0.5, device=str(device))
                    gt_dev = [{"boxes": all_gt[i]["boxes"].to(device),
                               "labels": all_gt[i]["labels"].to(device)}]
                    map_acc.update(po, gt_dev)
            m = map_acc.compute()
            srow["map_50"] = float(m["map_50"])
            srow["snr_db"] = args.snr
        sweep_rows.append(srow)
        print(f"[fim_metrics] tau={tau}: P_high={p_high:.4f} K={K} BPP={bpp:.4f} "
              f"P={p:.4f} R={r:.4f} F1={f1:.4f}"
              + (f" mAP@0.5={srow['map_50']:.4f}" if not args.skip_mAP else ""),
              flush=True)

    with open(args.sweep_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        w.writeheader()
        w.writerows(sweep_rows)
    print(f"[fim_metrics] WROTE {args.sweep_csv}")


if __name__ == "__main__":
    main()
