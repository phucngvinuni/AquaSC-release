# routing_eval_fixup.py — Wave 9a FIXUP driver (plan §0.2e corrections, R3 #5).
#
# Re-runs the routing strategies on the RUN-1 flagship (SHA b1d2eb25, best@E239)
# at K=63, SNR {0,10,20}, seed 42 (the V8-seeded proxy path: digital_seed =
# 42 + image_index), capturing BOTH the pooled mAP@0.5 (torchmetrics, conf 0.001)
# AND the per-image AP@0.5 (stats.per_image_ap50) — the unit of the paired
# Wilcoxon signed-rank. Strategies: fim (anchor), oracle_box, random (3 seeds for
# the pooled table; seed 42 for the per-image Wilcoxon), uniform, oracle_binary.
#
# The binomial-model claim ("96.2% of oracle") is RETIRED; the FIM claim is now
# supported by the paired Wilcoxon on per-image AP differences.
#
# Outputs:
#   routing_eval_fixup.csv               one row per (strategy, seed, snr) cell
#   routing_eval_fixup_perimage_ap.npz   per-image AP@0.5 [n_cells, 432]
#   routing_eval_fixup_stats.json        pooled mAP + Wilcoxon p-values

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import datasets
import metrics
import routing_baselines as rb
import stats
import utils
import model  # registers ViT_Reconstruction_Model_Default with timm
from ultralytics import YOLO
from torchmetrics.detection import MeanAveragePrecision

CEILING = 0.7292  # detector2 clean ceiling @ conf 0.001 (best.pt)

RUN1_CONFIG = dict(
    patch_size=8, encoder_embed_dim=512, encoder_depth=8, encoder_num_heads=8,
    decoder_embed_dim=512, decoder_depth=6, decoder_num_heads=8,
    fim_embed_dim=256, fim_depth=2, fim_num_heads=4, fim_drop_rate=0.1,
    fim_routing_threshold=0.6, routing_topk=63, eps_stab=1e-6,
    quantizer_dim=512, quantizer_type='rvq', rvq_stages=3, rvq_bits_per_stage=6,
    rvq_ema_decay=0.99, bits_vq_high=18, bits_vq_low=2, quantizer_commitment_cost=0.25,
    drop_rate=0.0, drop_path_rate=0.1,
)

H = W = 224
PS = 8
NP = 784
K = 63
SEED = 42
RANDOM_SEEDS = [42, 43, 44]

STRATEGIES = ["fim", "oracle_box", "random", "uniform", "oracle_binary"]


def build_model(checkpoint, device):
    ns = argparse.Namespace()
    ns.model = "ViT_Reconstruction_Model_Default"
    ns.input_size = 224
    for k, v in RUN1_CONFIG.items():
        setattr(ns, k, v)
    m = utils.get_model(ns)
    ck = torch.load(checkpoint, map_location="cpu")
    sd = ck.get("model", ck.get("state_dict", ck))
    m.load_state_dict(sd, strict=True)
    m.to(device).eval()
    return m


def preload(dataset):
    imgs, bms, gts = [], [], []
    for idx in range(len(dataset)):
        (img_t, bm_pos), (target, gt_dict, fim_map) = dataset[idx]
        imgs.append(img_t)
        bms.append(bm_pos if bm_pos is not None else torch.zeros(NP, dtype=torch.bool))
        gts.append(gt_dict)
    return imgs, bms, gts


def score_mask(strategy, boxes_list, batch_size, seed, device):
    """Return a [batch_size, NP] boolean high-routing mask for the strategy."""
    if strategy == "fim":
        return None
    if strategy == "oracle_box":
        scores = rb.oracle_box_scores(boxes_list, PS, NP, H, W)
    elif strategy == "random":
        scores = rb.random_scores(NP, seed=seed, batch_size=batch_size)
    elif strategy == "uniform":
        scores = rb.uniform_scores(NP, batch_size=batch_size)
    elif strategy == "oracle_binary":
        scores = rb.oracle_binary(boxes_list, PS, NP, H, W)
    else:
        raise ValueError(f"unknown strategy {strategy}")
    mask, _ = rb.apply_topk(scores, K)
    return mask.to(device)


def run_cell(model, yolo, imgs, bms, gts, strategy, snr, seed, device,
             conf=0.001, iou=0.5, chunk=16):
    """Evaluate one (strategy, snr, seed) cell. Returns pooled mAP@0.5 + per-image AP."""
    n = len(imgs)
    map_calc = MeanAveragePrecision(iou_type="bbox", class_metrics=True).to(device)
    map_calc.warn_on_many_detections = False
    ap_img = np.zeros(n, dtype=np.float32)
    psnr_all, ssim_all = [], []
    with torch.no_grad():
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            bsize = end - start
            img_b = torch.stack(imgs[start:end]).to(device)
            bm_b = torch.stack(bms[start:end]).to(device)
            override = score_mask(strategy, [gts[start + k]["boxes"] for k in range(bsize)],
                                  bsize, seed, device)
            ch_seed = SEED + start
            out = model(img=img_b, bm_pos=bm_b, _eval=True, eval_snr_db=float(snr),
                        digital_ldpc=False, digital_seed=ch_seed,
                        route_override=override)
            recon = out["reconstructed_image"]
            preds = metrics.yolo_predict(yolo, recon, conf=conf, iou=iou, device=device)
            gt_b = [{"boxes": gts[start + k]["boxes"].to(device),
                     "labels": gts[start + k]["labels"].to(device)}
                    for k in range(bsize)]
            map_calc.update(preds, gt_b)
            for k in range(bsize):
                idx = start + k
                ap_img[idx] = stats.per_image_ap50(
                    preds[k]["boxes"].cpu().numpy(), preds[k]["scores"].cpu().numpy(),
                    gts[idx]["boxes"].cpu().numpy(), iou_thr=0.5)
            orig_b = torch.stack(imgs[start:end]).to(device)
            psnr_all.extend(utils.calc_psnr(recon.detach().cpu(), orig_b.detach().cpu()))
            ssim_all.extend(utils.calc_ssim(recon.detach().cpu(), orig_b.detach().cpu()))
    res = map_calc.compute()
    map50 = float(res["map_50"]) if "map_50" in res and res["map_50"].numel() == 1 else float("nan")
    return {
        "map50": map50,
        "per_image_ap_mean": float(ap_img.mean()),
        "per_image_ap": ap_img,
        "psnr": float(np.mean(psnr_all)),
        "ssim": float(np.mean(ssim_all)),
    }


def analyze(per_image, cell_keys, out_json):
    """Paired Wilcoxon FIM vs {oracle, random, uniform} at each SNR."""
    keymap = {}
    for i, key in enumerate(cell_keys):
        keymap[key] = per_image[i]

    def get(strategy, snr, seed):
        k = (strategy, seed, snr)
        return keymap[k]

    report = {"ceiling": CEILING, "comparisons": []}
    for snr in (0, 10, 20):
        fim = get("fim", snr, SEED)
        for other in ("oracle_box", "random", "uniform"):
            o = get(other, snr, SEED)
            w = stats.wilcoxon_signed_rank(fim, o)
            entry = {
                "snr_db": snr, "fim_vs": other,
                "fim_mean": float(fim.mean()), "other_mean": float(o.mean()),
                "mean_diff": float((fim - o).mean()),
                "wilcoxon_statistic": w["statistic"], "wilcoxon_p": w["p_value"],
                "n": w["n"], "n_nonzero": w["n_nonzero"], "n_zero": w["n_zero"],
            }
            report["comparisons"].append(entry)
            print(f"  [SNR {snr:2d} dB] FIM vs {other:11s}: mean per-image AP "
                  f"{fim.mean():.4f} vs {o.mean():.4f}  diff={entry['mean_diff']:+.4f}  "
                  f"Wilcoxon W={w['statistic']:.1f} p={w['p_value']:.4g} "
                  f"(n={w['n']}, nonzero={w['n_nonzero']}, ties={w['n_zero']})")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS_FINAL/checkpoint-best_psnr.pth")
    ap.add_argument("--yolo", default="best.pt")
    ap.add_argument("--data_path", default="../yolo_fish_dataset_root/")
    ap.add_argument("--snr", type=float, nargs="+", default=[0, 10, 20])
    ap.add_argument("--out_csv", default="routing_eval_fixup.csv")
    ap.add_argument("--out_npz", default="routing_eval_fixup_perimage_ap.npz")
    ap.add_argument("--out_json", default="routing_eval_fixup_stats.json")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--analyze-only", action="store_true",
                    help="skip GPU eval; re-run Wilcoxon from the existing npz")
    args = ap.parse_args()

    cell_keys = []
    for strategy in STRATEGIES:
        seeds = RANDOM_SEEDS if strategy == "random" else [SEED]
        for seed in seeds:
            for snr in args.snr:
                cell_keys.append((strategy, seed, snr))

    if args.analyze_only:
        z = np.load(args.out_npz)
        per_image = z["per_image_ap"]
        keys = [(s, int(seed), float(snr)) for s, seed, snr in z["cell_keys"]]
        print("=" * 80)
        print("PAIRED WILCOXON (from existing npz)")
        print("=" * 80)
        analyze(per_image, keys, args.out_json)
        print("\nrouting_eval_fixup.py (analyze-only) DONE")
        return

    device = torch.device(args.device)
    model = build_model(args.checkpoint, device)
    print(f"loaded {args.checkpoint} (strict load OK)")

    yolo = YOLO(args.yolo)
    yolo.to(device)
    if hasattr(yolo, "fuse") and callable(yolo.fuse):
        yolo.fuse()
    yolo.eval()

    test_root = os.path.join(args.data_path, "test")
    ds = datasets.YOLODataset(
        img_root_dir_for_split=test_root, img_pixel_size=224, semcom_vit_patch_size=8,
        semcom_encoder_mask_ratio=0.0, is_train_split=False, num_object_classes=1,
        augment=False)
    assert len(ds) == 432, f"expected 432 test images, got {len(ds)}"
    print(f"test split: {len(ds)} images")

    imgs, bms, gts = preload(ds)
    print(f"preloaded {len(imgs)} images")

    n_cells = len(cell_keys)
    per_image = np.full((n_cells, len(imgs)), np.nan, dtype=np.float32)
    rows = []
    for ci, (strategy, seed, snr) in enumerate(cell_keys):
        r = run_cell(model, yolo, imgs, bms, gts, strategy, snr, seed, device,
                     conf=args.conf, chunk=args.chunk)
        per_image[ci] = r["per_image_ap"]
        rows.append({"strategy": strategy, "seed": seed, "snr_db": f"{snr:.1f}",
                     "map50": f"{r['map50']:.4f}",
                     "per_image_ap_mean": f"{r['per_image_ap_mean']:.4f}",
                     "psnr": f"{r['psnr']:.4f}", "ssim": f"{r['ssim']:.4f}",
                     "ceiling": f"{CEILING:.4f}"})
        print(f"[{strategy:>13s}] seed={seed} snr={snr:5.1f} dB  "
              f"mAP@0.5={r['map50']:.4f}  per-img-AP={r['per_image_ap_mean']:.4f}  "
              f"PSNR={r['psnr']:.4f}  SSIM={r['ssim']:.4f}", flush=True)

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["strategy", "seed", "snr_db", "map50",
                                          "per_image_ap_mean", "psnr", "ssim", "ceiling"])
        w.writeheader()
        w.writerows(rows)

    np.savez(args.out_npz, per_image_ap=per_image,
             cell_keys=np.array([[s, seed, snr] for s, seed, snr in cell_keys], dtype=object))

    print("\n" + "=" * 80)
    print("PAIRED WILCOXON (FIM vs oracle / random / uniform, per-image AP)")
    print("=" * 80)
    analyze(per_image, cell_keys, args.out_json)

    print(f"\nwrote {len(rows)} rows to {args.out_csv}, per-image AP to {args.out_npz}")
    print("routing_eval_fixup.py DONE")


if __name__ == "__main__":
    main()
