import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repo root (core modules)
# analysis/routing_eval.py — Wave 9a ROUTING-EVAL driver (plan §8.1 / §11.3, R3 #5).
#
# Runs the routing strategies from routing_baselines.py (fim anchor, oracle_box,
# random at matched K, uniform, oracle_binary) as drop-in score functions on the
# RUN-1 flagship checkpoint at K=63, SNR {0,10,20}, proxy (uncoded digital) path.
#
# Each strategy produces a [B, 784] score -> top-K(63) -> boolean mask, passed to
# model.forward(route_override=...) (eval-only hook added to model.py). The FIM
# anchor passes route_override=None so the model uses its internal FIM top-K.
#
# Channel seed convention matches engine1.py: digital_seed = 42 + image_index
# (batch_size=1). Chunking reproduces the identical per-image channel draws
# because transmit_and_receive_indices_batch uses seed + in-batch-index.
#
# Metrics (EVAL-FINAL protocol): torchmetrics MeanAveragePrecision (bbox,
# class_metrics=True) pooled mAP@0.5 at conf 0.001 / iou 0.5 (metrics.py's
# yolo_predict), global PSNR/SSIM via utils.calc_psnr / utils.calc_ssim.
#
# CAVEAT (must travel with the table): the decoder was trained on FIM-routed
# tokens, so swapping the score function at eval is a MISMATCHED train/test
# condition.

import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets
import metrics
import routing_baselines as rb
import utils
import model  # registers ViT_Reconstruction_Model_Default with timm
from ultralytics import YOLO
from torchmetrics.detection import MeanAveragePrecision

CEILING = 0.7292  # detector2 clean ceiling @ conf 0.001 (best.pt, wave-4 gate)

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


def build_model(checkpoint, device):
    import argparse as _ap
    ns = _ap.Namespace()
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
    imgs, bms, gts, boxes_list = [], [], [], []
    for idx in range(len(dataset)):
        (img_t, bm_pos), (target, gt_dict, fim_map) = dataset[idx]
        imgs.append(img_t)
        bms.append(bm_pos if bm_pos is not None else torch.zeros(NP, dtype=torch.bool))
        gts.append(gt_dict)
        boxes_list.append(gt_dict["boxes"])  # absolute-pixel xyxy on 224 grid
    return imgs, bms, gts, boxes_list


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


def run_cell(model, yolo, imgs, bms, gts, boxes_list, strategy, snr, seed,
             device, conf=0.001, iou=0.5, chunk=16):
    """Evaluate one (strategy, snr, seed) cell. Returns dict of aggregate metrics."""
    n = len(imgs)
    map_calc = MeanAveragePrecision(iou_type="bbox", class_metrics=True).to(device)
    map_calc.warn_on_many_detections = False
    psnr_all, ssim_all = [], []
    with torch.no_grad():
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            bsize = end - start
            img_b = torch.stack(imgs[start:end]).to(device)
            bm_b = torch.stack(bms[start:end]).to(device)
            override = score_mask(strategy, boxes_list[start:end], bsize, seed, device)
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
            orig_b = torch.stack(imgs[start:end]).to(device)
            psnr_all.extend(utils.calc_psnr(recon.detach().cpu(), orig_b.detach().cpu()))
            ssim_all.extend(utils.calc_ssim(recon.detach().cpu(), orig_b.detach().cpu()))
    res = map_calc.compute()
    map50 = float(res["map_50"]) if "map_50" in res and res["map_50"].numel() == 1 else float("nan")
    return {
        "map50": map50,
        "retention_50": map50 / CEILING if map50 == map50 else float("nan"),
        "psnr": float(np.mean(psnr_all)),
        "ssim": float(np.mean(ssim_all)),
    }


def compute_ceiling(yolo, imgs, gts, device, conf=0.001, iou=0.5, chunk=16):
    n = len(imgs)
    map_calc = MeanAveragePrecision(iou_type="bbox", class_metrics=True).to(device)
    map_calc.warn_on_many_detections = False
    with torch.no_grad():
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            img_b = torch.stack(imgs[start:end]).to(device)
            preds = metrics.yolo_predict(yolo, img_b, conf=conf, iou=iou, device=device)
            gt_b = [{"boxes": gts[start + k]["boxes"].to(device),
                     "labels": gts[start + k]["labels"].to(device)}
                    for k in range(end - start)]
            map_calc.update(preds, gt_b)
    res = map_calc.compute()
    return float(res["map_50"]) if "map_50" in res and res["map_50"].numel() == 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS_FINAL/checkpoint-best_psnr.pth")
    ap.add_argument("--yolo", default="best.pt")
    ap.add_argument("--data_path", default="../yolo_fish_dataset_root/")
    ap.add_argument("--snr", type=float, nargs="+", default=[0, 10, 20])
    ap.add_argument("--strategies", nargs="+",
                    default=["fim", "oracle_box", "random", "uniform", "oracle_binary"])
    ap.add_argument("--random_seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out_csv", default="results/routing/routing_eval_results.csv")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true",
                    help="run a single-image forward sanity check (route_override differs) and exit")
    args = ap.parse_args()

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

    imgs, bms, gts, boxes_list = preload(ds)
    print(f"preloaded {len(imgs)} images")

    if args.smoke:
        print("\n--- smoke test: single-image forward, fim vs uniform override ---")
        img1 = imgs[0].unsqueeze(0).to(device)
        bm1 = bms[0].unsqueeze(0).to(device)
        with torch.no_grad():
            out_fim = model(img=img1, bm_pos=bm1, _eval=True, eval_snr_db=10.0,
                            digital_ldpc=False, digital_seed=SEED)
            fim_scores = rb.model_fim_scores(model, img1)
            fim_mask, _ = rb.apply_topk(fim_scores, K)
            override = rb.apply_topk(rb.uniform_scores(NP, batch_size=1), K)[0].to(device)
            out_ov = model(img=img1, bm_pos=bm1, _eval=True, eval_snr_db=10.0,
                           digital_ldpc=False, digital_seed=SEED, route_override=override)
        n_diff = int((fim_mask != override).sum())
        recon_diff = float((out_fim["reconstructed_image"] - out_ov["reconstructed_image"]).abs().mean())
        print(f"  fim mask vs uniform mask differ in {n_diff} of {NP} tokens")
        print(f"  |recon(fim) - recon(override)| mean = {recon_diff:.6f}")
        assert n_diff > 0, "expected fim mask != uniform override mask"
        assert recon_diff > 0, "expected different reconstructions"
        print("  smoke test PASSED (override changes the mask and runs end-to-end)")
        return

    ceiling = compute_ceiling(yolo, imgs, gts, device, conf=args.conf)
    print(f"clean ceiling map_50_orig = {ceiling:.4f} (expected ~{CEILING:.4f})")

    fieldnames = ["strategy", "seed", "snr_db", "map50", "retention_50", "psnr", "ssim",
                  "ceiling"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

    rows = []
    for strategy in args.strategies:
        if strategy == "random":
            seeds = args.random_seeds
        else:
            seeds = [SEED]
        for sd in seeds:
            for snr in args.snr:
                r = run_cell(model, yolo, imgs, bms, gts, boxes_list, strategy, snr, sd,
                             device, conf=args.conf, chunk=args.chunk)
                row = {"strategy": strategy, "seed": sd, "snr_db": f"{snr:.1f}",
                       "map50": f"{r['map50']:.4f}", "retention_50": f"{r['retention_50']:.4f}",
                       "psnr": f"{r['psnr']:.4f}", "ssim": f"{r['ssim']:.4f}",
                       "ceiling": f"{ceiling:.4f}"}
                rows.append(row)
                print(f"[{strategy:>14s}] seed={sd} snr={snr:5.1f} dB  "
                      f"mAP@0.5={r['map50']:.4f}  retention={r['retention_50']:.4f}  "
                      f"PSNR={r['psnr']:.4f}  SSIM={r['ssim']:.4f}", flush=True)
    with open(args.out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {args.out_csv}")
    print("analysis/routing_eval.py DONE")


if __name__ == "__main__":
    main()
