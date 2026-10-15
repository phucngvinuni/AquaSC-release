import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repo root (core modules)
# analysis/stats_eval_bank.py — Wave 8 STATS step 5/6 driver (GPU, run AFTER evalfinal.done).
#
# Replays the seeded paired channel bank for the RUN-1 flagship (RVQ 3x6 + 2-bit
# low, patch 8) on the PROXY path at SNR {0,10,20} x M=10 realizations, scoring
# mAP@0.5 at conf 0.001 with the AquaSC detector.  The bank is replayed by
# passing  digital_seed = bank.subseed_base(snr, m) + chunk_start  to model.forward
# for an ascending-index chunk, which makes digital_channel.simulate_rayleigh_channel
# draw the SAME {h, n} frozen into channel_bank.npz (linear sub-seed derivation:
# subseed(i,snr,m) = base(snr,m) + i).
#
# Outputs (per (snr, m)): aggregate mAP@0.5 (torchmetrics, pooled) + per-image
# AP@0.5 (stats.per_image_ap50) -> saved to CSVs/npz, then t-CI + paired bootstrap.
#
# This script does NOT retrain and does NOT modify model.py / engine1.py; it
# reuses model.forward, datasets.YOLODataset and stats.py verbatim.  Images are
# pre-loaded once and processed in ascending-index chunks (batch CHUNK) so the
# Python-heavy proxy transmission is amortized.

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets
import stats
import utils
import model  # registers ViT_Reconstruction_Model_Default with timm
import channel_bank as CB
from ultralytics import YOLO

try:
    from torchmetrics.detection import MeanAveragePrecision
    _TM = True
except ImportError:
    _TM = False

CEILING = 0.7292  # detector2 clean ceiling @ conf 0.001; overridden by --ceiling for Site B

RUN1_CONFIG = dict(
    patch_size=8, encoder_embed_dim=512, encoder_depth=8, encoder_num_heads=8,
    decoder_embed_dim=512, decoder_depth=6, decoder_num_heads=8,
    fim_embed_dim=256, fim_depth=2, fim_num_heads=4, fim_drop_rate=0.1,
    fim_routing_threshold=0.6, routing_topk=63, eps_stab=1e-6,
    quantizer_dim=512, quantizer_type='rvq', rvq_stages=3, rvq_bits_per_stage=6,
    rvq_ema_decay=0.99, bits_vq_high=18, bits_vq_low=2, quantizer_commitment_cost=0.25,
    drop_rate=0.0, drop_path_rate=0.1,
)


def build_args(model_name, input_size=224):
    ns = argparse.Namespace()
    ns.model = model_name
    ns.input_size = input_size
    for k, v in RUN1_CONFIG.items():
        setattr(ns, k, v)
    return ns


def load_flagship(checkpoint, device):
    args = build_args("ViT_Reconstruction_Model_Default")
    model = utils.get_model(args)
    ck = torch.load(checkpoint, map_location="cpu")
    sd = ck.get("model", ck.get("state_dict", ck))
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    print(f"loaded {checkpoint} (strict load OK)")
    return model, args


def metrics_yolo_predict(yolo, images, conf, iou, device):
    out = []
    results = yolo(images.detach(), verbose=False, conf=conf, iou=iou)
    for r in results:
        if r.boxes is not None and len(r.boxes.xyxy) > 0:
            out.append({"boxes": r.boxes.xyxy.to(device), "scores": r.boxes.conf.to(device),
                        "labels": r.boxes.cls.to(torch.int64).to(device)})
        else:
            out.append({"boxes": torch.empty((0, 4), dtype=torch.float32, device=device),
                        "scores": torch.empty(0, dtype=torch.float32, device=device),
                        "labels": torch.empty(0, dtype=torch.int64, device=device)})
    return out


def preload(dataset, device):
    """Load all images/gt into memory once (avoids repeated cv2 per cell)."""
    imgs, bms, gts = [], [], []
    for idx in range(len(dataset)):
        (img_t, bm_pos), (target, gt_dict, fim_map) = dataset[idx]
        imgs.append(img_t)
        bms.append(bm_pos if bm_pos is not None else torch.zeros(784, dtype=torch.bool))
        gts.append(gt_dict)
    return imgs, bms, gts


def run_eval(model, yolo, imgs, bms, gts, bank, snr_list, m_list, device,
             out_prefix, conf=0.001, iou=0.5, chunk=16):
    if not _TM:
        raise ImportError("torchmetrics required")
    n = len(imgs)
    snr_keys = [(s, m) for s in snr_list for m in m_list]
    per_image = np.full((len(snr_keys), n), np.nan, dtype=np.float32)
    agg = {}
    for si, (snr, m) in enumerate(snr_keys):
        map_calc = MeanAveragePrecision(iou_type="bbox", class_metrics=True).to(device)
        map_calc.warn_on_many_detections = False
        ap_list = np.zeros(n, dtype=np.float32)
        base = CB.subseed_base(snr, m)
        with torch.no_grad():
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                img_b = torch.stack(imgs[start:end]).to(device)
                bm_b = torch.stack(bms[start:end]).to(device)
                seed = base + start
                out = model(img=img_b, bm_pos=bm_b, _eval=True, eval_snr_db=float(snr),
                            digital_ldpc=False, digital_seed=seed)
                recon = out["reconstructed_image"]
                preds = metrics_yolo_predict(yolo, recon, conf=conf, iou=iou, device=device)
                gt_b = [{"boxes": gts[start + k]["boxes"].to(device),
                         "labels": gts[start + k]["labels"].to(device)}
                        for k in range(end - start)]
                map_calc.update(preds, gt_b)
                for k in range(end - start):
                    idx = start + k
                    ap_list[idx] = stats.per_image_ap50(
                        preds[k]["boxes"].cpu().numpy(), preds[k]["scores"].cpu().numpy(),
                        gts[idx]["boxes"].cpu().numpy(), iou_thr=0.5)
        res = map_calc.compute()
        map50 = float(res["map_50"]) if "map_50" in res and res["map_50"].numel() == 1 else float("nan")
        agg[(snr, m)] = map50
        per_image[si] = ap_list
        print(f"snr={snr} m={m}: mAP@0.5={map50:.4f}  mean per-image AP={ap_list.mean():.4f}",
              flush=True)
    np.savez(f"{out_prefix}_perimage_ap.npz",
             per_image_ap=per_image, snr_keys=np.array(snr_keys))
    with open(f"{out_prefix}_agg.json", "w") as f:
        json.dump({f"{s}_{m}": v for (s, m), v in agg.items()}, f, indent=2)
    return per_image, snr_keys, agg


def summarize(per_image, snr_keys, agg, ceiling=CEILING):
    for snr in sorted({s for s, m in snr_keys}):
        mvals = [agg[(snr, m)] for m in sorted({mm for ss, mm in snr_keys if ss == snr})]
        ret = np.array([v / ceiling for v in mvals])
        tci = stats.t_confidence_interval(ret)
        print(f"\n[SNR {snr} dB] retention (M={len(ret)}): mean={tci['mean']:.4f} "
              f"+/- {tci['half_width']:.4f}  (95% t-CI [{tci['ci_low']:.4f}, {tci['ci_high']:.4f}])")
        for v in ret:
            print(f"    mAP={v * ceiling:.4f} retention={v:.4f}")
        idxs = [i for i, (ss, mm) in enumerate(snr_keys) if ss == snr]
        ap_img = per_image[idxs].mean(axis=0)
        ret_img = ap_img / ceiling
        bs = stats.bootstrap_ci(ret_img, n_boot=10000, seed=42)
        print(f"    bootstrap over images (retention): mean={bs['mean']:.4f} "
              f"95% CI [{bs['ci_low']:.4f}, {bs['ci_high']:.4f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--yolo", default="best.pt")
    ap.add_argument("--data_path", default="../yolo_fish_dataset_root/")
    ap.add_argument("--bank", default=CB.default_bank_path())
    ap.add_argument("--snr", type=float, nargs="+", default=[0, 10, 20])
    ap.add_argument("--m", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--out_prefix", default="stats_bank_run1")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ceiling", type=float, default=CEILING,
                     help="detector clean-channel ceiling for retention normalization "
                          "(default: Site A's 0.7292; pass 0.6044 for Site B pooled)")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, margs = load_flagship(args.checkpoint, device)

    yolo = YOLO(args.yolo)
    yolo.to(device)
    if hasattr(yolo, "fuse") and callable(yolo.fuse):
        yolo.fuse()
    yolo.eval()

    bank = CB.ChannelBank(args.bank)
    test_root = os.path.join(args.data_path, "test")
    ds = datasets.YOLODataset(
        img_root_dir_for_split=test_root, img_pixel_size=224, semcom_vit_patch_size=8,
        semcom_encoder_mask_ratio=0.0, is_train_split=False, num_object_classes=1,
        augment=False)
    assert len(ds) == bank.n_images, (
        f"dataset/bank size mismatch: {len(ds)} test images vs. bank's "
        f"{bank.n_images} images -- {args.bank} was not generated for "
        f"{test_root}'s split")
    print(f"test split: {len(ds)} images; bank: {bank.n_images} images, M={bank.M}")

    imgs, bms, gts = preload(ds, device)
    print(f"preloaded {len(imgs)} images")

    per_image, snr_keys, agg = run_eval(
        model, yolo, imgs, bms, gts, bank, args.snr, args.m, device, args.out_prefix,
        conf=args.conf, iou=0.5, chunk=args.chunk)
    summarize(per_image, snr_keys, agg, ceiling=args.ceiling)
    print("\nstats_eval_bank.py DONE")


if __name__ == "__main__":
    main()
