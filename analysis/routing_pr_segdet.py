#!/usr/bin/env python3
"""analysis/routing_pr_segdet.py — Plan 16.10 E-6: segmentation/detector-heatmap
routing precision/recall (R3 #5), same protocol as analysis/routing_pr_topk.py (E-5)
so the two numbers are directly comparable.

Score = routing_baselines.detector_heatmap_scores: run the SAME frozen
detector (best.pt) used for the paper's mAP metric on the ORIGINAL image (as
available at the transmitter/edge), score each patch by the confidence of the
highest-confidence predicted box it overlaps, then top-K(63) exactly like the
FIM router. Compared against the same box-oracle ground truth used in E-5.

CPU-only by design (avoids GPU contention with concurrent training jobs);
YOLO inference only, no SemCom model needed.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repo root (core modules)
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import stats
import metrics
import routing_baselines as rb
from ultralytics import YOLO

H = W = 224
PS = 8
NP = 784
K = 63

TEST_IMG_DIR = "__AQUASC_DATA__/yolo_fish_dataset_root/test/images"
TEST_LBL_DIR = "__AQUASC_DATA__/yolo_fish_dataset_root/test/labels"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--conf", type=float, default=0.001,
                    help="detector conf threshold for the heatmap (matches map_conf_thres convention)")
    args = ap.parse_args()

    device = torch.device(args.device)
    yolo = YOLO(args.weights)
    yolo.model.eval()

    from torchvision import transforms
    from PIL import Image
    names = sorted(f for f in os.listdir(TEST_IMG_DIR)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    t = transforms.Compose([transforms.Resize((H, W)), transforms.ToTensor()])

    precisions, recalls = [], []
    n_no_positive = 0
    n_no_detection = 0
    with torch.no_grad():
        for idx, name in enumerate(names):
            img = t(Image.open(os.path.join(TEST_IMG_DIR, name)).convert("RGB")).unsqueeze(0).to(device)
            stem = os.path.splitext(name)[0]
            boxes = rb.parse_yolo_label(os.path.join(TEST_LBL_DIR, stem + ".txt"), H=H, W=W)
            gt = rb.oracle_binary(boxes, PS, NP, H, W).to(device)  # [1, NP]

            preds = metrics.yolo_predict(yolo, img, conf=args.conf, iou=0.5, device=str(device))
            pb, ps_ = preds[0]["boxes"].detach().cpu(), preds[0]["scores"].detach().cpu()
            if pb.shape[0] == 0:
                n_no_detection += 1
            scores = rb.detector_heatmap_scores(pb, ps_, PS, NP, H, W).to(device)  # [1, NP]
            mask, topk_idx = rb.apply_topk(scores, K)

            hit = (mask & (gt > 0.5)).sum().item()
            n_pos = int((gt > 0.5).sum().item())
            precisions.append(hit / K)
            if n_pos > 0:
                recalls.append(hit / n_pos)
            else:
                n_no_positive += 1
            if (idx + 1) % 50 == 0 or (idx + 1) == len(names):
                print(f"  {idx+1}/{len(names)} images processed", flush=True)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    p_ci = stats.t_confidence_interval(precisions)
    r_ci = stats.t_confidence_interval(recalls)
    print(f"\nDetector-heatmap top-K={K} routing precision/recall (n={len(names)} test images, "
          f"{n_no_positive} skipped from recall avg for zero-positive-patch images, "
          f"{n_no_detection} images with zero detections at conf={args.conf}):")
    print(f"  precision: mean={p_ci['mean']:.4f} +/- {p_ci['half_width']:.4f} "
          f"(95% CI [{p_ci['ci_low']:.4f}, {p_ci['ci_high']:.4f}])")
    print(f"  recall (n={len(recalls)}): mean={r_ci['mean']:.4f} +/- {r_ci['half_width']:.4f} "
          f"(95% CI [{r_ci['ci_low']:.4f}, {r_ci['ci_high']:.4f}])")

    np.savez("results/routing/routing_pr_segdet_topk63_result.npz", precisions=precisions, recalls=recalls,
             n_no_positive=n_no_positive, n_no_detection=n_no_detection, K=K,
             weights=args.weights, conf=args.conf)
    print("Saved results/routing/routing_pr_segdet_topk63_result.npz")


if __name__ == "__main__":
    main()
