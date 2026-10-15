#!/usr/bin/env python3
"""analysis/routing_pr_topk.py — Track D + V8 (plan 16.10 E-5): flagship top-K=63 routing
precision/recall against the box-oracle ground truth, closing the open \\todo
in the response letter (R3 #3, R3 #5).

For each test image: FIM score -> top-K(63) mask (routing_baselines.apply_topk),
compared against oracle_binary (1 if patch overlaps any GT box). Precision =
|topk ∩ positive| / 63; recall = |topk ∩ positive| / |positive| (image skipped
from the recall average if it has zero positive patches, since recall is
undefined there).

CPU-only by design (avoids GPU contention with concurrent training jobs);
encoder+FIM forward only, no decoder/channel needed.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repo root (core modules)
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils
import stats
import routing_baselines as rb
import model  # registers ViT_Reconstruction_Model_Default with timm

H = W = 224
PS = 8
NP = 784
K = 63

RUN1_CONFIG = dict(
    patch_size=8, encoder_embed_dim=512, encoder_depth=8, encoder_num_heads=8,
    decoder_embed_dim=512, decoder_depth=6, decoder_num_heads=8,
    fim_embed_dim=256, fim_depth=2, fim_num_heads=4, fim_drop_rate=0.1,
    fim_routing_threshold=0.6, routing_topk=63, eps_stab=1e-6,
    quantizer_dim=512, quantizer_type='rvq', rvq_stages=3, rvq_bits_per_stage=6,
    rvq_ema_decay=0.99, bits_vq_high=18, bits_vq_low=2, quantizer_commitment_cost=0.25,
    drop_rate=0.0, drop_path_rate=0.1,
)

TEST_IMG_DIR = "__AQUASC_DATA__/yolo_fish_dataset_root/test/images"
TEST_LBL_DIR = "__AQUASC_DATA__/yolo_fish_dataset_root/test/labels"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=(
        "ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS_E2_ERASURE/checkpoint-best_psnr.pth"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    m = build_model(args.checkpoint, device)

    from torchvision import transforms
    from PIL import Image
    names = sorted(f for f in os.listdir(TEST_IMG_DIR)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    t = transforms.Compose([transforms.Resize((H, W)), transforms.ToTensor()])

    precisions, recalls = [], []
    n_no_positive = 0
    with torch.no_grad():
        for idx, name in enumerate(names):
            img = t(Image.open(os.path.join(TEST_IMG_DIR, name)).convert("RGB")).unsqueeze(0).to(device)
            stem = os.path.splitext(name)[0]
            boxes = rb.parse_yolo_label(os.path.join(TEST_LBL_DIR, stem + ".txt"), H=H, W=W)
            gt = rb.oracle_binary(boxes, PS, NP, H, W).to(device)  # [1, NP]

            scores = rb.model_fim_scores(m, img)  # [1, NP]
            mask, topk_idx = rb.apply_topk(scores, K)  # mask [1,NP] bool

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
    print(f"\nFlagship top-K={K} routing precision/recall (n={len(names)} test images, "
          f"{n_no_positive} skipped from recall avg for zero-positive-patch images):")
    print(f"  precision: mean={p_ci['mean']:.4f} +/- {p_ci['half_width']:.4f} "
          f"(95% CI [{p_ci['ci_low']:.4f}, {p_ci['ci_high']:.4f}])")
    print(f"  recall (n={len(recalls)}): mean={r_ci['mean']:.4f} +/- {r_ci['half_width']:.4f} "
          f"(95% CI [{r_ci['ci_low']:.4f}, {r_ci['ci_high']:.4f}])")

    np.savez("results/routing/routing_pr_topk63_result.npz", precisions=precisions, recalls=recalls,
             n_no_positive=n_no_positive, K=K, checkpoint=args.checkpoint)
    print("Saved results/routing/routing_pr_topk63_result.npz")


if __name__ == "__main__":
    main()
