#!/usr/bin/env python3
"""analysis/routing_pr_topk_fim_shrink.py -- 2026-09-23. Same protocol as
analysis/routing_pr_topk.py (patch-level precision/recall against oracle-box ground
truth, K=63, CPU-only) but parametrized for the Tiny (192d2L) encoder shape
and a SHRUNK fim_module shape, to evaluate training/train_fim_shrink.py's output
checkpoints. Compares against the ORIGINAL (pre-shrink) fim_module too, same
checkpoint's encoder, for an apples-to-apples before/after.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repo root (core modules)
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils  # noqa: E402
import stats  # noqa: E402
import routing_baselines as rb  # noqa: E402
import model  # noqa: E402
from model_util import FeatureImportanceTransformer  # noqa: E402

H = W = 224
PS = 8
NP = 784
K = 63

TEST_IMG_DIR = "__AQUASC_DATA__/yolo_fish_dataset_root/test/images"
TEST_LBL_DIR = "__AQUASC_DATA__/yolo_fish_dataset_root/test/labels"


def build_model(checkpoint, encoder_embed_dim, encoder_depth, encoder_num_heads,
                fim_embed_dim, fim_depth, fim_num_heads, device):
    ns = argparse.Namespace(
        model="ViT_Reconstruction_Model_Default", input_size=224, patch_size=8,
        encoder_embed_dim=encoder_embed_dim, encoder_depth=encoder_depth,
        encoder_num_heads=encoder_num_heads, decoder_embed_dim=512, decoder_depth=6,
        decoder_num_heads=8, quantizer_dim=512, bits_vq_high=10, bits_vq_low=2,
        quantizer_type="rvq", rvq_stages=3, rvq_bits_per_stage=6, rvq_ema_decay=.99,
        quantizer_commitment_cost=.25, drop_rate=0., drop_path_rate=.1,
        fim_embed_dim=fim_embed_dim, fim_depth=fim_depth, fim_num_heads=fim_num_heads,
        fim_drop_rate=.1, fim_routing_threshold=.7, routing_topk=63, eps_stab=1e-6,
        disable_fim=False,
    )
    m = utils.get_model(ns)
    ck = torch.load(checkpoint, map_location="cpu")
    sd = ck.get("model", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck
    m.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()}, strict=True)
    m.to(device).eval()
    return m


def eval_one(m, device, names):
    from torchvision import transforms
    from PIL import Image
    t = transforms.Compose([transforms.Resize((H, W)), transforms.ToTensor()])
    precisions, recalls = [], []
    n_no_positive = 0
    with torch.no_grad():
        for name in names:
            img = t(Image.open(os.path.join(TEST_IMG_DIR, name)).convert("RGB")).unsqueeze(0).to(device)
            stem = os.path.splitext(name)[0]
            boxes = rb.parse_yolo_label(os.path.join(TEST_LBL_DIR, stem + ".txt"), H=H, W=W)
            gt = rb.oracle_binary(boxes, PS, NP, H, W).to(device)
            scores = rb.model_fim_scores(m, img)
            mask, _ = rb.apply_topk(scores, K)
            hit = (mask & (gt > 0.5)).sum().item()
            n_pos = int((gt > 0.5).sum().item())
            precisions.append(hit / K)
            if n_pos > 0:
                recalls.append(hit / n_pos)
            else:
                n_no_positive += 1
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    p_ci = stats.t_confidence_interval(precisions)
    r_ci = stats.t_confidence_interval(recalls)
    return p_ci, r_ci, n_no_positive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shrunk_ckpt", required=True)
    ap.add_argument("--orig_ckpt", required=True,
                    help="the pre-shrink checkpoint (same encoder, original fim_module) for comparison")
    ap.add_argument("--encoder_embed_dim", type=int, default=192)
    ap.add_argument("--encoder_depth", type=int, default=2)
    ap.add_argument("--new_fim_embed_dim", type=int, default=64)
    ap.add_argument("--new_fim_depth", type=int, default=1)
    ap.add_argument("--new_fim_num_heads", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    names = sorted(f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    print(f"n={len(names)} test images")

    print("\n--- ORIGINAL fim_module (256/2/4) ---")
    m_orig = build_model(args.orig_ckpt, args.encoder_embed_dim, args.encoder_depth, 8,
                         256, 2, 4, device)
    p_ci, r_ci, n_skip = eval_one(m_orig, device, names)
    print(f"precision: {p_ci['mean']:.4f} +/- {p_ci['half_width']:.4f}  "
          f"recall: {r_ci['mean']:.4f} +/- {r_ci['half_width']:.4f}  (n_skip={n_skip})")

    print(f"\n--- SHRUNK fim_module ({args.new_fim_embed_dim}/{args.new_fim_depth}/{args.new_fim_num_heads}) ---")
    m_shrunk = build_model(args.shrunk_ckpt, args.encoder_embed_dim, args.encoder_depth, 8,
                           args.new_fim_embed_dim, args.new_fim_depth, args.new_fim_num_heads, device)
    p_ci2, r_ci2, n_skip2 = eval_one(m_shrunk, device, names)
    print(f"precision: {p_ci2['mean']:.4f} +/- {p_ci2['half_width']:.4f}  "
          f"recall: {r_ci2['mean']:.4f} +/- {r_ci2['half_width']:.4f}  (n_skip={n_skip2})")

    print(f"\nDelta (shrunk - original): precision {p_ci2['mean']-p_ci['mean']:+.4f}  "
          f"recall {r_ci2['mean']-r_ci['mean']:+.4f}")


if __name__ == "__main__":
    main()
