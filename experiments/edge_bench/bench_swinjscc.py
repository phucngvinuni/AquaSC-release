"""SwinJSCC encoder edge latency (arm A competitor #2). Random-init weights
(latency depends only on architecture): same encoder_kwargs as this project's
own build_net() (experiments/../competitors/swinnnn/eval_finetune_map_blockfade.py),
which itself never uses the checkpoint to determine architecture. rate=64 is
this project's SwinJSCC headline Site A operating point.

Usage: python3 bench_swinjscc.py --out logs/swinjscc_encoder.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "competitor_swinjscc" / "swinjscc_code"))

import numpy as np
import psutil
import torch
import torch.nn as nn
from net.encoder import create_encoder

IMG_SWIN = 256
N_WARMUP, N_CALLS = 20, 200

ENCODER_KWARGS = dict(
    model="SwinJSCC_w/_SAandRA", img_size=(IMG_SWIN, IMG_SWIN), patch_size=2, in_chans=3,
    embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10],
    C=None, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
    norm_layer=nn.LayerNorm, patch_norm=True,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=int, default=64, help="channel count C (64 = Site A headline)")
    ap.add_argument("--n_threads", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.n_threads:
        torch.set_num_threads(args.n_threads)

    encoder = create_encoder(**ENCODER_KWARGS).eval()
    encoder.update_resolution(IMG_SWIN, IMG_SWIN)  # one-time setup, not timed
    img = torch.rand(1, 3, IMG_SWIN, IMG_SWIN)

    with torch.no_grad():
        for _ in range(N_WARMUP):
            encoder(img, 10, args.rate, "SwinJSCC_w/_SAandRA")
        times = []
        for _ in range(N_CALLS):
            t0 = time.perf_counter()
            encoder(img, 10, args.rate, "SwinJSCC_w/_SAandRA")
            times.append((time.perf_counter() - t0) * 1000.0)

    a = np.asarray(times)
    result = {
        "system": "swinjscc", "rate": args.rate,
        "mean_ms": float(a.mean()), "std_ms": float(a.std()),
        "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
        "fps": float(1000.0 / a.mean()), "ram_mb": psutil.Process().memory_info().rss / 1024 / 1024,
    }
    print(f"[SwinJSCC encoder, rate={args.rate}] {result['mean_ms']:.2f}+-{result['std_ms']:.2f}ms "
          f"p95={result['p95_ms']:.2f}ms {result['fps']:.1f}fps RAM={result['ram_mb']:.1f}MB")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
