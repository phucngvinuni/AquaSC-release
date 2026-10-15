"""JSCC-CNN encoder edge latency (arm A competitor #1). Random-init weights:
latency depends only on architecture, not weight values -- same config (c=2,
the headline Site A rate) used throughout this project's real evaluations.
Run standalone (own process; avoids the AquaSC/ROI-JSCC `model` module name
collision noted in experiments/matched5/evaluate.py's docstring).

Usage: python3 bench_jscc.py --out logs/jscc_encoder.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "competitor_jscc"))

import numpy as np
import psutil
import torch
from model import DeepJSCC

N_WARMUP, N_CALLS = 20, 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=int, default=2, help="bottleneck channels (c=2 = Site A headline rate)")
    ap.add_argument("--n_threads", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.n_threads:
        torch.set_num_threads(args.n_threads)

    net = DeepJSCC(c=args.c, channel_type="Rayleigh", snr=10, P_encoder=1.0).eval()
    encoder = net.encoder
    img = torch.rand(1, 3, 224, 224)

    with torch.no_grad():
        for _ in range(N_WARMUP):
            encoder(img)
        times = []
        for _ in range(N_CALLS):
            t0 = time.perf_counter()
            encoder(img)
            times.append((time.perf_counter() - t0) * 1000.0)

    a = np.asarray(times)
    result = {
        "system": "jscc-cnn", "c": args.c,
        "mean_ms": float(a.mean()), "std_ms": float(a.std()),
        "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
        "fps": float(1000.0 / a.mean()), "ram_mb": psutil.Process().memory_info().rss / 1024 / 1024,
    }
    print(f"[JSCC-CNN encoder, c={args.c}] {result['mean_ms']:.2f}+-{result['std_ms']:.2f}ms "
          f"p95={result['p95_ms']:.2f}ms {result['fps']:.1f}fps RAM={result['ram_mb']:.1f}MB")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
