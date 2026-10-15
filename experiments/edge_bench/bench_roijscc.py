"""ROI-JSCC encoder edge latency (arm A competitor #3). Random-init weights
(latency depends only on architecture): same model_info/rcpp=32 config as
this project's own roijscc_fish_common.build_model() (already-proven code),
which itself never uses a checkpoint to determine architecture.

Only the encode-side (model.Encoder + the attention-mask preamble it needs,
matching ROIJSCC.forward()'s own call `self.Encoder(x, attention_mask)`) is
timed -- channel simulation and model.Decoder are server/base-station side,
same framing as arm A for every other system in this benchmark.

Usage: python3 bench_roijscc.py --out logs/roijscc_encoder.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent / "competitor_roijscc"
sys.path.insert(0, str(HERE / "aquasc_stubs"))
sys.path.insert(0, str(HERE))

import numpy as np
import psutil
import torch
from model.JSCC import ROIJSCC
from model.common_component import get_weighted_total_attention_mask

N_WARMUP, N_CALLS = 20, 200
ROI_RES = 256
PATCH_D = 4
ETA = 0.1


def build_model(rcpp=32, chan_type="Rayleigh"):
    model_info = {
        "chan_type": chan_type, "color_channel": 3, "rcpp": rcpp,
        "n_block_list": [2, 2, 2, 2], "n_feats_list": [40, 60, 80, 260],
        "num_heads_list": [4, 6, 8, 10], "window_size_list": [8, 8, 8, 8],
        "input_resolution": (ROI_RES, ROI_RES), "ratio1": 0.5, "ratio2": 0.5,
        "eta": ETA, "patch_d": PATCH_D, "SNR_info": 10,
    }
    return ROIJSCC(model_info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rcpp", type=int, default=32, help="Site A operating point")
    ap.add_argument("--n_threads", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.n_threads:
        torch.set_num_threads(args.n_threads)

    model = build_model(rcpp=args.rcpp).eval()
    encoder = model.Encoder
    img = torch.rand(1, 3, ROI_RES, ROI_RES) * 2 - 1  # [-1,1], matches real usage
    roi_index = (1, 1)  # fixed representative box; latency is shape-, not value-, driven

    with torch.no_grad():
        mask = get_weighted_total_attention_mask(roi_index, 1, PATCH_D, PATCH_D)
        for _ in range(N_WARMUP):
            encoder(img, mask)
        times = []
        for _ in range(N_CALLS):
            t0 = time.perf_counter()
            encoder(img, mask)
            times.append((time.perf_counter() - t0) * 1000.0)

    a = np.asarray(times)
    result = {
        "system": "roi-jscc", "rcpp": args.rcpp,
        "mean_ms": float(a.mean()), "std_ms": float(a.std()),
        "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
        "fps": float(1000.0 / a.mean()), "ram_mb": psutil.Process().memory_info().rss / 1024 / 1024,
    }
    print(f"[ROI-JSCC encoder, rcpp={args.rcpp}] {result['mean_ms']:.2f}+-{result['std_ms']:.2f}ms "
          f"p95={result['p95_ms']:.2f}ms {result['fps']:.1f}fps RAM={result['ram_mb']:.1f}MB")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
