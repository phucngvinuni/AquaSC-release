"""ONNX-Runtime edge benchmark for AquaSC encoder (arm A) + YOLO detector (arm
B). Mirrors this lab's existing sempose_fi/bench_inference_throttled.py
protocol exactly (same stat schema, same freq/cores CLI convention for use
with an external cpufreq-set + taskset sweep).

Run under cpufreq throttle + single-core affinity (on Pi), same recipe as
sempose_fi's own docstring:

  sudo apt install -y cpufrequtils
  sudo cpufreq-set -g userspace

  for f in 600 900 1200 1500 1800; do
    sudo cpufreq-set -f ${f}000 -c 0
    taskset -c 0 python3 bench_onnx.py --freq $f --cores 1 --n_threads 1 \
        --out logs/onnx_${f}mhz_1c.json
  done

  # Baseline (all cores, full speed):
  python3 bench_onnx.py --freq 1800 --cores 4 --n_threads 4 --out logs/onnx_1800mhz_4c.json

Usage from the payload directory (aquasc_encoder.onnx / yolo_fish.onnx in ./).
"""
from __future__ import annotations
import argparse
import json
import os
import time

import numpy as np
import onnxruntime as ort
import psutil

N_WARMUP = 20
N_CALLS = 200


def rss_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024


def stats_from(times_ms):
    a = np.asarray(times_ms)
    return {
        "mean_ms": float(a.mean()), "std_ms": float(a.std()),
        "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
        "fps": float(1000.0 / a.mean()),
    }


def load_session(path, n_threads):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = n_threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])


def bench_session(sess, feed, n_warmup=N_WARMUP, n_calls=N_CALLS):
    for _ in range(n_warmup):
        sess.run(None, feed)
    times = []
    for _ in range(n_calls):
        t0 = time.perf_counter()
        sess.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aquasc_onnx", default="aquasc_encoder.onnx")
    ap.add_argument("--yolo_onnx", default="yolo_fish.onnx")
    ap.add_argument("--freq", type=int, default=0, help="CPU freq MHz -- label only")
    ap.add_argument("--cores", type=int, default=1, help="active cores -- label only")
    ap.add_argument("--n_threads", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = {"backend": "onnxruntime", "freq_mhz": args.freq, "cores": args.cores,
              "n_threads": args.n_threads}
    print(f"\n=== ONNX Runtime edge bench @ {args.freq}MHz / {args.cores} core(s) ===")

    if os.path.exists(args.aquasc_onnx):
        sess = load_session(args.aquasc_onnx, args.n_threads)
        img = np.random.rand(1, 3, 224, 224).astype(np.float32)
        t = bench_session(sess, {"img": img})
        s = stats_from(t); s["ram_mb"] = rss_mb()
        result["aquasc_encoder"] = s
        print(f"  [A]   {s['mean_ms']:6.2f}+-{s['std_ms']:.2f}ms  p95={s['p95_ms']:.2f}ms "
              f"{s['fps']:5.0f}fps RAM={s['ram_mb']:.1f}MB")
    else:
        print(f"  [A] {args.aquasc_onnx} not found -- skip")

    if os.path.exists(args.yolo_onnx):
        sess = load_session(args.yolo_onnx, args.n_threads)
        in_name = sess.get_inputs()[0].name
        img = np.random.rand(1, 3, 224, 224).astype(np.float32)
        t = bench_session(sess, {in_name: img})
        s = stats_from(t); s["ram_mb"] = rss_mb()
        result["yolo"] = s
        print(f"  [B]   {s['mean_ms']:6.2f}+-{s['std_ms']:.2f}ms  p95={s['p95_ms']:.2f}ms "
              f"{s['fps']:5.0f}fps RAM={s['ram_mb']:.1f}MB")
    else:
        print(f"  [B] {args.yolo_onnx} not found -- skip")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
