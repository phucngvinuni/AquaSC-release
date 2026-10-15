"""PyTorch-eager edge benchmark: AquaSC encoder (arm A), AquaSC decoder +
round-trip (arm C, reference only), and the local YOLO fish detector (arm B).

Same protocol as this lab's existing sempose_fi/bench_inference_throttled.py:
batch=1, 20 warmup + 200 timed calls, mean/std/p50/p95 (ms) + FPS + RSS MB.
Run this script itself under `taskset -c <cores>` to pin cores (labels below
are informational only, matching bench_inference_throttled.py's convention).

Expects, alongside this file: model_util.py, edge_encoder_decoder.py,
encoder_weights.pt, decoder_weights.pt (from extract_edge_weights.py), and
(for arm B) a YOLO checkpoint path passed via --yolo.

Usage (on the Pi):
  python3 bench_pytorch.py --freq 1800 --cores 4 --n_threads 4 \
      --yolo best.pt --out logs/pytorch_1800mhz_4c.json
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from edge_encoder_decoder import AquaSCEdgeEncoder, AquaSCEdgeDecoder, FLAGSHIP_ARCH

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


def time_calls(fn, n_warmup=N_WARMUP, n_calls=N_CALLS):
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_calls):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def bench_one_aquasc(payload_dir: Path, name: str, arch: dict):
    enc = AquaSCEdgeEncoder(arch).eval()
    enc.load_state_dict(torch.load(payload_dir / f"encoder_{name}.pt", map_location="cpu"), strict=False)
    img = torch.rand(1, 3, 224, 224)

    result = {}
    with torch.no_grad():
        print(f"  [A:{name}] AquaSC encoder (transmitter, edge-side)...")
        t = time_calls(lambda: enc(img))
        s = stats_from(t); s["ram_mb"] = rss_mb()
        result["encoder"] = s
        print(f"      {s['mean_ms']:.2f}+-{s['std_ms']:.2f}ms p95={s['p95_ms']:.2f}ms "
              f"{s['fps']:.1f}fps RAM={s['ram_mb']:.1f}MB")
    result["encoder_disk_mb"] = (payload_dir / f"encoder_{name}.pt").stat().st_size / 1e6

    dec_path = payload_dir / f"decoder_{name}.pt"
    if not dec_path.exists():
        print(f"  [C:{name}] decoder_{name}.pt not shipped (reference-only arm, skipped to save transfer)")
        return result

    dec = AquaSCEdgeDecoder(arch).eval()
    dec.load_state_dict(torch.load(dec_path, map_location="cpu"), strict=False)
    with torch.no_grad():
        packed_high, idx_low, route = enc(img)

        print(f"  [C-decode:{name}] AquaSC decoder (base-station side, reference only)...")
        t = time_calls(lambda: dec(packed_high, idx_low, route))
        s = stats_from(t); s["ram_mb"] = rss_mb()
        result["decoder"] = s
        print(f"      {s['mean_ms']:.2f}+-{s['std_ms']:.2f}ms p95={s['p95_ms']:.2f}ms "
              f"{s['fps']:.1f}fps RAM={s['ram_mb']:.1f}MB")

        print(f"  [C-roundtrip:{name}] encoder+decoder, no channel (reference only)...")
        def roundtrip():
            ph, il, rt = enc(img)
            dec(ph, il, rt)
        t = time_calls(roundtrip)
        s = stats_from(t); s["ram_mb"] = rss_mb()
        result["roundtrip"] = s
        print(f"      {s['mean_ms']:.2f}+-{s['std_ms']:.2f}ms p95={s['p95_ms']:.2f}ms "
              f"{s['fps']:.1f}fps RAM={s['ram_mb']:.1f}MB")
    result["decoder_disk_mb"] = dec_path.stat().st_size / 1e6
    return result


def bench_aquasc(payload_dir: Path):
    manifest_path = payload_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = [dict(name="weights", encoder_embed_dim=512, encoder_depth=8, encoder_num_heads=8)]
    out = {}
    for m in manifest:
        arch = dict(FLAGSHIP_ARCH)
        arch["encoder_embed_dim"] = m["encoder_embed_dim"]
        arch["encoder_depth"] = m["encoder_depth"]
        arch["encoder_num_heads"] = m["encoder_num_heads"]
        out[m["name"]] = bench_one_aquasc(payload_dir, m["name"], arch)
    return out


def bench_yolo(yolo_path: str):
    from ultralytics import YOLO
    net = YOLO(yolo_path)
    net.fuse()
    dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    print("  [B] YOLO local detector (edge-side, replaces AquaSC entirely)...")
    t = time_calls(lambda: net.predict(dummy, imgsz=224, conf=0.001, iou=0.5,
                                       verbose=False, device="cpu"))
    s = stats_from(t); s["ram_mb"] = rss_mb()
    print(f"      {s['mean_ms']:.2f}+-{s['std_ms']:.2f}ms p95={s['p95_ms']:.2f}ms "
          f"{s['fps']:.1f}fps RAM={s['ram_mb']:.1f}MB")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload_dir", default=".")
    ap.add_argument("--yolo", default="best.pt")
    ap.add_argument("--freq", type=int, default=0, help="CPU freq MHz -- label only")
    ap.add_argument("--cores", type=int, default=os.cpu_count(), help="active cores -- label only")
    ap.add_argument("--n_threads", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    n_threads = args.n_threads or args.cores
    torch.set_num_threads(n_threads)

    print(f"=== PyTorch eager edge bench @ {args.freq}MHz / {args.cores} core(s) / "
          f"{n_threads} torch threads ===")
    result = {"backend": "pytorch", "freq_mhz": args.freq, "cores": args.cores,
              "n_threads": n_threads}
    result["aquasc"] = bench_aquasc(Path(args.payload_dir))

    def _save():
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Saved -> {args.out}")

    _save()  # write AquaSC-only results now, in case YOLO fails below
    if Path(args.yolo).exists():
        try:
            result["yolo"] = bench_yolo(args.yolo)
        except Exception as e:
            print(f"  [B] YOLO bench failed ({e!r}) -- AquaSC results above are still saved")
    else:
        print(f"  [B] {args.yolo} not found -- skipping YOLO arm")
    _save()


if __name__ == "__main__":
    main()
