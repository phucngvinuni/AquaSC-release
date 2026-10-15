#!/bin/bash
# run_webp_lowrates_full.sh -- 2026-09-24. Sequential full run of both
# WebP+LDPC low-rate real bit-level channel sims (q=1, q=81), 432 images x
# 6 SNRs x M=10 each. ~3.0h (q=1) + ~4.6h (q=81) sequential, per timed
# smoke test on 3 images. Runs in background, does not block GPU work.
set -e
cd "$(dirname "$0")"
SIONNA=__SIONNA_PY__

echo "=== q=1 starting $(date) ==="
$SIONNA webp_real_channel_sweep_lowrates_stage1.py --quality 1
echo "=== q=1 done $(date) ==="

echo "=== q=81 starting $(date) ==="
$SIONNA webp_real_channel_sweep_lowrates_stage1.py --quality 81
echo "=== q=81 done $(date) ==="

echo "===== WebP lowrates full sweep DONE ====="
