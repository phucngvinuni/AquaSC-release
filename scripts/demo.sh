#!/bin/bash
# One-command demo: needs NO checkpoint, dataset or GPU (CPU, a few minutes).
#   TORCH_PY=... SIONNA_PY=... bash scripts/demo.sh
# 1. unit tests of the statistics protocol, routing controls, bitstream (header/mask/CRC round trip) and the LDPC/QAM chain
# 2. recomputation of the paired confidence intervals from the shipped per-image arrays
# 3. regeneration of the manuscript figures from the shipped result files (written to ../figs/)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
echo "== 1/3 tests"
"$TORCH_PY" test_stats.py
"$TORCH_PY" test_routing_baselines.py
"$TORCH_PY" test_bitstream.py
"$SIONNA_PY" test_digital_pipeline.py
echo "== 2/3 statistics from stored per-image arrays"
"$TORCH_PY" verification/s3_s7/s3_recompute.py
echo "== 3/3 figures"
bash scripts/reproduce_figures.sh
echo "DEMO OK"
