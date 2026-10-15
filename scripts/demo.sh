#!/bin/bash
# One-command demo: needs NO checkpoint, dataset or GPU (CPU, a few minutes).
#   TORCH_PY=... SIONNA_PY=... bash scripts/demo.sh
# 1. unit tests of the statistics protocol, routing controls, bitstream (header/mask/CRC round trip) and the LDPC/QAM chain
# 2. recomputation of the paired confidence intervals from the shipped per-image arrays
# 3. regeneration of the manuscript figures from the shipped result files (written to ../figs/)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# tests/test_digital_pipeline.py crashed intermittently on the authors' desktop (roughly 1 run in 5-6, only when TensorFlow/Sionna is loaded: a TypeError/SystemError/segfault raised inside the
# interpreter, not an assertion failure; see docs/STATUS.md). A genuine test failure fails all three attempts.
retry() { for i in 1 2 3; do "$@" && return 0; echo "attempt $i/3 failed"; done; return 1; }
echo "== 1/3 tests"
"$TORCH_PY" tests/test_stats.py
"$TORCH_PY" tests/test_routing_baselines.py
retry "$TORCH_PY" tests/test_bitstream.py
retry "$SIONNA_PY" tests/test_digital_pipeline.py
echo "== 2/3 statistics from stored per-image arrays"
"$TORCH_PY" verification/s3_s7/s3_recompute.py
echo "== 3/3 figures"
bash scripts/reproduce_figures.sh
echo "DEMO OK"
