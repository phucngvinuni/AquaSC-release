#!/bin/bash
# WebP/JPEG at exactly 1536/3072/6144 channel uses (3/6/12 blocks), in-band length, AquaSC's receiver and fading.
#   TORCH_PY=... SIONNA_PY=... bash scripts/run_webp_jpeg_matched.sh
# Protocol + decisions: webp_jpeg_matched/IMPLEMENTATION_NOTES.md. Stored results: webp_jpeg_matched/RESULTS_matched_budget.md
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd webp_jpeg_matched
"$TORCH_PY" calibrate_quality_valid.py      # quality chosen on the 437 validation images only
bash run_stage1_all.sh                      # Sionna venv: transmit through LDPC/16-QAM/Rayleigh
"$TORCH_PY" gates.py                        # protocol gates (all must pass)
bash run_stage2_all.sh                      # torch env: decode + detector + PSNR/SSIM
"$TORCH_PY" aggregate.py
