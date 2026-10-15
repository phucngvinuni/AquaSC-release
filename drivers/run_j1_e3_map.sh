#!/bin/bash
# Plan v5.9 J1 (E-3 mAP): pooled mAP@0.5 + per-image AP for ROI-JSCC at
# rcpp=64 and rcpp=128, through the same seeded channel bank used for F1
# (SHA 369d33f6...), conf 0.001. Sequential (eval-only, GPU-light, "hours").
set -e
cd competitors/ROI-JSCC
PY=__TORCH_PY__
for rcpp in 64 128; do
    echo "===== J1: rcpp=${rcpp} ====="
    $PY stats_eval_roijscc_bank.py \
        --ckpt "saved_models/ROIIT_FISH_Rayleigh_SNR010_rcpp$(printf '%03d' ${rcpp})_PSNR_ROIJSCC_roi-largest_box_best.pt" \
        --rcpp ${rcpp} --roi_mode largest_box \
        --snr -5 0 5 10 15 20 \
        --out_dir "../../f1_perimage" \
        --device cuda
done
echo "J1 complete"
