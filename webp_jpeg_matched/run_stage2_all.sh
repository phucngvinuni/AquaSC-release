#!/bin/bash
# Stage 2 (decode + detector + PSNR/SSIM): waits for each config's six rx files, then scores it.
cd "$(dirname "$0")/.."
PY=__TORCH_PY__
for codec in webp jpeg; do for b in low mid high; do
  d=webp_jpeg_matched/runs/${codec}_${b}
  until [ -f $d/rx_snr-5.npz ] && [ -f $d/rx_snr0.npz ] && [ -f $d/rx_snr5.npz ] && [ -f $d/rx_snr10.npz ] && [ -f $d/rx_snr15.npz ] && [ -f $d/rx_snr20.npz ]; do sleep 20; done
  sleep 5
  rm -f $d/metrics.npz $d/summary.json
  taskset -c 28-31 $PY webp_jpeg_matched/stage2_eval.py --codec $codec --budget $b > webp_jpeg_matched/logs/stage2_${codec}_${b}.log 2>&1
done; done
echo STAGE2_DONE > webp_jpeg_matched/logs/stage2_done.flag
