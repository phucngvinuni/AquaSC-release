#!/bin/bash
# Stage 1 (PHY) for the whole protocol grid. WebP first (3 budgets in parallel), then JPEG.
cd "$(dirname "$0")/.."
PY=__SIONNA_PY__
export TF_NUM_INTRAOP_THREADS=6 TF_NUM_INTEROP_THREADS=2
for codec in webp jpeg; do
  taskset -c 0-7   $PY webp_jpeg_matched/stage1_phy.py --codec $codec --budget low  > webp_jpeg_matched/logs/stage1_${codec}_low.log  2>&1 &
  taskset -c 8-15  $PY webp_jpeg_matched/stage1_phy.py --codec $codec --budget mid  > webp_jpeg_matched/logs/stage1_${codec}_mid.log  2>&1 &
  taskset -c 16-27 $PY webp_jpeg_matched/stage1_phy.py --codec $codec --budget high > webp_jpeg_matched/logs/stage1_${codec}_high.log 2>&1 &
  wait
done
echo STAGE1_DONE > webp_jpeg_matched/logs/stage1_done.flag
