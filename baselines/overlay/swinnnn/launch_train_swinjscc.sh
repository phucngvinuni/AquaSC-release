#!/usr/bin/env bash
# Wave 9b SWINJSCC agent — sequential training launcher.
# Trains SwinJSCC_w/o_SAandRA on the honest split at C=20/33/64 (256x256,
# Rayleigh block-fading, SNR~U[0,20]), waiting for GPU headroom (<8.5GB) before
# each run.  Does NOT use the exclusive GPU lock (concurrent wave).
set -u
PY=__TORCH_PY__
DIR=__AQUASC_ROOT__/competitors/swinnnn
LOG=__AQUASC_DATA__/logs
mkdir -p "$LOG"

wait_headroom() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    used=${used:-0}
    if [ "$used" -lt 8500 ]; then
      echo "[launcher] GPU mem ${used} MiB < 8500, proceeding" | tee -a "$LOG/swinjscc_launcher.log"
      return 0
    fi
    echo "[launcher] GPU mem ${used} MiB >= 8500, waiting 60s" >> "$LOG/swinjscc_launcher.log"
    sleep 60
  done
}

for C in 20 33 64; do
  wait_headroom
  echo "[launcher] === starting C=${C} $(date) ===" | tee -a "$LOG/swinjscc_launcher.log"
  "$PY" "$DIR/train_swinjscc_honest.py" --C "$C" --epochs 100 --batch_size 2 \
      > "$LOG/swinjscc_train_C${C}.log" 2>&1
  rc=$?
  echo "[launcher] C=${C} finished rc=${rc} $(date)" | tee -a "$LOG/swinjscc_launcher.log"
done
echo "[launcher] ALL DONE $(date)" | tee -a "$LOG/swinjscc_launcher.log"
