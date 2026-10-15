#!/bin/bash
# Full coded evaluation of one AquaSC rate point on the Site A TEST split (432 images x 10 channel draws x 6 SNRs).
#   TORCH_PY=... SIONNA_PY=... bash scripts/eval_coded.sh {high|mid|low} [OUT_DIR] [SNRS="-5 0 5 10 15 20"] [M=10]
# high = 0.122449 uses/px (6144 symbols, flagship checkpoint)   mid = 0.061224 (3072 symbols, retrained r061)
# low  = 0.030612 (1536 symbols, retrained r030).   Needs: checkpoint (docs/CHECKPOINTS.md), dataset (docs/DATA.md), best.pt detector.
# Pipeline: encoder + RVQ (torch) -> serializer, CRC-32, BG2 LDPC, Gray QPSK, Rayleigh channel, APP demapper, 40-iteration decoder
# (Sionna venv) -> RVQ decoder + detector (torch). Frames whose CRC fails use the all-zero latent fallback and stay in the scoring.
# Runtime on an RTX 4070 / i9-13900K: about 2-3 minutes per SNR point per rate (docs/TIMING.md).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
RATE="${1:?rate: high|mid|low}"; OUT="${2:-$ROOT/runs_eval/$RATE}"; SNRS="${3:--5 0 5 10 15 20}"; M="${4:-10}"
case "$RATE" in
  high) CK=ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS_FINAL/checkpoint-best_psnr.pth; SYM=6144; ARCH="--rvq-bits-per-stage 6 --bits-vq-low 2 --routing-topk 63";;
  mid)  CK=ckpt_codec_r061_topk32_3x4_low1/checkpoint-best_map.pth;                       SYM=3072; ARCH="--rvq-bits-per-stage 4 --bits-vq-low 1 --routing-topk 32";;
  low)  CK=ckpt_codec_r030_topk24_3x6_low0/checkpoint-best_map.pth;                       SYM=1536; ARCH="--rvq-bits-per-stage 6 --bits-vq-low 0 --routing-topk 24";;
  *) echo "rate must be high|mid|low" >&2; exit 2;;
esac
[ -f "$CK" ] || { echo "missing checkpoint $CK -- see docs/CHECKPOINTS.md (scripts/attach_checkpoints.sh)" >&2; exit 1; }
[ -f best.pt ] || { echo "missing detector best.pt -- see docs/CHECKPOINTS.md" >&2; exit 1; }
M5=experiments/matched5
mkdir -p "$OUT"
"$TORCH_PY" $M5/evaluate.py --phase cache --split test --ckpt "$CK" $ARCH --out-dir "$OUT/cache"
for SNR in $SNRS; do
  "$SIONNA_PY" $M5/phy.py --transmit --source "$OUT/cache/source.npz" --split test --variant frame_qpsk --m "$M" --symbols "$SYM" --snr-db "$SNR" --out-dir "$OUT/phy_$SNR"
  "$TORCH_PY" $M5/evaluate.py --phase aquasc --split test --m "$M" --snr-db "$SNR" --receiver "$OUT/phy_$SNR/receiver.npz" --ckpt "$CK" $ARCH --out-dir "$OUT/aqua_$SNR"
done
echo "done: per-SNR summary.json / metrics.npz in $OUT/aqua_*  (pooled mAP@0.5, per-image AP50, frame-loss rate)"
