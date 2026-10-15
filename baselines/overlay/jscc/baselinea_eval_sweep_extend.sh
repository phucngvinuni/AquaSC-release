#!/bin/bash
# 2026-09-18: extend eval_sweep.csv to the 3 missing SNR points (-5, 5, 15)
# so JSCC Std/BBW's PSNR/SSIM cover the full 6-point grid, not just {0,10,20}.
# Exact same recipe as baselinea_eval_sweep_cpu.sh, best_psnr checkpoint only
# (the original script's own best_psnr/best_loss rows were "near-identical"
# per RESULTS_SUMMARY_TABLE.md's compilation -- one checkpoint is enough to
# fill the gap without doubling runtime).

export CUDA_VISIBLE_DEVICES=""

PY=__TORCH_PY__
JSCC=__AQUASC_ROOT__/competitors/jscc
AQ=__AQUASC_ROOT__
OUT=$JSCC/output_jscc_baselinea
TEST_IMG=__AQUASC_DATA__/yolo_fish_dataset_root/test/images
TEST_LBL=__AQUASC_DATA__/yolo_fish_dataset_root/test/labels
YOLO=$AQ/best.pt
LOGS=__AQUASC_DATA__/logs

CONFIGS=(
  "std_c2_snr10_ratio0.0833 2 best_psnr"
  "bbw_c2_snr10_ratio0.0833 2 best_psnr"
  "std_c1_snr10_ratio0.0417 1 best_psnr"
  "bbw_c1_snr10_ratio0.0417 1 best_psnr"
)

CSV=$OUT/eval_sweep.csv

for entry in "${CONFIGS[@]}"; do
  set -- $entry
  tag=$1; c=$2; ckpt=$3
  CKPTFILE=$OUT/$tag/checkpoints/$ckpt.pkl
  if [ ! -f "$CKPTFILE" ]; then
    echo "MISSING $CKPTFILE" >> "$LOGS/baselinea_eval_extend.log"
    continue
  fi
  for snr in -5 5 15; do
    RECON=$OUT/$tag/recon/$ckpt/snr$snr
    mkdir -p "$RECON"
    echo "===== recon $tag/$ckpt @ SNR$snr =====" >> "$LOGS/baselinea_eval_extend.log"
    cd "$JSCC"
    $PY baselinea_reconstruct.py --checkpoint "$CKPTFILE" --c "$c" --snr "$snr" \
      --channel Rayleigh --input_dir "$TEST_IMG" --output_dir "$RECON" >> "$LOGS/baselinea_eval_extend.log" 2>&1
    cd "$AQ"
    $PY baselinea_score.py --yolo "$YOLO" --orig_dir "$TEST_IMG" --recon_dir "$RECON" \
      --label_dir "$TEST_LBL" --conf 0.001 --out_csv "$RECON/score.csv" \
      --batch_size 16 > "$RECON/score.stdout" 2>&1
    $PY - "$RECON/score.json" "$tag" "$ckpt" "$snr" "$CSV" <<'EOF'
import json, sys, csv
j=json.load(open(sys.argv[1])); tag=sys.argv[2]; ckpt=sys.argv[3]; snr=sys.argv[4]; csvp=sys.argv[5]
row=[tag,ckpt,snr,j.get('psnr'),j.get('ssim'),j.get('in_box_psnr'),j.get('in_box_ssim'),
     j.get('map_50_orig'),j.get('map_50_recon'),j.get('map_orig'),j.get('map_recon'),
     j.get('retention_50'),j.get('retention'),j.get('precision_recon'),j.get('recall_recon'),j.get('f1_recon')]
with open(csvp,'a',newline='') as f: csv.writer(f).writerow(row)
print('row:', row)
EOF
  done
done
echo "DONE_EXTEND_SWEEP"
