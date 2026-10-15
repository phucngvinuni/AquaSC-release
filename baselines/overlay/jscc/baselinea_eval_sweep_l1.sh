#!/bin/bash
# BASELINE-A eval sweep for the L1-loss retrains (2026-09-23): reconstruct+score
# the std_c{1,2}_l1 best_psnr checkpoints at the full 6-point SNR grid
# (-5,0,5,10,15,20), matching the mAP grid already computed via
# stats_eval_jscc_bank.py, so PSNR/SSIM numbers exist at the same protocol
# as the original MSE eval_sweep.csv rows.
set -e

PY=__TORCH_PY__
JSCC=__AQUASC_ROOT__/competitors/jscc
AQ=__AQUASC_ROOT__
OUT=$JSCC/output_jscc_baselinea
TEST_IMG=__AQUASC_DATA__/yolo_fish_dataset_root/test/images
TEST_LBL=__AQUASC_DATA__/yolo_fish_dataset_root/test/labels
YOLO=$AQ/best.pt
LOGS=__AQUASC_DATA__/logs

CONFIGS=(
  "std_c2_snr10_ratio0.0833_l1 2 best_psnr"
  "std_c1_snr10_ratio0.0417_l1 1 best_psnr"
)

CSV=$OUT/eval_sweep_l1.csv
echo "tag,checkpoint,snr,psnr,ssim,in_box_psnr,in_box_ssim,map_50_orig,map_50_recon,map_orig,map_recon,retention_50,retention,precision_recon,recall_recon,f1_recon" > "$CSV"

for entry in "${CONFIGS[@]}"; do
  set -- $entry
  tag=$1; c=$2; ckpt=$3
  CKPTFILE=$OUT/$tag/checkpoints/$ckpt.pkl
  if [ ! -f "$CKPTFILE" ]; then
    echo "MISSING $CKPTFILE" >> "$LOGS/baselinea_eval_l1.log"
    continue
  fi
  for snr in -5 0 5 10 15 20; do
    RECON=$OUT/$tag/recon/$ckpt/snr$snr
    mkdir -p "$RECON"
    echo "===== recon $tag/$ckpt @ SNR$snr =====" >> "$LOGS/baselinea_eval_l1.log"
    cd "$JSCC"
    $PY baselinea_reconstruct.py --checkpoint "$CKPTFILE" --c "$c" --snr "$snr" \
      --channel Rayleigh --input_dir "$TEST_IMG" --output_dir "$RECON" >> "$LOGS/baselinea_eval_l1.log" 2>&1
    cd "$AQ"
    $PY analysis/baselinea_score.py --yolo "$YOLO" --orig_dir "$TEST_IMG" --recon_dir "$RECON" \
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
echo "===== BASELINE-A L1 eval sweep done ====="
