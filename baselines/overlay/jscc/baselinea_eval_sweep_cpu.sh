#!/bin/bash
# BASELINE-A eval sweep — CPU variant (runs concurrently with the CHANNEL-cont GPU
# fine-tune without touching the GPU; CPU work proceeds freely per wave4 rules §1).
# Reconstruct test split (432 imgs) for each trained checkpoint at SNR 0/10/20 dB
# (Rayleigh), score with best.pt @ conf 0.001 via metrics.py.

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
  "std_c2_snr10_ratio0.0833 2 best_loss"
  "bbw_c2_snr10_ratio0.0833 2 best_psnr"
  "bbw_c2_snr10_ratio0.0833 2 best_loss"
  "std_c1_snr10_ratio0.0417 1 best_psnr"
  "std_c1_snr10_ratio0.0417 1 best_loss"
  "bbw_c1_snr10_ratio0.0417 1 best_psnr"
  "bbw_c1_snr10_ratio0.0417 1 best_loss"
)

CSV=$OUT/eval_sweep.csv
echo "tag,checkpoint,snr,psnr,ssim,in_box_psnr,in_box_ssim,map_50_orig,map_50_recon,map_orig,map_recon,retention_50,retention,precision_recon,recall_recon,f1_recon" > "$CSV"

for entry in "${CONFIGS[@]}"; do
  set -- $entry
  tag=$1; c=$2; ckpt=$3
  CKPTFILE=$OUT/$tag/checkpoints/$ckpt.pkl
  if [ ! -f "$CKPTFILE" ]; then
    echo "MISSING $CKPTFILE" >> "$LOGS/baselinea_eval.log"
    continue
  fi
  for snr in 0 10 20; do
    RECON=$OUT/$tag/recon/$ckpt/snr$snr
    mkdir -p "$RECON"
    echo "===== recon $tag/$ckpt @ SNR$snr =====" >> "$LOGS/baselinea_eval.log"
    cd "$JSCC"
    $PY baselinea_reconstruct.py --checkpoint "$CKPTFILE" --c "$c" --snr "$snr" \
      --channel Rayleigh --input_dir "$TEST_IMG" --output_dir "$RECON" >> "$LOGS/baselinea_eval.log" 2>&1
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
echo "===== BASELINE-A eval sweep (CPU) done ====="
