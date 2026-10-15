#!/bin/bash
# BASELINE-A launcher: acquire GPU lock, train 4 JSCC-CNN configs sequentially,
# release lock on exit.  Run with: setsid nohup bash run_baselinea.sh & 

LOCK=__AQUASC_TMP__/gpu.lock
while [ -f "$LOCK" ]; do
  owner=$(cat "$LOCK" 2>/dev/null)
  if [ -n "$owner" ] && kill -0 "$owner" 2>/dev/null; then sleep 120; else rm -f "$LOCK"; fi
done
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

PY=__TORCH_PY__
JSCC=__AQUASC_ROOT__/competitors/jscc
LOGS=__AQUASC_DATA__/logs
OUT=$JSCC/output_jscc_baselinea
TRAIN_IMG=__AQUASC_DATA__/yolo_fish_dataset_root/train/images
TRAIN_LBL=__AQUASC_DATA__/yolo_fish_dataset_root/train/labels
VAL_IMG=__AQUASC_DATA__/yolo_fish_dataset_root/valid/images
VAL_LBL=__AQUASC_DATA__/yolo_fish_dataset_root/valid/labels

mkdir -p "$LOGS"

cd "$JSCC"

run() {
  local variant=$1 ratio=$2
  echo "===== BASELINE-A: $variant ratio=$ratio ====="
  $PY baselinea_train.py \
    --variant "$variant" --ratio "$ratio" --epochs 150 --batch_size 16 \
    --train_image_dir "$TRAIN_IMG" --train_label_dir "$TRAIN_LBL" \
    --val_image_dir "$VAL_IMG" --val_label_dir "$VAL_LBL" \
    --out "$OUT"
}

run std 1/12    > "$LOGS/baselinea_std_c2.log" 2>&1
run bbw 1/12    > "$LOGS/baselinea_bbw_c2.log" 2>&1
run std 1/24    > "$LOGS/baselinea_std_c1.log" 2>&1
run bbw 1/24    > "$LOGS/baselinea_bbw_c1.log" 2>&1

echo "===== BASELINE-A training all done ====="
