#!/bin/bash
# Wave 21 A-1 (2026-09-15): re-run multi-block UEP @0.122449 (bps=2, payload_rate=2/3)
# through 10 independent channel realizations per SNR, to get a real M=10-style CI
# for the 15/20dB "converged" claim in F7 panel (b) -- currently single-seed there.
#
# NOT a literal channel_bank.npz replay: the multi-block chain draws per-LDPC-block
# fades (up to 6 independent Rayleigh draws/image, digital_pipeline.py's
# simulate_digital_pipeline), not the bank's one-fade-per-image convention that
# joint-frame/ROI-JSCC/JSCC-CNN use -- see RESULTS_OF_RECORD.md section 6t. This
# script instead runs 10 widely-separated --seed values per SNR (real, independent
# channel draws each time, batch-size-invariant per engine1.py's digital_seed fix)
# and the per-image arrays are averaged across the 10 realizations afterward --
# the same STATISTICAL purpose as bank-M10 (reduce single-realization noise),
# without claiming bit-identical fading draws to the bank.
set -e
PY=__TORCH_PY__
SNR_LIST="-5 0 5 10 15 20"
SEED_LIST="1 2 3 4 5 6 7 8 9 10"
CHECKPOINT_PATH="ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS_FINAL/checkpoint-best_psnr.pth"
OUT_DIR="f1_perimage/matched5_mcs_m10"
mkdir -p "$OUT_DIR"

for snr in $SNR_LIST; do
  for m in $SEED_LIST; do
    out_file="${OUT_DIR}/flagship_matched5_mcs_snr${snr}_m${m}.npy"
    if [ -f "$out_file" ]; then
      echo "===== SKIP (already exists): SNR=${snr}dB m=${m} ====="
      continue
    fi
    echo "===== A-1: multi-block@0.1224, SNR=${snr}dB, realization m=${m}/10 ====="
    seed=$((44000000 + m * 1000000))
    PYTHONDONTWRITEBYTECODE=1 "$PY" run_class_main.py \
      --model ViT_Reconstruction_Model_Default \
      --output_dir evaluation_output_a1_m10 \
      --data_set fish --data_path "../yolo_fish_dataset_root/" \
      --num_object_classes 1 --yolo_weights best.pt \
      --batch_size 8 --input_size 224 --patch_size 8 \
      --encoder_embed_dim 512 --encoder_depth 8 --encoder_num_heads 8 \
      --decoder_embed_dim 512 --decoder_depth 6 --decoder_num_heads 8 \
      --fim_embed_dim 256 --fim_depth 2 --fim_num_heads 4 --quantizer_dim 512 \
      --quantizer_type rvq --rvq_stages 3 --rvq_bits_per_stage 6 --bits_vq_low 2 \
      --routing_topk 63 --num_workers 4 \
      --resume "${CHECKPOINT_PATH}" \
      --eval --snr_db_eval "${snr}" --seed "${seed}" \
      --digital_ldpc --digital_csi_mode perfect \
      --digital_bits_per_symbol 2 --digital_payload_rate 0.6666666666666666 \
      --digital_bridge_python __SIONNA_PY__ \
      --yolo_conf_thres 0.4 --yolo_iou_thres 0.5 --map_conf_thres 0.001 \
      --per_image_ap_out "${out_file}" 2>&1 | tail -4
    avail=$(free -m | awk '/^Mem:/{print $7}')
    echo "  [mem check] available=${avail}MiB"
    if [ "$avail" -lt 2000 ]; then
      echo "  [mem check] low memory, pausing 20s before next run"
      sleep 20
    fi
  done
done
echo "A-1 sweep complete: 60/60 runs"
