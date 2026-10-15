#!/bin/bash
# Track F / E3 (plan 16.9): MCS ladder, evaluation-only, existing E2 checkpoint.
# 16-QAM r1/2 is skipped here -- it's identical to the existing E2 coded sweep
# (results/eval/evalfinal_e2_coded_ldpc_snr_sweep.csv). This covers the other 5 configs.
set -e
SNR_LIST="-5 -3 -1 0 1 3 5 7 9 10 11 13 15 20"
CHECKPOINT_PATH="ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS_E2_ERASURE/checkpoint-best_psnr.pth"

# name, bits_per_symbol, payload_rate
CONFIGS=(
  "16qam_r13:4:0.3333333333333333"
  "qpsk_r12:2:0.5"
  "qpsk_r13:2:0.3333333333333333"
  "bpsk_r12:1:0.5"
  "bpsk_r13:1:0.3333333333333333"
)

for cfg in "${CONFIGS[@]}"; do
    IFS=':' read -r name bps rate <<< "$cfg"
    CSV_OUTPUT_FILE="evalfinal_e3_${name}_snr_sweep.csv"
    rm -f "${CSV_OUTPUT_FILE}"
    echo "##### E3 config ${name} (bits_per_symbol=${bps}, payload_rate=${rate}) #####"
    for snr in $SNR_LIST; do
        echo "===== E3 ${name} eval, SNR=${snr} dB ====="
        PYTHONDONTWRITEBYTECODE=1 python3 run_class_main.py \
            --model ViT_Reconstruction_Model_Default \
            --output_dir evaluation_output_e3_${name} \
            --data_set fish --data_path "../yolo_fish_dataset_root/" \
            --num_object_classes 1 --yolo_weights best.pt \
            --batch_size 8 --input_size 224 --patch_size 8 \
            --encoder_embed_dim 512 --encoder_depth 8 --encoder_num_heads 8 \
            --decoder_embed_dim 512 --decoder_depth 6 --decoder_num_heads 8 \
            --fim_embed_dim 256 --fim_depth 2 --fim_num_heads 4 --quantizer_dim 512 \
            --quantizer_type rvq --rvq_stages 3 --rvq_bits_per_stage 6 --bits_vq_low 2 \
            --routing_topk 63 --num_workers 4 \
            --resume "${CHECKPOINT_PATH}" \
            --eval --snr_db_eval ${snr} \
            --digital_ldpc --digital_csi_mode perfect \
            --digital_bridge_python __SIONNA_PY__ \
            --digital_bits_per_symbol ${bps} --digital_payload_rate ${rate} \
            --yolo_conf_thres 0.4 --yolo_iou_thres 0.5 --map_conf_thres 0.001 \
            --csv_log_file "${CSV_OUTPUT_FILE}" --append_csv
        if [ $? -ne 0 ]; then
            echo "ERROR: E3 ${name} eval failed at SNR=${snr} dB"
            exit 1
        fi
    done
    echo "E3 ${name} sweep complete: ${CSV_OUTPUT_FILE}"
done
echo "E3 MCS ladder sweep complete (all 5 configs)"
