#!/bin/bash
# Plan v7.4 handoff: fills the gap found while auditing RESULTS_OF_RECORD.md
# §2 -- the "Flagship, coded" row's per-image column comes from the true
# flagship (..._FINAL, SHA b1d2eb254194, via drivers/eval_flagship_perimage_bank.sh)
# but its pooled column pointed at results/eval/evalfinal_e2_coded_ldpc_snr_sweep.csv,
# which is actually the E2_ERASURE checkpoint (a different model). This script
# produces the pooled coded-LDPC sweep for the real flagship, at the same
# 6-point channel-bank SNR grid already used for the per-image arrays, so the
# row is internally consistent (same checkpoint for both columns).
set -e
SNR_LIST="-5 0 5 10 15 20"
CHECKPOINT_PATH="ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS_FINAL/checkpoint-best_psnr.pth"
CSV_OUTPUT_FILE="results/eval/evalfinal_flagship_coded_ldpc_snr_sweep.csv"

echo "SNR_dB,PSNR,SSIM,mAP_50,mAP" > "${CSV_OUTPUT_FILE}"

for snr in $SNR_LIST; do
    echo "===== flagship (FINAL) coded (LDPC) eval, SNR=${snr} dB ====="
    PYTHONDONTWRITEBYTECODE=1 __TORCH_PY__ run_class_main.py \
        --model ViT_Reconstruction_Model_Default \
        --output_dir evaluation_output_flagship_coded_pooled \
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
        --yolo_conf_thres 0.4 --yolo_iou_thres 0.5 --map_conf_thres 0.001 \
        --csv_log_file "${CSV_OUTPUT_FILE}" --append_csv
    if [ $? -ne 0 ]; then
        echo "ERROR: flagship coded eval failed at SNR=${snr} dB"
        exit 1
    fi
done
echo "Flagship coded SNR sweep complete: ${CSV_OUTPUT_FILE}"
