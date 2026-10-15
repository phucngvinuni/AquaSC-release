#!/bin/bash
# Plan v5.7 16.10 E-4 (B3): analog Rayleigh proxy channel, SAME checkpoint
# (E2_ERASURE flagship) and SAME SNR grid as eval_e2_sweep.sh's coded LDPC
# sweep (evalfinal_e2_coded_ldpc_snr_sweep.csv), so the two CSVs are a clean
# same-checkpoint/same-SNR proxy-vs-coded comparison. Eval-only, no --digital_ldpc.
set -e
SNR_LIST="-5 -3 -1 0 1 3 5 7 9 10 11 13 15 20"
CHECKPOINT_PATH="ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS_E2_ERASURE/checkpoint-best_psnr.pth"
CSV_OUTPUT_FILE="evalfinal_e4_proxy_snr_sweep.csv"

echo "SNR_dB,PSNR,SSIM,mAP_50,mAP" > "${CSV_OUTPUT_FILE}"

for snr in $SNR_LIST; do
    echo "===== E4 analog proxy eval, SNR=${snr} dB ====="
    PYTHONDONTWRITEBYTECODE=1 python3 run_class_main.py \
        --model ViT_Reconstruction_Model_Default \
        --output_dir evaluation_output_e4_proxy \
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
        --yolo_conf_thres 0.4 --yolo_iou_thres 0.5 --map_conf_thres 0.001 \
        --csv_log_file "${CSV_OUTPUT_FILE}" --append_csv
    if [ $? -ne 0 ]; then
        echo "ERROR: E4 proxy eval failed at SNR=${snr} dB"
        exit 1
    fi
done
echo "E4 proxy SNR sweep complete: ${CSV_OUTPUT_FILE}"
