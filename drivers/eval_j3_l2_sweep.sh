#!/bin/bash
# Plan v5.9 J3 (E-7) postamble: SNR sweep + per-image AP for L2 (RVQ 4x6,
# K=63, low=4 bits, 0.0438 uses/px), same 14-point SNR grid and coded LDPC
# chain as drivers/eval_e2_sweep.sh, so it's directly comparable to the flagship
# and to ROI-JSCC rcpp=128 (0.0469 uses/px, J1).
set -e
PY=__TORCH_PY__
SNR_LIST="-5 -3 -1 0 1 3 5 7 9 10 11 13 15 20"
CHECKPOINT_PATH="ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS_L2_RVQ4x6_LOW4/checkpoint-best_psnr.pth"
CSV_OUTPUT_FILE="results/eval/evalfinal_l2_coded_ldpc_snr_sweep.csv"

echo "SNR_dB,PSNR,SSIM,mAP_50,mAP" > "${CSV_OUTPUT_FILE}"

for snr in $SNR_LIST; do
    echo "===== L2 coded eval, SNR=${snr} dB ====="
    PYTHONDONTWRITEBYTECODE=1 $PY run_class_main.py \
        --model ViT_Reconstruction_Model_Default \
        --output_dir evaluation_output_l2_coded \
        --data_set fish --data_path "../yolo_fish_dataset_root/" \
        --num_object_classes 1 --yolo_weights best.pt \
        --batch_size 8 --input_size 224 --patch_size 8 \
        --encoder_embed_dim 512 --encoder_depth 8 --encoder_num_heads 8 \
        --decoder_embed_dim 512 --decoder_depth 6 --decoder_num_heads 8 \
        --fim_embed_dim 256 --fim_depth 2 --fim_num_heads 4 --quantizer_dim 512 \
        --quantizer_type rvq --rvq_stages 4 --rvq_bits_per_stage 6 --bits_vq_low 4 \
        --routing_topk 63 --num_workers 4 \
        --resume "${CHECKPOINT_PATH}" \
        --eval --snr_db_eval ${snr} \
        --digital_ldpc --digital_csi_mode perfect \
        --digital_bridge_python __SIONNA_PY__ \
        --yolo_conf_thres 0.4 --yolo_iou_thres 0.5 --map_conf_thres 0.001 \
        --csv_log_file "${CSV_OUTPUT_FILE}" --append_csv \
        --per_image_ap_out "f1_perimage/l2_coded_snr${snr}.npy"
    if [ $? -ne 0 ]; then
        echo "ERROR: L2 eval failed at SNR=${snr} dB"
        exit 1
    fi
done
echo "L2 SNR sweep complete: ${CSV_OUTPUT_FILE}"
