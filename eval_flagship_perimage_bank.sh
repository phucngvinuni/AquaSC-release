#!/bin/bash
# Plan v7.0: flagship per-image AP@0.5 at the 6 shared channel-bank SNRs
# (-5,0,5,10,15,20), same image-index pairing convention as F1's
# bpsk_r13/qpsk_r12 arrays and J1's ROI-JSCC bank arrays -- the one missing
# input for the unified paired crossover table (flagship vs rcpp64/128).
set -e
PY=__TORCH_PY__
SNR_LIST="-5 0 5 10 15 20"
CHECKPOINT_PATH="ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS_FINAL/checkpoint-best_psnr.pth"

for snr in $SNR_LIST; do
    echo "===== flagship per-image AP, SNR=${snr} dB ====="
    PYTHONDONTWRITEBYTECODE=1 $PY run_class_main.py \
        --model ViT_Reconstruction_Model_Default \
        --output_dir evaluation_output_flagship_perimage \
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
        --per_image_ap_out "f1_perimage/flagship_snr${snr}.npy"
    if [ $? -ne 0 ]; then
        echo "ERROR: flagship per-image eval failed at SNR=${snr} dB"
        exit 1
    fi
done
echo "flagship per-image bank sweep complete"
