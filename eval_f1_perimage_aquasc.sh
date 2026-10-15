#!/bin/bash
# Track F1 (plan 16.9): per-image AP@0.5 for AquaSC BPSK r1/3 and QPSK r1/2,
# at the channel_bank.npz SNR grid (-5,0,5,10,15,20), for paired bootstrap
# against ROI-JSCC. Reuses the existing E2 checkpoint, no retraining.
set -e
SNR_LIST="-5 0 5 10 15 20"
CHECKPOINT_PATH="ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS_E2_ERASURE/checkpoint-best_psnr.pth"

# name, bits_per_symbol, payload_rate
CONFIGS=(
  "bpsk_r13:1:0.3333333333333333"
  "qpsk_r12:2:0.5"
)

for cfg in "${CONFIGS[@]}"; do
    IFS=':' read -r name bps rate <<< "$cfg"
    for snr in $SNR_LIST; do
        OUT="f1_perimage/aquasc_${name}_snr${snr}.npy"
        echo "===== F1 per-image AP: AquaSC ${name}, SNR=${snr} dB ====="
        PYTHONDONTWRITEBYTECODE=1 python3 run_class_main.py \
            --model ViT_Reconstruction_Model_Default \
            --output_dir evaluation_output_f1_${name} \
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
            --per_image_ap_out "${OUT}"
        if [ $? -ne 0 ]; then
            echo "ERROR: F1 AquaSC ${name} failed at SNR=${snr} dB"
            exit 1
        fi
    done
done
echo "F1 AquaSC per-image capture complete"
