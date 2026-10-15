#!/bin/bash
# Wave 19 Part 2: AquaSC-Analog per-image AP@0.5 at the 6 shared SNRs
# (-5,0,5,10,15,20), same protocol/convention as eval_flagship_perimage_bank.sh
# (Rule 11: distinct output filenames per checkpoint tag, never colliding with
# flagship_snr*.npy or each other). No --digital_ldpc: there is no discrete
# index/codebook step for this variant, so the differentiable Rayleigh proxy
# (the same one used in training) IS the real eval-time channel -- see
# RESULTS_OF_RECORD.md section 6q.
#
# Evaluates BOTH checkpoint-best_psnr.pth and the final-epoch checkpoint --
# this project's own thesis is that PSNR does not track mAP (JSCC-CNN Std:
# ~33dB PSNR, mAP~=0), so PSNR-based checkpoint selection gets the same
# scrutiny here as it did for SwinJSCC, not a pass for our own experiment.
#
# Usage: bash eval_aquasc_analog_perimage_bank.sh <best_psnr|final> <ckpt_filename>
#   bash eval_aquasc_analog_perimage_bank.sh best_psnr checkpoint-best_psnr.pth
#   bash eval_aquasc_analog_perimage_bank.sh final checkpoint-299.pth
set -e
TAG="${1:?usage: eval_aquasc_analog_perimage_bank.sh <tag> <ckpt_filename>}"
CKPT_FILE="${2:?usage: eval_aquasc_analog_perimage_bank.sh <tag> <ckpt_filename>}"
PY=__TORCH_PY__
SNR_LIST="-5 0 5 10 15 20"
CHECKPOINT_PATH="ckpt_semcom_reconstruction_yolo_fish_AQUASC_ANALOG_0714/${CKPT_FILE}"

for snr in $SNR_LIST; do
    echo "===== AquaSC-Analog (${TAG}) per-image AP, SNR=${snr} dB ====="
    PYTHONDONTWRITEBYTECODE=1 $PY run_class_main.py \
        --model ViT_Reconstruction_Model_Default \
        --output_dir "evaluation_output_aquasc_analog_perimage_${TAG}" \
        --data_set fish --data_path "../yolo_fish_dataset_root/" \
        --num_object_classes 1 --yolo_weights best.pt \
        --batch_size 8 --input_size 224 --patch_size 8 \
        --encoder_embed_dim 512 --encoder_depth 8 --encoder_num_heads 8 \
        --decoder_embed_dim 512 --decoder_depth 6 --decoder_num_heads 8 \
        --fim_embed_dim 256 --fim_depth 2 --fim_num_heads 4 --quantizer_dim 512 \
        --quantizer_type analog --analog_symbols_per_token 9 \
        --routing_topk 63 --num_workers 4 \
        --resume "${CHECKPOINT_PATH}" \
        --eval --snr_db_eval ${snr} \
        --yolo_conf_thres 0.4 --yolo_iou_thres 0.5 --map_conf_thres 0.001 \
        --per_image_ap_out "f1_perimage/aquasc_analog_0714_${TAG}_snr${snr}.npy"
    if [ $? -ne 0 ]; then
        echo "ERROR: AquaSC-Analog (${TAG}) per-image eval failed at SNR=${snr} dB"
        exit 1
    fi
done
echo "AquaSC-Analog (${TAG}) per-image bank sweep complete"
