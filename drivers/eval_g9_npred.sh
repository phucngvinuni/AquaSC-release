#!/bin/bash
# G9b (plan v5.4): rerun all AquaSC digital-chain points + prior floor with
# npred instrumentation (engine1.py now saves a companion .npred.npy
# alongside --per_image_ap_out). Measures marginal npred=0, not AP=0 proxy.
set -e
CHECKPOINT="ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS_E2_ERASURE/checkpoint-best_psnr.pth"

run_digital() {
    local name=$1 bps=$2 rate=$3 snr=$4
    echo "===== G9b npred: ${name} SNR=${snr} ====="
    PYTHONDONTWRITEBYTECODE=1 python3 run_class_main.py \
        --model ViT_Reconstruction_Model_Default \
        --output_dir evaluation_output_g9_${name} \
        --data_set fish --data_path "../yolo_fish_dataset_root/" \
        --num_object_classes 1 --yolo_weights best.pt \
        --batch_size 8 --input_size 224 --patch_size 8 \
        --encoder_embed_dim 512 --encoder_depth 8 --encoder_num_heads 8 \
        --decoder_embed_dim 512 --decoder_depth 6 --decoder_num_heads 8 \
        --fim_embed_dim 256 --fim_depth 2 --fim_num_heads 4 --quantizer_dim 512 \
        --quantizer_type rvq --rvq_stages 3 --rvq_bits_per_stage 6 --bits_vq_low 2 \
        --routing_topk 63 --num_workers 4 \
        --resume "${CHECKPOINT}" \
        --eval --snr_db_eval ${snr} \
        --digital_ldpc --digital_csi_mode perfect \
        --digital_bridge_python __SIONNA_PY__ \
        --digital_bits_per_symbol ${bps} --digital_payload_rate ${rate} \
        --yolo_conf_thres 0.4 --yolo_iou_thres 0.5 --map_conf_thres 0.001 \
        --per_image_ap_out "f1_perimage/aquasc_${name}_snr${snr}.npy"
}

for snr in -10 -7.5 -5 0 5 10 15 20; do
    run_digital bpsk_r13 1 0.3333333333333333 ${snr}
done
for snr in -5 0 5 10 15 20; do
    run_digital qpsk_r12 2 0.5 ${snr}
done

echo "===== G9b npred: prior floor ====="
PYTHONDONTWRITEBYTECODE=1 python3 run_class_main.py \
    --model ViT_Reconstruction_Model_Default \
    --output_dir evaluation_output_g9_floor \
    --data_set fish --data_path "../yolo_fish_dataset_root/" \
    --num_object_classes 1 --yolo_weights best.pt \
    --batch_size 8 --input_size 224 --patch_size 8 \
    --encoder_embed_dim 512 --encoder_depth 8 --encoder_num_heads 8 \
    --decoder_embed_dim 512 --decoder_depth 6 --decoder_num_heads 8 \
    --fim_embed_dim 256 --fim_depth 2 --fim_num_heads 4 --quantizer_dim 512 \
    --quantizer_type rvq --rvq_stages 3 --rvq_bits_per_stage 6 --bits_vq_low 2 \
    --routing_topk 63 --num_workers 4 \
    --resume "${CHECKPOINT}" \
    --eval --snr_db_eval -5 --prior_floor_eval \
    --yolo_conf_thres 0.4 --yolo_iou_thres 0.5 --map_conf_thres 0.001 \
    --per_image_ap_out "f1_perimage/prior_floor_per_image.npy"

echo "G9b npred capture complete"
