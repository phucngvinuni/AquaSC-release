#!/bin/bash
# Track F / E3 mandatory companion (plan 16.9): the decoder's learned-prior
# floor. No SNR dependency (no channel involved) -- one pass over the full
# validation set with purely random codebook indices, no image input.
set -e
CHECKPOINT_PATH="ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS_E2_ERASURE/checkpoint-best_psnr.pth"
CSV_OUTPUT_FILE="evalfinal_prior_floor.csv"

python3 run_class_main.py \
    --model ViT_Reconstruction_Model_Default \
    --output_dir evaluation_output_prior_floor \
    --data_set fish --data_path "../yolo_fish_dataset_root/" \
    --num_object_classes 1 --yolo_weights best.pt \
    --batch_size 8 --input_size 224 --patch_size 8 \
    --encoder_embed_dim 512 --encoder_depth 8 --encoder_num_heads 8 \
    --decoder_embed_dim 512 --decoder_depth 6 --decoder_num_heads 8 \
    --fim_embed_dim 256 --fim_depth 2 --fim_num_heads 4 --quantizer_dim 512 \
    --quantizer_type rvq --rvq_stages 3 --rvq_bits_per_stage 6 --bits_vq_low 2 \
    --routing_topk 63 --num_workers 4 \
    --resume "${CHECKPOINT_PATH}" \
    --eval --snr_db_eval 0 --prior_floor_eval \
    --yolo_conf_thres 0.4 --yolo_iou_thres 0.5 --map_conf_thres 0.001 \
    --csv_log_file "${CSV_OUTPUT_FILE}"
echo "Prior floor eval complete: ${CSV_OUTPUT_FILE}"
