#!/bin/bash
# Plan v5.9 J3 (E-7): train L2 -- RVQ 4x6 (24-bit high path), K=63, low=4 bits
# -> 0.0438 uses/px. Exact RUN-1 recipe otherwise (configs/final_18_2.yaml),
# from scratch, 300 epochs. Does not wait for J1: rcpp=128 is already known to
# train (PSNR-only so far); L2 vs rcpp=128 (0.0469) is the tightest honest
# matched-rate pair with routing kept on our side.
# --num_workers 4 (not 8): J2 runs concurrently at num_workers=8 on the same
# GPU/host, and two 8-worker DataLoader pools crashed twice on 2026-09-09
# (/dev/shm bus errors) -- keeping this job's worker count lower bounds the
# combined pool size while J2's own training phase is active.
set -e
PY=__TORCH_PY__

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 $PY run_class_main.py \
    --model ViT_Reconstruction_Model_Default \
    --output_dir ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS_L2_RVQ4x6_LOW4 \
    --data_set fish --data_path "../yolo_fish_dataset_root/" \
    --num_object_classes 1 --batch_size 2 --input_size 224 --patch_size 8 \
    --encoder_embed_dim 512 --encoder_depth 8 --encoder_num_heads 8 \
    --decoder_embed_dim 512 --decoder_depth 6 --decoder_num_heads 8 \
    --fim_embed_dim 256 --fim_depth 2 --fim_num_heads 4 --quantizer_dim 512 \
    --quantizer_type rvq --rvq_stages 4 --rvq_bits_per_stage 6 --bits_vq_low 4 \
    --routing_topk 63 --no_augment --seed 42 \
    --lr 1e-4 --min_lr 1e-5 --warmup_epochs 20 --epochs 300 --opt adamw \
    --weight_decay 0.05 --clip_grad 1.0 --save_freq 10 --save_ckpt \
    --snr_db_train_min 0 --snr_db_train_max 20 --snr_db_eval 10 \
    --fim_loss_weight 0.5 --vq_loss_weight 0.15 --lpips_loss_weight 0 \
    --yolo_weights "" --yolo_conf_thres 0.4 --yolo_iou_thres 0.5 \
    --inside_box_loss_weight 10 --outside_box_loss_weight 0.1 \
    --num_workers 4
echo "J3 training complete"
