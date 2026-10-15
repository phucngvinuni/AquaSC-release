#!/bin/bash
# J2 retry: preambles (Base, Full per-image AP) already completed successfully
# before the OOM. This relaunches only the BB-NoFIM_V2 training, with
# --rvq_kmeans_images reduced (800->150) since J3 (RVQ 4x6 training) is
# running concurrently and the k-means init memory spike collided with it
# (CUDA OOM: J3 held 6.62GiB, only 312MiB free when k-means tried to
# allocate 1.2GiB).
set -e
PY=__TORCH_PY__

echo "===== J2 main (retry): BB-NoFIM retrain from scratch, 300 epochs, num_workers=8, rvq_kmeans_images=150 ====="
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 $PY run_class_main.py \
    --model ViT_Reconstruction_Model_Default \
    --output_dir ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS_ABL_BB_NOFIM_V2 \
    --data_set fish --data_path "../yolo_fish_dataset_root/" \
    --num_object_classes 1 --batch_size 2 --input_size 224 --patch_size 8 \
    --encoder_embed_dim 512 --encoder_depth 8 --encoder_num_heads 8 \
    --decoder_embed_dim 512 --decoder_depth 6 --decoder_num_heads 8 \
    --fim_embed_dim 256 --fim_depth 2 --fim_num_heads 4 --quantizer_dim 512 \
    --quantizer_type rvq --rvq_stages 3 --rvq_bits_per_stage 6 --bits_vq_low 2 \
    --routing_topk 63 --no_augment --seed 42 --disable_fim \
    --lr 1e-4 --min_lr 1e-5 --warmup_epochs 20 --epochs 300 --opt adamw \
    --weight_decay 0.05 --clip_grad 1.0 --save_freq 10 --save_ckpt \
    --snr_db_train_min 0 --snr_db_train_max 20 --snr_db_eval 10 \
    --fim_loss_weight 0.5 --vq_loss_weight 0.15 --lpips_loss_weight 0 \
    --yolo_weights "" --yolo_conf_thres 0.4 --yolo_iou_thres 0.5 \
    --inside_box_loss_weight 10 --outside_box_loss_weight 0.1 \
    --num_workers 8 --rvq_kmeans_images 150

echo "===== J2 postamble: BB-NoFIM_V2 per-image AP @10/20 dB ====="
for snr in 10 20; do
    PYTHONDONTWRITEBYTECODE=1 $PY run_class_main.py \
        --model ViT_Reconstruction_Model_Default \
        --output_dir evaluation_output_abl_BB_NOFIM_V2 \
        --data_set fish --data_path "../yolo_fish_dataset_root/" \
        --num_object_classes 1 --yolo_weights best.pt \
        --batch_size 8 --input_size 224 --patch_size 8 \
        --encoder_embed_dim 512 --encoder_depth 8 --encoder_num_heads 8 \
        --decoder_embed_dim 512 --decoder_depth 6 --decoder_num_heads 8 \
        --fim_embed_dim 256 --fim_depth 2 --fim_num_heads 4 --quantizer_dim 512 \
        --quantizer_type rvq --rvq_stages 3 --rvq_bits_per_stage 6 --bits_vq_low 2 \
        --routing_topk 63 --num_workers 4 \
        --resume "ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS_ABL_BB_NOFIM_V2/checkpoint-best_psnr.pth" \
        --eval --snr_db_eval ${snr} \
        --yolo_conf_thres 0.4 --yolo_iou_thres 0.5 --map_conf_thres 0.001 \
        --csv_log_file "results/ablation/ablation_BB_NOFIM_V2_eval.csv" --append_csv \
        --per_image_ap_out "f1_perimage/ablation_BB_NOFIM_V2_snr${snr}.npy" \
        --disable_fim
done
echo "J2 complete"
