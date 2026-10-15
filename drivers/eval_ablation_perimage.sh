#!/bin/bash
# Plan v5.7 E-2 eval (Part C of wave9b_ablation.md, extended with per-image AP
# for CIs per R2 #2). Same protocol as results/ablation/ablation_BASE_eval.csv: checkpoint-best_psnr,
# SNR {10,20}, best.pt, conf 0.001, ablation flags re-passed.
set -e
run_eval() {
    local cfg=$1 flags=$2 snr=$3
    echo "===== ${cfg} eval SNR=${snr} ====="
    PYTHONDONTWRITEBYTECODE=1 __TORCH_PY__ run_class_main.py \
        --model ViT_Reconstruction_Model_Default \
        --output_dir evaluation_output_abl_${cfg} \
        --data_set fish --data_path "../yolo_fish_dataset_root/" \
        --num_object_classes 1 --yolo_weights best.pt \
        --batch_size 8 --input_size 224 --patch_size 8 \
        --encoder_embed_dim 512 --encoder_depth 8 --encoder_num_heads 8 \
        --decoder_embed_dim 512 --decoder_depth 6 --decoder_num_heads 8 \
        --fim_embed_dim 256 --fim_depth 2 --fim_num_heads 4 --quantizer_dim 512 \
        --quantizer_type rvq --rvq_stages 3 --rvq_bits_per_stage 6 --bits_vq_low 2 \
        --routing_topk 63 --num_workers 4 \
        --resume "ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS_ABL_${cfg}/checkpoint-best_psnr.pth" \
        --eval --snr_db_eval ${snr} \
        --yolo_conf_thres 0.4 --yolo_iou_thres 0.5 --map_conf_thres 0.001 \
        --csv_log_file "ablation_${cfg}_eval.csv" --append_csv \
        --per_image_ap_out "f1_perimage/ablation_${cfg}_snr${snr}.npy" \
        ${flags}
}
run_eval FIM_NOBB "--disable_bbw" 10
run_eval FIM_NOBB "--disable_bbw" 20
run_eval BB_NOFIM "--disable_fim" 10
run_eval BB_NOFIM "--disable_fim" 20
echo "ablation eval complete"
