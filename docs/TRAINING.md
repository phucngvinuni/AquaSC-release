# Training the headline model


```bash
CUDA_VISIBLE_DEVICES=0 python3 run_class_main.py \
  --model ViT_Reconstruction_Model_Default \
  --output_dir ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS \
  --data_set fish --data_path ../yolo_fish_dataset_root/ \
  --num_object_classes 1 --yolo_weights best.pt \
  --batch_size 2 --input_size 224 --patch_size 8 \
  --encoder_embed_dim 512 --encoder_depth 8 --encoder_num_heads 8 \
  --decoder_embed_dim 512 --decoder_depth 6 --decoder_num_heads 8 \
  --fim_embed_dim 256 --fim_depth 2 --fim_num_heads 4 \
  --quantizer_dim 512 --quantizer_type rvq --rvq_stages 3 --rvq_bits_per_stage 6 \
  --bits_vq_low 2 --routing_topk 63 --no_augment --seed 42 \
  --lr 1e-4 --min_lr 1e-5 --warmup_epochs 20 --epochs 300 --opt adamw \
  --weight_decay 0.05 --clip_grad 1.0 --save_freq 10 --save_ckpt \
  --snr_db_train_min 0 --snr_db_train_max 20 --snr_db_eval 10
```

Notes:
- `--quantizer_type rvq` + `--rvq_stages 3 --rvq_bits_per_stage 6` gives an
  18-bit effective high path (`--bits_vq_high` is ignored for RVQ).
- `--no_augment` disables the Wave-4 training augmentation (the paper's final
  models train without it; augmentation collapsed detection — see
  `reports/wave6_sweep.md`, `wave7_audit_aug.md`).
- `--routing_topk 63` is the fixed-cardinality router (K = ⌈0.08·784⌉).
- `--yolo_weights ""` skips in-training mAP validation (used for the long runs
  to avoid VRAM spikes); post-hoc mAP comes from `--eval`.
- **Training reproducibility depends on `--batch_size` when `--train_channel_mode
  digital_uncoded` is used** (Track E1): the per-step channel seed in `model.py`
  (`args.seed + epoch*100003 + data_iter_step`) is derived from the training
  step counter, not from a batch-size-invariant global sample index (unlike the
  eval-time seed in `engine1.py`, fixed in plan v7.4 after a real defect was
  found in `transfer_eval.py` — see the internal audit notes of the revision). This
  does not bias any *reported* number (it only governs stochastic training-time
  channel augmentation), but it does mean **training with a different
  `--batch_size` than the released config gives each training example a
  different distribution of channel realizations across epochs, and is not
  guaranteed to reproduce the identical trained weights** — match `--batch_size`
  to the `configs/*.yaml` command exactly if bit-for-bit weight reproduction is
  the goal, not just similar final metrics. Not applicable to the default
  `analog`/`erasure` `--train_channel_mode`s, which do not derive their noise
  from this counter the same way.


The flags of every released checkpoint: `configs/README.md`.
