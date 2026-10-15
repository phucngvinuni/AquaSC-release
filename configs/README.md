# Configurations and experiment commands

Training flags live in two places:
* `configs/*.yaml` (hand-written, with seed, SHA-256 and the exact training command): `final_18_2.yaml` (headline), `final_18_4.yaml`, `final_op16_256d2.yaml`, `legacy_*.yaml`.
* `configs/generated/<name>.yaml`: flags **recovered from the `args` stored inside each released checkpoint** (only values that differ from the argparse defaults) and a runnable `command`. They are the flags of the *last* training session that wrote the checkpoint, so resumed runs also show `resume` / `start_epoch` (the earlier session used the same flags unless noted). Stage-1 distillation checkpoints (`training/train_distill_transmitter_only.py`) do **not** store their arguments: their YAML says `flags: not-stored`.
Paths in the YAML files use the placeholder tokens that `scripts/install_paths.sh` replaces.

## Experiment index
| Experiment | Config / checkpoint | Run |
|---|---|---|
| AquaSC high rate, 0.122449 uses/px (6144 symbols) | `final_18_2.yaml`, `generated/flagship_r122_best_psnr.yaml` | `bash scripts/eval_coded.sh high` |
| AquaSC mid rate, 0.061224 uses/px (3072 symbols) | `generated/codec_r061_best_map.yaml` | `bash scripts/eval_coded.sh mid` |
| AquaSC low rate, 0.030612 uses/px (1536 symbols) | `generated/codec_r030_best_map.yaml` | `bash scripts/eval_coded.sh low` |
| Ablations: Full (= headline), Base, BB-NoFIM, FIM-NoBB | `generated/ablation_*.yaml` | `bash experiments/matched5/run_ablation_routing_coded.sh` (SNR 10/20 dB, M=10), then `aggregate_ablation_routing_coded.py`, `paired_ci_ablation_routing_coded.py` |
| Routing controls through the coded pipeline: `oracle_box`, `random`, `uniform`, `detector_heatmap` (+ natural FIM) | headline checkpoint | same script (`--router ...`); precision/recall incl. `oracle_binary` ground truth: `analysis/routing_pr_topk.py`, `verification/s3_s7/s7_routing_pr.py` |
| K sensitivity (K = 40/48/56/63/70) | headline checkpoint | `experiments/matched5/run_k_sweep_valid.sh`, `experiments/matched5/run_k_sweep_test.sh` |
| Small (512d2L) / Tiny (192d2L): distillation + polish | `generated/{small,tiny}_stage{1,2}_*.yaml` | training: `training/train_distill_transmitter_only.py` (stage 1), `run_class_main.py` with the stage-2 YAML flags; coded evaluation: `experiments/matched5/run_stage1_vs_stage2_coded.sh` |
| Site B zero-shot and adaptation S-3 / S-4 | `generated/siteB_*.yaml` | `experiments/matched5/run_s4_siteb_honest_select.sh`, `run_s4_siteb_full_snr_sweep.sh` |
| WebP / JPEG at equal channel uses | `webp_jpeg_matched/runs/*/config.json` | `bash scripts/run_webp_jpeg_matched.sh` |
| SwinJSCC, JSCC-CNN, ROI-JSCC | `baselines/README.md` | `bash scripts/setup_baselines.sh`, then each baseline's evaluator |
| Frame format (header/mask/CRC) round trip, 3 operating points | — | `verification/s1/roundtrip_check.py` |
| Statistics (pooled mAP, mean per-image AP, paired bootstrap, mask-seed variability) | — | `stats.py`, `analysis/stats_analyze.py`, `tests/test_stats.py`; recomputation: `verification/s3_s7/s3_recompute.py` |
