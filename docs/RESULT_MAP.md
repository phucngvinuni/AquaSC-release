# Result map: where each reported result comes from and how to reproduce it

"Verified" means: executed in a fresh clone bound with `scripts/install_paths.sh` using only the shipped files.

## A. Manuscript figures (`../figs/` after running the script)

Figures used in the current manuscript draft: f3, f5, f6, f7, ablation_wave14, routing_pr_wave14, bitstream_jointframe, train_eval_pipeline_jointframe, qualitative_recon, main_comparison, aquasc_variants (the others are supplementary or retired).

| Figure file | Script | Inputs shipped here | Status |
|---|---|---|---|
| `figure_f3_jointframe_ceiling` | `plotting/plot_f3_jointframe_ceiling.py` | `experiments/matched5/runs/sweep_full_ci.json`, `f1_perimage/prior_floor_per_image.npy` | Verified; PNG pixel-identical to the manuscript file |
| `figure_f4_ablation` | `plotting/plot_f4_ablation.py` | `f1_perimage/*.npy` | Verified; identical |
| `figure_f5_rate_pareto` | `plotting/plot_f5_rate_pareto.py` | `experiments/matched5/runs/sweep_r030`, `experiments/matched5/runs/sweep_r061`, `experiments/matched5/runs/sweep_full_ci.json`, `f1_perimage/{jsccnn,roijscc,swinjscc}*` | Runs; **0.57% of pixels differ** from the manuscript file (one dotted SwinJSCC series): the manuscript file (2026-09-16 02:20) is older than the last edit of the script (02:37) — treat the regenerated figure as current |
| `figure_f6_headline` | `plotting/plot_f6_headline.py` | `experiments/matched5/runs/sweep_full_ci.json`, `f1_perimage/*` | Verified; identical |
| `figure_f7_two_levers` | `plotting/plot_f7_two_levers.py` | needs ten realizations from `drivers/run_a1_multiblock_bank_m10.sh` | **Not reproducible from the package**: the script refuses to draw panel (b) because those outputs (`evaluation_output_a1_m10/`) are absent in the source repository too |
| `figure_f8_failure_mechanism` | `plotting/plot_f8_failure_mechanism.py` | `experiments/matched5/runs/frame_loss_conditional_ap.json`, `experiments/matched5/runs/sweep` | Verified; identical |
| `figure_f9_siteb` | `plotting/plot_f9_siteb.py` | `experiments/matched5/runs/sweep_siteb` | Verified; identical |
| `figure_ablation_wave14`, `figure_routing_pr_wave14`, `figure_bitstream_jointframe`, `figure_train_eval_pipeline_jointframe`, `figure_crossover_thresholds_wave14`, `figure_flagship_coded_snr_wave14`, `figure_envelope_wave15` | `plotting/plot_wave14_manuscript_figures.py` (reads `f1_perimage/`, `results/routing/routing_pr_*result.npz`, `results/eval/evalfinal_flagship_coded_ldpc_snr_sweep.csv`; its retired `crossover` figure needs the internal ledger and is skipped here) | shipped | Verified; identical |
| `figure_qualitative_recon`, `figure_qualitative_recon_siteb` | `plotting/plot_qualitative_recon.py`, `plotting/plot_qualitative_recon_siteb.py` | need `receiver.npz` + flagship checkpoint | Not executed (needs checkpoint; see `CHECKPOINT_ACCESS.md`) |
| `figure_main_comparison`, `figure_aquasc_variants` | no generating script exists in the repository | — | Not reproducible: files date from the first submission |

## B. Numerical results

| Result | Script / command | Shipped outputs |
|---|---|---|
| Headline JointFrame SNR sweep (6144 symbols, QPSK), paired CIs vs ROI-JSCC | `bash scripts/eval_coded.sh high`; `experiments/matched5/` (`evaluate.py --phase cache`, `phy.py --transmit`, `evaluate.py --phase aquasc`; see its `README.md`) | `experiments/matched5/runs/sweep/aqua_*/{summary.json,metrics.npz}`, `experiments/matched5/runs/sweep_full_ci.json` |
| Retrained rate-point codecs r061 (0.061224 uses/px) and r030 (0.030612) | `bash scripts/eval_coded.sh mid` / `low`; training flags: `configs/generated/codec_r061_best_map.yaml`, `configs/generated/codec_r030_best_map.yaml` | `experiments/matched5/runs/jointframe_r061_bestmap`, `experiments/matched5/runs/jointframe_r030_bestmap` |
| Transport-truncated rate points | `phy.py --symbols` | `experiments/matched5/runs/sweep_r061`, `experiments/matched5/runs/sweep_r030` |
| FIM / bit-budget ablation and routing comparison through the coded pipeline | `experiments/matched5/run_ablation_routing_coded.sh`, `experiments/matched5/aggregate_ablation_routing_coded.py`, `experiments/matched5/paired_ci_ablation_routing_coded.py` | `experiments/matched5/runs/ablation_routing_coded` |
| Routing precision/recall incl. oracle, uniform, random controls | `analysis/routing_pr_topk.py`; `verification_2026-10/s3_s7/s7_routing_pr.py` | `verification_2026-10/s3_s7/s7_routing_pr.log` |
| Routing-cardinality (K) sweep | `experiments/matched5/run_k_sweep_valid.sh`, `experiments/matched5/run_k_sweep_test.sh` | `experiments/matched5/runs/k_sweep_valid`, `experiments/matched5/runs/k_sweep_test` |
| Site B zero-shot and fine-tuning (S-3, S-4) | `run_class_main.py` (flags: `configs/generated/siteB_*.yaml`), `experiments/matched5/run_s4_siteb_honest_select.sh`, `experiments/matched5/run_s4_siteb_full_snr_sweep.sh` | `experiments/matched5/runs/sweep_siteb`, `experiments/matched5/runs/s4_siteb_honest_select` |
| WebP/JPEG baselines at equal channel uses (fixed 1536/3072/6144 symbols, in-band length, same receiver and fading) | `baseline_rerun_20261015/run_stage1_all.sh`, `baseline_rerun_20261015/run_stage2_all.sh`, `baseline_rerun_20261015/aggregate.py` | `baseline_rerun_20261015/{RESULTS_matched_budget.md,runs/*/summary.json,runs/*/metrics.npz}` |
| Older WebP+LDPC points (variable block count, different receiver; history only) | `webp_real_channel_sweep_*_stage{1,2}.py`, `jpeg_real_channel_sweep_stage{1,2}.py` | `*_out/{manifest,results_by_snr}.json` |
| JSCC-CNN, ROI-JSCC, SwinJSCC baselines | `competitors/{jscc,ROI-JSCC,swinnnn}/` (their own training/eval scripts; bank-based evaluators `competitors/jscc/stats_eval_jscc_bank.py`, `competitors/ROI-JSCC/stats_eval_roijscc_bank.py`) | `f1_perimage/*.npy`, `competitors/jscc/output_jscc_baselinea/*/recon/*/*/score.json` (checkpoints not shipped) |
| Statistics protocol and recomputed CIs | `stats.py`, `analysis/stats_analyze.py`, `tests/test_stats.py`; `verification_2026-10/s3_s7/s3_recompute.py` | `verification_2026-10/s3_s7/s3_recompute.log` |
| Frame format (header/mask/CRC) round trip, three operating points | `verification_2026-10/s1/roundtrip_check.py` (needs `cache_test*/source.npz` from the cache phase) | `verification_2026-10/s1/{roundtrip.log,roundtrip_results.json}` |
| Sionna/LDPC configuration actually used | `verification_2026-10/s2/ldpc_config_dump.py` | `verification_2026-10/s2/{ldpc_config_dump.json,sionna_venv_freeze.txt}` |
| Transmitter parameter counts, memory, FLOPs | `verification_2026-10/s4/count_modules.py` | `verification_2026-10/s4/count_modules.log` |
| WebP byte/length/capacity accounting of the older runs | `verification_2026-10/s5/s5_counts.py`, `s5_trace.py` | `verification_2026-10/s5/*.json` |
| Edge-device (Raspberry Pi 4B) latency | `experiments/edge_bench/` (needs the device); the Pi network timing was measured interactively and is **not** scripted (see `docs/TIMING.md`) | no raw logs shipped |
| Distillation of the lightweight transmitter | `training/train_distill_transmitter_only.py`; stage-1 flags were not stored in the checkpoints (see `configs/README.md`) | — |

## C. Configurations and seeds
`configs/*.yaml` hold flags, seed and checkpoint SHA-256 for the legacy-training configurations; `configs/generated/*.yaml` hold the flags recovered from the checkpoints of every other model in the paper (`configs/README.md`). Channel draws are fixed by the seeded bank (`channel_bank.py`; master seed 42). The matched-budget rerun fixes quality on the validation split only (`baseline_rerun_20261015/valid_calib/`).
