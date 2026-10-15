# AquaSC

Reference implementation of **AquaSC: Task-Oriented Semantic Communication for Efficient and Intelligent Aquaculture IoT Systems**.
A learned Feature Importance Module (FIM) routes the patches that matter for fish detection to a high-fidelity residual vector quantiser and spends few bits on the background;
the resulting indices are serialised into a CRC-32-protected frame, protected by a 5G-NR LDPC code (BG2), and sent over a Rayleigh channel. Everything reported in the paper
(coded evaluation at three rates, ablations, routing controls, K sensitivity, Small/Tiny students, Site B adaptation, equal-channel-use WebP/JPEG, baselines, timing) can be traced from this repository.

This repository is a derivative of [RobustSemanComm](https://github.com/hqyyqh888/RobustSemanComm) (see `NOTICE.md`).

## 1. Install
Two interpreters are needed (details and reason in `requirements/README.md`):
```bash
git clone https://github.com/phucngvinuni/AquaSC-release.git && cd AquaSC-release
conda env create -f requirements/torch-env.yml                                   # torch 2.1.2 + CUDA 12.1  (env: siso-project-env)
python -m venv venv-sionna && venv-sionna/bin/pip install -r requirements/sionna-venv.txt   # Sionna 1.2.2 / TensorFlow 2.21.0, CPU is enough
export TORCH_PY=$(conda run -n siso-project-env which python)  SIONNA_PY=$PWD/venv-sionna/bin/python
bash scripts/install_paths.sh --torch-py "$TORCH_PY" --sionna-py "$SIONNA_PY" [--data-root DIR]   # once: binds path placeholders to this machine
```
`install_paths.sh` rewrites the `__AQUASC_*__` / `__TORCH_PY__` / `__SIONNA_PY__` placeholders in the text files of your working copy (scripts, configs); do not commit its output.

## 2. Demo (no checkpoint, no dataset, no GPU)
```bash
bash scripts/demo.sh
```
Runs the four unit tests (statistics protocol, routing controls, bitstream round trip with header/mask/CRC-32, LDPC/QAM chain), recomputes the paired confidence intervals from the shipped per-image arrays, and regenerates the manuscript figures from the shipped result files into `../figs/`.

## 3. Data and checkpoints
* Checkpoints and detector: `docs/CHECKPOINTS.md` (download link, SHA-256 manifest `artifacts/checkpoints_manifest.json`, `bash scripts/attach_checkpoints.sh DOWNLOAD_DIR` verifies every hash).
* Dataset (VinFish Site A, FishDet-NB2 Site B), annotation format, preprocessing and split lists: `docs/DATA.md`. Expected roots: `<DATA_ROOT>/yolo_fish_dataset_root/{train,valid,test}/{images,labels}` and `yolo_fishdet_nb2_vd/...`.

## 4. Full coded evaluation (one command per rate)
```bash
bash scripts/eval_coded.sh high      # 0.122449 uses/px, 6144 symbols; also: mid (0.061224, 3072), low (0.030612, 1536)
```
Site A test split, 432 images x 10 channel draws x SNR {-5, 0, 5, 10, 15, 20} dB, real LDPC/QPSK/Rayleigh chain with perfect receiver CSI (APP demapper, 40-iteration sum-product decoding, one fading coefficient per image).
A frame whose CRC fails is replaced by the all-zero-latent fallback and **stays in the scoring** (no failed image is dropped). Outputs per SNR: `summary.json` (pooled mAP@0.5, mean per-image AP50, frame-loss rate, hashes) and `metrics.npz` (per-image x per-draw AP).
The training proxy (analog Rayleigh channel with power normalisation, `channel.py`, `--train_channel_mode analog`) is only used during training and fast ablations; **all numbers in the paper come from the coded pipeline** (`transport` files below, `experiments/matched5/`).

## 5. What reproduces which figure / table
Full table with inputs and verification status: `docs/RESULT_MAP.md`. Configs and commands per experiment: `configs/README.md`.

| Paper item | Config / checkpoint | Command |
|---|---|---|
| Headline SNR sweep and CIs (high rate) | `configs/final_18_2.yaml`; `flagship_r122_best_psnr` | `bash scripts/eval_coded.sh high` ; figure: `plotting/plot_f6_headline.py`, `plotting/plot_f3_jointframe_ceiling.py` |
| Three rates (rate-Pareto) | `configs/generated/codec_r061_best_map.yaml`, `codec_r030_best_map.yaml` | `bash scripts/eval_coded.sh mid` / `low`; figure: `plotting/plot_f5_rate_pareto.py` |
| Ablations and routing controls | `configs/generated/ablation_*.yaml` | `bash experiments/matched5/run_ablation_routing_coded.sh`; figure: `plotting/plot_wave14_manuscript_figures.py --figs ablation,routing_pr` |
| K sensitivity | headline checkpoint | `experiments/matched5/run_k_sweep_{valid,test}.sh` |
| Small / Tiny | `configs/generated/{small,tiny}_*.yaml` | `experiments/matched5/run_stage1_vs_stage2_coded.sh` |
| Site B adaptation | `configs/generated/siteB_*.yaml` | `experiments/matched5/run_s4_siteb_*.sh`; figure: `plotting/plot_f9_siteb.py` |
| WebP / JPEG at equal channel uses | `webp_jpeg_matched/runs/*/config.json` | `bash scripts/run_webp_jpeg_matched.sh` (results: `webp_jpeg_matched/RESULTS_matched_budget.md`) |
| SwinJSCC, JSCC-CNN, ROI-JSCC | `baselines/README.md` | `bash scripts/setup_baselines.sh` |
| Frame format / CRC / routing cardinality / transmitted symbols | — | `python tests/test_bitstream.py`, `verification/s1/roundtrip_check.py`; `summary.json` records `symbols_per_image` / `info_bits` / `coded_bits` for every run; the WebP/JPEG rerun asserts the emitted symbol count for every record |
| Latency on Raspberry Pi 4B, network, server | `artifacts/results/timing/` | `docs/TIMING.md`, `experiments/edge_bench/` |
| Paired bootstrap / pooled mAP / mean per-image AP / mask-seed variability | — | `stats.py`, `analysis/stats_analyze.py`; `python verification/s3_s7/s3_recompute.py` |

## 6. Statistics protocol
Per-image AP50 is averaged over the 10 channel draws; differences between methods use a **paired bootstrap over the 432 test images** (10,000 resamples, seed 42, percentile 95 % interval, `stats.paired_bootstrap`). Pooled mAP@0.5 is computed per draw over all images and reported as mean +- std over the 10 draws.
The random-router control is repeated over mask seeds (`experiments/matched5/run_random_router_multiseed.sh`). Intervals are conditional on the released checkpoints and the test split: they do not include training-seed or video-level variability (one training seed per model).

## 7. Repository layout and code map
The root keeps the core library modules (they import each other by name); everything else lives in a folder.
```
model.py model_util.py engine1.py run_class_main.py base_args.py ...   core model / training / transport modules (table below)
analysis/      routing, transfer, codec and statistics analysis scripts
plotting/      figure scripts (plot_*.py, plotting/reproduce_figures.py)
tests/         unit tests (statistics, routing controls, bitstream, LDPC chain)
training/      detector, distillation and FIM-shrink training scripts
data_tools/    split building, channel-bank generation, leakage audit
drivers/       shell drivers of the older (uncoded / proxy) sweeps, kept as provenance of the CSV files in results/
scripts/       one-command entry points: demo, coded evaluation, figures, baselines, path binding
experiments/   matched5 (coded evaluation pipeline), edge_bench (Raspberry Pi and network timing)
webp_jpeg_matched/  equal-channel-use WebP/JPEG baselines     baselines/  third-party baselines (overlay + setup)
configs/ requirements/ docs/ artifacts/ verification/ splits/    configs, environments, documentation, manifests, audits, split manifests
results/       CSV / JSON / NPZ result files read by the plotting and analysis scripts (grouped by topic);  f1_perimage/  per-image AP arrays
```
Run every script from the repository root, e.g. `python plotting/plot_f6_headline.py`.

| Area | Files |
|---|---|
| Model: ViT encoder/decoder, FIM, projection, residual/hierarchical VQ, channel proxy | `model.py`, `model_util.py` (`ViTEncoder_Van`, `FeatureImportanceTransformer`, `ResidualQuantizer`, `HierarchicalQuantizer`, `ViTDecoder_ImageReconstruction`, `Channels`) |
| Losses and train/eval loops (reconstruction with bounding-box weighting, LPIPS, VQ codebook/commitment, FIM) | `engine1.py`, `run_class_main.py`, `base_args.py` |
| Training: AquaSC, ablations (`--disable_fim`, `--disable_bbw`), distillation (Small/Tiny), FIM shrink, detector | `run_class_main.py`, `training/train_distill_transmitter_only.py`, `training/train_fim_shrink.py`, `training/train_detector.py`, `configs/` |
| Datasets / preprocessing / splits | `datasets.py`, `dataset_folder.py`, `data_tools/splits.py`, `splits/`, `data_tools/build_combined_root.py` |
| Transport: serializer/parser, header, routing mask, CRC, LDPC, modulation, channel, demapper, decoder | `bitstream.py`, `digital_pipeline.py`, `digital_bridge.py`, `digital_client.py`, `digital_channel.py`, `channel.py`, `channel_bank.py`, `data_tools/gen_channel_bank.py`, `experiments/matched5/phy.py` |
| Evaluation (coded) | `experiments/matched5/{evaluate,protocol,phy}.py`, `metrics.py`, `fim_metrics.py`, `analysis/routing_eval.py`, `routing_baselines.py`, `analysis/routing_pr_topk.py` |
| Baselines | `baselines/`, `webp_jpeg_matched/`, `analysis/codec_baseline.py`, `analysis/codec_jpeg_eval.py` |
| Statistics | `stats.py`, `analysis/stats_analyze.py`, `analysis/stats_eval_bank.py`, `analysis/stats_eval_jscc.py` |
| Plotting | `plotting/` (`plot_*.py`, `plotting/plot_style.py`, `plotting/reproduce_figures.py`) |
| Tests / checks | `tests/`, `verification/` |

Frame-loss handling: a frame with a failed CRC-32 (header + mask + indices) produces an all-zero latent for the decoder; per-stage index partitioning, the 152-bit header and the enumerative mask encoding are defined in `bitstream.py`.

## 8. Runtime and hardware (reference)
Desktop with Intel Core i9-13900K, 31 GB RAM, NVIDIA RTX 4070 (12 GB): about 1.5-2 min for LDPC/QPSK transmission and 0.7-1.7 min for detector scoring per SNR point and rate (432 images x 10 draws), so one rate over six SNRs takes about 15-20 minutes. Unit tests and figure regeneration run on CPU in minutes. Edge timing hardware (Raspberry Pi 4B) and what the timing does and does not cover: `docs/TIMING.md`. These are single measurements, not a benchmark. Smoke test of this exact command in a fresh clone (`scripts/eval_coded.sh low OUT 10 1`: one SNR, one draw): 45 s including the encoder cache.

## 9. Status, limitations, licence
`docs/STATUS.md` lists what was verified and what was not (figures that cannot be regenerated, unrecoverable stage-1 distillation flags, unscripted Pi network timing, one training seed). Code is MIT (`LICENSE`); third-party material and the licence status of inherited code: `NOTICE.md`. Citation: `CITATION.cff`.
