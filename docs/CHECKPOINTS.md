# Checkpoints and detector

Weights are hosted outside the git repository. `artifacts/checkpoints_manifest.json` lists every checkpoint used in the paper
(file, SHA-256, size, role, config, stored results); `checkpoints_sha256.txt` lists the SHA-256 of all released files.

**Download:** `[to be filled in by the authors]`

```bash
# place the downloaded tree anywhere; this copies the files to their expected relative paths and verifies every SHA-256
bash scripts/attach_checkpoints.sh /path/to/downloaded/tree
bash scripts/attach_checkpoints.sh /path/to/downloaded/tree --verify-only
```
Expected locations are relative to the repository root, e.g. `ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS_FINAL/checkpoint-best_psnr.pth` and `best.pt`.

## Minimum set per reported result
| Result | Checkpoint(s) |
|---|---|
| Headline 0.122449 uses/px (high rate) | `ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS_FINAL/checkpoint-best_psnr.pth` (SHA-256 `b1d2eb25...2894`) + `best.pt` |
| 0.061224 uses/px (mid) | `ckpt_codec_r061_topk32_3x4_low1/checkpoint-best_map.pth` |
| 0.030612 uses/px (low) | `ckpt_codec_r030_topk24_3x6_low0/checkpoint-best_map.pth` |
| Ablations (Base, BB-NoFIM, FIM-NoBB; Full = headline) | `ckpt_..._ABL_BASE`, `..._ABL_BB_NOFIM_V2`, `..._ABL_FIM_NOBB` (`checkpoint-best_psnr.pth`) |
| Small / Tiny (two-stage distillation) | `ckpt_distill_tx_only_v2_512d2L/checkpoint-69.pth` -> `ckpt_polish_512d2L_from_tx_distill`; `ckpt_distill_tx_only_192d2L/checkpoint-149.pth` -> `ckpt_polish_192d2L_from_tx_distill` |
| Site B adaptation (S-3, S-4) | `ckpt_siteb_s3_full_finetune/checkpoint-29.pth`, `ckpt_siteb_s4_ema0p9/checkpoint-{4,9,14}.pth`, `best_siteB.pt` |
| Detector (Site A scoring detector) | `best.pt` (also `best.onnx` for the edge benchmark) |

`configs/generated/*.yaml` hold the training flags recovered from each checkpoint (see `configs/README.md` for what is and is not recoverable).
The detector files are Ultralytics checkpoints and may carry the training arguments (including original file paths) that Ultralytics pickles into them.
