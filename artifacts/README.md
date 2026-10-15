# artifacts/

| Path | Content |
|---|---|
| `checkpoints_manifest.json` | name, SHA-256, size, role, config and stored results of every checkpoint/detector used in the paper |
| `splits/` | per-file split lists of Site A and B and the Site B manifest (Site A manifest: `../splits/manifest.json`) |
| `results/timing/` | edge/network/server timing CSVs and the transcribed split-computing table (see `../docs/TIMING.md`) |

## Where the result files live
Scripts read their inputs by relative path, so the numerical results stay where the scripts expect them (they are tracked in this repository):
* per-image AP / detections arrays: `f1_perimage/*.npy`, `experiments/matched5/runs/**/metrics.npz`, `*perimage_ap.npz`, `routing_pr_*result.npz`, `webp_jpeg_matched/runs/*/metrics.npz`
* summaries: `experiments/matched5/runs/**/summary.json`, `experiments/matched5/runs/sweep_full_ci.json`, `webp_jpeg_matched/RESULTS_matched_budget.{md,json}`, `evalfinal_*.csv`, `ablation_*.csv`
* per-figure input map: `docs/RESULT_MAP.md`

## Channel bank
`channel_bank.npz` (Site A: `h` [432,6,10] complex64, `n` [432,6,10,12544] float32, ~1.3 GB) and `channel_bank_siteb.npz` are deterministic functions of master seed 42 and are not stored in git.
Regenerate Site A with `python gen_channel_bank.py` (CPU, a few minutes): the regenerated bank was verified bit-identical to the one used for the paper. Site B regeneration (`channel_bank_siteb.py`) was not independently verified.
Image order is the test-split order of the manifest; SNR grid {-5, 0, 5, 10, 15, 20} dB; 10 draws per (image, SNR); one Rayleigh coefficient per image and draw.
