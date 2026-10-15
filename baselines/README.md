# Baselines

| Method (paper) | Directory after `scripts/setup_baselines.sh` | Upstream @ commit | What we add (`overlay/`) |
|---|---|---|---|
| ROI-JSCC (rcpp32/64/128) | `competitors/ROI-JSCC` | hansung-choi/ROI-JSCC @ 6f947d1 (MIT) | `aqua_adapter.py`, `roijscc_fish_common.py`, `eval_roijscc_fish.py`, `train_roijscc_fish.py`, bank evaluators `stats_eval_roijscc_{bank,siteb}.py` |
| JSCC-CNN (Std / BBW) | `competitors/jscc` | fork of the JSCC-CNN baseline @ ed721bd | training/eval drivers (`baselinea_*.py|sh`), `stats_eval_jscc_bank.py` |
| SwinJSCC | `competitors/swinnnn` | fork of SwinJSCC (arXiv 2308.09361) @ 422d094 | `eval_swinjscc_bank*.py`, fine-tune and benchmark drivers |
| WebP / JPEG at equal channel uses | `webp_jpeg_matched/` (in this repo) | — | full protocol, code and stored results; `bash scripts/run_webp_jpeg_matched.sh` |

All learned baselines are scored with the same detector, the same 432 test images, and the same seeded channel bank as AquaSC
(`channel_bank.py`, master seed 42), so draws are paired. The bank-based evaluators produce the per-image AP arrays that the figure
and CI scripts consume (`f1_perimage/*.npy`, shipped). Baseline checkpoints are not shipped (`docs/CHECKPOINTS.md`); training them
follows each upstream README with the fish dataset (`docs/DATA.md`).
SD-JSCC is not part of the comparison and is not included in this release.
