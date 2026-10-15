# Figures in the revised AquaSC manuscript

## Run

The current quantitative figure generator needs only Python, NumPy, and Matplotlib.

```bash
python -m pip install -r requirements/plotting.txt
python plotting/plot_paper_figures.py --output-dir figs
# Equivalent shell entry point:
bash scripts/reproduce_figures.sh --output-dir figs
```

Run a subset with `--figures main rate`, or list choices with `--list`.
The Python entry point resolves inputs relative to its own repository and can be
called from any working directory. The default output is the repository's `figs/`.

Each selection writes a vector PDF and a 300-DPI PNG. A JSON output manifest
records file checksums and the resolved font. Times New Roman is used when
installed, with DejaVu Serif / Liberation Serif as portable serif fallbacks.
Both axis labels and panel captions use the same resolved font; subcaptions are
below panels, in regular weight. Different installed fonts can change spacing.

## Current figure map

| Figure | CLI name | Output stem | Evaluation represented |
|---|---|---|---|
| 3 | `main` | `figure_main_comparison` | Pooled mAP and PSNR at three channel-use rates; AquaSC coded evaluation and the stated baseline transports |
| 4 | `efficiency` | `figure_semantic_bandwidth` | Channel-use rates and raw pooled mAP per channel use at 10 dB |
| 5 | `ablation` | `figure_ablation_study` | Separately labeled uncoded-digital diagnostic; pooled mAP and mean per-image AP with bootstrap intervals |
| 6 | `rate` | `figure_rate_performance_snr_5db` | Rate/performance at 5 dB; fixed-budget WebP/JPEG rerun |
| 7 | `site-b` | `figure_domain_transfer` | Adapted models on Site B; 10-dB pooled mAP and retention over 0--20 dB |
| 8 | `lite` | `figure_aquasc_lite` | Small/Tiny accuracy and encoder-parameter comparison; complexity panel at 5 dB |

The introduction, architecture, testbed, and qualitative panels are authored
assets, supplied separately from these six quantitative plots.

## Inputs and scope

- The generator contains the frozen aggregate learned-model values and interval
  widths used in the paper. They are copied from the authors' current manuscript
  plotting source, preserving all plotted values, including the SwinJSCC and
  low-rate AquaSC series.
- `artifacts/figure_inputs/matched_digital_baselines_20261015.json` supplies the
  WebP/JPEG curves. These are the fixed 1536/3072/6144-symbol results with an
  in-band file length and shared receiver/channel settings. The protocol is
  included in that JSON. Baselines use 16-QAM and multiple LDPC blocks per image;
  AquaSC uses QPSK and one codeword per image.
- Figure 5 is the uncoded diagnostic; the coded ablation/routing replay is
  reported separately in Table VI. Its numbers should not be substituted for
  those in this diagnostic plot.
- This command redraws the submitted aggregate curves; it does not run model
  inference, train models, regenerate channel realizations, or recompute CIs.
  For numerical reruns, use `scripts/eval_coded.sh`, `experiments/matched5/`,
  `webp_jpeg_matched/`, and the statistics commands in `docs/RESULT_MAP.md`.
  Model/data reruns require the external assets described in `docs/CHECKPOINTS.md`
  and `docs/DATA.md`; those documents track access separately.
- Earlier experiment-specific plotting scripts remain in `plotting/` for their
  associated saved artifacts. Use this generator for the six current figures.

