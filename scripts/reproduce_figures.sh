#!/bin/bash
# Regenerate the manuscript figures from the shipped result files (no checkpoint needed). Output: ../figs/ (see docs/RESULT_MAP.md).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
for s in plot_f3_jointframe_ceiling plot_f5_rate_pareto plot_f6_headline plot_f7_two_levers; do
  echo "-- $s"; "$TORCH_PY" "$s.py" || echo "   (failed or input missing: see docs/RESULT_MAP.md for $s)"
done
"$TORCH_PY" plot_wave14_manuscript_figures.py --figs ablation,routing_pr,bitstream,train_eval
