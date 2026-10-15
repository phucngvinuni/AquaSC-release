#!/bin/bash
# Regenerate the manuscript figures from the shipped result files (no checkpoint needed). Output: ../figs/ (see docs/RESULT_MAP.md).
# Any failure of the figures below is an error, except figure_f7 panel (b), whose input is not part of the release (docs/STATUS.md).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
for s in plot_f3_jointframe_ceiling plot_f5_rate_pareto plot_f6_headline; do
  echo "-- $s"; "$TORCH_PY" "plotting/$s.py"
done
echo "-- plot_f7_two_levers (panel b needs data that is not in the release)"
"$TORCH_PY" plotting/plot_f7_two_levers.py || echo "   expected: figure_f7 panel (b) cannot be regenerated (see docs/STATUS.md)"
"$TORCH_PY" plotting/plot_wave14_manuscript_figures.py --figs ablation,routing_pr,bitstream,train_eval
