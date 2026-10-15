#!/bin/bash
# Draw Figs. 3--8 of the current revised manuscript from frozen aggregate inputs.
# Use PLOTTING_PY or TORCH_PY to select an interpreter with requirements/plotting.txt.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PLOTTING_PY:-${TORCH_PY:-python}}" "$REPO_ROOT/plotting/plot_paper_figures.py" "$@"
