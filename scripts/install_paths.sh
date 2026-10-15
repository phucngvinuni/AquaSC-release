#!/bin/bash
# install_paths.sh -- bind this supplement to the paths of YOUR machine (run once, from anywhere).
#
#   bash install_paths.sh --torch-py /path/to/envs/siso-project-env/bin/python \
#                         --sionna-py /path/to/venv-sionna/bin/python \
#                         [--data-root DIR] [--tmp DIR] [--dry-run]
#
# The shipped code contains placeholder tokens instead of absolute paths:
#   __AQUASC_ROOT__  this directory                         (default: where this script lives)
#   __AQUASC_DATA__  directory holding the sibling dataset folders yolo_fish_dataset_root/, yolo_fishdet_nb2_vd/, VinFish/, figs/
#                    (default: the parent of this directory)
#   __TORCH_PY__     python of the torch environment (environment.yml, "siso-project-env")
#   __SIONNA_PY__    python of the separate Sionna/TensorFlow venv (see README.md section 2.1)
#   __AQUASC_TMP__   scratch directory for temporary files (default: ${TMPDIR:-/tmp}/aquasc)
# Only *.py *.sh *.yaml *.yml *.md *.txt are rewritten; result files (*.json, *.csv, *.npz) are never modified --
# paths that appear inside them are provenance records of the original run and keep their tokens.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$(cd "$ROOT/.." && pwd)"; TMP="${TMPDIR:-/tmp}/aquasc"; TORCH=""; SIONNA=""; DRY=0
while [ $# -gt 0 ]; do case "$1" in
  --torch-py) TORCH="$2"; shift 2;; --sionna-py) SIONNA="$2"; shift 2;;
  --data-root) DATA="$(cd "$2" && pwd)"; shift 2;; --tmp) TMP="$2"; shift 2;; --dry-run) DRY=1; shift;;
  -h|--help) sed -n 2,17p "${BASH_SOURCE[0]}"; exit 0;; *) echo "unknown argument $1" >&2; exit 2;; esac; done
[ -x "$TORCH" ]  || { echo "--torch-py must point to an executable python (got '$TORCH')" >&2; exit 2; }
[ -x "$SIONNA" ] || { echo "--sionna-py must point to an executable python (got '$SIONNA')" >&2; exit 2; }
esc() { printf '%s' "$1" | sed -e 's/[\\|&]/\\&/g'; }
mapfile -d '' FILES < <(grep -rIlZE '__(AQUASC_ROOT|AQUASC_DATA|AQUASC_TMP|TORCH_PY|SIONNA_PY)__' "$ROOT" \
  --include='*.py' --include='*.sh' --include='*.yaml' --include='*.yml' --include='*.md' --include='*.txt' \
  --exclude=install_paths.sh --exclude=SUPPLEMENT_README.md 2>/dev/null || true)
echo "files with placeholders: ${#FILES[@]}"
[ $DRY -eq 1 ] && { printf '%s\n' "${FILES[@]}" | head -20; exit 0; }
mkdir -p "$TMP"
for f in "${FILES[@]}"; do
  sed -i -e "s|__AQUASC_ROOT__|$(esc "$ROOT")|g" -e "s|__AQUASC_DATA__|$(esc "$DATA")|g" -e "s|__AQUASC_TMP__|$(esc "$TMP")|g" \
         -e "s|__TORCH_PY__|$(esc "$TORCH")|g" -e "s|__SIONNA_PY__|$(esc "$SIONNA")|g" "$f"
done
echo "bound: root=$ROOT data=$DATA tmp=$TMP torch=$TORCH sionna=$SIONNA"
echo "expected data layout under $DATA: yolo_fish_dataset_root/{train,valid,test}/{images,labels}  (see README.md section 5)"
