#!/bin/bash
# Clone the third-party baselines at the commits used for the paper and overlay the files we added/modified (baselines/overlay/).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
mkdir -p competitors
while read -r name url commit; do
  [[ "$name" =~ ^# ]] || [ -z "$name" ] && continue
  [ -d "competitors/$name/.git" ] || git clone "$url" "competitors/$name"
  git -C "competitors/$name" checkout -q "$commit"
  cp -r "baselines/overlay/$name/." "competitors/$name/"
  echo "ok: $name @ $commit (+ overlay)"
done < baselines/BASELINES.txt
