#!/bin/bash
# attach_checkpoints.sh SOURCE_DIR [--verify-only]
# Copies every checkpoint/weights file listed in checkpoints_sha256.txt that exists under SOURCE_DIR (same relative
# path) into this directory and verifies its SHA-256. Files not found are reported, never fabricated.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; SRC="${1:?usage: attach_checkpoints.sh SOURCE_DIR [--verify-only]}"; VERIFY="${2:-}"
ok=0; bad=0; missing=0
while read -r h rel; do
  [[ "$h" =~ ^[0-9a-f]{64}$ ]] || continue
  dst="$ROOT/$rel"
  if [ "$VERIFY" != "--verify-only" ] && [ ! -f "$dst" ] && [ -f "$SRC/$rel" ]; then mkdir -p "$(dirname "$dst")"; cp "$SRC/$rel" "$dst"; fi
  if [ -f "$dst" ]; then
    if [ "$(sha256sum "$dst" | cut -d' ' -f1)" = "$h" ]; then ok=$((ok+1)); else echo "HASH MISMATCH: $rel" >&2; bad=$((bad+1)); fi
  else missing=$((missing+1)); fi
done < "$ROOT/checkpoints_sha256.txt"
echo "verified=$ok mismatched=$bad not_present=$missing"
[ $bad -eq 0 ]
