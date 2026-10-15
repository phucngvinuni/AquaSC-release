#!/bin/bash
# run_s4_siteb_honest_select.sh -- 2026-09-24.
#
# Fixes the audit-flagged issue (RESULTS_OF_RECORD.md item 5 / S-4 section):
# the S-4 "epoch 4" checkpoint was picked by comparing ep4/ep9/ep14 directly
# on the Site B TEST split (test-set-informed selection), through the OLD
# uncoded-digital/analog-proxy single-draw protocol, not the real coded
# joint-frame pipeline.
#
# This script:
#   1. Evaluates all 3 S-4 checkpoints (ep4/ep9/ep14) on the Site B VALID
#      split (487 images, video-disjoint from test per manifest.json) through
#      the REAL LDPC-coded joint-frame pipeline (experiments/matched5/), at
#      the headline/highest rate (symbols=6144, default flagship RVQ/routing
#      config -- 0.122449 uses/px, same as the Site A headline).
#   2. Whichever epoch wins on VALID is then evaluated on Site B TEST (455
#      images) at the same rate -- the number that should replace the old
#      test-selected "21.12%/14.21%" citation.
#
# SNR=10dB (project's standard "typical operating point", matches the
# original S-4 comparison point in RESULTS_OF_RECORD.md so the honest number
# is directly comparable to the flagged one). M=10 realizations (project
# standard).
set -e
cd "$(dirname "$0")"
ROOT=__AQUASC_ROOT__
SIONNA=__SIONNA_PY__
TORCH=__TORCH_PY__
RUNS=$ROOT/experiments/matched5/runs/s4_siteb_honest_select
mkdir -p "$RUNS/logs"

SNR=10
M=10

run_one () {
    name="$1"; ckpt="$2"; split="$3"

    cache_dir="$RUNS/cache_${name}_${split}"
    if [ ! -f "$cache_dir/source.npz" ]; then
        echo "=== [$name/$split] cache ==="
        rm -rf "$cache_dir"
        $TORCH evaluate.py --phase cache --split "$split" --site b \
            --ckpt "$ckpt" \
            --out-dir "$cache_dir" \
            > "$RUNS/logs/cache_${name}_${split}.log" 2>&1
    else
        echo "=== [$name/$split] cache already exists, skipping ==="
    fi

    phy_dir="$RUNS/phy_${name}_${split}_snr${SNR}"
    aqua_dir="$RUNS/aqua_${name}_${split}_snr${SNR}"
    if [ ! -f "$aqua_dir/summary.json" ]; then
        echo "=== [$name/$split] SNR=${SNR}dB transmit ==="
        rm -rf "$phy_dir"
        taskset -c 0-7,10-31 $SIONNA phy.py --transmit \
            --source "$cache_dir/source.npz" \
            --split "$split" --variant frame_qpsk --m $M --symbols 6144 --snr-db $SNR \
            --out-dir "$phy_dir" \
            > "$RUNS/logs/phy_${name}_${split}_snr${SNR}.log" 2>&1

        echo "=== [$name/$split] SNR=${SNR}dB evaluate ==="
        rm -rf "$aqua_dir"
        $TORCH evaluate.py --phase aquasc --split "$split" \
            --receiver "$phy_dir/receiver.npz" \
            --out-dir "$aqua_dir" \
            > "$RUNS/logs/aqua_${name}_${split}_snr${SNR}.log" 2>&1
    else
        echo "=== [$name/$split] SNR=${SNR}dB already done, skipping ==="
    fi
    echo "--- [$name/$split] summary ---"
    cat "$aqua_dir/summary.json"
    echo
}

# Step 1: select on VALID (honest, never-peeked split)
run_one "s4ep4"  "$ROOT/ckpt_siteb_s4_ema0p9/checkpoint-4.pth"  "valid"
run_one "s4ep9"  "$ROOT/ckpt_siteb_s4_ema0p9/checkpoint-9.pth"  "valid"
run_one "s4ep14" "$ROOT/ckpt_siteb_s4_ema0p9/checkpoint-14.pth" "valid"

echo "===== VALID-split selection done. Inspect the 3 summaries above, pick the winner, ====="
echo "===== then run: run_one <winner_name> <winner_ckpt> test  ====="
