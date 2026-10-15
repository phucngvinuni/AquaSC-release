#!/bin/bash
# run_stage1_vs_stage2_coded.sh -- 2026-09-23. Is the "polish" (stage 2)
# distillation fine-tune actually necessary, or is stage-1 (tx-only
# distillation) alone good enough? Neither stage has ever been evaluated
# through the REAL LDPC-coded joint-frame pipeline before (only the old
# uncoded/analog-proxy eval, per the v136/v138/v147/v148 logs) -- this
# answers it with the same real coded chain as the headline.
#
# 4 checkpoints: Tiny stage1/stage2, Small stage1/stage2. SNR 10/20dB, M=10.
set -e
cd "$(dirname "$0")"
ROOT=__AQUASC_ROOT__
SIONNA=__SIONNA_PY__
TORCH=__TORCH_PY__
RUNS=__AQUASC_ROOT__/experiments/matched5/runs/stage1_vs_stage2_coded
mkdir -p "$RUNS" "$RUNS/logs"

SNRS="10 20"
M=10

run_condition () {
    name="$1"; ckpt="$2"; dim="$3"; depth="$4"

    cache_dir="$RUNS/cache_${name}"
    if [ ! -f "$cache_dir/source.npz" ]; then
        echo "=== [$name] cache ==="
        rm -rf "$cache_dir"
        $TORCH evaluate.py --phase cache --split test \
            --out-dir "$cache_dir" --ckpt "$ckpt" \
            --encoder-embed-dim "$dim" --encoder-depth "$depth" --encoder-num-heads 8 \
            > "$RUNS/logs/cache_${name}.log" 2>&1
    else
        echo "=== [$name] cache already exists, skipping ==="
    fi

    for snr in $SNRS; do
        phy_dir="$RUNS/phy_${name}_snr${snr}"
        aqua_dir="$RUNS/aqua_${name}_snr${snr}"
        if [ ! -f "$aqua_dir/summary.json" ]; then
            echo "=== [$name] SNR=${snr}dB transmit ==="
            rm -rf "$phy_dir"
            taskset -c 0-7,10-31 $SIONNA phy.py --transmit \
                --source "$cache_dir/source.npz" \
                --split test --variant frame_qpsk --m $M --symbols 6144 --snr-db $snr \
                --out-dir "$phy_dir" \
                > "$RUNS/logs/phy_${name}_snr${snr}.log" 2>&1

            echo "=== [$name] SNR=${snr}dB evaluate ==="
            rm -rf "$aqua_dir"
            $TORCH evaluate.py --phase aquasc --split test \
                --receiver "$phy_dir/receiver.npz" \
                --out-dir "$aqua_dir" \
                > "$RUNS/logs/aqua_${name}_snr${snr}.log" 2>&1
            tail -1 "$RUNS/logs/aqua_${name}_snr${snr}.log"
        else
            echo "=== [$name] SNR=${snr}dB already done, skipping ==="
        fi
    done
}

run_condition "tiny_stage1"  "$ROOT/ckpt_distill_tx_only_192d2L/checkpoint-149.pth"          192 2
run_condition "tiny_stage2"  "$ROOT/ckpt_polish_192d2L_from_tx_distill/checkpoint-best_psnr.pth" 192 2
run_condition "small_stage1" "$ROOT/ckpt_distill_tx_only_v2_512d2L/checkpoint-69.pth"         512 2
run_condition "small_stage2" "$ROOT/ckpt_polish_512d2L_from_tx_distill/checkpoint-best_psnr.pth" 512 2

echo "=== ALL 4 CONDITIONS DONE ==="
