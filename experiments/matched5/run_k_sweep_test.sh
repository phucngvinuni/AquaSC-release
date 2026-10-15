#!/bin/bash
# run_k_sweep_test.sh -- 2026-10-03. Confirmatory re-run of §6al's K-sweep
# (40/48/56/63/70) on the Site A TEST split (432 images), alongside the
# already-done validation sweep -- NOT a replacement for it, and K=63 is
# NOT re-selected based on this: K=63 remains the shipped default regardless
# of outcome. K=63 on test is already on record exactly (runs/sweep/aqua_10,
# aqua_20 -- the existing headline sweep, info_bits=3073 matches K=63
# exactly) and is reused unchanged, not rerun. Only K=40/48/56/70 are new.
# Same protocol as the validation sweep: RVQ fixed 3x6-bit high/2-bit low,
# real coded QPSK pipeline, symbols=6144 fixed, SNR {10,20}dB, M=10, same
# channel draws across K (phy.py channel_draw() seeded by
# (split,image,draw) only).
set -e
cd "$(dirname "$0")"
ROOT=__AQUASC_ROOT__
SIONNA=__SIONNA_PY__
TORCH=__TORCH_PY__
RUNS=$ROOT/experiments/matched5/runs/k_sweep_test
mkdir -p "$RUNS/logs"

KS="40 48 56 70"
SNRS="10 20"
M=10

for K in $KS; do
    name="k${K}"
    cache_dir="$RUNS/cache_${name}"
    if [ ! -f "$cache_dir/source.npz" ]; then
        echo "=== [K=$K] cache (split=test, natural FIM routing at routing_topk=$K) ==="
        rm -rf "$cache_dir"
        taskset -c 0-7,10-31 $TORCH evaluate.py --phase cache --split test \
            --routing-topk $K --rvq-stages 3 --rvq-bits-per-stage 6 --bits-vq-low 2 \
            --out-dir "$cache_dir" \
            > "$RUNS/logs/cache_${name}.log" 2>&1
    else
        echo "=== [K=$K] cache already exists, skipping ==="
    fi

    for snr in $SNRS; do
        phy_dir="$RUNS/phy_${name}_snr${snr}"
        aqua_dir="$RUNS/aqua_${name}_snr${snr}"
        if [ ! -f "$aqua_dir/summary.json" ]; then
            echo "=== [K=$K] SNR=${snr}dB transmit (QPSK, symbols=6144) ==="
            rm -rf "$phy_dir"
            taskset -c 0-7,10-31 $SIONNA phy.py --transmit \
                --source "$cache_dir/source.npz" \
                --split test --variant frame_qpsk --m $M --symbols 6144 --snr-db $snr \
                --out-dir "$phy_dir" \
                > "$RUNS/logs/phy_${name}_snr${snr}.log" 2>&1

            echo "=== [K=$K] SNR=${snr}dB evaluate ==="
            rm -rf "$aqua_dir"
            taskset -c 0-7,10-31 $TORCH evaluate.py --phase aquasc --split test \
                --receiver "$phy_dir/receiver.npz" \
                --out-dir "$aqua_dir" \
                > "$RUNS/logs/aqua_${name}_snr${snr}.log" 2>&1
        else
            echo "=== [K=$K] SNR=${snr}dB already done, skipping ==="
        fi
        python3 -c "
import json
s=json.load(open('$aqua_dir/summary.json'))
rate = s['info_bits']/s['coded_bits']
print(f'  K=$K SNR=${snr}dB: info_bits={s[\"info_bits\"]} coded_bits={s[\"coded_bits\"]} eff_rate={rate:.4f} '
      f'pooled={s[\"pooled_map50\"]:.4f} per_image={s[\"mean_per_image_ap50\"]:.4f} frame_loss={s[\"frame_loss_rate\"]:.4f}')
"
    done
done

echo "===== K sweep (test) done ====="
