#!/bin/bash
# run_k_sweep_valid.sh -- 2026-10-03. Routing-cardinality (K) sensitivity
# sweep requested by the author: does reducing K below the default 63 cost
# accuracy, on the CURRENT true flagship checkpoint (b1d2eb25...), Site A
# VALIDATION split (437 images, never touched for any other reported
# number), real LDPC-coded joint-frame pipeline (not the training proxy),
# RVQ bit-width held fixed (3x6-bit high, 2-bit low) -- ONLY routing_topk
# varies. K in {40,48,56,63,70}, SNR {10,20}dB, M=10, SAME channel draws
# across K (phy.py's channel_draw() is seeded purely by split+image+draw
# index, independent of K/cache content -- verified earlier this session).
# symbols=6144 (QPSK) held fixed across all K per the author's request;
# effective code rate (info_bits/coded_bits) varies with K and is recorded,
# not held fixed.
set -e
cd "$(dirname "$0")"
ROOT=__AQUASC_ROOT__
SIONNA=__SIONNA_PY__
TORCH=__TORCH_PY__
RUNS=$ROOT/experiments/matched5/runs/k_sweep_valid
mkdir -p "$RUNS/logs"

KS="40 48 56 63 70"
SNRS="10 20"
M=10

for K in $KS; do
    name="k${K}"
    cache_dir="$RUNS/cache_${name}"
    if [ ! -f "$cache_dir/source.npz" ]; then
        echo "=== [K=$K] cache (split=valid, natural FIM routing at routing_topk=$K) ==="
        rm -rf "$cache_dir"
        taskset -c 0-7,10-31 $TORCH evaluate.py --phase cache --split valid \
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
                --split valid --variant frame_qpsk --m $M --symbols 6144 --snr-db $snr \
                --out-dir "$phy_dir" \
                > "$RUNS/logs/phy_${name}_snr${snr}.log" 2>&1

            echo "=== [K=$K] SNR=${snr}dB evaluate ==="
            rm -rf "$aqua_dir"
            taskset -c 0-7,10-31 $TORCH evaluate.py --phase aquasc --split valid \
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

echo "===== K sweep (validation) done ====="
