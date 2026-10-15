#!/bin/bash
# run_random_router_multiseed.sh -- 2026-09-25. §4b's random-router row
# (R1_main_TETCI.tex line ~907) used a single routing-mask seed (42, reused
# from cache_full_random/aqua_full_random_snr{10,20}). This runs 4 MORE
# routing-mask seeds (43,44,45,46) -- same Full/flagship checkpoint, same
# real coded joint-frame pipeline, same 10/20dB SNRs, SAME 10 channel draws
# per image (phy.py's channel RNG is seeded purely by split+image+draw index,
# `seed_base=44_000_000` for the test split -- confirmed independent of the
# router/cache content, so all 5 router seeds see IDENTICAL channel draws --
# isolating router-seed variance from channel variance, exactly as requested).
set -e
cd "$(dirname "$0")"
ROOT=__AQUASC_ROOT__
SIONNA=__SIONNA_PY__
TORCH=__TORCH_PY__
RUNS=$ROOT/experiments/matched5/runs/ablation_routing_coded
SNRS="10 20"
M=10
SEEDS="43 44 45 46"

for seed in $SEEDS; do
    name="full_random_seed${seed}"
    cache_dir="$RUNS/cache_${name}"
    if [ ! -f "$cache_dir/source.npz" ]; then
        echo "=== [$name] cache ==="
        rm -rf "$cache_dir"
        $TORCH evaluate.py --phase cache --split test \
            --out-dir "$cache_dir" --router random --router-seed $seed \
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
            taskset -c 0-7,10-31 $TORCH evaluate.py --phase aquasc --split test \
                --receiver "$phy_dir/receiver.npz" \
                --out-dir "$aqua_dir" \
                > "$RUNS/logs/aqua_${name}_snr${snr}.log" 2>&1
        else
            echo "=== [$name] SNR=${snr}dB already done, skipping ==="
        fi
        python3 -c "import json; s=json.load(open('$aqua_dir/summary.json')); print('  ', s['mean_per_image_ap50'], s['pooled_map50'], s['frame_loss_rate'])"
    done
done

echo "===== random-router multiseed sweep done ====="
