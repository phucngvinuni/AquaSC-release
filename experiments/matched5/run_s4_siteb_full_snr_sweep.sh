#!/bin/bash
# run_s4_siteb_full_snr_sweep.sh -- 2026-09-24.
#
# Extends §6ai's single-SNR (10dB) real-coded-pipeline re-evaluation of Site B
# S-3/S-4 to the full standard 6-point SNR grid, closing §6ai caveat (2).
# Reuses the existing caches (cache_s4ep9_test, cache_s3ep29_test) built by
# run_s4_siteb_honest_select.sh -- only the SNR-dependent transmit/eval steps
# are new. Also adds S-4 ep14 on test (closes caveat 3, for completeness --
# not the citable checkpoint, ep9 is, but completes the picture).
set -e
cd "$(dirname "$0")"
ROOT=__AQUASC_ROOT__
SIONNA=__SIONNA_PY__
TORCH=__TORCH_PY__
RUNS=$ROOT/experiments/matched5/runs/s4_siteb_honest_select
M=10
SNRS="-5 0 5 15 20"   # 10dB already done in run_s4_siteb_honest_select.sh

run_snr () {
    name="$1"; ckpt="$2"; split="$3"; snr="$4"

    cache_dir="$RUNS/cache_${name}_${split}"
    if [ ! -f "$cache_dir/source.npz" ]; then
        echo "=== [$name/$split] cache (missing, building) ==="
        rm -rf "$cache_dir"
        $TORCH evaluate.py --phase cache --split "$split" --site b \
            --ckpt "$ckpt" \
            --out-dir "$cache_dir" \
            > "$RUNS/logs/cache_${name}_${split}.log" 2>&1
    fi

    phy_dir="$RUNS/phy_${name}_${split}_snr${snr}"
    aqua_dir="$RUNS/aqua_${name}_${split}_snr${snr}"
    if [ ! -f "$aqua_dir/summary.json" ]; then
        echo "=== [$name/$split] SNR=${snr}dB transmit ==="
        rm -rf "$phy_dir"
        taskset -c 0-7,10-31 $SIONNA phy.py --transmit \
            --source "$cache_dir/source.npz" \
            --split "$split" --variant frame_qpsk --m $M --symbols 6144 --snr-db $snr \
            --out-dir "$phy_dir" \
            > "$RUNS/logs/phy_${name}_${split}_snr${snr}.log" 2>&1

        echo "=== [$name/$split] SNR=${snr}dB evaluate ==="
        rm -rf "$aqua_dir"
        $TORCH evaluate.py --phase aquasc --split "$split" \
            --receiver "$phy_dir/receiver.npz" \
            --out-dir "$aqua_dir" \
            > "$RUNS/logs/aqua_${name}_${split}_snr${snr}.log" 2>&1
    else
        echo "=== [$name/$split] SNR=${snr}dB already done, skipping ==="
    fi
    echo "--- [$name/$split] SNR=${snr}dB summary ---"
    python3 -c "import json; s=json.load(open('$aqua_dir/summary.json')); print(s['mean_per_image_ap50'], s['pooled_map50'], s['frame_loss_rate'])"
}

for snr in $SNRS; do
    run_snr "s4ep9"  "$ROOT/ckpt_siteb_s4_ema0p9/checkpoint-9.pth"       "test" "$snr"
    run_snr "s3ep29" "$ROOT/ckpt_siteb_s3_full_finetune/checkpoint-29.pth" "test" "$snr"
done

# ep14 on test at 10dB only, for completeness (not the citable checkpoint)
run_snr "s4ep14" "$ROOT/ckpt_siteb_s4_ema0p9/checkpoint-14.pth" "test" "10"

echo "===== full SNR sweep done ====="
