#!/bin/bash
# run_ablation_routing_coded.sh -- 2026-09-23. Re-runs the FIM/BBW ablation
# AND the routing-baseline comparison through the SAME real LDPC-coded
# joint-frame headline pipeline (experiments/matched5/), replacing the old
# uncoded-digital protocol (RESULTS_OF_RECORD.md section 3/4) that reviewer 3
# flagged as inconsistent with the headline's own coded channel.
#
# 8 unique conditions (Full+FIM is condition 1, already measured at the
# headline rate with M=10 -- runs/sweep/aqua_{10,20} -- reused, not rerun):
#   1. Full        + FIM              (reused, not run here)
#   2. FIM-NoBB    + FIM (natural)
#   3. Base        + random (natural, disable_fim's own seeded branch)
#   4. BB-NoFIM_V2 + random (natural, disable_fim's own seeded branch)
#   5. Full        + oracle_box (route_override)
#   6. Full        + random     (route_override)
#   7. Full        + uniform    (route_override)
#   8. Full        + detector_heatmap (route_override)
#
# SNR: 10, 20 dB (user's stated priority). M=10 realizations (project
# standard, matches the existing headline sweep -- runs/sweep/aqua_10,
# aqua_20 both have per_image_ap.shape==(432,10)).
set -e
cd "$(dirname "$0")"
ROOT=__AQUASC_ROOT__
SIONNA=__SIONNA_PY__
TORCH=__TORCH_PY__
RUNS=__AQUASC_ROOT__/experiments/matched5/runs/ablation_routing_coded
mkdir -p "$RUNS" "$RUNS/logs"

SNRS="10 20"
M=10

run_condition () {
    name="$1"; shift
    ckpt_args="$1"; shift
    router_args="$1"; shift

    cache_dir="$RUNS/cache_${name}"
    if [ ! -f "$cache_dir/source.npz" ]; then
        echo "=== [$name] cache ==="
        rm -rf "$cache_dir"
        $TORCH evaluate.py --phase cache --split test \
            --out-dir "$cache_dir" $ckpt_args $router_args \
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

FULL_CKPT=""   # empty = flagship default
ABL_FIM_NOBB="--ckpt $ROOT/ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS_ABL_FIM_NOBB/checkpoint-best_psnr.pth"
ABL_BASE="--ckpt $ROOT/ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS_ABL_BASE/checkpoint-best_psnr.pth --disable-fim"
ABL_BB_NOFIM="--ckpt $ROOT/ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS_ABL_BB_NOFIM_V2/checkpoint-best_psnr.pth --disable-fim"

run_condition "fim_nobb"     "$ABL_FIM_NOBB" ""
run_condition "base"         "$ABL_BASE"     ""
run_condition "bb_nofim"     "$ABL_BB_NOFIM" ""
run_condition "full_oracle"  "$FULL_CKPT"    "--router oracle_box"
run_condition "full_random"  "$FULL_CKPT"    "--router random"
run_condition "full_uniform" "$FULL_CKPT"    "--router uniform"
run_condition "full_detheat" "$FULL_CKPT"    "--router detector_heatmap"

echo "=== ALL 7 NEW CONDITIONS DONE ==="
