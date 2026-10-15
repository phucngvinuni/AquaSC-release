#!/bin/bash
# Run on the Raspberry Pi, from ~/aquasc_edge_bench (the transferred payload
# dir: encoder_weights.pt, decoder_weights.pt, aquasc_encoder.onnx,
# yolo_fish.onnx, best.pt, model_util.py, digital_channel.py,
# edge_encoder_decoder.py, bench_pytorch.py, bench_onnx.py, timm/).
#
# Baseline (PyTorch + ONNX, 1-core and 4-core, default governor/full speed)
# plus an ONNX single-core frequency sweep 600-1800MHz (same recipe as this
# lab's sempose_fi/bench_inference_throttled.py). Needs sudo for cpufreq-set.
set -e
cd "$(dirname "$0")"
mkdir -p logs

echo "=== Baseline: PyTorch, 4-core, full speed ==="
python3 bench_pytorch.py --payload_dir . --yolo best.pt --freq 1800 --cores 4 --n_threads 4 \
    --out logs/pytorch_1800mhz_4c.json

echo "=== Baseline: PyTorch, 1-core, full speed ==="
taskset -c 0 python3 bench_pytorch.py --payload_dir . --yolo best.pt --freq 1800 --cores 1 --n_threads 1 \
    --out logs/pytorch_1800mhz_1c.json

echo "=== Baseline: ONNX Runtime, 4-core, full speed ==="
python3 bench_onnx.py --freq 1800 --cores 4 --n_threads 4 --out logs/onnx_1800mhz_4c.json

echo "=== Baseline: competitor encoders (JSCC-CNN, SwinJSCC, ROI-JSCC), 1-core ==="
taskset -c 0 python3 bench_jscc.py --n_threads 1 --out logs/jscc_1c.json
taskset -c 0 python3 bench_swinjscc.py --n_threads 1 --out logs/swinjscc_1c.json
taskset -c 0 python3 bench_roijscc.py --n_threads 1 --out logs/roijscc_1c.json

echo "=== Baseline: competitor encoders, 4-core ==="
python3 bench_jscc.py --n_threads 4 --out logs/jscc_4c.json
python3 bench_swinjscc.py --n_threads 4 --out logs/swinjscc_4c.json
python3 bench_roijscc.py --n_threads 4 --out logs/roijscc_4c.json

echo "=== ONNX Runtime single-core frequency sweep ==="
sudo cpufreq-set -g userspace
for f in 600 900 1200 1500 1800; do
    sudo cpufreq-set -f ${f}000 -c 0
    echo "--- ${f} MHz, 1 core ---"
    vcgencmd measure_temp
    taskset -c 0 python3 bench_onnx.py --freq $f --cores 1 --n_threads 1 \
        --out logs/onnx_${f}mhz_1c.json
    vcgencmd measure_temp
    vcgencmd get_throttled
done
sudo cpufreq-set -g ondemand

echo "=== ALL DONE. Results in logs/*.json ==="
ls -la logs/
