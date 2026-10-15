# Timing: what was measured, how, and what it does not show

All timing files are in `artifacts/results/timing/`. Edge numbers come from a Raspberry Pi 4 Model B (4 x Cortex-A72 @ 1.8 GHz, 3.7 GB RAM, Debian 12 aarch64, torch 2.5.1 CPU, no accelerator).

## Edge compute (`edge_encoder_latency_pi4b.csv`; scripts in `experiments/edge_bench/`)
* Timed region: encoder forward only (`img_encoder` -> FIM -> top-K routing -> RVQ quantisers). **Not timed**: bitstream packing, header/mask rank, CRC-32, LDPC, modulation.
* PyTorch eager, batch 1, 224x224, 20 warm-up + 200 timed calls, mean +- std. AquaSC encoders use trained weights; JSCC-CNN / SwinJSCC / ROI-JSCC were timed with randomly initialised weights at their deployed architecture (latency depends only on shapes).
* Parity of the standalone encoder wrappers against the full model was verified before shipping them to the Pi (`experiments/edge_bench/extract_edge_weights.py`).
* Reference numbers for the other devices/runtimes: `results/edge/edge_onnx_latency.csv`, `results/edge/edge_complexity.csv`.

## Network and server (`network_transfer.csv`)
* Pi -> desktop: `scp` over one persistent `ssh -M` connection through a real Tailscale link (two WiFi networks), 20 trials. This measurement was run interactively and is **not scripted**; only 384 B and 6421 B payloads were timed.
* Corrected payloads: AquaSC 385 B (ceil(3073/8)); ROI-JSCC rcpp32 49,152 B (6144 complex symbols x 8 B). The 49,152 B payload was **not** timed on the Pi link. The laptop -> PC rows were measured because the Pi was unavailable; they are a different link and method and must not be compared with the Pi rows.
* Server-side decode + detection (desktop GPU): AquaSC 7.0 ms, JPEG baseline 2.3 ms.
* Scripts for the laptop -> PC probes: `experiments/edge_bench/net_latency_receiver.py`, `net_latency_sender.ps1`, `net_latency_scp.sh`.

## Split-computing table
`paper_table_split_computing.csv` is the table of the manuscript transcribed as printed (Edge / Network / Server / Total / payload / mAP@0.5 / PSNR at 10 dB). ROI-JSCC Network = 232.5 ms (value of the 50,176 B baseline rows), Total = 1461.9 ms.

## Reference hardware for the evaluation scripts (not the edge device)
Desktop: Intel Core i9-13900K (32 threads), 31 GB RAM, NVIDIA GeForce RTX 4070 (12 GB), Ubuntu 24.04-class Linux, CUDA 12.1 / torch 2.1.2; separate CPU venv for Sionna 1.2.2 / TensorFlow 2.21.0.
Wall clock from the stored `summary.json` files (432 images x 10 draws): LDPC/QPSK transmission 91-104 s per SNR point per rate; detector scoring 39-50 s per SNR point; so one rate point over 6 SNRs takes roughly 15-20 minutes plus ~2 minutes for the encoder cache. These are single measurements on this machine, not a benchmark.
