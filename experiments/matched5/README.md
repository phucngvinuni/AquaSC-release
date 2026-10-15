# Equal-rate digital AquaSC versus ROI-JSCC at 5 dB

Measured on Site A test (432 images, 10 shared channel realizations/image):

| Metric | AquaSC joint-frame QPSK | ROI-JSCC rcpp32 |
|---|---:|---:|
| Actual complex symbols/image | 6144 | 6144 |
| Complex uses / 224x224 pixel | 0.1224489796 | 0.1224489796 |
| Pooled mAP@0.5, mean over realizations | 0.6404676 | 0.5763720 |
| Mean per-image AP@0.5 | 0.6956193 | 0.6348151 |

Paired per-image AP improvement: **+0.0608042**, 95% image-bootstrap CI
**[0.0453224, 0.0765515]**, 10000 resamples. Pooled improvement is +0.0640956;
the CI above belongs to per-image AP, not pooled mAP. All ten realization-wise
mean-AP differences are positive. The baseline reproduces its released pooled
and mean-per-image metrics; maximum per-image mean discrepancy is 1.19e-7.

This is a new operating point, not the old 0.0714-uses/px flagship curve.
Compared with that old point, this profile spends 6144/3584 = 12/7 times the
symbols, while matching ROI-JSCC rcpp32 exactly. No new neural training was used.

## What changed

The source codec remains the released flagship RVQ 3x6 high / 2-bit low,
K=63, checkpoint SHA `b1d2eb25419af6ed6703a4aa886eb7143a8f1f350bd50e3371dacb3cde9e2894`.
The PHY protects a complete image in one BG2 LDPC codeword:

`152-bit header | 313-bit routing mask | 2576-bit RVQ indices | CRC32 = 3073 bits`

Sionna rate-matches these 3073 information bits to **12288 coded bits**, then
Gray QPSK emits exactly **6144 unit-energy complex symbols**. Receiver uses APP
demapping and sum-product (`boxplus-phi`) decoding with 40 fixed iterations.
The CRC covers header, mask, and all indices. Frame version is 2. CRC failure
uses the existing all-zero latent fallback; every failed image stays in scoring.
This profile gives up partial-stage UEP recovery, changes block length/MCS/decoder,
and has not been isolated into individual causal ablations.

Both systems use **one Rayleigh coefficient per image**, Es/N0=5 dB, perfect
receiver CSI, and the same fading/noise draws. There is no feedback, retransmission,
transmitter CSI, per-image MCS selection, or clean-token substitution at evaluation.
The original digital profile uses per-code-block fading; its old numbers are not
silently relabeled as results under this matched channel protocol.

## Selection and limitations

`PLAN.md` was written before quality measurements. Synthetic link screening
compared short/long codes, BPSK/QPSK, and NMS/sum-product. On all 437 validation
images with M=3 disjoint channel draws, long-frame QPSK scored 0.2672458 pooled
mAP versus BPSK 0.2587179 and ROI-JSCC 0.1615527. `LOCK.json` records the selection
before test. No transport parameter was retuned after test.

This result covers **one site, one SNR, and one training seed per model**.
Inherited training budgets are AquaSC 300 epochs and ROI-JSCC 100 epochs.
ROI-JSCC retains its ground-truth largest-box ROI protocol, with that side
information granted rather than charged. Image-bootstrap CIs do not measure
between-training-seed or independent-video uncertainty. No claim is made about
winning at all rates/SNRs or at the old flagship's smaller symbol budget.
Cached evaluation times are not end-to-end deployment latency measurements.

## Run through the normal model/bridge

Use the flagship's full architecture flags from `configs/final_18_2.yaml` and
add **`--eval --digital_ldpc --digital_joint_frame --digital_csi_mode perfect`**.
The opt-in joint profile fixes QPSK/BG2(3073,12288)/40 iterations and overrides
the legacy `digital_bits_per_symbol` / `digital_payload_rate` controls.
It requires the 18/2 RVQ configuration and perfect CSI. Existing default and
analog branches remain available. Example for one channel realization at 5 dB:

```bash
cd __AQUASC_ROOT__
__TORCH_PY__ run_class_main.py \
  --eval --digital_ldpc --digital_joint_frame --digital_csi_mode perfect \
  --digital_bridge_python __SIONNA_PY__ \
  --resume ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS_FINAL/checkpoint-best_psnr.pth \
  --model ViT_Reconstruction_Model_Default --data_set fish \
  --data_path ../yolo_fish_dataset_root --input_size 224 --patch_size 8 \
  --encoder_embed_dim 512 --encoder_depth 8 --encoder_num_heads 8 \
  --decoder_embed_dim 512 --decoder_depth 6 --decoder_num_heads 8 \
  --fim_embed_dim 256 --fim_depth 2 --fim_num_heads 4 --quantizer_dim 512 \
  --quantizer_type rvq --rvq_stages 3 --rvq_bits_per_stage 6 --bits_vq_low 2 \
  --routing_topk 63 --num_object_classes 1 --yolo_weights best.pt \
  --map_conf_thres 0.001 --yolo_iou_thres 0.5 --snr_db_eval 5 \
  --seed 44000000 --batch_size 1 --num_workers 0 --lpips_loss_weight 0 \
  --output_dir experiments/matched5/runs/main_replay_m0
```

`--seed 44000000 + 1000*m` reproduces test-bank realization `m=0..9` at 5 dB.
One realization is not the ten-realization aggregate reported above.

## Reproduce the paired test result

Run from `AquaSC-D2DB/`. Choose fresh output directories: scripts refuse to
overwrite a run. The source cache only avoids repeating the deterministic
encoder; `--transmit` still performs real LDPC/QPSK/noise/decoding for every frame.

```bash
TORCH_PY=__TORCH_PY__
SIONNA_PY=__SIONNA_PY__
MATCHED5_RUN=experiments/matched5/runs/replay_01
"$TORCH_PY" experiments/matched5/evaluate.py --phase cache --split test --out-dir "$MATCHED5_RUN/cache"
"$SIONNA_PY" experiments/matched5/phy.py --transmit --source "$MATCHED5_RUN/cache/source.npz" --split test --variant frame_qpsk --m 10 --out-dir "$MATCHED5_RUN/phy"
"$TORCH_PY" experiments/matched5/evaluate.py --phase aquasc --split test --receiver "$MATCHED5_RUN/phy/receiver.npz" --out-dir "$MATCHED5_RUN/aquasc"
"$TORCH_PY" experiments/matched5/evaluate.py --phase roi --split test --m 10 --out-dir "$MATCHED5_RUN/roi"
"$TORCH_PY" experiments/matched5/evaluate.py --phase compare --split test --aqua-dir "$MATCHED5_RUN/aquasc" --roi-dir "$MATCHED5_RUN/roi" --out-dir "$MATCHED5_RUN/comparison"
```

## Evidence and verification

- `runs/comparison_test_v1/comparison.json`: locked held-out comparison and hashes.
- `runs/comparison_valid_v1/comparison.json`: validation selection evidence.
- `runs/phy_test_qpsk_locked_v1/receiver.{npz,json}`: decoded fields, failure flags,
  actual rate, frame description, and channel seeds (no failed images omitted).
- `runs/aqua_test_qpsk_locked_v1/metrics.npz`, `runs/roi_test_single_v1/metrics.npz`:
  per-image/per-realization AP and pooled metrics.
- `runs/integration_smoke_v2/integration.json`: live model -> client -> bridge ->
  Sionna -> RVQ decoder agrees with cached PHY on 16 validation images (12 success,
  4 loss); reconstruction discrepancy <=1.19e-7. Legacy and new high-SNR
  round-trips pass and both rates are verified.
- `phy.py --self-test`: real-PHY BPSK/QPSK round-trips, CRC rejection including
  header/mask corruption, recovered routing mask, emitted-rate and legacy-rate checks.

Implementation lives in `phy.py`; the normal bridge imports its opt-in adapter.
Torch and Sionna environments remain separate (see `requirements/README.md`).

## Update 2026-09-15: full SNR sweep, `--snr-db`

`phy.py`'s `transmit_cache` and `evaluate.py`'s `evaluate_roi` originally
hardcoded 5dB (module constant `SNR_DB`) and their channel draws were only
ever verified against `channel_bank.npz` at `snr_idx=2`. Both now accept
`--snr-db <value>`, reading that SNR's own `h`/noise directly from the bank
(required for `--split test`; the grid is -5/0/5/10/15/20). Reusing the old
5dB-only shortcut at another SNR would have silently paired that point
against the 5dB fading draws -- the bank stores independent `h` per SNR
index, not one draw reused across the sweep. See `RESULTS_OF_RECORD.md`
section 6s for the full 6-point result, paired bootstrap CIs, and the
Holm-Bonferroni-corrected significance table. `run through the normal
model/bridge` and `Reproduce the paired test result` above both accept
`--snr-db`/`--snr_db_eval` accordingly; omit it for the original locked 5dB
point.
