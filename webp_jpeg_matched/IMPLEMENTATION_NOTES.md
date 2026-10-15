# WebP/JPEG matched-budget rerun: implementation notes (2026-10-15)

Implements `WEBP_JPEG_RERUN_PROTOCOL.md` (the file path cited by the author, `../audit-integration-20261015/`, does not exist on disk; the protocol text was supplied in the session, so this directory is the only record of how it was implemented).

## Files

| File | Role |
|---|---|
| `calibrate_quality_valid.py` -> `valid_calib/quality_selection.json`, `valid_sizes_{webp,jpeg}.npy` | Protocol §5: q=1..100 on the 437 validation images, highest q whose max whole-file size <= capacity. Test split is never read. |
| `stage1_phy.py` (Sionna venv) | Payload (4 B big-endian length + file + zero pad), 992+CRC32 blocks, BG2 k=1024/n=2048, 16-QAM, APP, boxplus-phi 40 it, llr_max 20, one h per image/draw, bank noise. Writes `runs/<codec>_<budget>/rx_snr*.npz`. |
| `gates.py` -> `gates_result.json`, `logs/gates.log` | Protocol §6 gates (all pass). |
| `stage2_eval.py` (torch env) | Decode, detector (best.pt, conf 0.001), per-image AP50, pooled mAP50, PSNR/SSIM; writes `metrics.npz`, `summary.json`. |
| `aggregate.py` -> `RESULTS_matched_budget.{md,json}` | Per-SNR tables and paired CI vs AquaSC at the same channel uses. |
| `run_stage1_all.sh`, `run_stage2_all.sh` | Exact commands that produced the results. |

## Decisions the protocol left open (made on validation only)

1. **JPEG encoder options: `optimize=True`.** With PIL defaults the validation floor is ~1440 B per file (standard Huffman tables + quantisation tables dominate), so the low and mid budgets would be 100% capacity failure by construction. `optimize=True` brings the floor to ~620-650 B. The mid budget (740 B) becomes feasible (q=5); the low budget (368 B) stays infeasible for every q, so it is q=1 with 100% capacity failure. This is reported as is, not hidden.
2. **WebP options**: `quality=q, method=4` (same as the earlier scripts).
3. **Selected q** (validation max size / cap): WebP low q=1 (548 B > 368, infeasible, 332/437 validation images over), WebP mid q=9 (738/740), WebP high q=51 (1476/1484); JPEG low q=1 (infeasible, 100%), JPEG mid q=5 (706/740), JPEG high q=24 (1483/1484).
4. **Channel draws**: h and noise come from `channel_bank.npz` by (image, SNR, draw), the same bank AquaSC's test runs use, with the same slicing convention as `phy.transmit_cache` (`noise = raw[:S] + j*raw[S:2S]`). The baselines therefore see the same fading coefficient and the same noise samples as AquaSC at the same budget (paired draws). Noise of a given image is split across its blocks (512 symbols per block), so blocks of one image see independent noise but the same h.
5. **Failure policies** (as in the protocol): a length block that fails CRC, a length > capacity, a length-0 marker, or a PIL decode failure -> black image. Failed-CRC data blocks are zero-filled. PIL decode additionally treats an absurd corrupted size (> 4096x4096) as a decode failure instead of allocating it.
6. **Exact-result cache in stage 2**: an image whose received bytes are identical to the clean compressed file, or a black image, reuses the deterministic detector/PSNR/SSIM result computed once for that image. Corrupted-but-decoded images are always scored fresh. All 25,920 records per config are present.
7. **Clean-channel reference** is stored per config (`clean_ap`, `clean_psnr`, `clean_ssim` in `metrics.npz`); capacity-failure images count as black in the clean reference too.
8. **AquaSC comparators**: headline JointFrame (`runs/sweep`, 6144 symbols), retrained r061 (`runs/jointframe_r061_bestmap`, 3072 symbols), retrained r030 (`runs/jointframe_r030_bestmap`, 1536 symbols), all Gray QPSK, one codeword per image. NOT the `sweep_r061`/`sweep_r030` directories (flagship codec with truncated transport).

## What is and is not a like-for-like comparison

Same: channel uses per image (all images, padding included), h and noise realisations, 432 images x 10 draws, detector, scoring code, perfect CSI, LDPC BG2, APP demapping, boxplus-phi 40 iterations, llr_max 20.

Different (must be stated when publishing): baselines use Gray 16-QAM with rate-1/2 per-block LDPC (k=1024), AquaSC uses QPSK with one codeword per image (rate 0.25 at the headline); source codecs differ. The comparison is of complete transmission configurations at equal channel uses, not an isolation of the source codec.
