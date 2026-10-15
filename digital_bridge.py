"""Subprocess bridge: torch eval loop -> Sionna digital pipeline (Wave 5 CHANNEL-cont).

Runs under the sionna venv (__SIONNA_PY__). The torch
training/eval env has tensorflow-free numpy 1.25 / torch 2.1.2 and cannot host
sionna 1.2.2 (which requires TF 2.21 -> numpy >= 1.26). Instead of installing
TF/sionna into the conda env (which would force numpy >= 1.26 and break torch),
the eval loop talks to this process over a JSON-lines protocol on stdin/stdout.

Protocol (one JSON object per line):
  request  {"op":"ping"}
  request  {"op":"transmit", "snr_db":10.0, "csi_mode":"perfect",
            "seed":42, "images":[{"tokens":[...n_tokens ints...], "high_mask":[...n_tokens 0/1...],
             "b_high":18, "b_low":2, "rvq_stage_uep":true}]}
  response {"ok":true, "images":[{"recovered":[...n_tokens ints...], "header_ok":bool,
            "frame_lost":bool, "blocks":int, "failed":int, "bler":float, "ber":float,
            "bit_accuracy":float, "erased_tokens":int, "iterations_mean":float}, ...],
            "stats": {...aggregate...}}

The legacy bridge mode transmits the full 3.3 frame: HEADER (152 b) + enumerative MASK
(ceil(log2 C(n_tokens,K)) bits: 313 for patch-8, 174 for patch-16) at rate 1/3
(UEP), INDEX (per-token bit fields) + PAYLOAD_CRC-32 at rate 1/2. The RVQ UEP
mode (`rvq_stage_uep=true`) instead sends stage 1 with HEADER+MASK at rate 1/3
and stages 2--3, low indices and the CRC at rate 1/2, via
digital_pipeline.simulate_digital_pipeline (5G NR BG2 LDPC,
Gray 16-QAM, Rayleigh block fading, per-block CRC-32 -> erasure). If the
header/mask segment is lost (any erased bit), the frame is lost and all tokens
are erased.

`recovered` is the per-token index values (length = n_tokens, e.g. 784 for
patch-8 or 196 for patch-16) with **-1 marking erased tokens**;
the torch side then does N4 interpolation in the codebook *vector* domain
(plan 3.4 "codeword interpolation"), so no index-domain concealment happens here.

Deviation (documented): simulate_digital_pipeline applies a per-block CRC-32, so
surviving blocks are bit-exact; the frame-level PAYLOAD_CRC-32 whole-payload
erasure used in digital_pipeline.parse_frame is NOT applied here -- erasure is
per-block, and the token's bits are checked for the -1 erase marker directly.
This is the granularity needed for a smooth mAP/BLER waterfall.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import digital_pipeline as dp
import bitstream as bs


def _pad_to_kcb(bits):
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    rem = bits.size % dp.K_CB
    if rem == 0:
        return bits
    return np.concatenate([bits, np.zeros(dp.K_CB - rem, dtype=np.uint8)])


def transmit_image(token_values, high_mask, b_high, b_low, snr_db, csi_mode,
                   seed, mask_mode=dp.MASK_MODE_ENUM, n_tokens=None,
                   grid_h=None, grid_w=None, bits_per_symbol=4, payload_rate=0.5):
    token_values = np.asarray(token_values, dtype=np.int64).reshape(-1)
    high_mask = np.asarray(high_mask, dtype=bool).reshape(-1)
    n_tokens = int(token_values.size) if n_tokens is None else int(n_tokens)
    assert token_values.size == n_tokens, token_values.size
    assert high_mask.size == n_tokens, high_mask.size

    index_bits = dp.build_index_bits(token_values, high_mask, b_high, b_low)
    k_high = int(np.count_nonzero(high_mask))
    if mask_mode == dp.MASK_MODE_ENUM:
        mask = dp.mask_bits_enumerative(high_mask, n_tokens, k_high)
    else:
        mask = dp.mask_bits_explicit(high_mask)
    header = dp.build_header(1, dp.CONFIG_18_2, mask_mode, n_tokens, k_high,
                             mask.size, index_bits.size)
    seg_a = np.concatenate([header, mask])
    payload_crc = dp.int_to_bits(dp.crc32_ieee(index_bits), 32)
    seg_b = np.concatenate([index_bits, payload_crc])

    seg_a_out, st_a = dp.simulate_digital_pipeline(
        seg_a, snr_db, rate=1 / 3, csi_mode=csi_mode, seed=seed,
        bits_per_symbol=bits_per_symbol)
    seg_b_out, st_b = dp.simulate_digital_pipeline(
        seg_b, snr_db, rate=payload_rate, csi_mode=csi_mode, seed=seed,
        bits_per_symbol=bits_per_symbol)

    header_ok = bool(np.all(seg_a_out[:dp.HEADER_LEN] >= 0))
    if header_ok:
        hdr = dp.parse_header(seg_a_out[:dp.HEADER_LEN].astype(np.uint8))
        header_ok = (hdr["magic"] == dp.MAGIC) and hdr["header_crc_ok"]
    mask_ok = bool(np.all(seg_a_out[dp.HEADER_LEN:dp.HEADER_LEN + mask.size] >= 0))
    frame_lost = not (header_ok and mask_ok)

    high_mask_rx = high_mask
    if not frame_lost:
        high_mask_rx = dp.mask_decode(
            seg_a_out[dp.HEADER_LEN:dp.HEADER_LEN + mask.size].astype(np.uint8),
            mask_mode, n_tokens, k_high)

    index_out = seg_b_out[:index_bits.size]

    tokens = [None] * n_tokens
    erased_count = 0
    bitpos = 0
    for t in range(n_tokens):
        w = b_high if high_mask_rx[t] else b_low
        if not frame_lost:
            chunk = index_out[bitpos:bitpos + w]
            if not np.any(chunk < 0):
                tokens[t] = dp.bits_to_int(chunk.astype(np.uint8))
        if tokens[t] is None:
            erased_count += 1
        bitpos += w

    # No index-domain concealment here: the torch side does N4 interpolation in
    # the codebook *vector* domain (plan 3.4 "codeword interpolation"). We return
    # -1 for erased tokens.
    recovered = np.array([(-1 if v is None else int(v)) for v in tokens],
                         dtype=np.int64)

    blocks = int(st_a["blocks"] + st_b["blocks"])
    failed = int(st_a["failed"] + st_b["failed"])
    bler = failed / blocks if blocks else 0.0
    n_bits = int(index_bits.size)
    wrong = int(np.sum(index_out < 0)) + int(np.sum((index_out >= 0) & (index_out != index_bits)))
    ber = wrong / n_bits if n_bits else 0.0
    bit_accuracy = 1.0 - ber
    iterations_mean = (
        (st_a["iterations_mean"] * st_a["blocks"] + st_b["iterations_mean"] * st_b["blocks"]) / blocks
        if blocks else 0.0
    )
    stats = dict(header_ok=bool(header_ok), frame_lost=bool(frame_lost),
                 blocks=blocks, failed=failed, bler=float(bler),
                 ber=float(ber), bit_accuracy=float(bit_accuracy),
                 erased_tokens=int(erased_count),
                 iterations_mean=float(iterations_mean),
                 n_tokens=n_tokens)
    stats["channel_uses_per_px"] = dp.channel_uses_per_px(
        [(1 / 3, st_a["blocks"]), (payload_rate, st_b["blocks"])], bits_per_symbol,
        csi_mode=csi_mode)
    return recovered, stats


def transmit_image_rvq_uep(token_values, high_mask, b_high, b_low, snr_db,
                            csi_mode, seed, mask_mode=dp.MASK_MODE_ENUM,
                            n_tokens=None, bits_per_symbol=4, payload_rate=0.5):
    """Transmit RVQ fields with stage-1 UEP at rate 1/3.

    The legacy path treats a high-token RVQ value as one 18-bit payload field.
    This path separates [stage1, stage2, stage3, low] and protects
    ``header|mask|stage1`` at rate 1/3 always (max protection for the fields
    that lose the whole frame if erased), while protecting
    ``stage2|stage3|low|payload_crc`` at ``payload_rate`` (default 1/2). A
    failed stage block erases only that stage; the torch side can reconstruct
    from surviving stages. ``bits_per_symbol`` (1=BPSK, 2=QPSK, 4=16-QAM)
    applies to both segments -- plan 16.9 Track F / E3 MCS ladder.
    """
    token_values = np.asarray(token_values, dtype=np.int64).reshape(-1)
    high_mask = np.asarray(high_mask, dtype=bool).reshape(-1)
    n_tokens = int(token_values.size) if n_tokens is None else int(n_tokens)
    if token_values.size != n_tokens or high_mask.size != n_tokens:
        raise ValueError("token_values/high_mask length mismatch")

    # bs.STAGE_BITS (6) is the fixed per-stage width across every RVQ config
    # in this codebase (--rvq_bits_per_stage 6); n_stages = b_high // 6 (3 for
    # 18/2b, 4 for L2's 24/4b). See bitstream.py's *_n functions (plan v5.9 J3).
    if b_high % bs.STAGE_BITS != 0:
        raise ValueError(f"per-stage UEP requires b_high a multiple of {bs.STAGE_BITS}, got {b_high}")
    n_stages = b_high // bs.STAGE_BITS

    if n_stages == bs.N_STAGES and b_low == bs.B_LOW:
        # Original, separately-verified 3-stage/low-2 path (18/2b flagship) --
        # kept byte-identical rather than routed through the generic path.
        stage1, stage2, stage3, low = bs.token_values_to_stages(
            token_values, high_mask, b_low=b_low)
        stages = [stage1, stage2, stage3]
        index_bits = bs.pack_rvq_indices(stage1, stage2, stage3, low,
                                         high_mask, n_tokens)
    else:
        stages, low = bs.token_values_to_stages_n(
            token_values, high_mask, n_stages, bs.STAGE_BITS, b_low)
        index_bits = bs.pack_rvq_indices_n(
            stages, low, high_mask, bs.STAGE_BITS, b_low, n_tokens)
    k_high = int(np.count_nonzero(high_mask))
    if mask_mode == dp.MASK_MODE_ENUM:
        mask = dp.mask_bits_enumerative(high_mask, n_tokens, k_high)
    else:
        mask = dp.mask_bits_explicit(high_mask)
    header = dp.build_header(1, dp.CONFIG_18_2, mask_mode, n_tokens, k_high,
                             mask.size, index_bits.size)

    def values_to_bits(values, width):
        if not values.size:
            return np.zeros(0, dtype=np.uint8)
        return np.concatenate([dp.int_to_bits(int(v) & ((1 << width) - 1), width)
                               for v in values])

    stage_bits = [values_to_bits(values, bs.STAGE_BITS) for values in stages]
    low_bits = values_to_bits(low, b_low)
    payload_crc = dp.int_to_bits(dp.crc32_ieee(index_bits), 32)
    seg_a_body = np.concatenate([header, mask, stage_bits[0]])
    seg_b_body = np.concatenate(stage_bits[1:] + [low_bits, payload_crc])
    seg_a = _pad_to_kcb(seg_a_body)
    seg_b = _pad_to_kcb(seg_b_body)
    seg_a_out, st_a = dp.simulate_digital_pipeline(
        seg_a, snr_db, rate=1 / 3, csi_mode=csi_mode, seed=seed,
        bits_per_symbol=bits_per_symbol)
    seg_b_out, st_b = dp.simulate_digital_pipeline(
        seg_b, snr_db, rate=payload_rate, csi_mode=csi_mode, seed=seed + 1,
        bits_per_symbol=bits_per_symbol)

    header_ok = bool(np.all(seg_a_out[:dp.HEADER_LEN] >= 0))
    if header_ok:
        hdr = dp.parse_header(seg_a_out[:dp.HEADER_LEN].astype(np.uint8))
        header_ok = (hdr["magic"] == dp.MAGIC) and hdr["header_crc_ok"]
    mask_start = dp.HEADER_LEN
    mask_ok = bool(np.all(seg_a_out[mask_start:mask_start + mask.size] >= 0))
    frame_lost = not (header_ok and mask_ok)
    high_mask_rx = high_mask.copy()
    if not frame_lost:
        high_mask_rx = dp.mask_decode(
            seg_a_out[mask_start:mask_start + mask.size].astype(np.uint8),
            mask_mode, n_tokens, k_high)

    stage_values = np.zeros((n_tokens, n_stages), dtype=np.int64)
    stage_erased = np.zeros((n_tokens, n_stages), dtype=bool)
    low_values = np.zeros(n_tokens, dtype=np.int64)
    low_erased = np.zeros(n_tokens, dtype=bool)

    def read_value(bits, offset, width):
        chunk = bits[offset:offset + width]
        if chunk.size < width or np.any(chunk < 0):
            return 0, True
        return dp.bits_to_int(chunk.astype(np.uint8)), False

    high_positions = np.where(high_mask_rx)[0]
    low_positions = np.where(~high_mask_rx)[0]
    a_stage_start = dp.HEADER_LEN + mask.size
    # seg_b holds stage_bits[1:] back-to-back, then low, then the CRC; the
    # cumulative offsets of stages 1..n_stages-1 within seg_b:
    seg_b_stage_offsets = np.cumsum([0] + [sb.size for sb in stage_bits[1:-1]]) \
        if n_stages > 1 else np.array([], dtype=np.int64)
    for stage_no in range(n_stages):
        source = seg_a_out if stage_no == 0 else seg_b_out
        offset = a_stage_start if stage_no == 0 else int(seg_b_stage_offsets[stage_no - 1])
        for j, pos in enumerate(high_positions):
            value, erased = read_value(source, offset + j * bs.STAGE_BITS,
                                       bs.STAGE_BITS)
            stage_values[pos, stage_no] = value
            stage_erased[pos, stage_no] = erased or frame_lost

    low_start = sum(sb.size for sb in stage_bits[1:])
    for j, pos in enumerate(low_positions):
        value, erased = read_value(seg_b_out, low_start + j * b_low, b_low)
        low_values[pos] = value
        low_erased[pos] = erased or frame_lost

    recovered = np.full(n_tokens, -1, dtype=np.int64)
    for pos in high_positions:
        valid = ~stage_erased[pos]
        if np.any(valid):
            recovered[pos] = bs.combine_high_value_n(
                [int(stage_values[pos, i]) for i in range(n_stages)], bs.STAGE_BITS)
    for pos in low_positions:
        if not low_erased[pos]:
            recovered[pos] = low_values[pos]

    blocks = int(st_a["blocks"] + st_b["blocks"])
    failed = int(st_a["failed"] + st_b["failed"])
    received = np.concatenate([seg_a_out[:seg_a_body.size],
                               seg_b_out[:seg_b_body.size]])
    expected = np.concatenate([seg_a_body, seg_b_body])
    valid_bits = received >= 0
    bit_accuracy = float(np.mean(received[valid_bits] == expected[valid_bits])) \
        if np.any(valid_bits) else 0.0
    stats = dict(
        header_ok=bool(header_ok), frame_lost=bool(frame_lost),
        payload_crc_ok=False, payload_crc_partial=True,
        blocks=blocks, failed=failed,
        bler=failed / blocks if blocks else 0.0,
        bit_accuracy=bit_accuracy, ber=1.0 - bit_accuracy,
        erased_tokens=int(np.sum(np.all(stage_erased[high_positions], axis=1)))
        + int(np.sum(low_erased[low_positions])),
        stage_erased_count=[int(np.sum(stage_erased[:, s])) for s in range(n_stages)],
        iterations_mean=(st_a["iterations_mean"] * st_a["blocks"]
                         + st_b["iterations_mean"] * st_b["blocks"]) / blocks
        if blocks else 0.0,
        n_tokens=n_tokens, rvq_stage_uep=True,
        # Diagnostic-only (plan v7.6): seg_b's OWN block failure, measured
        # before the `or frame_lost` OR-in below discards it -- lets a client
        # check whether seg_b's failure rate differs between frame_lost and
        # frame-alive images (independent-fading prediction: it should not).
        seg_b_blocks=int(st_b["blocks"]), seg_b_failed=int(st_b["failed"]),
    )
    stats["erased_token_fraction"] = stats["erased_tokens"] / n_tokens \
        if n_tokens else 0.0
    stats["channel_uses_per_px"] = dp.channel_uses_per_px(
        [(1 / 3, st_a["blocks"]), (payload_rate, st_b["blocks"])], bits_per_symbol,
        csi_mode=csi_mode)
    return recovered, stats, stage_values, stage_erased, low_values, low_erased


def _aggregate(agg, stats):
    if agg is None:
        out = dict(blocks=stats["blocks"], failed=stats["failed"],
                    erased_tokens=stats["erased_tokens"], ber_sum=stats["ber"],
                    bit_acc_sum=stats["bit_accuracy"],
                    iter_sum=stats["iterations_mean"] * stats["blocks"],
                    n_tokens_sum=stats["n_tokens"],
                    n_images=1)
        if "stage_erased_count" in stats:
            out["stage_erased_count"] = list(stats["stage_erased_count"])
        if "channel_uses_per_px" in stats:
            out["channel_uses_per_px_sum"] = stats["channel_uses_per_px"]
        if "seg_b_blocks" in stats:
            out["seg_b_blocks"] = stats["seg_b_blocks"]
            out["seg_b_failed"] = stats["seg_b_failed"]
        return out
    agg["blocks"] += stats["blocks"]
    agg["failed"] += stats["failed"]
    agg["erased_tokens"] += stats["erased_tokens"]
    agg["ber_sum"] += stats["ber"]
    agg["bit_acc_sum"] += stats["bit_accuracy"]
    agg["iter_sum"] += stats["iterations_mean"] * stats["blocks"]
    agg["n_tokens_sum"] += stats["n_tokens"]
    if "channel_uses_per_px" in stats:
        agg["channel_uses_per_px_sum"] = agg.get("channel_uses_per_px_sum", 0.0) + stats["channel_uses_per_px"]
    if "stage_erased_count" in stats:
        if "stage_erased_count" not in agg:
            agg["stage_erased_count"] = [0] * len(stats["stage_erased_count"])
        agg["stage_erased_count"] = [a + b for a, b in zip(agg["stage_erased_count"], stats["stage_erased_count"])]
    agg["n_images"] += 1
    return agg


def serve():
    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:  # noqa
            out.write(json.dumps({"ok": False, "error": f"bad json {e}"}) + "\n")
            out.flush()
            continue
        op = req.get("op")
        if op == "ping":
            out.write(json.dumps({"ok": True, "pong": True}) + "\n")
            out.flush()
            continue
        if op == "shutdown":
            out.write(json.dumps({"ok": True}) + "\n")
            out.flush()
            break
        if op == "transmit":
            snr_db = float(req["snr_db"])
            csi_mode = req.get("csi_mode", "perfect")
            seed = int(req.get("seed", 42))
            rvq_stage_uep = bool(req.get("rvq_stage_uep", False))
            joint_frame = bool(req.get("joint_frame", False))
            if joint_frame and not rvq_stage_uep:
                raise ValueError("joint-frame QPSK requires RVQ stage status")
            bits_per_symbol = int(req.get("bits_per_symbol", 4))
            payload_rate = float(req.get("payload_rate", 0.5))
            images = req["images"]
            results = []
            agg = None
            for i, im in enumerate(images):
                if rvq_stage_uep:
                    if joint_frame:
                        from experiments.matched5.phy import transmit_joint_image
                        rec, st, stage_values, stage_erased, low_values, low_erased = \
                            transmit_joint_image(im["tokens"], im["high_mask"],
                                int(im["b_high"]), int(im["b_low"]), snr_db,
                                csi_mode, seed + i)
                    else:
                        rec, st, stage_values, stage_erased, low_values, low_erased = \
                            transmit_image_rvq_uep(
                                im["tokens"], im["high_mask"], int(im["b_high"]),
                                int(im["b_low"]), snr_db, csi_mode, seed + i,
                                mask_mode=int(im.get("mask_mode", dp.MASK_MODE_ENUM)),
                                bits_per_symbol=bits_per_symbol, payload_rate=payload_rate)
                    results.append({
                        "recovered": rec.tolist(),
                        "stage_values": stage_values.tolist(),
                        "stage_erased": stage_erased.tolist(),
                        "low_values": low_values.tolist(),
                        "low_erased": low_erased.tolist(),
                        **st,
                    })
                else:
                    rec, st = transmit_image(
                        im["tokens"], im["high_mask"], int(im["b_high"]),
                        int(im["b_low"]), snr_db, csi_mode, seed + i,
                        mask_mode=int(im.get("mask_mode", dp.MASK_MODE_ENUM)),
                        bits_per_symbol=bits_per_symbol, payload_rate=payload_rate)
                    results.append({"recovered": rec.tolist(), **st})
                agg = _aggregate(agg, st)
            agg["bler"] = agg["failed"] / agg["blocks"] if agg["blocks"] else 0.0
            agg["ber"] = agg["ber_sum"] / agg["n_images"] if agg["n_images"] else 0.0
            agg["bit_accuracy"] = agg["bit_acc_sum"] / agg["n_images"] if agg["n_images"] else 0.0
            agg["iterations_mean"] = agg["iter_sum"] / agg["blocks"] if agg["blocks"] else 0.0
            agg["erased_token_fraction"] = agg["erased_tokens"] / agg["n_tokens_sum"] if agg["n_tokens_sum"] else 0.0
            if "channel_uses_per_px_sum" in agg:
                agg["channel_uses_per_px"] = agg["channel_uses_per_px_sum"] / agg["n_images"] if agg["n_images"] else 0.0
                agg.pop("channel_uses_per_px_sum", None)
            if "stage_erased_count" in agg:
                agg["stage_erased_fraction"] = [
                    x / agg["n_tokens_sum"] if agg["n_tokens_sum"] else 0.0
                    for x in agg["stage_erased_count"]
                ]
            agg.pop("ber_sum", None); agg.pop("bit_acc_sum", None); agg.pop("iter_sum", None)
            agg.pop("n_tokens_sum", None)
            out.write(json.dumps({"ok": True, "images": results, "stats": agg}) + "\n")
            out.flush()
            continue
        out.write(json.dumps({"ok": False, "error": f"unknown op {op}"}) + "\n")
        out.flush()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--serve":
        serve()
    else:
        # single-shot test mode (one JSON request per line, then exit after EOF)
        serve()
