"""Unit tests for digital_pipeline.py (run with the sionna venv python).

Usage:
    __SIONNA_PY__ tests/test_digital_pipeline.py
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repo root (core modules)
import numpy as np
import digital_pipeline as dp

rng = np.random.default_rng(42)
FAILED = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILED.append(name)


def test_crc():
    bits = rng.integers(0, 2, size=992).astype(np.uint8)
    c = dp.crc32_ieee(bits)
    check("crc32_ieee deterministic", c == dp.crc32_ieee(bits))
    bits2 = bits.copy()
    bits2[5] ^= 1
    check("crc32_ieee catches single-bit flip",
          dp.crc32_ieee(bits2) != c)
    h = dp.crc16_ccitt(bits[:136])
    bits3 = bits[:136].copy()
    bits3[0] ^= 1
    check("crc16_ccitt catches flip", dp.crc16_ccitt(bits3) != h)
    # known vector: CRC-32 of empty bit array (no bits) == 0x00000000 per IEEE? empty -> init
    check("crc32 empty = 0x00000000", dp.crc32_ieee(np.zeros(0, np.uint8)) == 0x00000000)
    check("crc32 all-zero 32 bits = zlib 0x2144DF1C",
          dp.crc32_ieee(np.zeros(32, np.uint8)) == 0x2144DF1C)


def test_enumerative_mask():
    n, k = 784, 63
    for _ in range(20):
        subset = np.sort(rng.choice(n, size=k, replace=False))
        rank = dp.subset_rank(subset)
        check(f"rank<2^313 (rank={rank})", rank < 2 ** dp.N_MASK_BITS_ENUM)
        back = dp.subset_unrank(rank, n, k)
        check("unrank(rank(subset)) == subset",
              np.array_equal(back, subset))
    check("log2 C(784,63) = 312.038 > 312 (313 bits required)",
          2 ** 312 < dp.math.comb(784, 63) < 2 ** 313)
    # boundary ranks
    check("rank 0 = first k", np.array_equal(dp.subset_unrank(0, 784, 63), np.arange(63)))
    check("rank max = last k",
          np.array_equal(dp.subset_unrank(dp.math.comb(784, 63) - 1, 784, 63),
                         np.arange(784 - 63, 784)))
    # explicit mask round trip
    hm = np.zeros(784, bool)
    hm[rng.choice(784, 63, replace=False)] = True
    check("explicit mask decode", np.array_equal(
        dp.mask_decode(dp.mask_bits_explicit(hm), 0, 784, 63), hm))
    check("enumerative mask decode", np.array_equal(
        dp.mask_decode(dp.mask_bits_enumerative(hm), 3, 784, 63), hm))


def test_frame_roundtrip():
    cfg = dp.CONFIGS[dp.CONFIG_18_2]
    for mask_mode in (dp.MASK_MODE_EXPLICIT, dp.MASK_MODE_ENUM):
        tv = rng.integers(0, 2 ** cfg["b_low"], size=784)
        high_idx = rng.choice(784, cfg["k_high"], replace=False)
        hm = np.isin(np.arange(784), high_idx)
        tv[high_idx] = rng.integers(0, 2 ** cfg["b_high"], size=cfg["k_high"])
        f = dp.build_frame(tv, hm, mask_mode=mask_mode)
        check(f"frame mode {mask_mode}: header len 152",
              f["header"].size == dp.HEADER_LEN)
        check(f"frame mode {mask_mode}: index len = K*18 + (784-K)*2 = 2576",
              f["index"].size == 2576)
        check(f"frame mode {mask_mode}: mask len",
              f["mask"].size == (784 if mask_mode == 0 else dp.N_MASK_BITS_ENUM))
        check(f"frame mode {mask_mode}: seg_a padded to 1024",
              f["segment_a"].size % 1024 == 0 and f["n_blocks_a"] == 1)
        check(f"frame mode {mask_mode}: seg_b = 3 blocks",
              f["segment_b"].size == 3072 and f["n_blocks_b"] == 3)
        # clean parse
        p = dp.parse_frame([f["segment_a"]], [f["segment_b"][0:1024], f["segment_b"][1024:2048], f["segment_b"][2048:]])
        check(f"frame mode {mask_mode}: header_ok", p["header_ok"])
        check(f"frame mode {mask_mode}: crc_ok", p["payload_crc_ok"])
        check(f"frame mode {mask_mode}: tokens equal",
              np.array_equal(np.asarray([int(v) for v in p["tokens"]]), tv))
        check(f"frame mode {mask_mode}: high_mask equal",
              np.array_equal(p["high_mask"], hm))
        # header with all-one and all-zero boundary token values
        tv0 = np.zeros(784, np.int64)
        tv0[high_idx] = (1 << cfg["b_high"]) - 1
        f0 = dp.build_frame(tv0, hm, mask_mode=mask_mode)
        p0 = dp.parse_frame([f0["segment_a"]],
                            [f0["segment_b"][0:1024], f0["segment_b"][1024:2048], f0["segment_b"][2048:]])
        check(f"frame mode {mask_mode}: boundary values roundtrip",
              np.array_equal(np.asarray([int(v) for v in p0["tokens"]]), tv0))
        # header block erased -> frame lost
        pl = dp.parse_frame([None], [f["segment_b"][0:1024]])
        check(f"frame mode {mask_mode}: header erased -> frame_lost", pl["frame_lost"])
        # middle payload block erased -> payload crc fails, all tokens erased
        pe = dp.parse_frame([f["segment_a"]], [f["segment_b"][0:1024], None, f["segment_b"][2048:]])
        check(f"frame mode {mask_mode}: payload block erased -> all tokens erased",
              pe["erased_tokens"] == 784 and not pe["payload_crc_ok"])
        # corrupted header -> crc16 catches
        bad = f["segment_a"].copy()
        bad[40] ^= 1
        pb = dp.parse_frame([bad], [f["segment_b"][0:1024], f["segment_b"][1024:2048], f["segment_b"][2048:]])
        check(f"frame mode {mask_mode}: corrupted header detected", not pb["header_ok"])


def test_concealment():
    tv = rng.integers(0, 2 ** 18, size=784)
    erased = rng.choice(784, 200, replace=False)
    tokens = [int(v) for v in tv]
    for e in erased:
        tokens[e] = None
    filled, st = dp.conceal_tokens(tokens)
    check("conceal: all tokens filled", np.isnan(filled.astype(float)).sum() == 0)
    check("conceal: stats", st["erased_before"] == 200 and st["erased_after"] == 0)
    for e in erased:
        r, c = e // 28, e % 28
        nb = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < 28 and 0 <= cc < 28 and rr * 28 + cc not in erased:
                nb.append(float(tv[rr * 28 + cc]))
        # iterative fill converges to mean of neighbours after a few passes
        check(f"conceal: token {e} finite", np.isfinite(float(filled[r, c])))
    # all erased -> fallback 0
    filled0, st0 = dp.conceal_tokens([None] * 784)
    check("conceal: all-erased -> 0 fill",
          st0["fallback_used"] == 1 and (filled0 == 0).all())
    # neighbours mean for a single erased centre
    tokens1 = [int(v) for v in tv]
    tokens1[14 * 28 + 14] = None
    f1, _ = dp.conceal_tokens(tokens1)
    expect = np.mean([float(tv[14 * 28 + 13]), float(tv[14 * 28 + 15]),
                      float(tv[13 * 28 + 14]), float(tv[15 * 28 + 14])])
    check("conceal: single centre = mean of N4 (int truncation)",
          f1[14, 14] == int(expect), f"got={f1[14,14]} expect={expect}")


def test_patch16_n_tokens():
    """G3 (plan v5.9): n_tokens is parameterized throughout digital_pipeline.py,
    not hardcoded to 784 -> patch-16 (196 tokens, 14x14 grid) must round-trip
    identically to patch-8 (784 tokens, 28x28 grid). This is what unblocks the
    RUN-2 (patch-16, OP16_256d2) low-rate curve point through the digital chain."""
    n, k = 196, 63
    cfg = dp.CONFIGS[dp.CONFIG_18_2]
    # enumerative mask rank/unrank at n=196
    for _ in range(10):
        subset = np.sort(rng.choice(n, size=k, replace=False))
        rank = dp.subset_rank(subset)
        bits_needed = dp.enum_mask_bits(n, k)
        check(f"patch16: rank<2^{bits_needed} (rank={rank})", rank < 2 ** bits_needed)
        back = dp.subset_unrank(rank, n, k)
        check("patch16: unrank(rank(subset)) == subset", np.array_equal(back, subset))
    check("patch16: enum_mask_bits(196,63) == 174",
          dp.enum_mask_bits(196, 63) == 174, f"got {dp.enum_mask_bits(196, 63)}")

    # frame build/parse round-trip at n=196
    tv = rng.integers(0, 2 ** cfg["b_low"], size=n)
    high_idx = rng.choice(n, cfg["k_high"], replace=False)
    hm = np.isin(np.arange(n), high_idx)
    tv[high_idx] = rng.integers(0, 2 ** cfg["b_high"], size=cfg["k_high"])
    f = dp.build_frame(tv, hm, mask_mode=dp.MASK_MODE_ENUM, n_tokens=n)
    check("patch16: index len = K*18 + (196-K)*2 = 1400",
          f["index"].size == 63 * 18 + (196 - 63) * 2, f"got {f['index'].size}")
    p = dp.parse_frame([f["segment_a"]],
                       [f["segment_b"][i * 1024:(i + 1) * 1024] for i in range(f["n_blocks_b"])])
    check("patch16: header_ok", p["header_ok"])
    check("patch16: crc_ok", p["payload_crc_ok"])
    check("patch16: tokens equal",
          np.array_equal(np.asarray([int(v) for v in p["tokens"]]), tv))

    # N4 concealment on a 14x14 grid
    tokens = [int(v) for v in tv]
    erased = rng.choice(n, 40, replace=False)
    for e in erased:
        tokens[e] = None
    filled, st = dp.conceal_tokens(tokens, n_tokens=n, grid_h=14, grid_w=14)
    check("patch16 conceal: shape 14x14", filled.shape == (14, 14))
    check("patch16 conceal: all filled, no fallback",
          st["erased_before"] == 40 and st["erased_after"] == 0 and st["fallback_used"] == 0)

    # end-to-end one_image_demo at n_tokens=196, grid 14x14 (smoke, no crash)
    res = dp.one_image_demo(snr_list=(20.0,), seed=7, frames_per_snr=5,
                            n_tokens=196, grid_h=14, grid_w=14)
    check("patch16 one_image_demo: ran without error", len(res) == 1 and res[0]["frames"] == 5)


def test_block_roundtrip_20db():
    payload = rng.integers(0, 2, size=992).astype(np.uint8)
    info = dp.make_block_payload(payload)
    out, stats = dp.transmit_and_receive_block(info, snr_db=20.0, rate=0.5,
                                               csi_mode="perfect", seed=7)
    check("20dB round trip: payload returned", out is not None)
    if out is not None:
        check("20dB round trip: bits equal",
              np.array_equal(out, payload), f"errs={np.sum(out != payload)}")
    check("20dB round trip: stats", stats["blocks"] == 1 and stats["bit_accuracy"] == 1.0)


def test_block_roundtrip_1000_blocks():
    n = 1000
    payload = rng.integers(0, 2, size=n * 992).astype(np.uint8)
    out, stats = dp.simulate_digital_pipeline(payload, 20.0, rate=0.5,
                                              csi_mode="perfect", seed=3)
    bler = stats["failed"] / stats["blocks"]
    check("1000 blocks @20dB: BLER ~ 0",
          bler < 0.05, f"bler={bler:.4f} failed={stats['failed']}")
    ok_bits = out != -1
    check("1000 blocks @20dB: bit_accuracy ~ 1",
          stats["bit_accuracy"] > 0.999, f"acc={stats['bit_accuracy']:.5f}")
    # outage law: with gamma_th ~ 4.9 (6.9 dB) measured P_out(20 dB) ~ 0.048;
    # assert the erased fraction is outage-consistent (not a broken chain)
    check("1000 blocks @20dB: erasures outage-consistent (< 8%)",
          stats["erased_tokens"] / (n * 992) < 0.08,
          f"erased_frac={stats['erased_tokens'] / (n * 992):.4f} bler={bler:.4f}")


def test_crc_catches_injection():
    payload = rng.integers(0, 2, size=992).astype(np.uint8)
    info = dp.make_block_payload(payload)
    info[10] ^= 1
    info[900] ^= 1
    out, stats = dp.transmit_and_receive_block(info, snr_db=25.0, rate=0.5,
                                               csi_mode="perfect", seed=11)
    check("CRC catches injected errors", out is None and stats["failed"] == 1)
    check("CRC erasure stats", stats["erased_tokens"] == 992)


def test_pilot_and_rates():
    payload = rng.integers(0, 2, size=992).astype(np.uint8)
    info = dp.make_block_payload(payload)
    for rate in (1 / 3, 1 / 2, 2 / 3):
        out, stats = dp.transmit_and_receive_block(info, snr_db=25.0, rate=rate,
                                                   csi_mode="pilot", seed=13)
        check(f"pilot CSI rate {rate} @25dB round trip",
              out is not None and np.array_equal(out, payload))
    out, stats = dp.transmit_and_receive_block(info, snr_db=0.0, rate=1 / 3,
                                               csi_mode="pilot", seed=13)
    check("rate 1/3 pilot @0dB survives (threshold ~4 dB AWGN)",
          out is not None, f"failed={stats['failed']}")


def test_frame_end_to_end_30db():
    """Transmit the §3.3 frame over the digital chain (UEP rates) at 20 dB."""
    import tensorflow as tf
    cfg = dp.CONFIGS[dp.CONFIG_18_2]
    tv = rng.integers(0, 2 ** cfg["b_low"], size=784)
    high_idx = rng.choice(784, cfg["k_high"], replace=False)
    hm = np.isin(np.arange(784), high_idx)
    tv[high_idx] = rng.integers(0, 2 ** cfg["b_high"], size=cfg["k_high"])
    f = dp.build_frame(tv, hm, mask_mode=dp.MASK_MODE_ENUM)
    rng2 = np.random.default_rng(5)
    enc_a, dec_a = dp.get_coder(1 / 3)
    enc_b, dec_b = dp.get_coder(1 / 2)
    h_a = (rng2.standard_normal(f["n_blocks_a"]) + 1j * rng2.standard_normal(f["n_blocks_a"])) / np.sqrt(2)
    h_b = (rng2.standard_normal(f["n_blocks_b"]) + 1j * rng2.standard_normal(f["n_blocks_b"])) / np.sqrt(2)
    seg_a, seg_b = [], []
    for i in range(f["n_blocks_a"]):
        blk = f["segment_a"][i * 1024:(i + 1) * 1024]
        cw = enc_a(tf.convert_to_tensor(blk.astype(np.float32)[None, :], tf.float32)).numpy()
        llr, _ = dp.channel_pass(cw, 30.0, np.asarray([h_a[i]]), csi_mode="perfect", rng=rng2)
        u, ok, _ = dec_a.incremental_decode(llr)
        seg_a.append(u[0].astype(np.uint8) if ok[0] else None)
    for i in range(f["n_blocks_b"]):
        blk = f["segment_b"][i * 1024:(i + 1) * 1024]
        cw = enc_b(tf.convert_to_tensor(blk.astype(np.float32)[None, :], tf.float32)).numpy()
        llr, _ = dp.channel_pass(cw, 30.0, np.asarray([h_b[i]]), csi_mode="perfect", rng=rng2)
        u, ok, _ = dec_b.incremental_decode(llr)
        seg_b.append(u[0].astype(np.uint8) if ok[0] else None)
    p = dp.parse_frame(seg_a, seg_b)
    check("frame e2e @30dB: header ok", p["header_ok"] and not p["frame_lost"])
    check("frame e2e @30dB: crc ok", p["payload_crc_ok"])
    got = np.asarray([int(v) for v in p["tokens"]])
    check("frame e2e @30dB: tokens exact", np.array_equal(got, tv),
          f"diff={int(np.sum(got != tv))}")


def main():
    test_crc()
    test_enumerative_mask()
    test_frame_roundtrip()
    test_concealment()
    test_patch16_n_tokens()
    test_block_roundtrip_20db()
    test_block_roundtrip_1000_blocks()
    test_crc_catches_injection()
    test_pilot_and_rates()
    test_frame_end_to_end_30db()
    if FAILED:
        print(f"\n{len(FAILED)} FAILED: {FAILED}")
        raise SystemExit(1)
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
