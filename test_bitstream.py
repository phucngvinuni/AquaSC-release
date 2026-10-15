"""Bitstream layer tests (R3 #1). Run with:

    __TORCH_PY__ test_bitstream.py

Part 1: pure-numpy round-trip / boundary / CRC / padding / BPP asserts.
Part 2: cross-check vs digital_pipeline.py (shells out to the sionna venv) --
        verifies header/mask/CRC/padding are bit-identical and the index field
        is a pure per-token -> per-stage permutation (the §3.3 fix).
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import bitstream as bs

SIONNA_PY = "__SIONNA_PY__"
N_FAIL = [0]


def check(cond, msg):
    if not cond:
        N_FAIL[0] += 1
        print(f"FAIL: {msg}")
    else:
        print(f"ok: {msg}")


def rng(seed=42):
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------
# Part 1 -- pure numpy tests
# --------------------------------------------------------------------------
print("=" * 70)
print("PART 1: pure-numpy round-trip / boundary / CRC / padding / BPP")
print("=" * 70)

# 1. header round-trip
h = bs.build_header(1, 0, 0, 784, 63, 784, 2576)
check(h.size == bs.HEADER_LEN == 152, f"header is 152 bits (got {h.size})")
ph = bs.parse_header(h)
check(ph["magic"] == 0x41515343, f"magic == 'AQSC' ({hex(ph['magic'])})")
check(ph["version"] == 1 and ph["config_id"] == 0 and ph["mask_mode"] == 0,
      "version/config_id/mask_mode parsed")
check(ph["reserved"] == 0, "reserved == 0")
check(ph["n_tokens"] == 784 and ph["k_high"] == 63, "n_tokens/k_high parsed")
check(ph["mask_bits"] == 784 and ph["payload_bits"] == 2576, "mask_bits/payload_bits parsed")
check(ph["header_crc_ok"], "header CRC valid")

# 2. header field widths sum to 152
wsum = sum(w for _, _, w in bs.HEADER_FIELDS)
check(wsum == 152, f"header field widths sum to 152 (got {wsum})")

# 3. CRC check vectors
bits_123456789 = np.array([int(c) for c in "".join(f"{ord(x):08b}" for x in "123456789")],
                          dtype=np.uint8)
check(bs.crc16_ccitt(bits_123456789) == 0x29B1, "CRC-16('123456789') == 0x29B1")
check(bs.crc32_ieee(np.zeros(32, dtype=np.uint8)) == 0x2144DF1C,
      "CRC-32(4 zero bytes) == 0x2144DF1C")

# 4. subset rank/unrank round-trip (random + boundary)
for seed in (0, 1, 42):
    r = rng(seed)
    for k in (0, 1, 63, 784):
        sub = np.sort(r.choice(784, size=k, replace=False))
        rk = bs.subset_rank(sub)
        back = bs.subset_unrank(rk, 784, k)
        check(np.array_equal(sub, back), f"subset rank/unrank round-trip k={k} seed={seed}")

# 5. mask round-trip (both modes)
for mode in (bs.MASK_MODE_EXPLICIT, bs.MASK_MODE_ENUM):
    r = rng(7)
    high = r.choice(784, size=63, replace=False)
    high_mask = np.isin(np.arange(784), high)
    if mode == bs.MASK_MODE_EXPLICIT:
        m = bs.mask_bits_explicit(high_mask)
        check(m.size == 784, "explicit mask is 784 bits")
    else:
        m = bs.mask_bits_enumerative(high_mask)
        check(m.size == bs.N_MASK_BITS_ENUM == 313, "enumerative mask is 313 bits")
    back = bs.mask_decode(m, mode, 784, 63)
    check(np.array_equal(back, high_mask), f"mask round-trip mode {mode}")

# enumerative rank fits in 313 bits but not 312 (max rank = C(784,63)-1)
max_rank = bs.subset_rank(np.arange(721, 784))
check(max_rank == int(__import__("math").comb(784, 63)) - 1, "max rank == C(784,63)-1")
check(max_rank < 2 ** 313, "max combinadic rank fits 313 bits")
check(max_rank >= 2 ** 312, "max combinadic rank exceeds 312 bits (why 313 is needed)")

# 6. pack/unpack rvq round-trip boundary cases
def roundtrip_stages(stage1, stage2, stage3, low, high_mask):
    payload = bs.pack_rvq_indices(stage1, stage2, stage3, low, high_mask)
    return payload, *bs.unpack_rvq_indices(payload, high_mask)

# 6a. all-low tokens (K=0)
hm = np.zeros(784, dtype=bool)
low = rng(1).integers(0, 4, size=784)
_, _, _, _, lo = roundtrip_stages(np.array([], dtype=np.int64), np.array([], dtype=np.int64),
                                   np.array([], dtype=np.int64), low, hm)
check(np.array_equal(lo, low), "all-low tokens round-trip (K=0)")

# 6b. all-high tokens (K=784)
hm = np.ones(784, dtype=bool)
s1 = rng(2).integers(0, 64, size=784)
s2 = rng(3).integers(0, 64, size=784)
s3 = rng(4).integers(0, 64, size=784)
_, r1, r2, r3, _ = roundtrip_stages(s1, s2, s3, np.array([], dtype=np.int64), hm)
check(np.array_equal(r1, s1) and np.array_equal(r2, s2) and np.array_equal(r3, s3),
      "all-high tokens round-trip (K=784)")

# 6c. token value 0 everywhere (K=63)
hm = np.zeros(784, dtype=bool); hm[:63] = True
payload, r1, r2, r3, lo = roundtrip_stages(np.zeros(63, dtype=np.int64),
                                           np.zeros(63, dtype=np.int64),
                                           np.zeros(63, dtype=np.int64),
                                           np.zeros(721, dtype=np.int64), hm)
check(np.all(r1 == 0) and np.all(r2 == 0) and np.all(r3 == 0) and np.all(lo == 0),
      "all-zero token values round-trip")
check(payload.size == 2576, f"payload is 2576 bits (got {payload.size})")

# 6d. max value (stage=63, low=3)
s1 = np.full(63, 63); s2 = np.full(63, 63); s3 = np.full(63, 63); low = np.full(721, 3)
tv = bs.stages_to_token_values(s1, s2, s3, low, hm)
check(np.all(tv[hm] == 262143) and np.all(tv[~hm] == 3), "max token value 262143 / 3")
_, r1, r2, r3, lo = roundtrip_stages(s1, s2, s3, low, hm)
check(np.array_equal(r1, s1) and np.array_equal(lo, low), "max-value round-trip")

# 6e. K=63 high at both extremes
for hi in (np.arange(63), np.arange(721, 784)):
    hm2 = np.zeros(784, dtype=bool); hm2[hi] = True
    s1 = rng(5).integers(0, 64, size=63)
    s2 = rng(6).integers(0, 64, size=63)
    s3 = rng(7).integers(0, 64, size=63)
    lo = rng(8).integers(0, 4, size=721)
    _, r1, r2, r3, l2 = roundtrip_stages(s1, s2, s3, lo, hm2)
    ok = (np.array_equal(r1, s1) and np.array_equal(r2, s2)
          and np.array_equal(r3, s3) and np.array_equal(l2, lo))
    check(ok, f"K=63 high at {'start' if hi[0] == 0 else 'end'} round-trip")

# 7. token_values <-> stages conversion round-trip
hm = np.zeros(784, dtype=bool); hm[rng(9).choice(784, size=63, replace=False)] = True
tv = rng(10).integers(0, 4, size=784)
tv[hm] = rng(11).integers(0, 262144, size=63)
s1, s2, s3, low = bs.token_values_to_stages(tv, hm)
tv2 = bs.stages_to_token_values(s1, s2, s3, low, hm)
check(np.array_equal(tv, tv2), "token_values <-> stages round-trip")

# 8. full build_frame / parse_frame round-trip (both mask modes)
for mode in (bs.MASK_MODE_EXPLICIT, bs.MASK_MODE_ENUM):
    fr = bs.build_frame(s1, s2, s3, low, hm, mask_mode=mode)
    check(fr["index"].size == 2576, f"frame index 2576 bits (mode {mode})")
    check(fr["header"].size == 152, f"frame header 152 bits (mode {mode})")
    parsed = bs.parse_frame(fr["frame_unpadded"])
    check(parsed["header_ok"], f"frame parse header_ok (mode {mode})")
    check(parsed["payload_crc_ok"], f"frame parse payload_crc_ok (mode {mode})")
    check(np.array_equal(parsed["high_mask"], hm), f"frame parse mask exact (mode {mode})")
    check(np.array_equal(parsed["stage1_idx"], s1) and np.array_equal(parsed["stage2_idx"], s2)
          and np.array_equal(parsed["stage3_idx"], s3) and np.array_equal(parsed["low_idx"], low),
          f"frame parse stage indices exact (mode {mode})")
    check(np.array_equal(parsed["token_values"], tv), f"frame parse token_values exact (mode {mode})")
    check(parsed["erased_tokens"] == 0, f"frame parse zero erased (mode {mode})")

# 9. CRC detects a single flipped bit
fr = bs.build_frame(s1, s2, s3, low, hm, mask_mode=bs.MASK_MODE_EXPLICIT)
uf = fr["frame_unpadded"].copy()
uf_hdr = uf.copy(); uf_hdr[10] ^= 1
check(not bs.parse_frame(uf_hdr)["header_ok"], "single flipped header bit detected (CRC-16)")
uf_pay = uf.copy(); uf_pay[bs.HEADER_LEN + 784 + 100] ^= 1
check(not bs.parse_frame(uf_pay)["payload_crc_ok"], "single flipped payload bit detected (CRC-32)")

# 10. padding / depadding exact
for mode in (bs.MASK_MODE_EXPLICIT, bs.MASK_MODE_ENUM):
    fr = bs.build_frame(s1, s2, s3, low, hm, mask_mode=mode)
    check(fr["frame"].size % bs.K_CB == 0, f"frame padded to k_cb multiple (mode {mode})")
    check(fr["frame_unpadded"].size == 152 + fr["mask"].size + 2576 + 32,
          f"unpadded frame length exact (mode {mode})")
    stripped = fr["frame"][:fr["frame_unpadded"].size]
    check(np.array_equal(stripped, fr["frame_unpadded"]), f"padding strip exact (mode {mode})")
    check(np.all(fr["frame"][fr["frame_unpadded"].size:] == 0), f"padding bits are zero (mode {mode})")

# 11. measured BPP
for mode, name in ((bs.MASK_MODE_EXPLICIT, "explicit"), (bs.MASK_MODE_ENUM, "enumerative")):
    d = bs.measure_bpp_breakdown(bs.CONFIG_18_2, hm, mask_mode=mode)
    check(d["index_bits"] == 2576, f"index_bits == 2576 ({name})")
    check(d["header_bits"] == 152, f"header_bits == 152 ({name})")
    check(d["mask_bits"] == (784 if mode == bs.MASK_MODE_EXPLICIT else 313),
          f"mask_bits == {'784' if mode == bs.MASK_MODE_EXPLICIT else '313'} ({name})")
    check(d["crc_bits"] == 32, f"crc_bits == 32 ({name})")
    check(d["emitted"] == d["total"] + d["padding"], f"emitted == total+padding ({name})")
    check(d["emitted"] % bs.K_CB == 0, f"emitted is k_cb multiple ({name})")
    print(f"    [{name}] header={d['header_bits']} mask={d['mask_bits']} index={d['index_bits']} "
          f"crc={d['crc_bits']} total={d['total']} padding={d['padding']} "
          f"emitted={d['emitted']} bpp={d['bpp']:.6f} (no-pad {d['bpp_no_padding']:.6f})")

print(f"\nPART 1 failures: {N_FAIL[0]}")

# --------------------------------------------------------------------------
# Part 2 -- cross-check vs digital_pipeline.py (sionna venv)
# --------------------------------------------------------------------------
print("=" * 70)
print("PART 2: cross-check vs digital_pipeline.py (sionna venv)")
print("=" * 70)

if not os.path.exists(SIONNA_PY):
    print("SKIP: sionna venv not found; cannot run cross-check")
else:
    r = rng(42)
    token_values = r.integers(0, 4, size=784).astype(np.int64)
    high = r.choice(784, size=63, replace=False)
    high_mask = np.isin(np.arange(784), high)
    token_values[high] = r.integers(0, 262144, size=63)

    data_dir = tempfile.mkdtemp(prefix="bitstream_xcheck_", dir="__AQUASC_TMP__")
    in_npz = os.path.join(data_dir, "inputs.npz")
    out_npz = os.path.join(data_dir, "dp_out.npz")
    np.savez(in_npz, token_values=token_values, high_mask=high_mask)

    helper = os.path.join(data_dir, "dp_build.py")
    with open(helper, "w") as f:
        f.write(f"""
import numpy as np, sys
sys.path.insert(0, {REPO!r})
from digital_pipeline import build_frame, crc16_ccitt, crc32_ieee
d = np.load({in_npz!r})
tv, hm = d['token_values'], d['high_mask']
probe16 = np.arange(40) % 2
probe32 = np.arange(256) % 2
out = {{}}
for mode in (0, 3):
    fr = build_frame(tv, hm, mask_mode=mode)
    out[f'hdr_{{mode}}'] = fr['header']
    out[f'mask_{{mode}}'] = fr['mask']
    out[f'index_{{mode}}'] = fr['index']
    out[f'crc_{{mode}}'] = fr['payload_crc']
out['crc16_probe'] = np.array([crc16_ccitt(probe16)])
out['crc32_probe'] = np.array([crc32_ieee(probe32)])
np.savez({out_npz!r}, **out)
print('dp build ok')
""")

    r1 = subprocess.run([SIONNA_PY, helper], capture_output=True, text=True)
    if r1.returncode != 0:
        print("FAIL: sionna helper failed:\n", r1.stderr)
        N_FAIL[0] += 1
    else:
        dp = np.load(out_npz)
        st = bs.token_values_to_stages(token_values, high_mask)
        for mode in (0, 3):
            fr = bs.build_frame(*st, high_mask, mask_mode=mode)
            check(np.array_equal(fr["header"], dp[f"hdr_{mode}"]),
                  f"header bit-identical vs digital_pipeline (mode {mode})")
            check(np.array_equal(fr["mask"], dp[f"mask_{mode}"]),
                  f"mask bit-identical vs digital_pipeline (mode {mode})")

        # CRC implementations cross-validated on identical probe inputs
        check(int(dp["crc16_probe"][0]) == bs.crc16_ccitt(np.arange(40) % 2),
              "crc16_ccitt identical to sionna impl")
        check(int(dp["crc32_probe"][0]) == bs.crc32_ieee(np.arange(256) % 2),
              "crc32_ieee identical to sionna impl")

        # index: same length; per-stage is a pure permutation of per-token
        dp_index = dp["index_0"]
        check(dp_index.size == 2576, "digital_pipeline index length 2576")
        # regroup digital_pipeline's per-token index into per-stage fields
        stage1_bits, stage2_bits, stage3_bits, low_bits = [], [], [], []
        pos = 0
        for t in range(784):
            w = 18 if high_mask[t] else 2
            chunk = dp_index[pos:pos + w]
            pos += w
            if high_mask[t]:
                stage1_bits.append(chunk[0:6])
                stage2_bits.append(chunk[6:12])
                stage3_bits.append(chunk[12:18])
            else:
                low_bits.append(chunk)
        regrouped = np.concatenate([np.concatenate(stage1_bits), np.concatenate(stage2_bits),
                                    np.concatenate(stage3_bits), np.concatenate(low_bits)])
        fr = bs.build_frame(*st, high_mask, mask_mode=0)
        check(np.array_equal(regrouped, fr["index"]),
              "index is pure per-token -> per-stage permutation (identical info, reordered)")
        # digital_pipeline index decodes to the same token values
        rec = np.zeros(784, dtype=np.int64)
        pos = 0
        for t in range(784):
            w = 18 if high_mask[t] else 2
            rec[t] = bs.bits_to_int(dp_index[pos:pos + w])
            pos += w
        check(np.array_equal(rec, token_values),
              "digital_pipeline index decodes to same token values")

        # bitstream payload_crc == crc32(its index); digital_pipeline's payload_crc
        # differs ONLY because the index ordering differs (documented §3.3 change)
        check(bs.crc32_ieee(fr["index"]) == bs.bits_to_int(fr["payload_crc"]),
              "bitstream payload_crc == crc32(index)")
        check(dp["crc_0"].size == 32, "digital_pipeline payload_crc is 32 bits")

    print("\nPART 2 done")

print(f"\nTOTAL failures: {N_FAIL[0]}")
sys.exit(1 if N_FAIL[0] else 0)
