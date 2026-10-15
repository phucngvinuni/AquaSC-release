"""Checklist §1 evidence: header/mask/CRC/round-trip for the three AquaSC operating points.

Run (sionna venv, light):  taskset -c 0-7 __SIONNA_PY__ roundtrip_check.py
Independent cross-checks use only python stdlib (zlib, binascii, itertools, math) + numpy.
"""
import binascii, itertools, json, math, sys, zlib
from pathlib import Path
import numpy as np

ROOT = Path("__AQUASC_ROOT__")
sys.path.insert(0, str(ROOT / "experiments" / "matched5"))
sys.path.insert(0, str(ROOT))
import bitstream as bs                      # noqa: E402
import phy                                   # noqa: E402

out = {}

# ---- 1.1 header field table -------------------------------------------------
fields = bs.HEADER_FIELDS
pos = 0
for name, off, w in fields:
    assert off == pos, (name, off, pos)
    pos += w
assert pos == bs.HEADER_LEN == 152
out["header_fields"] = [(n, o, w) for n, o, w in fields]
out["header_total_bits"] = pos
# bit ordering: MSB-first. build a header and read raw bytes independently.
h = bs.build_header(2, 0, 3, 784, 63, 313, 2576)
hb = np.packbits(h.astype(np.uint8)).tobytes()
assert hb[:4] == b"AQSC", hb[:4]
assert hb[4] == 2 and hb[5] == 0 and hb[6] == 3 and hb[7] == 0
assert int.from_bytes(hb[8:10], "big") == 784 and int.from_bytes(hb[10:12], "big") == 63
assert int.from_bytes(hb[12:14], "big") == 313 and int.from_bytes(hb[14:17], "big") == 2576
assert binascii.crc_hqx(hb[:17], 0xFFFF) == int.from_bytes(hb[17:19], "big")  # independent CCITT-FALSE
out["header_independent_check"] = "magic/version/config/mask_mode/reserved/n_tokens/k_high/mask_bits/payload_bits big-endian MSB-first; CRC16 = binascii.crc_hqx(.,0xFFFF)"

# ---- 1.3 CRC check values ---------------------------------------------------
def bits_of(b):
    return np.unpackbits(np.frombuffer(b, np.uint8))
assert bs.crc16_ccitt(bits_of(b"123456789")) == 0x29B1
assert bs.crc32_ieee(bits_of(b"\x00\x00\x00\x00")) == 0x2144DF1C
assert bs.crc32_ieee(bits_of(b"123456789")) == zlib.crc32(b"123456789") == 0xCBF43926
out["crc"] = {"crc16": "CCITT-FALSE poly 0x1021 init 0xFFFF MSB-first no-reflect xorout 0 check('123456789')=0x29B1",
              "crc32": "IEEE 802.3 reflected poly 0xEDB88320 init 0xFFFFFFFF xorout 0xFFFFFFFF; bits packed MSB-first into bytes, final partial byte zero-padded on the right; check('123456789')=0xCBF43926, check(00000000)=0x2144DF1C",
              "crc32_coverage": "header+mask+index (everything before the 32 CRC bits) in the JointFrame (phy.pack_joint_frame)",
              "crc16_coverage": "first 136 header bits"}
rng = np.random.default_rng(0)
for L in (3041, 1478, 736):   # odd bit lengths -> partial last byte
    b = rng.integers(0, 2, L, dtype=np.uint8)
    ref = zlib.crc32(np.packbits(b).tobytes())      # packbits right-pads zeros == the module's convention
    assert bs.crc32_ieee(b) == ref
out["crc32_partial_byte_padding_verified"] = True

# ---- 1.2 combinadic convention ---------------------------------------------
n, k = 9, 4
subs = sorted(itertools.combinations(range(n), k), key=lambda s: tuple(reversed(s)))  # colex order
for r, s in enumerate(subs):
    assert bs.subset_rank(np.array(s)) == r
    assert tuple(bs.subset_unrank(r, n, k)) == s
out["rank_convention"] = "rank = sum_i C(s_i, i+1), s_0<s_1<..<s_{k-1} ascending 0-indexed token positions = colexicographic index among k-subsets of {0..n-1}; verified exhaustively n=9,k=4"
for kk in (24, 32, 63):
    for _ in range(50):
        s = np.sort(rng.choice(784, kk, replace=False))
        r = bs.subset_rank(s)
        assert 0 <= r < math.comb(784, kk) and np.array_equal(bs.subset_unrank(r, 784, kk), s)
    out[f"mask_bits_K{kk}"] = bs.enum_mask_bits(784, kk)
out["mask_bits_note"] = "mask field = enum_mask_bits(784,K)=ceil(log2 C(784,K)) bits, rank written MSB-first; JointFrame always mask_mode=3"

# ---- 1.4 round trip per operating point -------------------------------------
OPS = {
    "headline_K63_3x6_low2": ("cache_test_v1", dict(n_stages=3, stage_bits=6, b_low=2, k_high=63)),
    "r061_K32_3x4_low1":     ("cache_test_r061_bestmap_v1", dict(n_stages=3, stage_bits=4, b_low=1, k_high=32)),
    "r030_K24_3x6_low0":     ("cache_test_r030_bestmap_v1", dict(n_stages=3, stage_bits=6, b_low=0, k_high=24)),
}
res = {}
for name, (cache, cfg) in OPS.items():
    d = np.load(ROOT / "experiments/matched5/runs" / cache / "source.npz", allow_pickle=True)
    toks, masks = d["tokens"], d["high_mask"]
    ns, sb, bl, K = cfg["n_stages"], cfg["stage_bits"], cfg["b_low"], cfg["k_high"]
    info = phy.joint_frame_info_bits(ns, sb, bl, K)
    mb = bs.enum_mask_bits(784, K)
    idx_bits = ns * K * sb + (784 - K) * bl
    assert info == 152 + mb + idx_bits + 32
    n_ok = n_len = n_mask = n_tok = n_hdr = n_stage_major = 0
    for i in range(toks.shape[0]):
        fr = phy.pack_joint_frame(toks[i], masks[i], ns, sb, bl, K)
        n_len += (len(fr) == info)
        hdr = bs.parse_header(fr[:152])
        n_hdr += bool(hdr["header_crc_ok"] and hdr["mask_mode"] == 3 and hdr["k_high"] == K
                      and hdr["mask_bits"] == mb and hdr["payload_bits"] == idx_bits and hdr["version"] == 2)
        # boundaries: stage-major: first K*sb bits after the mask = stage-1 indices of the K high tokens in ascending token order
        hi = np.where(masks[i])[0]
        s1_expected = (toks[i][hi] >> ((ns - 1) * sb)) & ((1 << sb) - 1)
        s1_bits = fr[152 + mb:152 + mb + K * sb].reshape(K, sb)
        s1_got = (s1_bits * (1 << np.arange(sb - 1, -1, -1))).sum(1)
        n_stage_major += bool(np.array_equal(s1_expected, s1_got))
        crc_ok = bs.crc32_ieee(fr[:-32]) == bs.bits_to_int(fr[-32:])
        vals, mask, ok = phy.parse_joint_frame(fr, ns, sb, bl, K)
        n_ok += bool(ok and crc_ok)
        n_mask += bool(np.array_equal(mask, masks[i]))
        # tokens: high = full packed value; low = b_low-bit value; compare element-wise
        n_tok += bool(np.array_equal(vals, toks[i]))
    # corruption must be rejected (one flipped bit in each segment)
    rej = {}
    fr = phy.pack_joint_frame(toks[0], masks[0], ns, sb, bl, K)
    for seg, p in (("header", 10), ("mask", 152 + mb // 2), ("index", 152 + mb + idx_bits // 2), ("crc", info - 5)):
        bad = fr.copy(); bad[p] ^= 1
        rej[seg] = (not phy.parse_joint_frame(bad, ns, sb, bl, K)[2])
    short = phy.parse_joint_frame(fr[:-1], ns, sb, bl, K)[2]
    rej["truncated_by_1_bit"] = (not short)
    res[name] = dict(images=int(toks.shape[0]), frame_bits=int(info), segments=dict(header=152, mask=mb, index=idx_bits, crc32=32),
                     length_ok=int(n_len), header_ok=int(n_hdr), stage_major_ok=int(n_stage_major), crc_parse_ok=int(n_ok),
                     mask_roundtrip_exact=int(n_mask), token_roundtrip_exact=int(n_tok), corruption_rejected=rej)
    print(name, json.dumps(res[name]))
out["roundtrip"] = res

# ---- 1.2b independent INDEX layout: [stage1 of high tokens][stage2][stage3][low tokens], tokens ascending, MSB-first
lay = {}
for name, (cache, cfg) in OPS.items():
    d = np.load(ROOT / "experiments/matched5/runs" / cache / "source.npz", allow_pickle=True)
    ns, sb, bl, K = cfg["n_stages"], cfg["stage_bits"], cfg["b_low"], cfg["k_high"]
    mb = bs.enum_mask_bits(784, K)
    bad = 0
    for i in range(0, 432, 9):
        t, m = d["tokens"][i], d["high_mask"][i]
        hi, lo = np.where(m)[0], np.where(~m)[0]
        exp = []
        for s in range(ns):                                   # stage-major; stage 1 = most-significant sb bits of the packed value
            sh = (ns - 1 - s) * sb
            for tok in hi:
                v = (int(t[tok]) >> sh) & ((1 << sb) - 1)
                exp += [(v >> b) & 1 for b in range(sb - 1, -1, -1)]
        for tok in lo:
            v = int(t[tok]) & ((1 << bl) - 1) if bl else 0
            exp += [(v >> b) & 1 for b in range(bl - 1, -1, -1)]
        fr = phy.pack_joint_frame(t, m, ns, sb, bl, K)
        bad += not np.array_equal(np.array(exp, np.uint8), fr[152 + mb:-32])
    lay[name] = {"images_checked": len(range(0, 432, 9)), "mismatches": int(bad)}
    assert bad == 0
out["index_layout_independent"] = lay
out["index_layout"] = "INDEX = [stage1 x K high tokens][stage2 x K][stage3 x K][low x (784-K) low tokens]; tokens in ascending position; each value MSB-first; stage 1 = most-significant sb bits of the packed value; low field absent when b_low=0"
print("index layout independent check:", lay)
(Path(__file__).parent / "roundtrip_results.json").write_text(json.dumps(out, indent=2, default=str))
print("ALL ASSERTIONS PASSED; wrote roundtrip_results.json")
