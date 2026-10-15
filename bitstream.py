"""AquaSC-side bitstream layer (master plan R3 #1 / §3.3 frame format).

Wave 5, BITSTREAM agent -- R3 #1 (bitstream specification and the routing mask).

Pure numpy (no tensorflow/sionna) so the conda training env
(``siso-project-env``) can import it directly. This module is the *source-format*
counterpart of ``digital_pipeline.py`` (which lives in the sionna venv): it packs
and unpacks the same §3.3 frame and MUST match ``digital_pipeline.py`` bit-for-bit
on every shared field (header, mask, CRC-16, CRC-32, padding).

Two deliberate deviations from ``digital_pipeline.py`` (both escalated corrections,
see ``reports/wave4_gate.md`` and the Wave-4 CHANNEL report):

1. **Header is 152 bits, not 128.** The plan's own field list
   (32+8+8+8+8+16+16+16+24+16) sums to 152; we follow the field list.
2. **Per-stage RVQ index fields (plan §3.3 "currently unresolved").** The RVQ
   packs the 3 stage indices into ONE 18-bit integer per high token. ``bitstream.py``
   emits SEPARATE per-stage fields instead::

       INDEX = [stage1: K*6][stage2: K*6][stage3: K*6][low: (784-K)*2]

   so stage-1 can be protected more strongly than stages 2-3. Unequal error
   protection (UEP) split used by the channel layer::

       segment A (rate 1/3): HEADER | MASK | STAGE1
       segment B (rate 1/2): STAGE2 | STAGE3 | LOW | PAYLOAD_CRC-32

   ``digital_pipeline.py`` still emits the per-token 18-bit field; the cross-check
   (test_bitstream.py) verifies that header/mask/CRC/padding are bit-identical and
   that the index field is a pure permutation (per-token -> per-stage).

Layout (18/2b config, K=63, n_tokens=784, b_high=18 as 3x6, b_low=2)::

    HEADER      152 b  (magic 32 | version 8 | config_id 8 | mask_mode 8 |
                       reserved 8 | n_tokens 16 | k_high 16 | mask_bits 16 |
                       payload_bits 24 | header_crc 16)
    MASK        mode 0: 784 b explicit  |  mode 3: 313 b enumerative rank
    INDEX       2576 b = stage1 378 | stage2 378 | stage3 378 | low 1442
    PAYLOAD_CRC 32 b (CRC-32 IEEE over the whole INDEX)
    PADDING     zeros to k_cb * ceil(total / k_cb), k_cb = 1024

All integer <-> bit conversions are MSB-first (bit[0] is the most significant bit),
matching ``digital_pipeline.py``.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Constants (identical values to digital_pipeline.py where shared)
# --------------------------------------------------------------------------
K_CB = 1024                     # info bits per LDPC code block
GRID_H = GRID_W = 28
N_TOKENS = GRID_H * GRID_W      # 784
MAGIC = 0x41515343              # "AQSC" (big-endian)
HEADER_LEN = 152                # bits; see module docstring
MASK_MODE_EXPLICIT = 0          # 784-bit explicit mask
MASK_MODE_ENUM = 3              # enumerative top-K rank
N_MASK_BITS_ENUM = 313          # ceil(log2 C(784,63)) = 313 (plan said 312; corrected)
CONFIG_18_2 = 0                 # config_id 0: b_high=18, b_low=2, K=63
IMAGE_PIXELS = 224 * 224        # 50176
CRC16_POLY = 0x1021             # CRC-16-CCITT (CCITT-FALSE, init 0xFFFF, MSB-first)
CRC32_POLY = 0xEDB88320         # CRC-32 IEEE (reflected, init/xorout 0xFFFFFFFF)
CRC16_BITS = 16
CRC32_BITS = 32

# 18/2b RVQ decomposition: b_high = N_STAGES * STAGE_BITS, b_low single stage
N_STAGES = 3                    # high path has 3 RVQ stages
STAGE_BITS = 6                  # 6 bits per stage (2^6 = 64 codewords)
B_LOW = 2                       # low path single 2-bit index

CONFIGS = {
    CONFIG_18_2: dict(b_high=18, b_low=2, k_high=63, name="18/2b",
                      n_stages=N_STAGES, stage_bits=STAGE_BITS),
}


# --------------------------------------------------------------------------
# Bit <-> integer helpers (MSB-first, identical to digital_pipeline.py)
# --------------------------------------------------------------------------
def bits_to_int(bits: np.ndarray) -> int:
    """MSB-first: bits[0] is the most significant bit."""
    v = 0
    for b in np.asarray(bits).tolist():
        v = (v << 1) | int(b)
    return v


def int_to_bits(value: int, nbits: int) -> np.ndarray:
    return np.array([(value >> (nbits - 1 - i)) & 1 for i in range(nbits)],
                    dtype=np.uint8)


def pack_bytes(value: int, nbytes: int) -> np.ndarray:
    return int_to_bits(value, 8 * nbytes)


# --------------------------------------------------------------------------
# CRCs (bit-for-bit identical to digital_pipeline.py)
# --------------------------------------------------------------------------
def crc16_ccitt(bits: np.ndarray) -> int:
    """CRC-16-CCITT (CCITT-FALSE): poly 0x1021, init 0xFFFF, MSB-first, no reflection.
    Check vector: crc16('123456789') = 0x29B1."""
    crc = 0xFFFF
    for b in np.asarray(bits).tolist():
        crc ^= int(b) << 15
        crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def crc32_ieee(bits: np.ndarray) -> int:
    """CRC-32 (IEEE 802.3): reflected poly 0xEDB88320, init/xorout 0xFFFFFFFF.
    MSB-first bit stream packed into bytes; check vector:
    crc32(b'\\x00\\x00\\x00\\x00') = 0x2144DF1C (zlib-compatible)."""
    bits = np.asarray(bits)
    crc = 0xFFFFFFFF
    n = int(bits.size)
    for byte_start in range(0, n, 8):
        byte = 0
        for i in range(8):
            byte = (byte << 1) | (int(bits[byte_start + i]) if byte_start + i < n else 0)
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xEDB88320 if crc & 1 else crc >> 1
    return crc ^ 0xFFFFFFFF


# --------------------------------------------------------------------------
# Enumerative subset coding (mask mode 3) -- identical to digital_pipeline.py
# --------------------------------------------------------------------------
def _comb(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def enum_mask_bits(n_tokens: int, k_high: int) -> int:
    """Number of bits for the enumerative top-K mask = ceil(log2 C(n_tokens, k_high))."""
    if k_high <= 0 or k_high >= n_tokens:
        return 0
    return int(math.ceil(math.log2(_comb(n_tokens, k_high))))


def subset_rank(subset: np.ndarray) -> int:
    """Combinadic rank of an ascending k-subset of {0..n-1} (Savage)."""
    rank = 0
    for i, s in enumerate(sorted(np.asarray(subset).tolist())):
        rank += _comb(s, i + 1)
    return rank


def subset_unrank(rank: int, n: int, k: int) -> np.ndarray:
    """Inverse of subset_rank. rank in [0, C(n,k))."""
    out = []
    r = rank
    for i in range(k - 1, -1, -1):
        s = i
        while s < n and _comb(s, i + 1) <= r:
            s += 1
        s -= 1
        out.append(s)
        r -= _comb(s, i + 1)
    return np.array(sorted(out), dtype=np.int64)


# --------------------------------------------------------------------------
# Header (152 bits) -- identical to digital_pipeline.py
# --------------------------------------------------------------------------
def build_header(version: int, config_id: int, mask_mode: int, n_tokens: int,
                 k_high: int, mask_bits: int, payload_bits: int) -> np.ndarray:
    """152-bit header per the plan field list (see module docstring)."""
    fields = [pack_bytes(MAGIC, 4), int_to_bits(version & 0xFF, 8),
              int_to_bits(config_id & 0xFF, 8), int_to_bits(mask_mode & 0xFF, 8),
              int_to_bits(0, 8),
              int_to_bits(n_tokens & 0xFFFF, 16), int_to_bits(k_high & 0xFFFF, 16),
              int_to_bits(mask_bits & 0xFFFF, 16), int_to_bits(payload_bits & 0xFFFFFF, 24)]
    head = np.concatenate(fields)
    crc = int_to_bits(crc16_ccitt(head), 16)
    return np.concatenate([head, crc])


def parse_header(head_bits: np.ndarray) -> Dict:
    assert head_bits.size == HEADER_LEN
    crc = crc16_ccitt(head_bits[:HEADER_LEN - 16])
    out = dict(magic=bits_to_int(head_bits[0:32]),
               version=bits_to_int(head_bits[32:40]),
               config_id=bits_to_int(head_bits[40:48]),
               mask_mode=bits_to_int(head_bits[48:56]),
               reserved=bits_to_int(head_bits[56:64]),
               n_tokens=bits_to_int(head_bits[64:80]),
               k_high=bits_to_int(head_bits[80:96]),
               mask_bits=bits_to_int(head_bits[96:112]),
               payload_bits=bits_to_int(head_bits[112:136]),
               header_crc_ok=crc == bits_to_int(head_bits[136:152]))
    return out


# Field offsets within the 152-bit header (for the paper packet figure).
HEADER_FIELDS = [
    ("magic", 0, 32),
    ("version", 32, 8),
    ("config_id", 40, 8),
    ("mask_mode", 48, 8),
    ("reserved", 56, 8),
    ("n_tokens", 64, 16),
    ("k_high", 80, 16),
    ("mask_bits", 96, 16),
    ("payload_bits", 112, 24),
    ("header_crc", 136, 16),
]


# --------------------------------------------------------------------------
# Mask encode / decode (identical to digital_pipeline.py)
# --------------------------------------------------------------------------
def mask_bits_explicit(high_mask: np.ndarray) -> np.ndarray:
    return np.asarray(high_mask, dtype=np.uint8)


def mask_bits_enumerative(high_mask: np.ndarray, n_tokens: Optional[int] = None,
                          k_high: Optional[int] = None) -> np.ndarray:
    high_mask = np.asarray(high_mask)
    n_tokens = high_mask.size if n_tokens is None else int(n_tokens)
    k_high = int(np.count_nonzero(high_mask)) if k_high is None else int(k_high)
    subset = np.where(high_mask)[0]
    assert subset.size == np.count_nonzero(high_mask)
    return int_to_bits(subset_rank(subset), enum_mask_bits(n_tokens, k_high))


def mask_bits_count(mask_mode: int, n_tokens: int = N_TOKENS,
                    k_high: int = 63) -> int:
    if mask_mode == MASK_MODE_EXPLICIT:
        return n_tokens
    return enum_mask_bits(n_tokens, k_high)


def mask_decode(mask_bits: np.ndarray, mode: int, n_tokens: int, k_high: int) -> np.ndarray:
    if mode == MASK_MODE_EXPLICIT:
        assert mask_bits.size == n_tokens
        return np.asarray(mask_bits, dtype=bool)
    if mode == MASK_MODE_ENUM:
        rank = bits_to_int(mask_bits)
        return np.isin(np.arange(n_tokens), subset_unrank(rank, n_tokens, k_high))
    raise ValueError(f"unsupported mask_mode {mode}")


# --------------------------------------------------------------------------
# RVQ per-stage index fields (the §3.3 resolution -- SEPARATE per-stage fields)
# --------------------------------------------------------------------------
def split_high_value(value18: int) -> Tuple[int, int, int]:
    """18-bit high token value -> (stage1, stage2, stage3), each 6 bits.
    stage1 occupies the HIGH-order 6 bits (most error-sensitive)."""
    v = int(value18)
    return (v >> 12) & 0x3F, (v >> 6) & 0x3F, v & 0x3F


def combine_high_value(stage1: int, stage2: int, stage3: int) -> int:
    return ((int(stage1) & 0x3F) << 12) | ((int(stage2) & 0x3F) << 6) | (int(stage3) & 0x3F)


def token_values_to_stages(token_values: np.ndarray, high_mask: np.ndarray,
                           b_low: int = B_LOW) -> Tuple[np.ndarray, np.ndarray,
                                                        np.ndarray, np.ndarray]:
    """Split ascending token values into per-stage index arrays.

    Returns (stage1, stage2, stage3, low): stage* are K-element arrays in
    ascending high-token order (values 0..63); low is a (784-K)-element array
    in ascending low-token order (values 0..2^b_low-1)."""
    high_mask = np.asarray(high_mask, dtype=bool)
    token_values = np.asarray(token_values)
    high_idx = np.where(high_mask)[0]
    low_idx = np.where(~high_mask)[0]
    stage1 = np.array([split_high_value(token_values[t])[0] for t in high_idx], dtype=np.int64)
    stage2 = np.array([split_high_value(token_values[t])[1] for t in high_idx], dtype=np.int64)
    stage3 = np.array([split_high_value(token_values[t])[2] for t in high_idx], dtype=np.int64)
    low = np.array([int(token_values[t]) & ((1 << b_low) - 1) for t in low_idx], dtype=np.int64)
    return stage1, stage2, stage3, low


def stages_to_token_values(stage1: np.ndarray, stage2: np.ndarray, stage3: np.ndarray,
                           low: np.ndarray, high_mask: np.ndarray,
                           n_tokens: int = N_TOKENS) -> np.ndarray:
    """Inverse of token_values_to_stages."""
    high_mask = np.asarray(high_mask, dtype=bool)
    high_idx = np.where(high_mask)[0]
    low_idx = np.where(~high_mask)[0]
    token_values = np.zeros(n_tokens, dtype=np.int64)
    for j, t in enumerate(high_idx):
        token_values[t] = combine_high_value(stage1[j], stage2[j], stage3[j])
    for j, t in enumerate(low_idx):
        token_values[t] = int(low[j]) & 0x3
    return token_values


def pack_rvq_indices(stage1_idx: np.ndarray, stage2_idx: np.ndarray,
                     stage3_idx: np.ndarray, low_idx: np.ndarray,
                     high_mask: np.ndarray, n_tokens: int = N_TOKENS) -> np.ndarray:
    """Pack per-stage RVQ indices into the payload (INDEX) bit field.

    Layout: [stage1: K*6][stage2: K*6][stage3: K*6][low: (n_tokens-K)*2] (MSB-first,
    ascending token order within each stage block)."""
    high_mask = np.asarray(high_mask, dtype=bool)
    k = int(np.count_nonzero(high_mask))
    stage1_idx = np.asarray(stage1_idx).reshape(-1)
    stage2_idx = np.asarray(stage2_idx).reshape(-1)
    stage3_idx = np.asarray(stage3_idx).reshape(-1)
    low_idx = np.asarray(low_idx).reshape(-1)
    assert stage1_idx.size == stage2_idx.size == stage3_idx.size == k
    assert low_idx.size == n_tokens - k
    parts = []
    for arr in (stage1_idx, stage2_idx, stage3_idx):
        for v in arr:
            parts.append(int_to_bits(int(v) & 0x3F, STAGE_BITS))
    for v in low_idx:
        parts.append(int_to_bits(int(v) & ((1 << B_LOW) - 1), B_LOW))
    return np.concatenate(parts)


def unpack_rvq_indices(payload_bits: np.ndarray, high_mask: np.ndarray,
                       n_tokens: int = N_TOKENS) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of pack_rvq_indices. Returns (stage1, stage2, stage3, low)."""
    high_mask = np.asarray(high_mask, dtype=bool)
    k = int(np.count_nonzero(high_mask))
    payload = np.asarray(payload_bits).reshape(-1)
    stage1_bits = k * STAGE_BITS
    stage2_bits = k * STAGE_BITS
    stage3_bits = k * STAGE_BITS
    low_bits = (n_tokens - k) * B_LOW
    assert payload.size == stage1_bits + stage2_bits + stage3_bits + low_bits, \
        f"payload size {payload.size} != expected {stage1_bits + stage2_bits + stage3_bits + low_bits}"
    off = 0
    def _read(n_vals: int, w: int) -> np.ndarray:
        nonlocal off
        vals = np.zeros(n_vals, dtype=np.int64)
        for j in range(n_vals):
            vals[j] = bits_to_int(payload[off:off + w])
            off += w
        return vals
    stage1 = _read(k, STAGE_BITS)
    stage2 = _read(k, STAGE_BITS)
    stage3 = _read(k, STAGE_BITS)
    low = _read(n_tokens - k, B_LOW)
    return stage1, stage2, stage3, low


# --------------------------------------------------------------------------
# Field layout of the unpadded frame (exact spec for the paper packet figure)
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Generalized N-stage RVQ packing (plan v5.9 J3/E-7: L2 uses 4 stages x 6 bits
# + 4-bit low, which the fixed 3-stage functions above cannot represent).
# These are separate functions, not a rewrite of the 3-stage path above, so
# the already-verified 18/2b behavior (test_bitstream.py) is untouched.
# --------------------------------------------------------------------------
def split_high_value_n(value: int, n_stages: int, stage_bits: int) -> Tuple[int, ...]:
    """Generalized split_high_value: MSB-first, stage 1 = highest-order bits.
    Matches split_high_value exactly when (n_stages, stage_bits) == (3, 6)."""
    v = int(value)
    mask = (1 << stage_bits) - 1
    shift = (n_stages - 1) * stage_bits
    out = []
    for _ in range(n_stages):
        out.append((v >> shift) & mask)
        shift -= stage_bits
    return tuple(out)


def combine_high_value_n(stages: "Sequence[int]", stage_bits: int) -> int:
    """Inverse of split_high_value_n."""
    v = 0
    for s in stages:
        v = (v << stage_bits) | (int(s) & ((1 << stage_bits) - 1))
    return v


def token_values_to_stages_n(token_values: np.ndarray, high_mask: np.ndarray,
                             n_stages: int, stage_bits: int,
                             b_low: int) -> Tuple[List[np.ndarray], np.ndarray]:
    """N-stage generalization of token_values_to_stages.

    Returns (stages, low): ``stages`` is a list of ``n_stages`` K-element
    arrays (stage[0] = most-significant / most error-sensitive); ``low`` is a
    (n_tokens-K)-element array of ``b_low``-bit values."""
    high_mask = np.asarray(high_mask, dtype=bool)
    token_values = np.asarray(token_values)
    high_idx = np.where(high_mask)[0]
    low_idx = np.where(~high_mask)[0]
    split = [split_high_value_n(token_values[t], n_stages, stage_bits) for t in high_idx]
    stages = [np.array([s[i] for s in split], dtype=np.int64) for i in range(n_stages)]
    low = np.array([int(token_values[t]) & ((1 << b_low) - 1) for t in low_idx], dtype=np.int64)
    return stages, low


def stages_to_token_values_n(stages: "Sequence[np.ndarray]", low: np.ndarray,
                             high_mask: np.ndarray, stage_bits: int,
                             b_low: int, n_tokens: int = N_TOKENS) -> np.ndarray:
    """Inverse of token_values_to_stages_n."""
    high_mask = np.asarray(high_mask, dtype=bool)
    high_idx = np.where(high_mask)[0]
    low_idx = np.where(~high_mask)[0]
    token_values = np.zeros(n_tokens, dtype=np.int64)
    for j, t in enumerate(high_idx):
        token_values[t] = combine_high_value_n([s[j] for s in stages], stage_bits)
    low_mask = (1 << b_low) - 1
    for j, t in enumerate(low_idx):
        token_values[t] = int(low[j]) & low_mask
    return token_values


def pack_rvq_indices_n(stages: "Sequence[np.ndarray]", low_idx: np.ndarray,
                       high_mask: np.ndarray, stage_bits: int, b_low: int,
                       n_tokens: int = N_TOKENS) -> np.ndarray:
    """N-stage generalization of pack_rvq_indices.

    Layout: [stage[0]: K*stage_bits]...[stage[n-1]: K*stage_bits][low: (n_tokens-K)*b_low]
    (MSB-first, ascending token order within each stage block)."""
    high_mask = np.asarray(high_mask, dtype=bool)
    k = int(np.count_nonzero(high_mask))
    stages = [np.asarray(s).reshape(-1) for s in stages]
    low_idx = np.asarray(low_idx).reshape(-1)
    for s in stages:
        assert s.size == k
    assert low_idx.size == n_tokens - k
    stage_mask = (1 << stage_bits) - 1
    low_mask = (1 << b_low) - 1
    parts = []
    for s in stages:
        for v in s:
            parts.append(int_to_bits(int(v) & stage_mask, stage_bits))
    for v in low_idx:
        parts.append(int_to_bits(int(v) & low_mask, b_low))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint8)


def unpack_rvq_indices_n(payload_bits: np.ndarray, high_mask: np.ndarray,
                         n_stages: int, stage_bits: int, b_low: int,
                         n_tokens: int = N_TOKENS) -> Tuple[List[np.ndarray], np.ndarray]:
    """Inverse of pack_rvq_indices_n. Returns (stages, low)."""
    high_mask = np.asarray(high_mask, dtype=bool)
    k = int(np.count_nonzero(high_mask))
    payload = np.asarray(payload_bits).reshape(-1)
    expected = n_stages * k * stage_bits + (n_tokens - k) * b_low
    assert payload.size == expected, \
        f"payload size {payload.size} != expected {expected}"
    off = 0
    def _read(n_vals: int, w: int) -> np.ndarray:
        nonlocal off
        vals = np.zeros(n_vals, dtype=np.int64)
        for j in range(n_vals):
            vals[j] = bits_to_int(payload[off:off + w])
            off += w
        return vals
    stages = [_read(k, stage_bits) for _ in range(n_stages)]
    low = _read(n_tokens - k, b_low)
    return stages, low


def index_field_widths(k_high: int, b_high: int, b_low: int,
                       n_tokens: int = N_TOKENS) -> Dict[str, int]:
    """Per-stage width of the INDEX field."""
    stage_bits = N_STAGES * STAGE_BITS if b_high == N_STAGES * STAGE_BITS else None
    stage_w = (b_high // N_STAGES) if b_high % N_STAGES == 0 else b_high
    return dict(
        stage1=k_high * stage_w,
        stage2=k_high * stage_w,
        stage3=k_high * stage_w,
        low=(n_tokens - k_high) * b_low,
    )


def field_layout(mask_mode: int, k_high: int, b_high: int, b_low: int,
                 n_tokens: int = N_TOKENS) -> List[Tuple[str, int, int]]:
    """Ordered (name, bit_offset, bit_width) of the unpadded frame. Offsets are
    relative to the start of the frame (bit 0 = first magic bit)."""
    mask_w = mask_bits_count(mask_mode, n_tokens, k_high)
    widths = index_field_widths(k_high, b_high, b_low, n_tokens)
    index_w = widths["stage1"] + widths["stage2"] + widths["stage3"] + widths["low"]
    off = 0
    rows = []
    for name, o, w in HEADER_FIELDS:
        rows.append((name, o, w))
    off = HEADER_LEN
    rows.append(("mask", off, mask_w))
    off += mask_w
    rows.append(("stage1", off, widths["stage1"])); off += widths["stage1"]
    rows.append(("stage2", off, widths["stage2"])); off += widths["stage2"]
    rows.append(("stage3", off, widths["stage3"])); off += widths["stage3"]
    rows.append(("low", off, widths["low"])); off += widths["low"]
    rows.append(("payload_crc", off, CRC32_BITS)); off += CRC32_BITS
    assert off == HEADER_LEN + mask_w + index_w + CRC32_BITS
    return rows


# --------------------------------------------------------------------------
# Frame build / parse
# --------------------------------------------------------------------------
def _pad_to_kcb(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    rem = bits.size % K_CB
    if rem == 0:
        return bits
    return np.concatenate([bits, np.zeros(K_CB - rem, dtype=np.uint8)])


def build_frame(stage1_idx: np.ndarray, stage2_idx: np.ndarray, stage3_idx: np.ndarray,
                low_idx: np.ndarray, high_mask: np.ndarray,
                mask_mode: int = MASK_MODE_EXPLICIT, config_id: int = CONFIG_18_2,
                version: int = 1, n_tokens: int = N_TOKENS) -> Dict:
    """Build the §3.3 frame from per-stage RVQ indices.

    Returns a dict with every section plus the two UEP protection segments and
    the whole padded frame::

        segment_a (rate 1/3): header | mask | stage1        -> padded to k_cb
        segment_b (rate 1/2): stage2 | stage3 | low | crc32 -> padded to k_cb
        frame                 : header | mask | index | crc32 -> padded to k_cb
    """
    cfg = CONFIGS[config_id]
    b_high, b_low = cfg["b_high"], cfg["b_low"]
    k_high = int(np.count_nonzero(high_mask))
    index = pack_rvq_indices(stage1_idx, stage2_idx, stage3_idx, low_idx, high_mask,
                             n_tokens)
    if mask_mode == MASK_MODE_EXPLICIT:
        mask = mask_bits_explicit(high_mask)
    elif mask_mode == MASK_MODE_ENUM:
        mask = mask_bits_enumerative(high_mask, n_tokens, k_high)
    else:
        raise ValueError(mask_mode)
    header = build_header(version, config_id, mask_mode, n_tokens, k_high,
                          mask.size, index.size)
    payload_crc = int_to_bits(crc32_ieee(index), CRC32_BITS)

    widths = index_field_widths(k_high, b_high, b_low, n_tokens)
    stage1 = index[0:widths["stage1"]]
    stage2 = index[widths["stage1"]:widths["stage1"] + widths["stage2"]]
    stage3 = index[widths["stage1"] + widths["stage2"]:
                   widths["stage1"] + widths["stage2"] + widths["stage3"]]
    low = index[widths["stage1"] + widths["stage2"] + widths["stage3"]:]

    frame_unpadded = np.concatenate([header, mask, index, payload_crc])
    frame = _pad_to_kcb(frame_unpadded)

    seg_a_body = np.concatenate([header, mask, stage1])
    seg_b_body = np.concatenate([stage2, stage3, low, payload_crc])
    segment_a = _pad_to_kcb(seg_a_body)
    segment_b = _pad_to_kcb(seg_b_body)

    layout = field_layout(mask_mode, k_high, b_high, b_low, n_tokens)
    return dict(
        header=header, mask=mask, mask_mode=mask_mode,
        stage1=stage1, stage2=stage2, stage3=stage3, low=low,
        index=index, payload_crc=payload_crc,
        frame_unpadded=frame_unpadded, frame=frame,
        segment_a=segment_a, segment_b=segment_b, seg_a_body=seg_a_body,
        seg_b_body=seg_b_body,
        config_id=config_id, version=version, k_high=k_high,
        b_high=b_high, b_low=b_low, n_tokens=n_tokens,
        n_blocks_a=segment_a.size // K_CB, n_blocks_b=segment_b.size // K_CB,
        n_blocks_frame=frame.size // K_CB, layout=layout,
    )


def parse_frame(frame_bits: np.ndarray) -> Dict:
    """Inverse of build_frame at the bitstream level (no channel, no erasure).

    ``frame_bits`` is the UNPADDED concatenation header|mask|index|payload_crc.
    Returns a dict with header fields, high_mask, per-stage index arrays,
    token_values, and CRC flags. Padding (if present) is ignored.
    """
    frame_bits = np.asarray(frame_bits, dtype=np.uint8).reshape(-1)
    head_bits = frame_bits[:HEADER_LEN]
    hdr = parse_header(head_bits)
    out = dict(header_ok=(hdr["magic"] == MAGIC and hdr["header_crc_ok"]),
               payload_crc_ok=False, frame_lost=True, high_mask=None,
               stage1_idx=None, stage2_idx=None, stage3_idx=None, low_idx=None,
               token_values=None, erased_tokens=N_TOKENS)
    out.update(hdr)
    n_tokens = int(out.get("n_tokens", N_TOKENS))
    out["erased_tokens"] = n_tokens
    if not out["header_ok"]:
        return out
    mask_off = HEADER_LEN
    mask_len = hdr["mask_bits"]
    mask_bits = frame_bits[mask_off:mask_off + mask_len]
    high_mask = mask_decode(mask_bits, hdr["mask_mode"], hdr["n_tokens"], hdr["k_high"])
    out["high_mask"] = high_mask
    out["frame_lost"] = False
    cfg = CONFIGS.get(out["config_id"], CONFIGS[CONFIG_18_2])
    b_high, b_low = cfg["b_high"], cfg["b_low"]
    index_off = mask_off + mask_len
    index_len = hdr["payload_bits"]
    index = frame_bits[index_off:index_off + index_len]
    crc_bits = frame_bits[index_off + index_len:index_off + index_len + CRC32_BITS]
    out["payload_crc_ok"] = (crc_bits.size == CRC32_BITS
                             and crc32_ieee(index) == bits_to_int(crc_bits))
    stage1, stage2, stage3, low = unpack_rvq_indices(index, high_mask, n_tokens)
    out["stage1_idx"] = stage1
    out["stage2_idx"] = stage2
    out["stage3_idx"] = stage3
    out["low_idx"] = low
    out["token_values"] = stages_to_token_values(stage1, stage2, stage3, low,
                                                 high_mask, n_tokens)
    if not out["payload_crc_ok"]:
        out["token_values"] = None
        out["erased_tokens"] = n_tokens
    else:
        out["erased_tokens"] = 0
    return out


# --------------------------------------------------------------------------
# Measured BPP
# --------------------------------------------------------------------------
def measure_bpp(config, high_mask: np.ndarray, mask_mode: int = MASK_MODE_EXPLICIT,
                image_pixels: int = IMAGE_PIXELS, k_cb: int = K_CB,
                n_tokens: int = N_TOKENS) -> float:
    """Fully-loaded measured BPP (source/info bits) for one frame.

    Counts header 152 + mask + all index fields + CRC-32 + zero-padding to the
    k_cb boundary, divided by image_pixels (224*224 = 50176)."""
    return measure_bpp_breakdown(config, high_mask, mask_mode,
                                 image_pixels, k_cb, n_tokens)["bpp"]


def measure_bpp_breakdown(config, high_mask: np.ndarray,
                          mask_mode: int = MASK_MODE_EXPLICIT,
                          image_pixels: int = IMAGE_PIXELS,
                          k_cb: int = K_CB,
                          n_tokens: int = N_TOKENS) -> Dict:
    cfg = CONFIGS[config] if isinstance(config, int) else config
    b_high, b_low = cfg["b_high"], cfg["b_low"]
    k_high = int(np.count_nonzero(high_mask))
    header_bits = HEADER_LEN
    mask_bits = mask_bits_count(mask_mode, n_tokens, k_high)
    index_bits = k_high * b_high + (n_tokens - k_high) * b_low
    crc_bits = CRC32_BITS
    total = header_bits + mask_bits + index_bits + crc_bits
    padding = (k_cb - total % k_cb) % k_cb
    emitted = total + padding
    return dict(
        header_bits=header_bits, mask_bits=mask_bits, index_bits=index_bits,
        crc_bits=crc_bits, total=total, padding=padding, emitted=emitted,
        bpp=emitted / image_pixels,
        bpp_no_padding=total / image_pixels,
        bpp_header=header_bits / image_pixels,
        bpp_mask=mask_bits / image_pixels,
        bpp_index=index_bits / image_pixels,
        bpp_crc=crc_bits / image_pixels,
        bpp_padding=padding / image_pixels,
    )


# --------------------------------------------------------------------------
# Per-segment (UEP) padded totals -- what actually goes through the channel coder
# --------------------------------------------------------------------------
def segment_padded_bits(config, high_mask: np.ndarray,
                        mask_mode: int = MASK_MODE_EXPLICIT, k_cb: int = K_CB,
                        n_tokens: int = N_TOKENS) -> Dict:
    """Info-bit size of the two UEP segments after k_cb padding.

    segment_a (rate 1/3) = header + mask + stage1; segment_b (rate 1/2) =
    stage2 + stage3 + low + payload_crc. Each is zero-padded to k_cb separately.
    """
    cfg = CONFIGS[config] if isinstance(config, int) else config
    b_high, b_low = cfg["b_high"], cfg["b_low"]
    k_high = int(np.count_nonzero(high_mask))
    widths = index_field_widths(k_high, b_high, b_low, n_tokens)
    seg_a = HEADER_LEN + mask_bits_count(mask_mode, n_tokens, k_high) + widths["stage1"]
    seg_b = widths["stage2"] + widths["stage3"] + widths["low"] + CRC32_BITS
    pad_a = (k_cb - seg_a % k_cb) % k_cb
    pad_b = (k_cb - seg_b % k_cb) % k_cb
    return dict(seg_a=seg_a, seg_b=seg_b, pad_a=pad_a, pad_b=pad_b,
                seg_a_padded=seg_a + pad_a, seg_b_padded=seg_b + pad_b,
                total_padded=seg_a + pad_a + seg_b + pad_b)
