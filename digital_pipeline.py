"""Digital PHY pipeline for AquaSC (master plan §3.3 frame format, §5.3 digital chain).

Wave 4, CHANNEL agent -- R3 #3 (digital pipeline). Standalone module; integration
into the eval path (model.py/channel.py/digital_channel.py) is Wave 5.

Scope
-----
* 5G NR LDPC, base graph BG2 (Sionna ``LDPC5GEncoder``/``LDPC5GDecoder``,
  v1.2.2, tensorflow 2.21.0, CPU). ``k_cb = 1024`` info bits per code block.
* Payload rate 1/2 and header+mask rate 1/3 (UEP); the integrated RVQ path
  additionally places the first 6-bit RVQ stage in the rate-1/3 segment.
* Decoder: normalized min-sum with scale 0.75 (``cn_update_minsum`` scaled by
  0.75), max 25 iterations, syndrome early termination. Sionna exposes the
  decoder state (``msg_v2c``) and the pruned parity-check matrix; we run the
  BP decoder in chunks of 5 iterations and stop as soon as the syndrome
  (pruned PCM times hard decisions, mod 2) is all zero. When no syndrome is
  reached the decoder runs the full 25-iteration cap.
* Gray 16-QAM at unit average symbol energy (constellation normalized by
  1/sqrt(10)); max-log demapper.
* SISO Rayleigh block fading h ~ CN(0,1), constant per code block
  (one fade per 512 symbols at rate 1/2). SNR = Es/N0 with Es = 1.
* CSI: (a) perfect: ZF equalization z = y/h with noise variance N0/|h|^2;
  (b) pilot: one pilot symbol every 10 emitted positions (indices
  0,10,20,...), fixed pilot symbol x_p = (1+1j)/sqrt(10), LS estimate
  h_hat = mean(y_p/x_p) over the block's pilots, then ZF as in (a).
  Pilot overhead 10% (57/569 symbols at 512 data symbols; 10.02%).
* Error handling: LDPC syndrome failure -> block erased. At frame level the
  payload carries a trailing CRC-32 (IEEE) over the index bits; per-block
  operation appends CRC-32 to each 1024-bit info block (992 payload + 32 CRC,
  effective payload rate 992/1024 x 1/2 = 0.4844). Erased tokens are
  concealed by iterative spatial N4 averaging on the 28x28 token grid
  (row-major token order; erased token replaced by the mean of already
  known/filled 4-neighbours, repeated until fixpoint; if no neighbour is
  available, fall back to the mean of all known tokens, then 0).

Deviations from the plan text (documented, deliberate)
------------------------------------------------------
* Header is 152 bits, not "16 B = 128 b": the field list (32+8+8+8+8+16+16+16
  +24+16) sums to 152. We follow the field list.
* Mode-3 mask rank needs 313 bits, not 312: log2(C(784,63)) = 312.0382 > 312,
  so 312 bits cannot represent every 63-subset of 784. We use
  ceil(log2(C(784,63))) = 313 bits (slight rate penalty vs plan).
* Measured AWGN threshold of BG2 rate 1/2 at k_cb = 1024 (16 filler bits,
  208 punctured systematic bits out of 1024) is ~6.5 dB, i.e. the fading
  BLER follows P_out(gamma_th) with gamma_th ~= 6.9 dB (min-sum, 25 it),
  vs the 4.77 dB (gamma_th = 3) assumed by the outage bound in the plan.
  This is a property of the specified code, not of the implementation;
  BLER is reported per SNR with the bound side by side.

Environment: run with __SIONNA_PY__
(sionna 1.2.2, tensorflow 2.21.0, numpy 2.2.6, CPU).
"""
from __future__ import annotations

import csv
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder, LDPCBPDecoder, cn_update_minsum
from sionna.phy.mapping import Constellation, Mapper, Demapper

# --------------------------------------------------------------------------
# Constants / configuration (plan 5.3 table)
# --------------------------------------------------------------------------
K_CB = 1024                     # info bits per LDPC code block
NUM_ITER_MAX = 25               # decoder iteration cap
NUM_ITER_CHUNK = 5              # incremental decoding chunk (early termination)
LLR_MAX = 20.0
NMS_SCALE = 0.75                # normalized min-sum scale
PILOT_FRAC = 0.1                # pilot overhead
PILOT_EVERY = 10                # one pilot every 10 emitted symbols
PILOT_SYMBOL = (1.0 + 1.0j) / np.sqrt(10.0)
GRID_H = GRID_W = 28
N_TOKENS = GRID_H * GRID_W      # 784
MAGIC = 0x41515343              # "AQSC" (big-endian)
HEADER_LEN = 152                # bits; see deviations note
MASK_MODE_EXPLICIT = 0          # 784-bit explicit mask
MASK_MODE_ENUM = 3              # enumerative top-K rank
N_MASK_BITS_ENUM = 313          # ceil(log2 C(784,63)) = 313 (plan says 312; see note)
CONFIG_18_2 = 0                 # config_id 0: b_high=18, b_low=2, K=63
CONFIGS = {
    CONFIG_18_2: dict(b_high=18, b_low=2, k_high=63, name="18/2b"),
}
RATES = {1 / 3: 3072, 1 / 2: 2048, 2 / 3: 1536}   # rate -> coded n for k_cb=1024


def channel_uses_per_px(segments, bits_per_symbol: int,
                        img_h: int = 224, img_w: int = 224,
                        csi_mode: str = "perfect",
                        pilot_frac: float = PILOT_FRAC,
                        pilot_every: int = PILOT_EVERY,
                        coded_block_lengths: Optional[List[int]] = None) -> float:
    """Canonical, single source of truth for the real digital-chain channel-use
    rate (plan v6.1 P-2 -- three different rate figures were in circulation
    because nothing computed this consistently; this function is now the only
    place that does).

    ``segments`` is a list of ``(rate, n_blocks)`` pairs, one per LDPC-coded
    segment actually transmitted for one image (e.g. the per-stage UEP split:
    ``[(1/3, blocks_a), (payload_rate, blocks_b)]``). Every ``k_cb=1024``-bit
    LDPC block emits ``RATES[rate]`` coded bits **regardless of how much of
    that block is real payload versus whole-frame padding or the 32-bit
    per-block CRC overhead** -- that padding is a real, non-negotiable channel
    cost (the block is transmitted either way) and must not be netted out.
    This is why this figure differs from a naive ``source_bits / rate /
    bits_per_symbol`` formula: the formula assumes perfectly divisible,
    unpadded blocks, which the real LDPC5G block structure does not give you.

    ``csi_mode``: 'perfect' emits exactly ``coded_bits / bits_per_symbol``
    symbols (verified against ``channel_pass``'s own ``n_sym`` return value --
    no pilot insertion). 'pilot' inserts one pilot symbol every
    ``pilot_every`` emitted positions (``channel_pass``'s own formula:
    ``n_emit = n_sym + ceil(n_sym / (pilot_every - 1))``), which is a real
    ~1/(pilot_every-1) additional channel-use cost this function must include
    or it silently undercounts any pilot-mode result by that fraction.

    ``coded_block_lengths`` optionally supplies actual encoder output lengths
    for variable-length codewords. Use empty ``segments`` and perfect CSI;
    lengths must be positive integer multiples of ``bits_per_symbol``.
    """
    # Optional actual encoder lengths for experimental rate-matching; legacy
    # calls remain identical. Arbitrary lengths currently support perfect CSI.
    if coded_block_lengths is not None:
        if segments or csi_mode != "perfect":
            raise ValueError("explicit coded lengths require empty segments and perfect CSI")
        if bits_per_symbol not in (1, 2, 4):
            raise ValueError("supported bits_per_symbol: 1, 2, 4")
        if any(not isinstance(n, (int, np.integer)) or n <= 0
               or n % bits_per_symbol for n in coded_block_lengths):
            raise ValueError("coded lengths must be positive and divisible by bits_per_symbol")
        total_coded_bits = sum(coded_block_lengths)
    else:
        total_coded_bits = sum(RATES[rate] * n_blocks for rate, n_blocks in segments)
    n_sym = total_coded_bits / bits_per_symbol
    if csi_mode == "pilot" and pilot_frac > 0:
        n_sym = n_sym + np.ceil(n_sym / (pilot_every - 1))
    return n_sym / (img_h * img_w)
CRC16_POLY = 0x1021             # CRC-16-CCITT (CCITT-FALSE, init 0xFFFF, MSB-first)
CRC32_POLY = 0xEDB88320         # CRC-32 IEEE (reflected poly, init/xorout 0xFFFFFFFF)
OUTAGE_BOUND_GAMMA = 3.0        # plan bound P_out = 1 - exp(-3/gamma_bar)
CRC_PER_BLOCK_BITS = 32

# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _rng(seed) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(int(seed) & 0xFFFFFFFF)

def bits_to_int(bits: np.ndarray) -> int:
    """MSB-first: bits[0] is the most significant bit."""
    v = 0
    for b in bits.tolist():
        v = (v << 1) | int(b)
    return v

def int_to_bits(value: int, nbits: int) -> np.ndarray:
    return np.array([(value >> (nbits - 1 - i)) & 1 for i in range(nbits)], dtype=np.uint8)

def pack_bytes(value: int, nbytes: int) -> np.ndarray:
    return int_to_bits(value, 8 * nbytes)

def crc16_ccitt(bits: np.ndarray) -> int:
    """CRC-16-CCITT (CCITT-FALSE): poly 0x1021, init 0xFFFF, MSB-first, no reflection.
    Check vector: crc16('123456789') = 0x29B1."""
    crc = 0xFFFF
    for b in bits.tolist():
        crc ^= int(b) << 15
        crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc

def crc32_ieee(bits: np.ndarray) -> int:
    """CRC-32 (IEEE 802.3): reflected poly 0xEDB88320, init/xorout 0xFFFFFFFF.
    MSB-first bit stream packed into bytes; check vector:
    crc32(b'\\x00\\x00\\x00\\x00') = 0x2144DF1C (zlib-compatible)."""
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
# Enumerative subset coding (mask mode 3)
# --------------------------------------------------------------------------
def _comb(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def enum_mask_bits(n_tokens: int, k_high: int) -> int:
    """Number of bits for the enumerative top-K mask = ceil(log2 C(n_tokens, k_high)).

    For n_tokens=784, k_high=63 this is 313 (log2 C(784,63) = 312.038 > 312);
    for n_tokens=196, k_high=63 it is 174 (log2 C(196,63) = 173.1... > 173).
    """
    if k_high <= 0 or k_high >= n_tokens:
        return 0
    return int(math.ceil(math.log2(_comb(n_tokens, k_high))))

def subset_rank(subset: np.ndarray) -> int:
    """Combinadic rank of an ascending k-subset of {0..n-1} (Savage)."""
    rank = 0
    for i, s in enumerate(sorted(subset.tolist())):
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
# LDPC encoder / decoder objects (lazily built, cached; no global mutable state
# other than this read-mostly cache keyed by (k, n, cn_update)).
# --------------------------------------------------------------------------
_CODER_CACHE: Dict[Tuple[int, int, str], Tuple[LDPC5GEncoder, "AquaSCLDPCDecoder"]] = {}

def cn_update_nms(msg_v2c_rag, llr_clipping=None):
    """Normalized min-sum, scale 0.75."""
    return tf.multiply(cn_update_minsum(msg_v2c_rag, llr_clipping=llr_clipping), NMS_SCALE)


class AquaSCLDPCDecoder(LDPC5GDecoder):
    """5G BG2 decoder wrapper exposing syndrome check and incremental decoding.

    Sionna's LDPC5GDecoder runs a fixed number of BP iterations and returns
    only the rate-matched bits. We subclass it to (i) return the hard
    decisions of the pruned graph so the syndrome can be computed exactly
    (the pruned PCM ``self._pcm`` is the matrix the BP core checks), and
    (ii) resume decoding from the internal decoder state (``msg_v2c``) so
    that syndrome early termination can be implemented without re-decoding.
    """

    def __init__(self, encoder: LDPC5GEncoder, num_iter: int = NUM_ITER_MAX,
                 llr_max: float = LLR_MAX):
        super().__init__(encoder, cn_update=cn_update_nms, vn_update="sum",
                         cn_schedule="flooding", hard_out=True,
                         return_infobits=True, num_iter=num_iter,
                         llr_max=llr_max, prune_pcm=True, return_state=True)
        self._Hp = np.asarray(self._pcm.todense()).astype(np.int64)

    # -- internal: build the full-length LLR vector exactly as LDPC5GDecoder
    #    does (punctured systematic bits -> 0, filler -> -llr_max logits)
    def _prepare(self, llr_ch: tf.Tensor) -> tf.Tensor:
        B = tf.shape(tf.reshape(llr_ch, [-1, self.encoder.n]))[0]
        llr = tf.reshape(llr_ch, [-1, self.encoder.n])
        k_filler = self.encoder.k_ldpc - self.encoder.k
        nb_punc = (self.encoder.n_ldpc - k_filler) - self.encoder.n - 2 * self.encoder.z
        llr5 = tf.concat([tf.zeros([B, 2 * self.encoder.z], self.rdtype), llr,
                          tf.zeros([B, nb_punc - self._nb_pruned_nodes], self.rdtype)], axis=1)
        x1 = llr5[:, :self.encoder.k]
        nb_par = self.encoder.n_ldpc - k_filler - self.encoder.k - self._nb_pruned_nodes
        x2 = llr5[:, self.encoder.k:self.encoder.k + nb_par]
        z = -tf.cast(self._llr_max, self.rdtype) * tf.ones([B, k_filler], self.rdtype)
        return tf.concat([x1, z, x2], axis=1)

    def syndrome(self, x_hat: np.ndarray) -> np.ndarray:
        """Per-block syndrome (pruned PCM, mod 2); True == valid codeword."""
        syn = (self._Hp @ x_hat.T) % 2
        return ~np.any(syn, axis=0)

    def incremental_decode(self, llr_ch: np.ndarray, num_iter: int = NUM_ITER_MAX,
                           chunk: int = NUM_ITER_CHUNK) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode with syndrome early termination.

        Returns (u_hat[B,k_cb] hard decisions, ok[B] syndrome-valid flags,
        iters_used[B] BP iterations consumed, capped at num_iter).
        """
        llr_t = tf.convert_to_tensor(np.asarray(llr_ch, dtype=np.float32), tf.float32)
        llr_t = tf.clip_by_value(llr_t, -self._llr_max, self._llr_max)
        llr5 = self._prepare(llr_t)
        B = llr5.shape[0]
        state = None
        u_hat = np.zeros((B, self.encoder.k), dtype=np.uint8)
        ok = np.zeros(B, dtype=bool)
        iters = np.zeros(B, dtype=np.int32)
        it = 0
        while it < num_iter:
            step = min(chunk, num_iter - it)
            x_hat, state = LDPCBPDecoder.call(self, llr5, num_iter=step, msg_v2c=state)
            x_hat_np = x_hat.numpy()
            good = self.syndrome(x_hat_np)
            it += step
            new_good = good & ~ok
            iters[new_good] = it
            u_hat[new_good] = x_hat_np[new_good, :self.encoder.k].astype(np.uint8)
            ok |= good
            if ok.all():
                break
        if not ok.all():
            iters[~ok] = num_iter
            u_hat[~ok] = x_hat_np[~ok, :self.encoder.k].astype(np.uint8)
        return u_hat, ok, iters


def get_coder(rate: float) -> Tuple[LDPC5GEncoder, AquaSCLDPCDecoder]:
    if rate not in RATES:
        raise ValueError(f"unsupported rate {rate}; supported {sorted(RATES)}")
    n = RATES[rate]
    key = (K_CB, n, "nms075")
    if key not in _CODER_CACHE:
        enc = LDPC5GEncoder(k=K_CB, n=n, bg="bg2")
        dec = AquaSCLDPCDecoder(enc)
        _CODER_CACHE[key] = (enc, dec)
    return _CODER_CACHE[key]


# --------------------------------------------------------------------------
# Channel: Rayleigh block fading + 16-QAM + CSI
# --------------------------------------------------------------------------
def modulate_symbols(cw_bits: np.ndarray, bits_per_symbol: int = 4) -> np.ndarray:
    """[B, n] bits -> [B, n/bits_per_symbol] complex symbols, unit energy.

    bits_per_symbol: 1 = BPSK (Gray PAM), 2 = QPSK (Gray QAM), 4 = 16-QAM
    (Gray QAM). Track F / E3 (plan 16.9): the MCS ladder varies this to trade
    rate for robustness at low SNR without retraining -- the source coder and
    LDPC layer are untouched.
    """
    ctype = "qam" if bits_per_symbol >= 2 else "pam"
    mapp = Mapper(constellation_type=ctype, num_bits_per_symbol=bits_per_symbol)
    sym = mapp(tf.convert_to_tensor(np.asarray(cw_bits, dtype=np.float32), tf.float32))
    return sym.numpy().reshape(cw_bits.shape[0], -1)


def qam4_map(cw_bits: np.ndarray) -> np.ndarray:
    """[B, n] bits -> [B, n/4] complex 16-QAM symbols, unit energy. Back-compat wrapper."""
    return modulate_symbols(cw_bits, bits_per_symbol=4)


def _demap_llr(z: np.ndarray, no: np.ndarray, bits_per_symbol: int = 4) -> np.ndarray:
    ctype = "qam" if bits_per_symbol >= 2 else "pam"
    demap = Demapper(demapping_method="maxlog", constellation_type=ctype,
                     num_bits_per_symbol=bits_per_symbol)
    llr = demap(tf.convert_to_tensor(z, tf.complex64),
                tf.convert_to_tensor(np.asarray(no, dtype=np.float32), tf.float32))
    return llr.numpy()


def channel_pass(cw_bits: np.ndarray, snr_db: float, h: np.ndarray,
                 csi_mode: str = "perfect", pilot_frac: float = PILOT_FRAC,
                 rng: Optional[np.random.Generator] = None,
                 bits_per_symbol: int = 4) -> Tuple[np.ndarray, int]:
    """Push coded bits through the block-fading channel.

    Args:
        cw_bits: [B, n] coded bits.
        snr_db: Es/N0 in dB (Es = 1).
        h: [B] complex block gains, CN(0,1).
        csi_mode: 'perfect' or 'pilot'.
        pilot_frac: pilot overhead (0 disables pilots).
        rng: RNG for noise.
        bits_per_symbol: 1 (BPSK), 2 (QPSK), or 4 (16-QAM) -- plan 16.9 Track F.

    Returns (llr [B, n], n_emitted_symbols) where n_emitted_symbols counts the
    symbols actually emitted including pilots (for BPP accounting).
    """
    if rng is None:
        rng = _rng(42)
    B, n = cw_bits.shape
    n_sym = n // bits_per_symbol
    sym = modulate_symbols(cw_bits, bits_per_symbol=bits_per_symbol)  # [B, n_sym]
    N0 = 10 ** (-snr_db / 10.0)
    if csi_mode == "perfect" or pilot_frac <= 0:
        y = h[:, None] * sym + (rng.standard_normal((B, n_sym))
                                + 1j * rng.standard_normal((B, n_sym))) * np.sqrt(N0 / 2)
        z = y / h[:, None]
        no = np.full((B, n_sym), N0, dtype=np.float64) / (np.abs(h) ** 2)[:, None]
        return _demap_llr(z, no, bits_per_symbol=bits_per_symbol), n_sym
    if csi_mode == "pilot":
        # pilot pattern: one pilot every PILOT_EVERY emitted positions
        n_emit = n_sym + int(np.ceil(n_sym / (PILOT_EVERY - 1)))
        pilot_pos = np.arange(0, n_emit, PILOT_EVERY)
        data_pos = np.setdiff1d(np.arange(n_emit), pilot_pos)
        assert data_pos.size >= n_sym, "pilot pattern under-provisions data slots"
        x = np.zeros((B, n_emit), dtype=np.complex128)
        x[:, data_pos[:n_sym]] = sym
        x[:, pilot_pos] = PILOT_SYMBOL
        y = h[:, None] * x + (rng.standard_normal((B, n_emit))
                              + 1j * rng.standard_normal((B, n_emit))) * np.sqrt(N0 / 2)
        # LS estimate per block: mean over pilots
        h_hat = np.mean(y[:, pilot_pos] / PILOT_SYMBOL, axis=1)
        z = y[:, data_pos[:n_sym]] / h_hat[:, None]
        no = np.full((B, n_sym), N0, dtype=np.float64) / (np.abs(h_hat) ** 2)[:, None]
        return _demap_llr(z, no, bits_per_symbol=bits_per_symbol), n_emit
    raise ValueError(f"unknown csi_mode {csi_mode}")


# --------------------------------------------------------------------------
# Block-level transmit/receive
# --------------------------------------------------------------------------
def transmit_and_receive_block(info_bits: np.ndarray, snr_db: float, rate: float,
                               csi_mode: str = "perfect", seed: int = 42,
                               num_iter: int = NUM_ITER_MAX,
                               crc_per_block: bool = True,
                               h: Optional[complex] = None) -> Tuple[Optional[np.ndarray], Dict]:
    """One 1024-bit info block over the digital chain.

    Args:
        info_bits: k_cb = 1024 bits (uint8 0/1). If crc_per_block, the last 32
            bits are the CRC-32 of the first 992 (caller must have appended it;
            see make_block_payload/crc).
        snr_db, rate (1/3, 1/2, 2/3), csi_mode ('perfect'|'pilot'), seed.
        h: optional explicit block gain; otherwise drawn h ~ CN(0,1).

    Returns (payload_or_erased, stats). payload_or_erased is None on failure.
    stats = {blocks, failed, erased_tokens, bit_accuracy, iterations}.
    """
    info_bits = np.asarray(info_bits, dtype=np.uint8).reshape(-1)
    assert info_bits.size == K_CB, f"info block must be {K_CB} bits"
    enc, dec = get_coder(rate)
    rng = _rng(seed)
    if h is None:
        h = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
    cw = enc(tf.convert_to_tensor(info_bits.astype(np.float32)[None, :], tf.float32)).numpy()
    llr, _ = channel_pass(cw, snr_db, np.asarray([h]), csi_mode=csi_mode, rng=rng)
    u_hat, ok, iters = dec.incremental_decode(llr, num_iter=num_iter)
    u_hat = u_hat[0]
    ok = bool(ok[0])
    crc_ok = True
    if crc_per_block and ok:
        crc_ok = crc32_ieee(u_hat[:K_CB - CRC_PER_BLOCK_BITS]) == bits_to_int(
            u_hat[K_CB - CRC_PER_BLOCK_BITS:])
    ok = ok and crc_ok
    payload = u_hat[:K_CB - CRC_PER_BLOCK_BITS] if ok else None
    stats = dict(blocks=1, failed=int(not ok),
                 erased_tokens=0 if ok else (K_CB - CRC_PER_BLOCK_BITS),
                 bit_accuracy=1.0 if ok else 0.0,
                 iterations=int(iters[0]))
    return payload, stats


def make_block_payload(payload_bits: np.ndarray) -> np.ndarray:
    """payload_bits (<= 992) -> 1024-bit info block with trailing CRC-32."""
    payload_bits = np.asarray(payload_bits, dtype=np.uint8).reshape(-1)
    assert payload_bits.size <= K_CB - CRC_PER_BLOCK_BITS
    pad = np.zeros(K_CB - CRC_PER_BLOCK_BITS - payload_bits.size, dtype=np.uint8)
    body = np.concatenate([payload_bits, pad])
    crc = int_to_bits(crc32_ieee(body), 32)
    return np.concatenate([body, crc])


def simulate_digital_pipeline(bits: np.ndarray, snr_db: float, k_cb: int = K_CB,
                              rate: float = 0.5, csi_mode: str = "perfect",
                              seed: int = 42, num_iter: int = NUM_ITER_MAX,
                              crc_per_block: bool = True,
                              bits_per_symbol: int = 4,
                              ) -> Tuple[np.ndarray, Dict]:
    """Run many info blocks through the digital chain (vectorized).

    Args:
        bits: payload bits; segmented into (k_cb - 32) pieces when
            crc_per_block=True, else k_cb pieces (trailing block zero-padded).
        rate: payload code rate (1/3, 1/2, 2/3).
        csi_mode: 'perfect' | 'pilot'.
        bits_per_symbol: 1 (BPSK), 2 (QPSK), or 4 (16-QAM) -- plan 16.9 Track F.

    Returns (bits_out, stats):
        bits_out: np.ndarray same length as bits; -1 marks erased bits.
        stats: {blocks, failed, erased_tokens, bit_accuracy, ber, iterations_mean}.
    """
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    rng = _rng(seed)
    enc, dec = get_coder(rate)
    block_payload = k_cb - (CRC_PER_BLOCK_BITS if crc_per_block else 0)
    n_blocks = int(np.ceil(bits.size / block_payload))
    pad_len = n_blocks * block_payload - bits.size
    bits_pad = np.concatenate([bits, np.zeros(pad_len, dtype=np.uint8)])
    info_blocks = bits_pad.reshape(n_blocks, block_payload)
    if crc_per_block:
        info_blocks = np.stack([make_block_payload(r) for r in info_blocks])
    h = (rng.standard_normal(n_blocks) + 1j * rng.standard_normal(n_blocks)) / np.sqrt(2)
    cw = enc(tf.convert_to_tensor(info_blocks.astype(np.float32), tf.float32)).numpy()
    llr, _ = channel_pass(cw, snr_db, h, csi_mode=csi_mode, rng=rng,
                          bits_per_symbol=bits_per_symbol)
    u_hat, ok, iters = dec.incremental_decode(llr, num_iter=num_iter)
    if crc_per_block:
        for b in range(n_blocks):
            if ok[b]:
                body = u_hat[b, :k_cb - CRC_PER_BLOCK_BITS]
                ok[b] = crc32_ieee(body) == bits_to_int(u_hat[b, k_cb - CRC_PER_BLOCK_BITS:])
    body = u_hat[:, :block_payload] if crc_per_block else u_hat
    out = body.reshape(-1)[:bits.size].copy()
    out = out.astype(np.int16)
    erased_mask = np.repeat(~ok, block_payload)[:bits.size]
    out[erased_mask] = -1
    n_erased_bits = int(np.sum(erased_mask))
    ok_bits = bits[:out.size].astype(int) == out[:bits.size].astype(int)
    ok_bits = ok_bits[~erased_mask]
    stats = dict(
        blocks=n_blocks,
        failed=int(np.sum(~ok)),
        erased_tokens=n_erased_bits,
        bit_accuracy=float(np.mean(ok_bits)) if ok_bits.size else 0.0,
        ber=float(1.0 - np.mean(ok_bits)) if ok_bits.size else 1.0,
        iterations_mean=float(np.mean(iters)),
    )
    return out, stats


# --------------------------------------------------------------------------
# Frame format (plan 3.3): HEADER | MASK | INDEX | PAYLOAD_CRC | PADDING
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

def mask_decode(mask_bits: np.ndarray, mode: int, n_tokens: int, k_high: int) -> np.ndarray:
    if mode == MASK_MODE_EXPLICIT:
        assert mask_bits.size == n_tokens
        return np.asarray(mask_bits, dtype=bool)
    if mode == MASK_MODE_ENUM:
        rank = bits_to_int(mask_bits)
        return np.isin(np.arange(n_tokens), subset_unrank(rank, n_tokens, k_high))
    raise ValueError(f"unsupported mask_mode {mode}")


def build_index_bits(token_values: np.ndarray, high_mask: np.ndarray,
                     b_high: int, b_low: int) -> np.ndarray:
    """Ascending token order: high tokens K x b_high, then low tokens x b_low."""
    parts = []
    for t in range(token_values.size):
        w = b_high if high_mask[t] else b_low
        parts.append(int_to_bits(int(token_values[t]) & ((1 << w) - 1), w))
    return np.concatenate(parts)


def build_frame(token_values: np.ndarray, high_mask: np.ndarray,
                mask_mode: int = MASK_MODE_EXPLICIT, config_id: int = CONFIG_18_2,
                version: int = 1, n_tokens: Optional[int] = None) -> Dict:
    """Build the §3.3 frame. Returns dict with header/mask/index/crc sections
    and the two protection segments (segment_a: header+mask padded to k_cb,
    segment_b: index+payload_crc padded to k_cb multiples)."""
    token_values = np.asarray(token_values).reshape(-1)
    high_mask = np.asarray(high_mask).reshape(-1)
    if n_tokens is None:
        n_tokens = int(token_values.size)
    n_tokens = int(n_tokens)
    cfg = CONFIGS[config_id]
    b_high, b_low = cfg["b_high"], cfg["b_low"]
    k_high = int(np.count_nonzero(high_mask))
    assert token_values.size == n_tokens
    assert high_mask.size == n_tokens
    index = build_index_bits(token_values, high_mask, b_high, b_low)
    if mask_mode == MASK_MODE_EXPLICIT:
        mask = mask_bits_explicit(high_mask)
    elif mask_mode == MASK_MODE_ENUM:
        mask = mask_bits_enumerative(high_mask, n_tokens, k_high)
    else:
        raise ValueError(mask_mode)
    header = build_header(version, config_id, mask_mode, n_tokens, k_high,
                          mask.size, index.size)
    payload_crc = int_to_bits(crc32_ieee(index), 32)
    seg_a = np.concatenate([header, mask])
    seg_a = np.concatenate([seg_a, np.zeros(K_CB - seg_a.size % K_CB, dtype=np.uint8)]) \
        if seg_a.size % K_CB else seg_a
    seg_b = np.concatenate([index, payload_crc])
    seg_b = np.concatenate([seg_b, np.zeros(K_CB - seg_b.size % K_CB, dtype=np.uint8)]) \
        if seg_b.size % K_CB else seg_b
    return dict(header=header, mask=mask, mask_mode=mask_mode, index=index,
                payload_crc=payload_crc, segment_a=seg_a, segment_b=seg_b,
                config_id=config_id, k_high=k_high, b_high=b_high, b_low=b_low,
                n_tokens=n_tokens, n_blocks_a=seg_a.size // K_CB,
                n_blocks_b=seg_b.size // K_CB)


def parse_frame(seg_a_blocks: List[Optional[np.ndarray]],
                seg_b_blocks: List[Optional[np.ndarray]]) -> Dict:
    """Parse a received frame. Blocks are k_cb-bit arrays or None (erased).

    Returns dict with header fields, high_mask (None if header lost), token
    values (np.ndarray of size n_tokens with None entries for erased tokens),
    and flags (header_ok, payload_crc_ok, frame_lost).

    Erasure rule (documented): a code block that fails its LDPC syndrome is
    passed as None and its bits are unknown; tokens overlapping erased blocks
    are marked erased. If the frame-level PAYLOAD_CRC-32 fails (or cannot be
    checked because a block is erased), all payload tokens are marked erased.
    """
    out = dict(header_ok=False, payload_crc_ok=False, frame_lost=True,
               high_mask=None, tokens=None, erased_tokens=0)
    if not seg_a_blocks or seg_a_blocks[0] is None:
        return out
    head_bits = seg_a_blocks[0][:HEADER_LEN]
    hdr = parse_header(head_bits)
    if hdr["magic"] != MAGIC or not hdr["header_crc_ok"]:
        return out
    out["header_ok"] = True
    out.update(hdr)
    n_tokens = int(out["n_tokens"])
    mask_bits = seg_a_blocks[0][HEADER_LEN:HEADER_LEN + hdr["mask_bits"]]
    if mask_bits.size < hdr["mask_bits"]:
        return out
    try:
        high_mask = mask_decode(mask_bits, hdr["mask_mode"],
                                hdr["n_tokens"], hdr["k_high"])
    except Exception:
        return out
    out["high_mask"] = high_mask
    out["frame_lost"] = False
    cfg = CONFIGS.get(out["config_id"], CONFIGS[CONFIG_18_2])
    b_high, b_low = cfg["b_high"], cfg["b_low"]
    index_len = out["payload_bits"]
    payload_total = index_len + 32
    n_blocks = int(np.ceil(payload_total / K_CB))
    index_bits = np.zeros(index_len, dtype=np.uint8)
    erased_block = np.zeros(n_blocks, dtype=bool)
    for i in range(n_blocks):
        blk = seg_b_blocks[i] if i < len(seg_b_blocks) else None
        if blk is None:
            erased_block[i] = True
            continue
        lo = i * K_CB
        if lo < index_len:
            seg_len = min(K_CB, index_len - lo)
            index_bits[lo:lo + seg_len] = blk[:seg_len]
    crc_ok = False
    if not np.any(erased_block):
        crc_bits = []
        for i in range(n_blocks):
            lo = i * K_CB
            take = min(K_CB, payload_total - lo)
            crc_bits.append(seg_b_blocks[i][:take])
        all_payload = np.concatenate(crc_bits)
        crc_ok = crc32_ieee(all_payload[:index_len]) == bits_to_int(
            all_payload[index_len:index_len + 32])
    out["payload_crc_ok"] = crc_ok
    # map bits -> tokens (ascending order, per-token width)
    tokens = [None] * n_tokens
    bitpos = 0
    for t in range(n_tokens):
        w = b_high if high_mask[t] else b_low
        blk_idx = bitpos // K_CB
        if bitpos + w <= index_len and not erased_block[blk_idx]:
            tokens[t] = bits_to_int(index_bits[bitpos:bitpos + w])
        bitpos += w
    if not crc_ok:
        tokens = [None] * n_tokens
    out["tokens"] = np.array(tokens, dtype=object)
    out["erased_tokens"] = int(sum(v is None for v in tokens))
    return out


# --------------------------------------------------------------------------
# Spatial N4 concealment on the 28x28 token grid
# --------------------------------------------------------------------------
def conceal_tokens(tokens: List[Optional[int]], n_tokens: int = N_TOKENS,
                   grid_h: int = GRID_H, grid_w: int = GRID_W) -> Tuple[np.ndarray, Dict]:
    """Iterative N4 (4-neighbour) mean interpolation of erased token values.

    Rule: while any erased tokens remain, each erased token is replaced by the
    mean of its non-erased (or already filled) 4-neighbours' values; tokens
    with no available neighbour wait for the next pass. When a pass makes no
    progress, remaining erased tokens take the mean of all known values; if
    all tokens are erased the grid is filled with 0. Row-major token order:
    token t -> grid (t // grid_w, t % grid_w). Defaults to 28x28 (patch-8,
    784 tokens); pass grid_h=grid_w=14 for patch-16 (196 tokens) — see
    tests/test_digital_pipeline.py:test_patch16_n_tokens for the regression check.
    """
    grid = np.full((grid_h, grid_w), np.nan, dtype=np.float64)
    for t in range(n_tokens):
        v = tokens[t]
        if v is not None:
            grid[t // grid_w, t % grid_w] = float(v)
    erased = np.isnan(grid)
    stats = dict(erased_before=int(erased.sum()), passes=0, erased_after=0,
                 fallback_used=0)
    if not erased.any():
        return grid.astype(np.int64), stats
    n_erased = int(erased.sum())
    for _ in range(n_erased + 2):          # at most one fill per token
        if not np.isnan(grid).any():
            break
        new = grid.copy()
        progress = False
        for (r, c) in np.argwhere(np.isnan(grid)):
            nb = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < grid_h and 0 <= cc < grid_w and not np.isnan(grid[rr, cc]):
                    nb.append(grid[rr, cc])
            if nb:
                new[r, c] = float(np.mean(nb))
                progress = True
        grid = new
        stats["passes"] += 1
        if not progress:
            break
    if np.isnan(grid).any():
        known = grid[~np.isnan(grid)]
        fill = float(np.mean(known)) if known.size else 0.0
        grid[np.isnan(grid)] = fill
        stats["fallback_used"] = 1
    stats["erased_after"] = 0
    return grid.astype(np.int64), stats


# --------------------------------------------------------------------------
# Sweeps / demos
# --------------------------------------------------------------------------
def outage_bound(snr_db: float) -> float:
    gamma = 10 ** (snr_db / 10.0)
    return 1.0 - math.exp(-OUTAGE_BOUND_GAMMA / gamma)


def run_bler_sweep(snr_list: List[float], blocks: int = 500, rate: float = 0.5,
                   csi_mode: str = "perfect", seed: int = 42,
                   csv_path: Optional[str] = None, log: bool = True) -> List[Dict]:
    """BLER vs SNR sweep (acceptance). Writes CSV with columns
    snr_db, blocks, failed, bler, ber, erased_fraction, bit_accuracy,
    iterations_mean, outage_bound."""
    rows = []
    payload_per_block = K_CB - CRC_PER_BLOCK_BITS
    for snr in snr_list:
        rng = _rng(seed + int(round(snr * 100)))
        payload = rng.integers(0, 2, size=blocks * payload_per_block).astype(np.uint8)
        _, stats = simulate_digital_pipeline(payload, snr, rate=rate,
                                             csi_mode=csi_mode, seed=seed)
        rows.append(dict(
            snr_db=snr, blocks=stats["blocks"], failed=stats["failed"],
            bler=stats["failed"] / stats["blocks"],
            ber=stats["ber"],
            erased_fraction=stats["erased_tokens"] / (blocks * payload_per_block),
            bit_accuracy=stats["bit_accuracy"],
            iterations_mean=stats["iterations_mean"],
            outage_bound=outage_bound(snr)))
        if log:
            print(f"[sweep] snr={snr:+.0f}dB {csi_mode} blocks={stats['blocks']} "
                  f"failed={stats['failed']} bler={rows[-1]['bler']:.4f} "
                  f"bound={rows[-1]['outage_bound']:.4f} "
                  f"bit_acc={stats['bit_accuracy']:.4f} iter={stats['iterations_mean']:.1f}")
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    return rows


def one_image_demo(snr_list: Tuple[float, ...] = (5.0, 15.0), seed: int = 42,
                   config_id: int = CONFIG_18_2,
                   mask_mode: int = MASK_MODE_ENUM,
                   frames_per_snr: int = 50,
                   n_tokens: Optional[int] = None,
                   grid_h: Optional[int] = None,
                   grid_w: Optional[int] = None) -> List[Dict]:
    """End-to-end one-image demo (plan 5.5/5.6 acceptance).

    Builds a plausible 18/2b frame (K=63 high tokens x 18b, 721 low tokens x 2b,
    enumerative mask), runs header->LDPC(UEP)->16-QAM->Rayleigh->equalize->
    demap->decode->CRC->parse->conceal, reports BER/BLER and BPP counted from
    the ACTUAL emitted bits (LDPC parity + pilots + header included, 224x224
    image = 50176 px). Each SNR runs ``frames_per_snr`` independent channel
    realizations of the same frame (block fading per code block).
    """
    rng = _rng(seed)
    if n_tokens is None:
        n_tokens = N_TOKENS
    if grid_h is None:
        grid_h = GRID_H
    if grid_w is None:
        grid_w = GRID_W
    cfg = CONFIGS[config_id]
    k_high, b_high, b_low = cfg["k_high"], cfg["b_high"], cfg["b_low"]
    token_values = rng.integers(0, 2 ** b_low, size=n_tokens)
    high_idx = rng.choice(n_tokens, size=k_high, replace=False)
    high_mask = np.isin(np.arange(n_tokens), high_idx)
    token_values[high_idx] = rng.integers(0, 2 ** b_high, size=k_high)
    frame = build_frame(token_values, high_mask, mask_mode=mask_mode,
                        config_id=config_id, n_tokens=n_tokens)
    enc_a, dec_a = get_coder(1 / 3)
    enc_b, dec_b = get_coder(1 / 2)
    results = []
    for snr in snr_list:
        rng2 = _rng(seed + int(round(snr * 10)))
        header_ok, frame_lost, crc_ok = 0, 0, 0
        erased_tokens, total_index_err = 0, 0
        total_symbols = 0
        for _ in range(frames_per_snr):
            h_a = (rng2.standard_normal(frame["n_blocks_a"])
                   + 1j * rng2.standard_normal(frame["n_blocks_a"])) / np.sqrt(2)
            h_b = (rng2.standard_normal(frame["n_blocks_b"])
                   + 1j * rng2.standard_normal(frame["n_blocks_b"])) / np.sqrt(2)
            seg_a_blocks, seg_b_blocks = [], []
            for i in range(frame["n_blocks_a"]):
                blk = frame["segment_a"][i * K_CB:(i + 1) * K_CB]
                cw = enc_a(tf.convert_to_tensor(blk.astype(np.float32)[None, :], tf.float32)).numpy()
                llr, n_sym = channel_pass(cw, snr, np.asarray([h_a[i]]),
                                          csi_mode="pilot", rng=rng2)
                u, ok, _ = dec_a.incremental_decode(llr)
                seg_a_blocks.append(u[0].astype(np.uint8) if ok[0] else None)
                total_symbols += n_sym
            for i in range(frame["n_blocks_b"]):
                blk = frame["segment_b"][i * K_CB:(i + 1) * K_CB]
                cw = enc_b(tf.convert_to_tensor(blk.astype(np.float32)[None, :], tf.float32)).numpy()
                llr, n_sym = channel_pass(cw, snr, np.asarray([h_b[i]]),
                                          csi_mode="pilot", rng=rng2)
                u, ok, _ = dec_b.incremental_decode(llr)
                seg_b_blocks.append(u[0].astype(np.uint8) if ok[0] else None)
                total_symbols += n_sym
            parsed = parse_frame(seg_a_blocks, seg_b_blocks)
            recovered = [None] * n_tokens
            if parsed["tokens"] is not None:
                recovered = list(parsed["tokens"])
            filled, conceal_stats = conceal_tokens(recovered, n_tokens=n_tokens,
                                                   grid_h=grid_h, grid_w=grid_w)
            rx_index_parts = []
            for t in range(n_tokens):
                w = b_high if high_mask[t] else b_low
                rx_index_parts.append(int_to_bits(int(filled[t // grid_w, t % grid_w]), w))
            rx_index = np.concatenate(rx_index_parts)
            total_index_err += int(np.sum(rx_index != frame["index"]))
            erased_tokens += conceal_stats["erased_before"]
            header_ok += int(parsed["header_ok"])
            frame_lost += int(parsed["frame_lost"])
            crc_ok += int(parsed["payload_crc_ok"])
        n_frames = frames_per_snr
        bpp = 4 * (total_symbols // n_frames) / 50176.0
        results.append(dict(
            snr_db=snr, frames=n_frames,
            header_ok_rate=header_ok / n_frames,
            frame_lost_rate=frame_lost / n_frames,
            payload_crc_ok_rate=crc_ok / n_frames,
            index_bits=frame["index"].size,
            mean_index_ber=total_index_err / (n_frames * frame["index"].size),
            mean_erased_tokens=erased_tokens / n_frames,
            emitted_symbols_per_frame=total_symbols // n_frames,
            emitted_bits_per_frame=4 * (total_symbols // n_frames),
            bpp=bpp,
            source_bits=K_CB * (frame["n_blocks_a"] + frame["n_blocks_b"])))
        print(f"[demo] snr={snr:+.0f}dB header_ok={results[-1]['header_ok_rate']:.2f} "
              f"frame_lost={results[-1]['frame_lost_rate']:.2f} "
              f"crc_ok={results[-1]['payload_crc_ok_rate']:.2f} "
              f"mean_index_ber={results[-1]['mean_index_ber']:.4f} "
              f"mean_erased={results[-1]['mean_erased_tokens']:.1f} "
              f"emitted={results[-1]['emitted_symbols_per_frame']} sym "
              f"({results[-1]['bpp']:.4f} bpp)")
    return results
