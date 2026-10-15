"""Sionna-only matched5 probes; run with __SIONNA_PY__.

This module is isolated from the released transport. No production defaults
are changed. Probe output records actual calls and block counts.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import digital_pipeline as dp
import digital_bridge as bridge
import bitstream as bs
if __package__:
    from .protocol import channel_draw, SYMBOLS, SNR_DB
else:
    from protocol import channel_draw, SYMBOLS, SNR_DB
import tensorflow as tf
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper


class Codec:
    """Real BG2 encoding, normalized Gray modulation, fixed-iteration BP.

    CRC32 in the framing caller determines acceptance. Transmitted bits are
    never used to accept a received message. Perfect receiver CSI only.
    """
    def __init__(self, k, n, bps, decoder="boxplus-phi", iterations=40):
        self.k, self.n, self.bps = int(k), int(n), int(bps)
        self.encoder = LDPC5GEncoder(k=self.k, n=self.n, bg="bg2")
        self.decoder = LDPC5GDecoder(
            self.encoder, cn_update=dp.cn_update_nms if decoder == "nms" else decoder,
            hard_out=True, return_infobits=True, num_iter=iterations, llr_max=20)
        ctype = "pam" if bps == 1 else "qam"
        self.mapper = Mapper(constellation_type=ctype, num_bits_per_symbol=bps)
        self.demapper = Demapper("app", constellation_type=ctype, num_bits_per_symbol=bps)
        self.run_graph = tf.function(self._run, reduce_retracing=True)

    def _run(self, info, h, noise, n0):
        coded = self.encoder(info)
        symbols = self.mapper(coded)
        rx = h[:, None] * symbols + noise * tf.cast(tf.sqrt(n0 / 2), tf.complex64)
        equalized = rx / h[:, None]
        no = n0 / tf.square(tf.abs(h[:, None]))
        llr = self.demapper(equalized, no)
        return self.decoder(llr), tf.shape(coded)[1], tf.shape(symbols)[1]

    def transmit(self, info, h, noise, snr_db):
        info = np.asarray(info, dtype=np.float32)
        h = np.asarray(h, dtype=np.complex64).reshape(-1)
        noise = np.asarray(noise, dtype=np.complex64)
        assert info.shape == (h.size, self.k)
        assert noise.shape == (h.size, self.n // self.bps)
        decoded, n, n_sym = self.run_graph(
            tf.constant(info), tf.constant(h), tf.constant(noise),
            tf.constant(10 ** (-float(snr_db) / 10), tf.float32))
        assert int(n) == self.n and int(n_sym) == self.n // self.bps
        return decoded.numpy().astype(np.uint8), int(n), int(n_sym)


def link_sweep(out_dir, count):
    """Synthetic link selection; no image/test data are read."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    configs = [(3073, 6144, 1, "frame_bpsk"),
               (3073, 12288, 2, "frame_qpsk"),
               (1024, 3072, 2, "compact_block_qpsk")]
    results = []
    for k, n, bps, name in configs:
        for algorithm in ("nms", "boxplus-phi"):
            codec = Codec(k, n, bps, algorithm, 40)
            rng = np.random.default_rng(19092026)
            errors, failures, symbols = 0, 0, 0
            start = time.monotonic()
            for offset in range(0, count, 32):
                batch = min(32, count - offset)
                info = rng.integers(0, 2, (batch, k), dtype=np.uint8)
                h = (rng.standard_normal(batch) + 1j*rng.standard_normal(batch)) / np.sqrt(2)
                noise = rng.standard_normal((batch, n//bps)) + 1j*rng.standard_normal((batch, n//bps))
                rx, emitted_bits, symbols = codec.transmit(info, h, noise, 5)
                errors += int(np.sum(rx != info))
                failures += int(np.sum(np.any(rx != info, axis=1)))
            row = dict(config=name, decoder=algorithm, iterations=40, k=k,
                       coded_bits=emitted_bits, bps=bps, symbols=symbols,
                       blocks=count, bler=failures/count, ber=errors/(count*k),
                       seconds=time.monotonic()-start, snr_db=5,
                       data="synthetic, not image validation or test")
            results.append(row)
            print(json.dumps(row), flush=True)
            (out_dir / "link.json").write_text(json.dumps(results, indent=2)+"\n")


def pack_joint_frame(tokens, mask, n_stages=3, stage_bits=6, b_low=2, k_high=63,
                     n_tokens=784):
    """Version 2: header | enumerative mask | stage-major indices | CRC32.

    Single-frame CRC covers the header AND mask as well as all index bits.
    No outer segment padding, no inter-image aggregation, no missing side cost.

    2026-09-19: n_stages/stage_bits/b_low/k_high/n_tokens are now overridable
    (defaults = flagship 3/6/2/63/784) for rate-matched codec variants (e.g.
    r061: 3/4/1/32/784) using bitstream.py's generalized `_n` pack/unpack
    functions -- byte-identical to the old hardcoded path at the defaults
    (verified: split_high_value_n(v,3,6) computes the same shifts/mask as
    the old split_high_value(v), and pack_rvq_indices_n's field layout
    matches pack_rvq_indices's exactly for n_stages=3,b_low=2).
    """
    assert tokens.shape == mask.shape == (n_tokens,) and mask.sum() == k_high
    stages, low = bs.token_values_to_stages_n(tokens, mask, n_stages, stage_bits, b_low)
    index = bs.pack_rvq_indices_n(stages, low, mask, stage_bits, b_low, n_tokens)
    mask_bits = bs.mask_bits_enumerative(mask, n_tokens, k_high)
    header = bs.build_header(2, 0, 3, n_tokens, k_high, mask_bits.size, index.size)
    body = np.concatenate([header, mask_bits, index])
    frame = np.concatenate([body, bs.int_to_bits(bs.crc32_ieee(body), 32)])
    return frame


def joint_frame_info_bits(n_stages=3, stage_bits=6, b_low=2, k_high=63, n_tokens=784):
    """Total frame size (bits) for a given codec config -- 3073 at the
    flagship defaults, matching pack_joint_frame's old hardcoded assert."""
    mask_bits = bs.enum_mask_bits(n_tokens, k_high)
    payload_bits = n_stages * k_high * stage_bits + (n_tokens - k_high) * b_low
    return bs.HEADER_LEN + mask_bits + payload_bits + 32


def parse_joint_frame(received, n_stages=3, stage_bits=6, b_low=2, k_high=63,
                      n_tokens=784):
    """No access to transmitted tokens, masks, or labels on this side.

    2026-09-19: generalized the same way as pack_joint_frame (see there);
    defaults reproduce the old hardcoded flagship-only behavior exactly.
    """
    values = np.zeros(n_tokens, dtype=np.int32)
    mask = np.zeros(n_tokens, dtype=bool)
    info_bits = joint_frame_info_bits(n_stages, stage_bits, b_low, k_high, n_tokens)
    if len(received) != info_bits:
        return values, mask, False
    if bs.crc32_ieee(received[:-32]) != bs.bits_to_int(received[-32:]):
        return values, mask, False
    hdr = bs.parse_header(received[:152])
    mask_bits_len = bs.enum_mask_bits(n_tokens, k_high)
    payload_bits_len = n_stages * k_high * stage_bits + (n_tokens - k_high) * b_low
    expected = dict(magic=bs.MAGIC, version=2, config_id=0, mask_mode=3,
                    n_tokens=n_tokens, k_high=k_high, mask_bits=mask_bits_len,
                    payload_bits=payload_bits_len, reserved=0, header_crc_ok=True)
    if any(hdr[k] != v for k, v in expected.items()):
        return values, mask, False
    try:
        mask = bs.mask_decode(received[152:152+mask_bits_len], 3, n_tokens, k_high)
        stages, low = bs.unpack_rvq_indices_n(received[152+mask_bits_len:-32], mask,
                                              n_stages, stage_bits, b_low, n_tokens)
        values = bs.stages_to_token_values_n(stages, low, mask, stage_bits, b_low,
                                             n_tokens).astype(np.int32)
    except (ValueError, AssertionError, IndexError):
        return np.zeros(n_tokens,dtype=np.int32), np.zeros(n_tokens,dtype=bool), False
    return values, mask, True


_JOINT_CODEC = None


def transmit_joint_image(token_values, high_mask, b_high, b_low, snr_db,
                         csi_mode, seed):
    """Opt-in real-time bridge adapter for the validation-locked QPSK profile.

    Same frame, code, decoder, and draw order as the cached evaluation. No
    training or test data are read. Input -> actual PHY -> parsed receiver fields.
    BER here includes failed frames, measured before CRC erasure.
    """
    global _JOINT_CODEC
    if (b_high,b_low)!=(18,2) or csi_mode!="perfect":
        raise ValueError("joint QPSK requires flagship RVQ 18/2 and perfect receiver CSI")
    tokens=np.asarray(token_values,dtype=np.int64)
    mask=np.asarray(high_mask,dtype=bool)
    frame=pack_joint_frame(tokens,mask)
    if _JOINT_CODEC is None:
        _JOINT_CODEC=Codec(3073,12288,2,"boxplus-phi",40)
    rng=np.random.default_rng(int(seed))
    h=np.complex64((rng.standard_normal()+1j*rng.standard_normal())/np.sqrt(2))
    raw=rng.standard_normal(2*SYMBOLS).astype(np.float32)
    noise=raw[:SYMBOLS]+1j*raw[SYMBOLS:]
    rx,n_bits,n_sym=_JOINT_CODEC.transmit(frame[None],np.array([h]),noise[None],snr_db)
    values,received_mask,ok=parse_joint_frame(rx[0])
    stages=np.zeros((784,3),dtype=np.int64)
    stage_bad=np.ones((784,3),dtype=bool)
    low=np.zeros(784,dtype=np.int64)
    low_bad=np.ones(784,dtype=bool)
    if ok:
        for stage,shift in enumerate((12,6,0)):
            stages[received_mask,stage]=(values[received_mask] >> shift) & 63
            stage_bad[received_mask,stage]=False
        low[~received_mask]=values[~received_mask]
        low_bad[~received_mask]=False
    recovered=values.astype(np.int64) if ok else np.full(784,-1,dtype=np.int64)
    ber=float(np.mean(rx[0]!=frame))
    st=dict(header_ok=ok,frame_lost=not ok,payload_crc_ok=ok,blocks=1,
        failed=int(not ok),bler=float(not ok),ber=ber,bit_accuracy=1-ber,
        erased_tokens=0 if ok else 784,erased_token_fraction=float(not ok),
        stage_erased_count=[0 if ok else 63]*3,iterations_mean=40.,n_tokens=784,
        joint_frame=True,channel_uses_per_px=dp.channel_uses_per_px(
            [],2,coded_block_lengths=[n_bits]))
    assert n_sym==6144
    return recovered,st,stages,stage_bad,low,low_bad


def transmit_cache(cli):
    # SNR sweep (2026-09-15): SNR_DB (module constant, 5.0) was the only point
    # this file could evaluate -- the LOCK.json selection was made at 5dB only.
    # --snr-db generalizes this. channel_bank.npz stores INDEPENDENT h per SNR
    # index (verified: h[:, s, :] differs across s, not reused across the sweep
    # -- see RESULTS_OF_RECORD.md), so channel_draw()'s fixed-formula shortcut
    # (designed and asserted to match ONLY bank snr_idx=2, i.e. 5dB) cannot be
    # reused at other SNRs by just rescaling noise -- that would silently pair
    # every point against the 5dB fading draws, not the SNR-matched draws the
    # already-recorded ROI-JSCC numbers at other SNRs were computed against.
    # For split=="test", read h and noise DIRECTLY from the bank at the
    # requested SNR's index instead -- exact, no seed-arithmetic re-derivation.
    snr_db = float(getattr(cli, "snr_db", 5.0))
    # Wave 21 (2026-09-15): --symbols overrides the module constant SYMBOLS
    # (6144, the 0.122449-uses/px operating point) to build a SECOND joint-frame
    # rate point without touching the already-verified default path -- omitting
    # --symbols reproduces the exact prior behavior byte-for-byte (symbols=SYMBOLS).
    symbols = int(getattr(cli, "symbols", None) or SYMBOLS)
    out = Path(cli.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    src = np.load(cli.source, allow_pickle=False)
    source_meta = json.loads(Path(cli.source).with_name("meta.json").read_text())
    assert source_meta["split"] == cli.split
    # Wave 21 F9 (2026-09-15): site="b" reads channel_bank_siteb.npz instead of
    # channel_bank.npz -- zero-shot, same flagship checkpoint/codec, only the
    # data root and bank differ (propagated from cache_source's own meta.json).
    site = source_meta.get("site", "a")
    bank_path = ROOT / ("channel_bank_siteb.npz" if site == "b" else "channel_bank.npz")
    bps = {"frame_bpsk":1, "frame_qpsk":2, "frame_16qam":4}[cli.variant]
    # 2026-09-19: codec config (n_stages/stage_bits/b_low/k_high), read from
    # cache_source's own meta.json -- defaults reproduce the flagship exactly
    # (3/6/2/63 -> info_bits=3073, the old hardcoded value). For a
    # rate-matched variant (e.g. r061: 3/4/1/32 -> info_bits=1510), this is
    # the ONLY change needed to transmit its actual (smaller) frame -- the
    # LDPC encoder/decoder already support arbitrary k<n.
    n_stages = source_meta.get("rvq_stages", 3)
    stage_bits = source_meta.get("rvq_bits_per_stage", 6)
    b_low = source_meta.get("bits_vq_low", 2)
    k_high = source_meta.get("routing_topk", 63)
    info_bits = joint_frame_info_bits(n_stages, stage_bits, b_low, k_high)
    codec = Codec(info_bits, symbols*bps, bps, "boxplus-phi", 40)
    frames = np.stack([pack_joint_frame(t,m,n_stages,stage_bits,b_low,k_high)
                       for t,m in zip(src["tokens"],src["high_mask"])])
    count = len(frames)
    bank_h = bank_n = None
    if cli.split == "test":
        bank = np.load(bank_path, allow_pickle=False)
        assert src["names"].tolist() == bank["image_names"].tolist()
        bank_snr = [float(s) for s in bank["snr_db"]]
        if snr_db not in bank_snr:
            raise ValueError(f"snr_db={snr_db} not in channel_bank.npz grid {bank_snr}")
        s_idx = bank_snr.index(snr_db)
        need_reals = 2 * symbols
        assert bank["n"].shape[-1] >= need_reals, (
            f"bank noise budget {bank['n'].shape[-1]} < required {need_reals} reals/image")
        bank_h = bank["h"][:, s_idx, :]           # [count, M] complex64
        bank_n = bank["n"][:, s_idx, :, :need_reals]  # [count, M, 2*symbols] float32
        # Sanity, not an assumption: at 5dB (bank snr_idx=2), with the DEFAULT
        # symbols count, this must reproduce the already-verified channel_draw()
        # shortcut exactly. Only meaningful when symbols==SYMBOLS (channel_draw's
        # own noise vector is sized 2*SYMBOLS, not 2*symbols, for any override)
        # AND site=="a" (channel_draw's seed formula matches only Site A's
        # subseed base 44,000,000, not Site B's 55,000,000 -- channel_bank_siteb.py).
        if snr_db == 5.0 and symbols == SYMBOLS and site == "a":
            for i in (0, count-1):
                for m in range(cli.m):
                    assert bank_h[i, m] == channel_draw(cli.split, i, m)[0]
    values = np.zeros((count,cli.m,784),dtype=np.int32)
    masks = np.zeros((count,cli.m,784),dtype=bool)
    ok = np.zeros((count,cli.m),dtype=bool)
    start = time.monotonic()
    for m in range(cli.m):
        for offset in range(0,count,32):
            end = min(count,offset+32)
            if cli.split == "test":
                h = bank_h[offset:end, m]
                raw = bank_n[offset:end, m, :]  # [batch, 2*symbols] float32
                noise = raw[:, :symbols] + 1j*raw[:, symbols:]
            else:
                draws = [channel_draw(cli.split,i,m) for i in range(offset,end)]
                h, noise = np.asarray([d[0] for d in draws]), np.stack([d[1] for d in draws])
            rx,n_bits,n_sym = codec.transmit(frames[offset:end],h,noise,snr_db)
            for j,row in enumerate(rx):
                i = offset+j
                values[i,m],masks[i,m],ok[i,m] = parse_joint_frame(
                    row,n_stages,stage_bits,b_low,k_high)
                if ok[i,m] and not np.array_equal(row,frames[i]):
                    raise RuntimeError("undetected CRC error; retain and audit before reporting")
        print(json.dumps(dict(variant=cli.variant,realization=m,snr_db=snr_db,
                              frame_loss=float(1-ok[:,m].mean()))),flush=True)
    rate = dp.channel_uses_per_px([],bps,coded_block_lengths=[n_bits])
    assert n_sym == symbols and rate == symbols/(224*224)
    np.savez_compressed(out/"receiver.npz",tokens=values,mask=masks,frame_ok=ok,
                        names=src["names"])
    meta = dict(split=cli.split,site=site,variant=cli.variant,source=str(Path(cli.source).resolve()),
        symbols_per_image=n_sym,channel_uses_per_px=rate,info_bits=info_bits,coded_bits=n_bits,
        bits_per_symbol=bps,ldpc_base_graph="BG2",decoder="boxplus-phi",iterations=40,
        demapper="app",frame_crc="CRC32 over header+mask+indices",frame_version=2,
        snr_db=snr_db,csi="perfect_receiver",fading="one CN(0,1) h per image",
        bank=str(bank_path),
        M=cli.m,n=count,frame_loss_rate=float(1-ok.mean()),
        seconds=time.monotonic()-start,seed_base=44_000_000 if cli.split=="test" else 914_000_000,
        # 2026-09-18 (AquaSC-lite): propagate the encoder identity from
        # cache_source's own meta.json so evaluate.py's evaluate_aquasc()
        # rebuilds the SAME (possibly non-flagship) model for scoring,
        # without needing separate CLI flags on that phase too.
        checkpoint=source_meta.get("checkpoint"),
        checkpoint_sha256=source_meta.get("checkpoint_sha256"),
        encoder_embed_dim=source_meta.get("encoder_embed_dim",512),
        encoder_depth=source_meta.get("encoder_depth",8),
        encoder_num_heads=source_meta.get("encoder_num_heads",8),
        # 2026-09-19: same propagation for the RVQ/routing config (rate-
        # matched codec variants, e.g. r061/r030) -- defaults preserve the
        # flagship (3/6/2/63).
        rvq_stages=n_stages,rvq_bits_per_stage=stage_bits,
        bits_vq_low=b_low,routing_topk=k_high)
    (out/"receiver.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps(meta),flush=True)


def self_test():
    rng=np.random.default_rng(125)
    mask=np.zeros(784,dtype=bool)
    mask[rng.choice(784,63,replace=False)]=True
    tokens=rng.integers(0,4,784,dtype=np.int64)
    tokens[mask]=rng.integers(0,2**18,63)
    frame=pack_joint_frame(tokens,mask)
    # CRC rejects payload, header and mask corruption; loss returns no source mask.
    for index in (0,200,600,3072):
        corrupt=frame.copy()
        corrupt[index]^=1
        values,m,ok=parse_joint_frame(corrupt)
        assert not ok and not m.any() and not values.any()
    for bps in (1,2):
        codec=Codec(3073,SYMBOLS*bps,bps)
        h,noise=channel_draw("valid",0,0)
        rx,n,symbols=codec.transmit(frame[None],np.array([h]),noise[None],40)
        recovered,received_mask,ok=parse_joint_frame(rx[0])
        assert ok and np.array_equal(recovered,tokens) and np.array_equal(received_mask,mask)
        assert symbols==SYMBOLS and dp.channel_uses_per_px([],bps,coded_block_lengths=[n])==SYMBOLS/50176
    # Verify old rate interface is unchanged.
    assert dp.channel_uses_per_px([(1/3,2),(.5,4)],4)==3584/50176
    print("PASS: frame CRC, authenticated mask, BPSK/QPSK real-PHY roundtrip, emitted rate, legacy rate",flush=True)


def probe():
    rng = np.random.default_rng(20260914)
    mask = np.zeros(784, dtype=bool)
    mask[rng.choice(784, 63, replace=False)] = True
    tokens = rng.integers(0, 4, 784, dtype=np.int64)
    tokens[mask] = rng.integers(0, 2**18, 63, dtype=np.int64)
    pad_original = bridge._pad_to_kcb
    sim_original = dp.simulate_digital_pipeline
    rows = []
    def traced_pad(bits):
        out = pad_original(bits)
        rows.append({"body_bits": int(bits.size), "outer_padded_bits": int(out.size)})
        return out
    def traced_sim(bits, *args, **kwargs):
        out, st = sim_original(bits, *args, **kwargs)
        row = next(r for r in rows if "blocks" not in r)
        row.update(blocks=st["blocks"], rate=kwargs["rate"],
                   coded_bits=int(st["blocks"] * dp.RATES[kwargs["rate"]]))
        return out, st
    try:
        bridge._pad_to_kcb = traced_pad
        dp.simulate_digital_pipeline = traced_sim
        result = bridge.transmit_image_rvq_uep(tokens, mask, 18, 2, 40,
                                             "perfect", 20260914)
    finally:
        bridge._pad_to_kcb = pad_original
        dp.simulate_digital_pipeline = sim_original
    assert np.array_equal(result[0], tokens), "high-SNR legacy round trip failed"
    print(json.dumps({"kind": "actual_legacy_probe", "segments": rows,
                      "symbols": sum(r["coded_bits"] for r in rows) // 4,
                      "stats": result[1]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--link-sweep", action="store_true")
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--out-dir")
    parser.add_argument("--transmit", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--source")
    parser.add_argument("--split", choices=("valid","test"))
    parser.add_argument("--variant", choices=("frame_bpsk","frame_qpsk","frame_16qam"))
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--symbols", type=int, default=None,
                        help="Wave 21: override the default 6144-symbol operating point "
                             "(0.122449 uses/px) to build a second joint-frame rate. "
                             "channel_uses_per_px = symbols/50176. Omit to reproduce the "
                             "original 0.122449 point exactly.")
    parser.add_argument("--snr-db", type=float, default=5.0,
                        help="Test split only: must be one of channel_bank.npz's grid "
                             "(-5,0,5,10,15,20). Reads that SNR's own h/noise draws "
                             "directly from the bank -- see transmit_cache's docstring "
                             "note for why this cannot be a simple noise rescale.")
    cli = parser.parse_args()
    if cli.probe:
        probe()
    elif cli.link_sweep:
        if not cli.out_dir:
            parser.error("--link-sweep requires a new --out-dir")
        link_sweep(cli.out_dir, cli.count)
    elif cli.self_test:
        self_test()
    elif cli.transmit:
        if not all((cli.source,cli.split,cli.variant,cli.out_dir)):
            parser.error("--transmit requires --source --split --variant --out-dir")
        transmit_cache(cli)
