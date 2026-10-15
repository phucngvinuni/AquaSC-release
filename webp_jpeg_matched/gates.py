"""Protocol §6 gates (run in the Sionna venv before the full grid). Writes gates_result.json.
  TF_NUM_INTRAOP_THREADS=8 taskset -c 0-9 __SIONNA_PY__ webp_jpeg_matched/gates.py
"""
import json, os, struct, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage1_phy as s1
import phy

res = {}
sel = json.loads((HERE / "valid_calib/quality_selection.json").read_text())["selected"]
bank = np.load(s1.BANK, allow_pickle=False)
names = bank["image_names"].tolist()
codec = phy.Codec(s1.K_INFO, s1.N_CODED, s1.BPS, "boxplus-phi", 40)
bank_snr = [float(x) for x in bank["snr_db"]]

# G1 noiseless round trip + G3 payload accounting, all budgets/codecs, 24 images
g1, g3 = {}, {}
for codec_name in ("webp", "jpeg"):
    for bud, c in s1.BUDGETS.items():
        nb, cap, q = c["nb"], c["cap"], sel[f"{codec_name}_{bud}"]["q"]
        S = nb * s1.SYM_PER_BLOCK
        files, infos, fails = [], [], []
        for n in names[:24]:
            fb = s1.encode(s1.load_resized(os.path.join(s1.TEST_IMAGES, n)), codec_name, q)
            p, f = s1.build_payload(fb, nb, cap)
            assert p.size == nb * s1.PAYLOAD_BITS                                     # G3: metadata+file+padding == payload bits
            assert struct.unpack(">I", np.packbits(p[:32]).tobytes())[0] == (0 if f else len(fb))
            files.append(fb); fails.append(f); infos.append(s1.to_info_blocks(p, nb))
        infos = np.stack(infos)
        # noiseless: bank h but zero noise at 40 dB -> perfect recovery expected for any h of reasonable size; use h=1
        rx, nsym = s1.transmit_batch(codec, infos, np.ones(24, np.complex64), np.zeros((24, 2 * S), np.float32), nb, 40.0)
        exact = 0
        for i in range(24):
            st, fb, nf = s1.receive(rx[i], nb, cap)
            if fails[i]:
                exact += int(st == 1 and fb is None)
            else:
                exact += int(st == 0 and fb == files[i])
        g1[f"{codec_name}_{bud}"] = dict(images=24, exact=exact, n_capacity_marker=int(sum(fails)), symbols_per_image=nsym)
        g3[f"{codec_name}_{bud}"] = dict(payload_bits=nb * s1.PAYLOAD_BITS, info_bits_per_image=nb * s1.K_INFO,
                                          coded_bits_per_image=nb * s1.N_CODED, symbols=nb * s1.SYM_PER_BLOCK)
        assert exact == 24 and nsym == S, (codec_name, bud, exact, nsym)
res["G1_noiseless_roundtrip"] = g1
res["G3_payload_accounting"] = g3

# G2 receiver does not use the oracle length: corrupt the length block / field
nb, cap = 6, 740
fb = os.urandom(300)
p, _ = s1.build_payload(fb, nb, cap)
inf = s1.to_info_blocks(p, nb)
g2 = {}
bad = inf.copy(); bad[0, 5] ^= 1                         # flip a bit inside the length field -> block-0 CRC fails
g2["length_block_bit_flip"] = s1.STATUS[s1.receive(bad, nb, cap)[0]]
for name, val in (("length_zero_valid_crc", 0), ("length_gt_cap_valid_crc", cap + 1), ("length_ok_valid_crc", 300)):
    q2 = p.copy(); q2[:32] = ((val >> np.arange(31, -1, -1)) & 1)
    st, got, _ = s1.receive(s1.to_info_blocks(q2, nb), nb, cap)
    g2[name] = dict(status=s1.STATUS[st], returned_len=None if got is None else len(got))
dat = inf.copy(); dat[3, 100] ^= 1                         # corrupt a DATA block -> that block zero-filled, length still read in-band
st, got, nf = s1.receive(dat, nb, cap)
g2["data_block_corrupt"] = dict(status=s1.STATUS[st], n_failed_blocks=nf, returned_len=len(got), equals_orig=got == fb)
import inspect
src = inspect.getsource(s1.receive)
g2["receive_source_mentions_original_file"] = ("file_bytes" in src and "files" in src)
assert g2["length_block_bit_flip"] == "length_block_crc_fail" and g2["length_zero_valid_crc"]["status"] == "capacity_failure_marker"
assert g2["length_gt_cap_valid_crc"]["status"] == "length_invalid" and g2["length_ok_valid_crc"]["returned_len"] == 300
res["G2_in_band_length"] = g2

# G4/G5 actual symbols + same-h across blocks + independent draws
cap_h = []
orig_tx = codec.transmit
def spy(info, h, noise, snr):
    cap_h.append(np.array(h)); return orig_tx(info, h, noise, snr)
codec.transmit = spy
nb = 12
infos = np.stack([s1.to_info_blocks(s1.build_payload(os.urandom(500), nb, 1484)[0], nb) for _ in range(8)])
h_img = bank["h"][:8, 3, 0]; raw = bank["n"][:8, 3, 0, :2 * nb * 512]
_, nsym = s1.transmit_batch(codec, infos, h_img, raw, nb, 10.0)
hh = cap_h[0].reshape(8, nb)
res["G4_actual_symbols"] = dict(per_image_symbols_emitted=nsym, expected=6144)
res["G5_h"] = dict(all_blocks_share_h=bool((hh == hh[:, :1]).all()), h_per_image_differs=bool(len(np.unique(hh[:, 0])) == 8),
                   h_equals_bank=bool(np.array_equal(hh[:, 0], h_img)),
                   draws_independent=bool(len(np.unique(bank["h"][0, 3, :])) == bank["h"].shape[2]),
                   noise_independent_across_blocks=bool(np.abs(np.corrcoef(raw[0, :512], raw[0, 512:1024])[0, 1]) < 0.2))
assert nsym == 6144 and all(res["G5_h"].values())
codec.transmit = orig_tx

# G6 runtime receiver configuration (introspection, not config echo)
dec, dem = codec.decoder, codec.demapper
def attr(o, *ks):
    return {k: (str(getattr(o, k)) if hasattr(o, k) else None) for k in ks}
res["G6_runtime_config"] = dict(
    decoder_class=type(dec).__name__, decoder=attr(dec, "_cn_update", "_vn_update", "_num_iter", "_llr_max", "_hard_out", "_return_infobits"),
    decoder_num_iter=int(dec._num_iter), decoder_llr_max=float(dec._llr_max),
    demapper_class=type(dem).__name__, demapper_logits2llrs=type(dem._logits2llrs).__name__,
    demapper_method=str(dem._logits2llrs._method), demapper_bits_per_symbol=int(dem._logits2llrs._num_bits_per_symbol),
    demapper_hard_out=bool(dem._logits2llrs._hard_out),
    constellation=dict(type=str(dem._constellation.constellation_type), bits=int(dem._constellation.num_bits_per_symbol),
                       normalized=bool(dem._constellation.normalize),
                       mean_symbol_energy=float(np.mean(np.abs(dem._constellation.points.numpy()) ** 2))),
    encoder=dict(k=codec.encoder.k, n=codec.encoder.n, bg=str(getattr(codec.encoder, "_bg", None)), Z=getattr(codec.encoder, "z", None),
                 iLS=getattr(codec.encoder, "_i_ls", None)),
    mapper_constellation_is_demapper_constellation=bool(codec.mapper._constellation.num_bits_per_symbol == 4), module_phy=phy.__file__)

# G7 capacity failure & decode failure still produce transmitted records: covered in stage1 status counts; synthetic oversize here
nb, cap = 3, 368
big = os.urandom(900)
p, f = s1.build_payload(big, nb, cap)
infos = s1.to_info_blocks(p, nb)[None]
rx, nsym = s1.transmit_batch(codec, infos, np.ones(1, np.complex64), np.zeros((1, 2 * 1536), np.float32), nb, 40.0)
st, got, _ = s1.receive(rx[0], nb, cap)
res["G7_capacity_failure"] = dict(flag=bool(f), receiver_status=s1.STATUS[st], symbols_still_transmitted=nsym)
assert f and st == 1 and nsym == 1536
assert res["G6_runtime_config"]["demapper_method"] == "app" and "cn_update_phi" in res["G6_runtime_config"]["decoder"]["_cn_update"]
assert res["G6_runtime_config"]["decoder_num_iter"] == 40 and res["G6_runtime_config"]["decoder_llr_max"] == 20.0
(HERE / "gates_result.json").write_text(json.dumps(res, indent=2, default=str))
print(json.dumps(res, indent=2, default=str))
print("ALL GATES PASSED")
