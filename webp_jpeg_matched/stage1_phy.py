"""WebP/JPEG matched-budget rerun, stage 1 (PHY).  Sionna venv:
  TF_NUM_INTRAOP_THREADS=8 __SIONNA_PY__ webp_jpeg_matched/stage1_phy.py \
      --codec webp --budget mid [--snr 10] [--limit N] [--out-dir DIR]

Implements WEBP_JPEG_RERUN_PROTOCOL.md (copy in this directory):
  * fixed budget: nb in {3,6,12} blocks of (992 payload + CRC32 = 1024 info bits -> 2048 coded bits -> 512 16-QAM
    symbols) = 1536/3072/6144 symbols for EVERY image (padding and failure markers are transmitted too);
  * in-band length: first 4 payload bytes = uint32 big-endian whole-file length (0 = capacity-failure marker);
    receiver reads only the CRC-valid first block, never len(original bytes);
  * receiver = phy.Codec (BG2 k=1024 n=2048, Gray 16-QAM, demapper "app", boxplus-phi 40 it, llr_max 20),
    perfect CSI, ONE Rayleigh h per image/draw shared by all blocks;
  * h and noise are read from channel_bank.npz by (image, snr, draw) -- the SAME bank AquaSC's test runs use
    (phy.transmit_cache: noise = raw[:S] + j raw[S:2S], S = symbols), so fading/noise are paired with AquaSC;
  * failed-CRC data blocks are zero-filled; failed/invalid length block or capacity failure -> black image.
Test split only; quality q comes from valid_calib/quality_selection.json (validation-selected).
"""
import argparse, hashlib, io, json, os, struct, sys, time, zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "experiments" / "matched5"))
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import phy                      # noqa: E402  (also fixes CUDA_VISIBLE_DEVICES="" for TF)
import bitstream as bs          # noqa: E402

IMG = 224
TEST_IMAGES = "__AQUASC_DATA__/yolo_fish_dataset_root/test/images"
BANK = ROOT / "channel_bank.npz"
K_INFO, N_CODED, BPS = 1024, 2048, 4
PAYLOAD_BITS = K_INFO - 32          # 992
SYM_PER_BLOCK = N_CODED // BPS      # 512
BUDGETS = {"low": dict(nb=3, cap=368), "mid": dict(nb=6, cap=740), "high": dict(nb=12, cap=1484)}
SNRS = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
STATUS = {0: "ok", 1: "capacity_failure_marker", 2: "length_block_crc_fail", 3: "length_invalid"}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_resized(path):
    im = Image.open(path).convert("RGB")
    if im.size != (IMG, IMG):
        im = im.resize((IMG, IMG), Image.BICUBIC)
    return im


def encode(im, codec, q):
    buf = io.BytesIO()
    if codec == "webp":
        im.save(buf, "WEBP", quality=q, method=4)
    else:
        im.save(buf, "JPEG", quality=q, optimize=True)
    return buf.getvalue()


def crc32_bits(bits):               # == bs.crc32_ieee for byte-multiple lengths (asserted in self_check)
    return zlib.crc32(np.packbits(bits.astype(np.uint8)).tobytes()) & 0xFFFFFFFF


def crc_bits(v):
    return ((v >> np.arange(31, -1, -1)) & 1).astype(np.uint8)


def self_check():
    rng = np.random.default_rng(1)
    for _ in range(5):
        b = rng.integers(0, 2, PAYLOAD_BITS, dtype=np.uint8)
        assert crc32_bits(b) == bs.crc32_ieee(b)
        assert np.array_equal(crc_bits(crc32_bits(b)), bs.int_to_bits(crc32_bits(b), 32))


def build_payload(file_bytes, nb, cap):
    """-> (payload uint8 bits [nb*992], capacity_failure). Marker (length 0) when the file does not fit."""
    total_bytes = nb * PAYLOAD_BITS // 8
    assert total_bytes * 8 == nb * PAYLOAD_BITS
    fail = len(file_bytes) > cap
    body = b"" if fail else file_bytes
    length = 0 if fail else len(file_bytes)
    raw = struct.pack(">I", length) + body
    assert len(raw) <= total_bytes
    raw = raw + b"\x00" * (total_bytes - len(raw))
    return np.unpackbits(np.frombuffer(raw, np.uint8)), fail


def to_info_blocks(payload, nb):
    pb = payload.reshape(nb, PAYLOAD_BITS)
    out = np.zeros((nb, K_INFO), np.uint8)
    for b in range(nb):
        out[b, :PAYLOAD_BITS] = pb[b]
        out[b, PAYLOAD_BITS:] = crc_bits(crc32_bits(pb[b]))
    return out


def receive(dec_blocks, nb, cap):
    """dec_blocks [nb,1024] decoded info bits -> (status, file_bytes|None, n_failed_data_blocks, n_failed_blocks, n_undetected)."""
    ok = np.array([crc32_bits(dec_blocks[b, :PAYLOAD_BITS]) == bs.bits_to_int(dec_blocks[b, PAYLOAD_BITS:]) for b in range(nb)])
    n_fail = int((~ok).sum())
    if not ok[0]:
        return 2, None, n_fail
    payload = np.where(ok[:, None], dec_blocks[:, :PAYLOAD_BITS], 0).astype(np.uint8)   # zero-fill failed blocks
    raw = np.packbits(payload.reshape(-1)).tobytes()
    length = struct.unpack(">I", raw[:4])[0]
    if length == 0:
        return 1, None, n_fail
    if length > cap:
        return 3, None, n_fail
    return 0, raw[4:4 + length], n_fail


def transmit_batch(codec, infos, h_img, raw, nb, snr):
    """infos [n,nb,1024]; h_img [n] (ONE h per image, tiled over its nb blocks); raw [n,2S] bank noise reals.
    Returns decoded info bits [n,nb,1024] and the number of channel symbols actually emitted per image."""
    S = nb * SYM_PER_BLOCK
    n = infos.shape[0]
    h = np.repeat(h_img, nb)
    noise = (raw[:, :S] + 1j * raw[:, S:2 * S]).reshape(-1, SYM_PER_BLOCK)
    rx, n_bits, n_sym = codec.transmit(infos.reshape(-1, K_INFO), h, noise, snr)
    assert n_bits == N_CODED and n_sym == SYM_PER_BLOCK
    return rx.reshape(n, nb, K_INFO), nb * n_sym


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec", choices=("webp", "jpeg"), required=True)
    ap.add_argument("--budget", choices=tuple(BUDGETS), required=True)
    ap.add_argument("--snr", type=float, nargs="*", default=SNRS)
    ap.add_argument("--m", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N images only")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--batch-images", type=int, default=24)
    cli = ap.parse_args()
    self_check()

    nb, cap = BUDGETS[cli.budget]["nb"], BUDGETS[cli.budget]["cap"]
    S = nb * SYM_PER_BLOCK
    sel_path = HERE / "valid_calib" / "quality_selection.json"
    sel = json.loads(sel_path.read_text())["selected"][f"{cli.codec}_{cli.budget}"]
    assert sel["cap_B"] == cap
    q = sel["q"]
    out = Path(cli.out_dir) if cli.out_dir else HERE / "runs" / f"{cli.codec}_{cli.budget}"
    out.mkdir(parents=True, exist_ok=True)

    bank = np.load(BANK, allow_pickle=False)
    names = bank["image_names"].tolist()
    assert sorted(names) == sorted(f for f in os.listdir(TEST_IMAGES) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    bank_snr = [float(s) for s in bank["snr_db"]]
    assert bank["n"].shape[-1] >= 2 * S and bank["h"].shape[2] >= cli.m
    n_img = len(names) if cli.limit is None else cli.limit

    t0 = time.monotonic()
    files, payloads, infos, fails = [], [], [], []
    for n in names[:n_img]:
        fb = encode(load_resized(os.path.join(TEST_IMAGES, n)), cli.codec, q)
        files.append(fb)
        p, f = build_payload(fb, nb, cap)
        payloads.append(p); fails.append(f); infos.append(to_info_blocks(p, nb))
    infos = np.stack(infos)                                  # [n,nb,1024]
    file_sizes = np.array([len(f) for f in files]); fails = np.array(fails)
    meta = dict(protocol="WEBP_JPEG_RERUN_PROTOCOL.md", codec=cli.codec, budget=cli.budget, quality=q,
        options=json.loads(sel_path.read_text())["options"][cli.codec], quality_selection=sel,
        quality_selection_sha256=sha256(sel_path), blocks_per_image=nb, symbols_per_image=S,
        channel_uses_per_px=S / (IMG * IMG), bits_per_symbol=BPS, modulation="Gray 16-QAM, unit average symbol energy",
        k_info=K_INFO, n_coded=N_CODED, payload_bits_per_block=PAYLOAD_BITS, crc="CRC-32 IEEE per block over the 992 payload bits",
        capacity_B_after_4B_length=cap, raw_payload_capacity_B=nb * PAYLOAD_BITS // 8,
        length_field="uint32 big-endian, whole compressed file incl. container; 0 = capacity-failure marker (still transmitted)",
        ldpc="Sionna 5G NR BG2, k=1024 n=2048", decoder="boxplus-phi", iterations=40, llr_max=20, demapper="app",
        csi="perfect receiver CSI", fading="ONE Rayleigh CN(0,1) h per image/draw shared by all blocks of the image",
        noise="channel_bank.npz n[image,snr,draw]: complex noise = raw[:S] + j raw[S:2S], split into nb slices of 512 symbols",
        bank=str(BANK), bank_sha256=sha256(BANK), failed_data_block_policy="payload bits zero-filled",
        failed_length_block_policy="black image", M=cli.m, n_images=n_img, test_images=TEST_IMAGES,
        file_bytes_min=int(file_sizes.min()), file_bytes_mean=float(file_sizes.mean()), file_bytes_max=int(file_sizes.max()),
        n_capacity_failures=int(fails.sum()), frac_capacity_failures=float(fails.mean()),
        git_phy_py_sha256=sha256(ROOT / "experiments/matched5/phy.py"), script_sha256=sha256(__file__),
        argv=sys.argv, prep_seconds=time.monotonic() - t0)
    try:
        import sionna, tensorflow as tf
        meta.update(sionna=sionna.__version__, tensorflow=tf.__version__)
    except Exception:
        pass
    (out / "config.json").write_text(json.dumps(meta, indent=2) + "\n")
    np.savez_compressed(out / "source_files.npz", names=np.array(names[:n_img]), sizes=file_sizes, capacity_failure=fails)
    print(json.dumps({k: meta[k] for k in ("codec", "budget", "quality", "symbols_per_image", "file_bytes_mean",
                                           "file_bytes_max", "n_capacity_failures")}), flush=True)

    codec = phy.Codec(K_INFO, N_CODED, BPS, "boxplus-phi", 40)
    for snr in cli.snr:
        dst = out / f"rx_snr{int(snr)}.npz"
        if dst.exists():
            print(f"snr {snr}: exists, skip", flush=True); continue
        s_idx = bank_snr.index(float(snr))
        status = np.zeros((n_img, cli.m), np.int8); n_failed = np.zeros((n_img, cli.m), np.int16)
        lens = np.full((n_img, cli.m), -1, np.int32); blob = []
        sym_emitted = np.zeros((n_img, cli.m), np.int32); n_undetected = 0
        t1 = time.monotonic()
        for m in range(cli.m):
            h_all = bank["h"][:n_img, s_idx, m]
            raw_all = bank["n"][:n_img, s_idx, m, :2 * S]
            for o in range(0, n_img, cli.batch_images):
                e = min(n_img, o + cli.batch_images)
                rx, n_sym_img = transmit_batch(codec, infos[o:e], h_all[o:e], raw_all[o:e], nb, snr)
                sym_emitted[o:e, m] = n_sym_img
                for j in range(e - o):
                    i = o + j
                    st, fb, nf = receive(rx[j], nb, cap)
                    # undetected-error audit: any CRC-passing block that differs from what was sent
                    okb = [crc32_bits(rx[j, b, :PAYLOAD_BITS]) == bs.bits_to_int(rx[j, b, PAYLOAD_BITS:]) for b in range(nb)]
                    for b in range(nb):
                        if okb[b] and not np.array_equal(rx[j, b], infos[i, b]):
                            n_undetected += 1
                    status[i, m], n_failed[i, m] = st, nf
                    if fb is not None:
                        lens[i, m] = len(fb); blob.append(np.frombuffer(fb, np.uint8))
        assert (sym_emitted == S).all(), "transmitted symbol count differs from budget"
        np.savez_compressed(dst, status=status, n_failed_blocks=n_failed, lens=lens,
                            blob=np.concatenate(blob) if blob else np.zeros(0, np.uint8),
                            symbols_emitted=sym_emitted, n_undetected_crc=n_undetected)
        print(json.dumps(dict(snr=snr, seconds=round(time.monotonic() - t1, 1), n_undetected_crc=n_undetected,
              frac_black=float((lens < 0).mean()), frac_any_block_failed=float((n_failed > 0).mean()),
              status_counts={STATUS[k]: int((status == k).sum()) for k in STATUS},
              symbols_min_mean_max=[int(sym_emitted.min()), float(sym_emitted.mean()), int(sym_emitted.max())])), flush=True)
        assert n_undetected == 0, "undetected CRC error: retain and audit"


if __name__ == "__main__":
    main()
