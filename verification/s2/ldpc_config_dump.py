"""Checklist §2 evidence: dump the LDPC/modulation/decoder configuration from the objects the experiments actually instantiate.
Run: taskset -c 0-7 __SIONNA_PY__ ldpc_config_dump.py"""
import json, sys, importlib.metadata as md
from pathlib import Path
ROOT = Path("__AQUASC_ROOT__")
sys.path.insert(0, str(ROOT / "experiments/matched5")); sys.path.insert(0, str(ROOT))
import phy, digital_pipeline as dp

def enc_info(e, n_sent, bps, sym):
    k, z = e.k, e.z
    k_ldpc, n_ldpc = e.k_ldpc, e.n_ldpc
    filler = k_ldpc - k
    return dict(base_graph="BG2", Z=z, lifting_set_index_iLS=e._i_ls, k_b_info_columns=k_ldpc // z,
                mother_code=dict(k_ldpc=k_ldpc, n_ldpc=n_ldpc, rows_x_cols_of_expanded_H=f"{(n_ldpc-k_ldpc)}x{n_ldpc}"),
                info_bits_k=k, filler_bits=filler, punctured_systematic_bits_2Z=2*z,
                rate_matched_coded_bits_n=n_sent, mother_parity_bits_not_sent=(n_ldpc - filler - 2*z) - n_sent,
                code_rate_k_over_n=round(k / n_sent, 4), bits_per_symbol=bps, complex_symbols=sym,
                encoder_num_bits_per_symbol_arg=e.num_bits_per_symbol)

out = dict(env=dict(sionna=md.version("sionna"), tensorflow=md.version("tensorflow"), numpy=md.version("numpy"),
                    python=sys.version.split()[0], venv="<venv-sionna>", frozen_list="sionna_venv_freeze.txt"))
# --- AquaSC JointFrame (phy.Codec) ---
aq = {}
for name, k, n, bps in (("headline_0.122449", 3073, 12288, 2), ("r061_0.061224", 1510, 6144, 2), ("r030_0.030612", 768, 3072, 2)):
    c = phy.Codec(k, n, bps, "boxplus-phi", 40)
    d = enc_info(c.encoder, n, bps, n // bps)
    dec = c.decoder
    d["decoder"] = dict(cls=type(dec).__name__, cn_update=str(getattr(dec, "_cn_update", "boxplus-phi (ctor arg)")),
                        num_iter=getattr(dec, "num_iter", None), llr_max=getattr(dec, "_llr_max", None),
                        hard_out=getattr(dec, "_hard_out", None), return_infobits=getattr(dec, "_return_infobits", None),
                        prune_pcm=getattr(dec, "_prune_pcm", None), early_termination="none (fixed iterations)")
    d["mapper"] = f"sionna.phy.mapping.Mapper(constellation_type='qam', num_bits_per_symbol={bps}) (default Gray labeling, unit average energy)"
    d["demapper"] = "sionna.phy.mapping.Demapper('app', 'qam')"
    d["channel"] = "one CN(0,1) Rayleigh coefficient per image (single codeword), perfect CSI zero-forcing, Es/N0 with Es=1"
    aq[name] = d
out["aquasc_jointframe"] = aq
# --- WebP/JPEG baselines (digital_pipeline.simulate_digital_pipeline) ---
bl = {}
enc, dec = dp.get_coder(0.5)
for mod, bps in (("16QAM(default; WebP q=8 cap744, lowrates q=1/q=81, midrate16qam q=9, JPEG)", 4), ("QPSK(explicit; midrate q=1 cap372)", 2)):
    d = enc_info(enc, 2048, bps, 2048 // bps)
    d["per_block_payload_bits"] = f"{dp.K_CB - dp.CRC_PER_BLOCK_BITS} + 32-bit CRC = K_CB {dp.K_CB}"
    d["decoder"] = dict(cls=type(dec).__name__, cn_update="normalized min-sum, scale %s" % dp.NMS_SCALE, vn_update="sum", schedule="flooding",
                        max_iterations=dp.NUM_ITER_MAX, chunk=dp.NUM_ITER_CHUNK, early_termination="syndrome check every chunk",
                        llr_max=dp.LLR_MAX, prune_pcm=True, hard_out=True, return_infobits=True)
    d["demapper"] = "sionna Demapper('maxlog')  (digital_pipeline._demap_llr)"
    d["channel"] = "independent CN(0,1) Rayleigh coefficient PER LDPC BLOCK, perfect CSI, Es/N0 with Es=1"
    d["block_acceptance"] = "per-block CRC-32 AND syndrome-valid"
    bl[mod] = d
out["webp_jpeg_multiblock"] = bl
Path("ldpc_config_dump.json").write_text(json.dumps(out, indent=2, default=str))
print(json.dumps(out, indent=1, default=str)[:6000])
