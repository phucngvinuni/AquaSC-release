"""PSNR/SSIM for the joint-frame protocol (2026-09-18) -- the joint-frame
evaluation pipeline (evaluate.py) only ever computed mAP/frame-loss-rate, never
reconstructed-image PSNR/SSIM (confirmed by reading evaluate.py: no psnr/ssim
anywhere in it). This script reuses decode_frame() and the already-computed
receiver.npz for each (rate, SNR) sweep point -- NO new channel simulation --
to compute PSNR/SSIM the same way every other system in this project does
(utils.calc_psnr/calc_ssim), for a real, comparable number.

Run (siso-project-env, from AquaSC-D2DB/experiments/matched5/):
  python3 eval_psnr_ssim.py --receiver runs/sweep/phy_10/receiver.npz --split test --out_json runs/sweep/phy_10/psnr_ssim.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch

import utils as aq_utils
from evaluate import decode_frame, dataset, load_aquasc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--receiver", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    rx = np.load(args.receiver, allow_pickle=False)
    meta = json.loads(Path(args.receiver).with_suffix(".json").read_text())
    site = meta.get("site", "a")
    assert meta["split"] == args.split

    # 2026-09-18 (AquaSC-lite): rebuild the SAME (possibly non-flagship)
    # model the receiver was encoded with, matching evaluate_aquasc()'s
    # convention -- read back from phy.py's meta, not re-guessed.
    ckpt = meta.get("checkpoint")
    enc_dim = meta.get("encoder_embed_dim", 512)
    enc_depth = meta.get("encoder_depth", 8)
    enc_heads = meta.get("encoder_num_heads", 8)
    rvq_stages = meta.get("rvq_stages", 3)
    rvq_bits_per_stage = meta.get("rvq_bits_per_stage", 6)
    bits_vq_low = meta.get("bits_vq_low", 2)
    routing_topk = meta.get("routing_topk", 63)
    ds = dataset(args.split, site)
    net = load_aquasc(args.split, site, ckpt, enc_dim, enc_depth, enc_heads,
                       rvq_stages, rvq_bits_per_stage, bits_vq_low, routing_topk)
    assert rx["names"].tolist() == [Path(p).name for p in ds.img_files]
    originals = [ds[i][1][0] for i in range(len(ds))]  # targets_tuple[0] = original_images_for_loss

    n, m_count = rx["frame_ok"].shape
    psnr_per_mi = np.zeros((n, m_count), dtype=np.float64)
    ssim_per_mi = np.zeros((n, m_count), dtype=np.float64)
    with torch.no_grad():
        for m in range(m_count):
            for offset in range(0, n, 8):
                end = min(n, offset + 8)
                recon = decode_frame(net, rx["tokens"][offset:end, m],
                                     rx["mask"][offset:end, m], rx["frame_ok"][offset:end, m])
                recon = recon.clamp(0.0, 1.0).cpu()
                for j in range(offset, end):
                    orig = originals[j]
                    p = aq_utils.calc_psnr(recon[j - offset], orig)
                    s = aq_utils.calc_ssim(recon[j - offset], orig)
                    psnr_per_mi[j, m] = p[0] if isinstance(p, list) else p
                    ssim_per_mi[j, m] = s[0] if isinstance(s, list) else s

    out = {
        "receiver_file": str(Path(args.receiver).resolve()),
        "psnr_mean": float(psnr_per_mi.mean()),
        "ssim_mean": float(ssim_per_mi.mean()),
        "psnr_mean_per_m": psnr_per_mi.mean(axis=0).tolist(),
        "ssim_mean_per_m": ssim_per_mi.mean(axis=0).tolist(),
        "n_images": int(n), "M": int(m_count),
        "note": ("Computed from the existing receiver.npz (no new channel "
                 "simulation) via decode_frame(), scored with utils.calc_psnr/"
                 "calc_ssim -- the same convention every other system in "
                 "RESULTS_SUMMARY_TABLE.md uses."),
    }
    out_path = Path(args.out_json) if args.out_json else Path(args.receiver).with_name("psnr_ssim.json")
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
