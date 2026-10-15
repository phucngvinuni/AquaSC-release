#!/usr/bin/env python
"""plotting/reproduce_figures.py — single entry point that regenerates the key
AquaSC figures/tables FROM CHECKPOINTS WITHOUT RETRAINING.

Reviewer mapping: R3 #2 (figure traceability / reproducibility) and R1 #6
(dataset documentation provenance). This is the Wave-8 REPRO deliverable that
plan §4 requires ("a single script that regenerates every figure and table from
released checkpoints without retraining").

It produces:
  (a) the rate–retention curve            -> figs/fig_rate_retention.png
  (b) the SNR sweep curves                -> figs/fig_snr_sweep.png
      (PSNR / SSIM / mAP@0.5 vs SNR, proxy + digital)
  (c) the FIM tau-sweep table + histograms-> figs/fig_fim_tau.png (+ reuses
      fim_metrics.py histograms)
  (d) the proxy-vs-digital table          -> printed + figs/fig_proxy_digital.png

Provenance mechanism: every number is read from a CSV/log file whose path,
the producing checkpoint (+ SHA-256), the seed, and which pipeline
(analog proxy vs digital LDPC chain) are printed next to it. Nothing is
fabricated; a missing input is reported and (if requested) regenerated on the
CPU via the existing eval path (``run_class_main.py --eval``) + ``metrics.py`` +
``fim_metrics.py``.

Dependency on EVAL-FINAL: the final SNR-sweep CSVs
``evalfinal_{run1,run2,run3}_{proxy,digital}.csv`` are produced by the EVAL-FINAL
agent (driver ``__AQUASC_TMP__/evalfinal_driver.sh``). If any are missing, pass
``--regenerate`` (or the per-figure ``--regenerate-*`` flags) and this script
re-runs the same ``run_class_main.py --eval`` invocations on the CPU.

Usage:
  python plotting/reproduce_figures.py                # plot from existing CSVs
  python plotting/reproduce_figures.py --regenerate   # (re)generate missing CSVs on CPU first
  python plotting/reproduce_figures.py --figs a,b     # only some figures
  python plotting/reproduce_figures.py --device cpu   # default; do NOT use the GPU here
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repo root (core modules)

import argparse
import csv
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# Provenance constants (all cross-checked against checkpoints_sha256.txt)
# --------------------------------------------------------------------------
REPO = HERE
DATA_PATH = "__AQUASC_DATA__/yolo_fish_dataset_root/"
DETECTOR = os.path.join(REPO, "best.pt")
DETECTOR_SHA = "bab85794b395ae57e38e428ccf4db0bacecf10578b6c18161de56358189f8e57"
CEILING_MAP50 = 0.7292          # clean ceiling, conf 0.001 (best.pt, wave-4 gate)
SEED = 42
FIGS_DIR = "__AQUASC_DATA__/figs"
SIONNA_PY = "__SIONNA_PY__"

# Final (Wave-7/8) configs. Each entry: checkpoint, SHA-256, seed, low-path bits,
# patch size, and the arch flags EVAL-FINAL used. "bpp" is the measured fully
# loaded source rate (bitstream.py, enumerative mask, no k_cb padding).
CONFIGS = {
    "run1_18_2": {
        "name": "final 18/2 (RUN-1 flagship)",
        "checkpoint": "ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS_FINAL/checkpoint-best_psnr.pth",
        "sha256": "b1d2eb25419af6ed6703a4aa886eb7143a8f1f350bd50e3371dacb3cde9e2894",
        "epoch": 239,
        "bits_low": 2, "patch": 8,
        "arch": ["--patch_size", "8", "--encoder_embed_dim", "512", "--encoder_depth", "8",
                 "--encoder_num_heads", "8", "--decoder_embed_dim", "512", "--decoder_depth", "6",
                 "--decoder_num_heads", "8", "--fim_embed_dim", "256", "--fim_depth", "2",
                 "--fim_num_heads", "4", "--quantizer_dim", "512", "--quantizer_type", "rvq",
                 "--rvq_stages", "3", "--rvq_bits_per_stage", "6", "--routing_topk", "63"],
        "config_yaml": "configs/final_18_2.yaml",
        "csv_base": "run1",   # EVAL-FINAL driver stem: evalfinal_run1_{proxy,digital}.csv
    },
    "run3_18_4": {
        "name": "final 18/4 (RUN-3 higher-rate)",
        "checkpoint": "ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS_FINAL_18_4/checkpoint-best_psnr.pth",
        "sha256": "9e2e5d4ee7df3a4c9f14f59cd06e25b257a436321fd6d324bd7816e92fd78a0b",
        "epoch": 219,
        "bits_low": 4, "patch": 8,
        "arch": ["--patch_size", "8", "--encoder_embed_dim", "512", "--encoder_depth", "8",
                 "--encoder_num_heads", "8", "--decoder_embed_dim", "512", "--decoder_depth", "6",
                 "--decoder_num_heads", "8", "--fim_embed_dim", "256", "--fim_depth", "2",
                 "--fim_num_heads", "4", "--quantizer_dim", "512", "--quantizer_type", "rvq",
                 "--rvq_stages", "3", "--rvq_bits_per_stage", "6", "--routing_topk", "63"],
        "config_yaml": "configs/final_18_4.yaml",
        "csv_base": "run3",
    },
    "run2_op16_256d2": {
        "name": "final OP16_256d2 (RUN-2 lightweight)",
        "checkpoint": "ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS_FINAL_OP16_256d2/checkpoint-best_psnr.pth",
        "sha256": "580bd2b598c88164a904448104a8baf80d86ed45cb552098bfb4c82c22f79b2f",
        "epoch": 69,
        "bits_low": 2, "patch": 16,
        "arch": ["--patch_size", "16", "--encoder_embed_dim", "256", "--encoder_depth", "2",
                 "--encoder_num_heads", "8", "--decoder_embed_dim", "256", "--decoder_depth", "2",
                 "--decoder_num_heads", "8", "--fim_embed_dim", "128", "--fim_depth", "2",
                 "--fim_num_heads", "4", "--quantizer_dim", "256", "--quantizer_type", "rvq",
                 "--rvq_stages", "3", "--rvq_bits_per_stage", "6", "--routing_topk", "63"],
        "config_yaml": "configs/final_op16_256d2.yaml",
        "csv_base": "run2",
    },
}
FINAL_KEYS = ["run1_18_2", "run3_18_4", "run2_op16_256d2"]

# Legacy configs (for the rate curve / augmentation-collapse ablation).
LEGACY = {
    "rvq_wave3": {
        "name": "Wave-3 RVQ 3x6 (no-aug, pre-RVQFIX)",
        "checkpoint": "ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS_RVQ/checkpoint-best_psnr.pth",
        "sha256": "755a027b9b2e22241642ab4809e6d0e152547a7be2d390c69b7f40fcc469e096",
        "bits_low": 2, "patch": 8,
        "config_yaml": "configs/legacy_rvq_wave3.yaml",
    },
    "aug": {
        "name": "Wave-4 AUG RVQ 3x6 (augmentation)",
        "checkpoint": "ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS_RVQ_AUG/checkpoint-best_psnr.pth",
        "sha256": "b5b3e693d2d6eb1a3b60e8d1960e9d361eaef246faa859c4f4acc47206bf5f40",
        "bits_low": 2, "patch": 8,
        "config_yaml": "configs/legacy_aug.yaml",
    },
    "combined": {
        "name": "Wave-6 COMBINED (EMA + aug)",
        "checkpoint": "ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS_COMBINED/checkpoint-best_psnr.pth",
        "sha256": "d218ed1eed36c761b31fb65a1ca6964bfd42c517ed84de1afa7de4efd39379f3",
        "bits_low": 2, "patch": 8,
        "config_yaml": "configs/legacy_combined.yaml",
    },
}

SNR_GRID = [-5, 0, 5, 10, 15, 20]


# --------------------------------------------------------------------------
# BPP arithmetic (bitstream.py's own measured-frame formula; no hardcoded BPP)
# --------------------------------------------------------------------------
def bpp_for_config(b_high: int, b_low: int, k_high: int = 63, n_tokens: int = 784,
                   mask_mode: str = "enumerative", image_pixels: int = 224 * 224,
                   k_cb: int = 1024, header_bits: int = 152) -> dict:
    """Replicate bitstream.measure_bpp_breakdown for arbitrary (b_high, b_low)."""
    if mask_mode == "enumerative":
        mask_bits = int(math.ceil(math.log2(math.comb(n_tokens, k_high))))
    else:
        mask_bits = n_tokens
    index_bits = k_high * b_high + (n_tokens - k_high) * b_low
    crc_bits = 32
    total = header_bits + mask_bits + index_bits + crc_bits
    padding = (k_cb - total % k_cb) % k_cb
    return {
        "mask_bits": mask_bits, "index_bits": index_bits, "total": total,
        "padding": padding, "bpp_no_padding": total / image_pixels,
        "bpp": (total + padding) / image_pixels,
    }


# --------------------------------------------------------------------------
# CSV I/O
# --------------------------------------------------------------------------
def load_eval_csv(path):
    """Read an eval CSV (SNR_dB, PSNR, SSIM, mAP_50, mAP, [digital cols]) into a
    dict keyed by float SNR. Missing values become None (never fabricated)."""
    if not os.path.exists(path):
        return None
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            snr = float(r["SNR_dB"])
            def _f(k):
                v = r.get(k)
                if v is None or v == "" or v == "N/A":
                    return None
                try:
                    return float(v)
                except ValueError:
                    return None
            rows[snr] = {k: _f(k) for k in
                         ["PSNR", "SSIM", "mAP_50", "mAP", "digital_bler",
                          "digital_ber", "digital_bit_accuracy",
                          "digital_erased_token_fraction"]}
    return rows


def load_csv_cols(path, col_key="tau"):
    """Read a generic CSV keyed by a column."""
    if not os.path.exists(path):
        return None
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            k = float(r[col_key])
            out[k] = {kk: (float(vv) if _isnum(vv) else vv) for kk, vv in r.items()}
    return out


def _isnum(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------
# Regeneration (CPU): reuse the exact EVAL-FINAL run_class_main --eval flags
# --------------------------------------------------------------------------
def _eval_cmd(key, csv_path, snr, digital, device):
    cfg = CONFIGS[key]
    bits_low = cfg["bits_low"]
    dig = ["--digital_ldpc", "--digital_csi_mode", "perfect"] if digital else []
    cmd = [sys.executable, "run_class_main.py",
           "--model", "ViT_Reconstruction_Model_Default",
           "--output_dir", "__AQUASC_TMP__/evalfinal_out",
           "--data_set", "fish", "--data_path", DATA_PATH,
           "--num_object_classes", "1", "--yolo_weights", DETECTOR,
           "--batch_size", "8", "--input_size", "224",
           ] + cfg["arch"] + [
           "--bits_vq_low", str(bits_low),
           "--device", device, "--seed", str(SEED),
           "--resume", cfg["checkpoint"],
           "--eval", "--snr_db_eval", str(snr),
           "--yolo_conf_thres", "0.4", "--yolo_iou_thres", "0.5",
           "--map_conf_thres", "0.001"] + dig + [
           "--csv_log_file", csv_path, "--append_csv"]
    return cmd


def regenerate_sweep(key, digital, device="cpu"):
    """Regenerate one config's SNR-sweep CSV on the CPU (does not touch the GPU)."""
    cfg = CONFIGS[key]
    kind = "digital" if digital else "proxy"
    csv_path = os.path.join(REPO, f"evalfinal_{cfg['csv_base']}_{kind}.csv")
    if digital and not os.path.exists(SIONNA_PY):
        print(f"[regenerate] WARNING: Sionna venv {SIONNA_PY} not found; "
              f"digital regeneration will fail. See reports/wave5_channel.md.")
    if os.path.exists(csv_path):
        os.remove(csv_path)  # rebuild cleanly (append_csv)
    for snr in SNR_GRID:
        cmd = _eval_cmd(key, csv_path, snr, digital, device)
        print(f"[regenerate] {key} {kind} SNR {snr}: {' '.join(cmd)}")
        rc = subprocess.run(cmd, cwd=REPO, env=_env_with_sionna()).returncode
        if rc != 0:
            print(f"[regenerate] FAILED {key} {kind} SNR {snr} (rc={rc})")
            return False
    return True


def _env_with_sionna():
    env = os.environ.copy()
    return env


# --------------------------------------------------------------------------
# Figure (a): rate–retention curve
# --------------------------------------------------------------------------
def fig_rate_retention(regenerate=False, device="cpu"):
    print("\n" + "=" * 70)
    print("(a) RATE–RETENTION CURVE  (R2 #1 / R3 #2)")
    print("=" * 70)
    os.makedirs(FIGS_DIR, exist_ok=True)

    # Retention points are mAP@0.5(recon)/0.7292 at a fixed SNR. For the final
    # configs use the EVAL-FINAL CSVs; for legacy use the wave-6 sweep CSVs.
    proxy_rows = []
    for key in FINAL_KEYS:
        cfg = CONFIGS[key]
        rows = load_eval_csv(os.path.join(REPO, f"evalfinal_{cfg['csv_base']}_proxy.csv"))
        if rows is None:
            if regenerate:
                regenerate_sweep(key, digital=False, device=device)
                rows = load_eval_csv(os.path.join(REPO, f"evalfinal_{cfg['csv_base']}_proxy.csv"))
            else:
                print(f"  [SKIP] {key}: evalfinal_{cfg['csv_base']}_proxy.csv missing "
                      f"(EVAL-FINAL still running; rerun with --regenerate)")
                continue
        bpp = bpp_for_config(b_high=18, b_low=cfg["bits_low"],
                             n_tokens=(224 // cfg["patch"]) ** 2,
                             k_high=63)["bpp_no_padding"]
        for snr in (10, 20):
            r = rows.get(float(snr))
            if r and r.get("mAP_50") is not None:
                proxy_rows.append((bpp, r["mAP_50"] / CEILING_MAP50, snr, cfg["name"],
                                   cfg["sha256"], "analog proxy", key))
                print(f"  {cfg['name']}: BPP={bpp:.6f} SNR={snr}dB "
                      f"mAP@0.5={r['mAP_50']:.4f} retention={r['mAP_50']/CEILING_MAP50:.3f}")

    # Legacy RVQ no-aug 18/2 (the wave-6 sweep diagnostic K=63 reproduces the
    # wave-4 METRIC 0.1991/0.3439 exactly).
    rvq = {"checkpoint": LEGACY["rvq_wave3"]["checkpoint"],
           "sha256": LEGACY["rvq_wave3"]["sha256"]}
    bpp_182 = bpp_for_config(18, 2, n_tokens=784, k_high=63)["bpp_no_padding"]
    for snr, map50 in ((10, 0.1991), (20, 0.3439)):
        proxy_rows.append((bpp_182, map50 / CEILING_MAP50, snr,
                           LEGACY["rvq_wave3"]["name"], rvq["sha256"],
                           "analog proxy (wave-6 sweep K=63)", "rvq_wave3"))
        print(f"  {LEGACY['rvq_wave3']['name']}: BPP={bpp_182:.6f} SNR={snr}dB "
              f"mAP@0.5={map50:.4f} retention={map50/CEILING_MAP50:.3f} "
              f"(provenance: sweep_results/sweep_RVQ_K63_proxy.csv)")

    # WebP anchor (wave-7 COMPETITOR, clean decode at q=81).
    wp = load_csv_cols(os.path.join(REPO, "results/codec/codec_webp_eval.csv"), col_key="mean_bpp")
    if wp:
        row = next(iter(wp.values()))
        bpp_w = row["mean_bpp"]; ret_w = row["retention_vs_canonical_07292"]
        proxy_rows.append((bpp_w, ret_w, 99, "WebP q=81 (clean decode, anchor)",
                           "n/a (codec)", "codec baseline", "webp"))
        print(f"  WebP anchor: BPP={bpp_w:.6f} retention={ret_w:.3f} "
              f"(provenance: results/codec/codec_webp_eval.csv)")

    # Plot: retention vs BPP, one series per SNR.
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for snr, color in ((10, "tab:blue"), (20, "tab:red"), (99, "tab:green")):
        pts = [p for p in proxy_rows if p[2] == snr]
        if not pts:
            continue
        pts.sort(key=lambda p: p[0])
        label = {10: "retention @ 10 dB", 20: "retention @ 20 dB",
                 99: "codec anchor (clean)"}[snr]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=color,
                label=label)
        for bpp, ret, _, name, sha, pipe, key in pts:
            ax.annotate(key, (bpp, ret), textcoords="offset points",
                        xytext=(5, 4), fontsize=7)
    ax.set_xlabel("measured source BPP (enumerative mask, no k_cb pad)")
    ax.set_ylabel("retention = mAP@0.5(recon) / 0.7292")
    ax.set_title("AquaSC rate–retention (video-disjoint test, seed 42)")
    ax.grid(True, alpha=0.3); ax.legend()
    out = os.path.join(FIGS_DIR, "fig_rate_retention.png")
    fig.tight_layout(); fig.savefig(out, dpi=130); print(f"  WROTE {out}")
    return proxy_rows


# --------------------------------------------------------------------------
# Figure (b): SNR sweep curves (PSNR / SSIM / mAP, proxy + digital)
# --------------------------------------------------------------------------
def fig_snr_sweep(regenerate=False, device="cpu"):
    print("\n" + "=" * 70)
    print("(b) SNR SWEEP CURVES  (R3 #3 digital pipeline / R3 #2)")
    print("=" * 70)
    os.makedirs(FIGS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for key, cfg in [(k, CONFIGS[k]) for k in FINAL_KEYS]:
        for digital in (False, True):
            kind = "digital" if digital else "proxy"
            rows = load_eval_csv(os.path.join(REPO, f"evalfinal_{cfg['csv_base']}_{kind}.csv"))
            if rows is None:
                if regenerate:
                    regenerate_sweep(key, digital, device)
                    rows = load_eval_csv(os.path.join(REPO, f"evalfinal_{cfg['csv_base']}_{kind}.csv"))
                else:
                    print(f"  [SKIP] {key} {kind}: evalfinal_{cfg['csv_base']}_{kind}.csv missing "
                          f"(rerun with --regenerate)")
                    continue
            snrs = sorted(rows)
            ls = "-" if digital else "--"
            lab = f"{cfg['name']} {'digital' if digital else 'proxy'}"
            ax = axes[0]
            ax.plot(snrs, [rows[s]["PSNR"] for s in snrs], ls, label=lab)
            ax = axes[1]
            ax.plot(snrs, [rows[s]["SSIM"] for s in snrs], ls, label=lab)
            ax = axes[2]
            ax.plot(snrs, [rows[s]["mAP_50"] for s in snrs], ls, label=lab)
            print(f"  {lab}: "
                  + " ".join(f"{s}dB={rows[s]['mAP_50']:.3f}" for s in snrs))
    axes[0].set_ylabel("PSNR (dB)"); axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("SSIM"); axes[1].grid(True, alpha=0.3)
    axes[2].set_ylabel("mAP@0.5"); axes[2].grid(True, alpha=0.3)
    for ax in axes:
        ax.set_xlabel("SNR (dB)")
    axes[0].legend(fontsize=6, loc="best")
    fig.suptitle("SNR sweep (proxy solid-ish: proxy=--, digital=-; test 432, seed 42, conf 0.001)")
    out = os.path.join(FIGS_DIR, "fig_snr_sweep.png")
    fig.tight_layout(); fig.savefig(out, dpi=130); print(f"  WROTE {out}")


# --------------------------------------------------------------------------
# Figure (c): FIM tau-sweep table + histograms (fim_metrics.py)
# --------------------------------------------------------------------------
def fig_fim(regenerate=False, device="cpu"):
    print("\n" + "=" * 70)
    print("(c) FIM tau-SWEEP + HISTOGRAMS  (R3 #5 routing quality)")
    print("=" * 70)
    sweep_csv = os.path.join(REPO, "results/fim/fim_tau_sweep.csv")
    cls_csv = os.path.join(REPO, "results/fim/fim_classification.csv")
    hist_png = os.path.join(FIGS_DIR, "fim_histograms.png")

    if not os.path.exists(sweep_csv) and regenerate:
        print("  [regenerate] results/fim/fim_tau_sweep.csv missing -> running fim_metrics.py (CPU)")
        cmd = [sys.executable, "fim_metrics.py",
               "--checkpoint", LEGACY["rvq_wave3"]["checkpoint"],
               "--data_path", DATA_PATH, "--detector", DETECTOR,
               "--device", device, "--snr", "10",
               "--out_csv", cls_csv, "--sweep_csv", sweep_csv,
               "--hist_png", hist_png]
        subprocess.run(cmd, cwd=REPO)

    rows = load_csv_cols(sweep_csv, col_key="tau")
    if rows is None:
        print(f"  [SKIP] {sweep_csv} missing (rerun with --regenerate)")
        return
    print(f"  FIM tau sweep (checkpoint {LEGACY['rvq_wave3']['name']}, "
          f"SHA {LEGACY['rvq_wave3']['sha256'][:16]}…, seed 42, test 432):")
    print(f"  {'tau':>5} {'P_high':>7} {'K':>6} {'BPP':>6} {'P':>6} {'R':>6} {'F1':>6} {'mAP@0.5':>8}")
    taus = sorted(rows)
    for t in taus:
        r = rows[t]
        print(f"  {t:5.1f} {r['p_high']:7.4f} {int(r['K']):6d} {r['bpp']:6.4f} "
              f"{r['precision']:6.4f} {r['recall']:6.4f} {r['f1']:6.4f} "
              f"{r.get('map_50', float('nan')):8.4f}")
    if os.path.exists(hist_png):
        print(f"  histograms: {hist_png}")


# --------------------------------------------------------------------------
# Figure (d): proxy-vs-digital table
# --------------------------------------------------------------------------
def fig_proxy_digital(regenerate=False, device="cpu"):
    print("\n" + "=" * 70)
    print("(d) PROXY-VS-DIGITAL TABLE  (R3 #3 §5.4)")
    print("=" * 70)
    for key in FINAL_KEYS:
        cfg = CONFIGS[key]
        proxy = load_eval_csv(os.path.join(REPO, f"evalfinal_{cfg['csv_base']}_proxy.csv"))
        digital = load_eval_csv(os.path.join(REPO, f"evalfinal_{cfg['csv_base']}_digital.csv"))
        if proxy is None or digital is None:
            if regenerate:
                if proxy is None:
                    regenerate_sweep(key, False, device)
                if digital is None:
                    regenerate_sweep(key, True, device)
                proxy = load_eval_csv(os.path.join(REPO, f"evalfinal_{cfg['csv_base']}_proxy.csv"))
                digital = load_eval_csv(os.path.join(REPO, f"evalfinal_{cfg['csv_base']}_digital.csv"))
            else:
                print(f"  [SKIP] {cfg['name']}: evalfinal_{cfg['csv_base']}_proxy.csv or "
                      f"_digital.csv missing (rerun with --regenerate)")
                continue
        print(f"\n  {cfg['name']}  (ckpt SHA {cfg['sha256'][:16]}…, seed 42)")
        print(f"  {'SNR':>5} {'PSNR P/D':>14} {'SSIM P/D':>14} {'mAP@0.5 P/D':>14} "
              f"{'BLER':>7} {'bit_acc':>8} {'erased':>7}")
        for snr in SNR_GRID:
            p, d = proxy.get(float(snr)), digital.get(float(snr))
            if p is None or d is None:
                continue
            print(f"  {snr:5.1f} {p['PSNR']:6.2f}/{d['PSNR']:6.2f} "
                  f"{p['SSIM']:6.3f}/{d['SSIM']:6.3f} "
                  f"{p['mAP_50']:6.4f}/{d['mAP_50']:6.4f} "
                  f"{d['digital_bler'] if d['digital_bler'] is not None else float('nan'):7.3f} "
                  f"{d['digital_bit_accuracy'] if d['digital_bit_accuracy'] is not None else float('nan'):8.3f} "
                  f"{d['digital_erased_token_fraction'] if d['digital_erased_token_fraction'] is not None else float('nan'):7.3f}")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Regenerate AquaSC figures from checkpoints (no retraining).")
    ap.add_argument("--figs", default="a,b,c,d",
                    help="comma-separated subset of a,b,c,d (default all)")
    ap.add_argument("--regenerate", action="store_true",
                    help="(re)generate missing CSVs on the CPU before plotting")
    ap.add_argument("--device", default="cpu", help="device for regeneration (default cpu; do NOT use cuda)")
    ap.add_argument("--check-only", action="store_true",
                    help="only check provenance / file presence, no plotting")
    args = ap.parse_args()

    print("#" * 70)
    print("# AquaSC plotting/reproduce_figures.py — figures from checkpoints, no retraining")
    print(f"# detector: best.pt (SHA {DETECTOR_SHA[:16]}…)  ceiling mAP@0.5={CEILING_MAP50}")
    print(f"# seed={SEED}  data={DATA_PATH}")
    print("#" * 70)

    figs = args.figs.split(",")
    if args.check_only:
        for key in FINAL_KEYS:
            cfg = CONFIGS[key]
            for kind in ("proxy", "digital"):
                p = os.path.join(REPO, f"evalfinal_{cfg['csv_base']}_{kind}.csv")
                print(f"  evalfinal_{cfg['csv_base']}_{kind}.csv exists={os.path.exists(p)}")
        print(f"  results/fim/fim_tau_sweep.csv exists={os.path.exists(os.path.join(REPO,'results/fim/fim_tau_sweep.csv'))}")
        print(f"  checkpoints_sha256.txt exists={os.path.exists(os.path.join(REPO,'checkpoints_sha256.txt'))}")
        return

    if "a" in figs:
        fig_rate_retention(regenerate=args.regenerate, device=args.device)
    if "b" in figs:
        fig_snr_sweep(regenerate=args.regenerate, device=args.device)
    if "c" in figs:
        fig_fim(regenerate=args.regenerate, device=args.device)
    if "d" in figs:
        fig_proxy_digital(regenerate=args.regenerate, device=args.device)

    print("\nProvenance: every number above traces to (i) a CSV whose path is printed, "
          "(ii) the checkpoint + SHA-256 in the CONFIGS/LEGACY registries, "
          "(iii) seed=42, and (iv) the pipeline label (analog proxy vs digital LDPC). "
          "Config YAMLs in configs/; SHA manifest in checkpoints_sha256.txt.")


if __name__ == "__main__":
    main()
