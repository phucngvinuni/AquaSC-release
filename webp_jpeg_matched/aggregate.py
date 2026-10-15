"""Aggregate the matched-budget WebP/JPEG rerun and compare with AquaSC at the same channel uses (paired by image).
  __TORCH_PY__ webp_jpeg_matched/aggregate.py
Writes RESULTS_matched_budget.json / .md.  AquaSC references: headline JointFrame (6144 sym), retrained r061 (3072), r030 (1536).
Paired CI: stats.paired_bootstrap on per-image mean-over-10-draws AP50, 10,000 resamples, seed 42, percentile 95%.
"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import stats

RUNS = ROOT / "experiments/matched5/runs"
AQUA = {"low": "jointframe_r030_bestmap", "mid": "jointframe_r061_bestmap", "high": "sweep"}
SNRS = [-5, 0, 5, 10, 15, 20]
sdir = lambda s: "m5" if s == -5 else str(s)
RATE = {"low": 0.030612, "mid": 0.061224, "high": 0.122449}

def aqua(budget, snr):
    d = RUNS / AQUA[budget]
    mt = np.load(d / f"aqua_{sdir(snr)}/metrics.npz")
    sm = json.loads((d / f"aqua_{sdir(snr)}/summary.json").read_text())
    ps = json.loads((d / f"phy_{sdir(snr)}/psnr_ssim.json").read_text()) if (d / f"phy_{sdir(snr)}/psnr_ssim.json").exists() else {}
    return mt["per_image_ap"], sm, ps

out, md = {}, []
md.append("# Matched-budget WebP/JPEG rerun (protocol WEBP_JPEG_RERUN_PROTOCOL.md) vs AquaSC\n")
md.append("Test split, 432 images x 10 draws per cell; all records scored (capacity/length/CRC/decode failures -> black image). "
          "CI = paired bootstrap over images (AquaSC - baseline), per-image AP50 averaged over draws. "
          "AquaSC = Gray QPSK JointFrame (headline / retrained r061 / retrained r030); baselines = Gray 16-QAM, BG2 k=1024 rate-1/2 blocks.\n")
for budget in ("low", "mid", "high"):
    md.append(f"\n## Budget {budget}: {RATE[budget]} uses/px, {int(RATE[budget]*50176+0.5)} symbols/image\n")
    for codec in ("webp", "jpeg"):
        run = HERE / "runs" / f"{codec}_{budget}"
        if not (run / "metrics.npz").exists():
            md.append(f"- {codec}: not available\n"); continue
        mt = np.load(run / "metrics.npz"); cfg = json.loads((run / "config.json").read_text())
        snrs = mt["snr_db"].tolist()
        md.append(f"\n### {codec.upper()} q={cfg['quality']} ({cfg['options']}); capacity {cfg['capacity_B_after_4B_length']} B; "
                  f"test capacity failures {cfg['n_capacity_failures']}/{cfg['n_images']} ({100*cfg['frac_capacity_failures']:.2f}%); "
                  f"clean-channel AP50 {mt['clean_ap'].mean():.4f}, PSNR {mt['clean_psnr'].mean():.2f}, SSIM {mt['clean_ssim'].mean():.4f}\n")
        md.append("| SNR | base AP50 | base pooled mAP50 | base PSNR | base SSIM | black % | AquaSC AP50 | AquaSC PSNR | diff AP50 [95% CI] |\n|---|---|---|---|---|---|---|---|---|")
        for si, snr in enumerate(snrs):
            b_img = mt["per_image_ap"][si].mean(1)                # per-image mean over draws
            a_arr, sm, ps = aqua(budget, snr)
            a_img = a_arr.mean(1)
            pb = stats.paired_bootstrap(a_img, b_img, n_boot=10000, seed=42)
            row = dict(snr=snr, base_ap50=float(b_img.mean()), base_pooled=float(mt["pooled_map50"][si].mean()),
                       base_psnr=float(mt["psnr"][si].mean()), base_ssim=float(mt["ssim"][si].mean()),
                       frac_black=float((mt["recon_class"][si] == 0).mean()), aqua_ap50=float(a_img.mean()),
                       aqua_pooled=float(sm["pooled_map50"]), aqua_psnr=ps.get("psnr_mean"), aqua_ssim=ps.get("ssim_mean"),
                       diff=pb["mean_diff"], ci=[pb["ci_low"], pb["ci_high"]], p=pb["p_value"],
                       symbols_emitted=[int(mt["symbols_emitted"][si].min()), float(mt["symbols_emitted"][si].mean()), int(mt["symbols_emitted"][si].max())],
                       aqua_symbols=sm["symbols_per_image"], frac_any_block_failed=float((mt["n_failed_blocks"][si] > 0).mean()),
                       frac_decode_failed_after_length_ok=float(((mt["status"][si] == 0) & (~mt["decode_ok"][si])).mean()))
            out.setdefault(f"{codec}_{budget}", []).append(row)
            ap = f"{row['aqua_psnr']:.2f}" if row["aqua_psnr"] is not None else "n/a"
            md.append(f"| {snr:+d} | {row['base_ap50']:.4f} | {row['base_pooled']:.4f} | {row['base_psnr']:.2f} | {row['base_ssim']:.4f} | "
                      f"{100*row['frac_black']:.1f} | {row['aqua_ap50']:.4f} | {ap} | {row['diff']:+.4f} [{row['ci'][0]:+.4f}, {row['ci'][1]:+.4f}] |")
        sy = [r["symbols_emitted"] for r in out[f"{codec}_{budget}"]]
        md.append(f"\nActual transmitted symbols/image (min/mean/max over all records): {min(s[0] for s in sy)}/{np.mean([s[1] for s in sy]):.1f}/{max(s[2] for s in sy)}; AquaSC {out[f'{codec}_{budget}'][0]['aqua_symbols']}.\n")
(HERE / "RESULTS_matched_budget.json").write_text(json.dumps(out, indent=2))
(HERE / "RESULTS_matched_budget.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
