"""aggregate_ablation_routing_coded.py -- 2026-09-23. Reads the 8 conditions
(7 new, run by run_ablation_routing_coded.sh, + the existing Full+FIM headline
reused unchanged from runs/sweep/aqua_{10,20}) and produces one consolidated
table: pooled mAP@0.5, mean per-image AP@0.5, frame rejection rate, and
frame-loss-conditional AP (lost vs. alive), at 10 and 20 dB, through the real
LDPC-coded joint-frame pipeline -- the coded-channel counterpart to
RESULTS_OF_RECORD.md sections 3 (FIM/BBW ablation) and 4 (routing
precision/recall), which used the uncoded-digital protocol.

Run in siso-project-env (numpy/torch only, no GPU needed).
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path("__AQUASC_ROOT__/experiments/matched5")
RUNS = ROOT / "runs" / "ablation_routing_coded"
SNRS = [10, 20]

# name -> (phy_dir_template or None if reusing the existing headline sweep,
#          aqua_dir_template or None, description)
CONDITIONS = [
    ("full_fim",     "Full (flagship)", "FIM (natural)"),
    ("fim_nobb",     "FIM-NoBB",        "FIM (natural)"),
    ("base",         "Base",            "random, disable_fim (natural)"),
    ("bb_nofim",     "BB-NoFIM_V2",     "random, disable_fim (natural)"),
    ("full_oracle",  "Full (flagship)", "oracle_box (override)"),
    ("full_random",  "Full (flagship)", "random (override)"),
    ("full_uniform", "Full (flagship)", "uniform (override)"),
    ("full_detheat", "Full (flagship)", "detector_heatmap (override)"),
]

# Denominators, matching RESULTS_OF_RECORD.md section 1/3's own convention.
POOLED_CEILING = 0.7292211
PERIMAGE_CEILING = 0.7743


def paths_for(name, snr):
    if name == "full_fim":
        # Reuse the existing headline sweep, unchanged -- do not rerun.
        phy_dir = ROOT / "runs" / "sweep" / f"phy_{snr}"
        aqua_dir = ROOT / "runs" / "sweep" / f"aqua_{snr}"
        return phy_dir, aqua_dir
    return RUNS / f"phy_{name}_snr{snr}", RUNS / f"aqua_{name}_snr{snr}"


def load_cell(name, snr):
    phy_dir, aqua_dir = paths_for(name, snr)
    # receiver.npz holds frame_ok; metrics.npz holds per_image_ap [n, M] + pooled_map50 [M]
    recv_path = phy_dir / "receiver.npz"
    metrics_path = aqua_dir / "metrics.npz"
    if not recv_path.exists() or not metrics_path.exists():
        return None
    recv = np.load(recv_path, allow_pickle=False)
    metrics = np.load(metrics_path, allow_pickle=False)
    frame_ok = recv["frame_ok"]  # [n, M] bool (or [n] if M folded differently -- handle both)
    per_image_ap = metrics["per_image_ap"]  # [n, M]
    pooled = metrics["pooled_map50"]  # [M]
    if frame_ok.ndim == 1:
        frame_ok = frame_ok[:, None].repeat(per_image_ap.shape[1], axis=1)
    ok_flat = frame_ok.reshape(-1)
    ap_flat = per_image_ap.reshape(-1)
    frame_loss_rate = float(1 - ok_flat.mean())
    lost_mean = float(ap_flat[~ok_flat].mean()) if (~ok_flat).any() else float("nan")
    lost_n = int((~ok_flat).sum())
    alive_mean = float(ap_flat[ok_flat].mean()) if ok_flat.any() else float("nan")
    alive_n = int(ok_flat.sum())
    return dict(
        pooled_map50_mean=float(np.mean(pooled)), pooled_map50_std=float(np.std(pooled)),
        per_image_ap_mean=float(np.mean(per_image_ap)), per_image_ap_std=float(np.std(np.mean(per_image_ap, axis=0))),
        frame_loss_rate=frame_loss_rate, lost_mean=lost_mean, lost_n=lost_n,
        alive_mean=alive_mean, alive_n=alive_n, M=per_image_ap.shape[1], n_images=per_image_ap.shape[0],
    )


def main():
    all_results = {}
    print(f"{'Condition':28s} {'Router':30s} {'SNR':>4s} {'pooled_mAP':>11s} {'per-img_AP':>11s} "
          f"{'frame_loss%':>11s} {'lost_AP':>8s} {'alive_AP':>8s}")
    for name, arm_label, router_label in CONDITIONS:
        for snr in SNRS:
            cell = load_cell(name, snr)
            if cell is None:
                print(f"{arm_label:28s} {router_label:30s} {snr:4d}  MISSING (not run yet or paths wrong)")
                continue
            all_results[f"{name}_snr{snr}"] = dict(name=name, arm=arm_label, router=router_label, snr=snr, **cell)
            print(f"{arm_label:28s} {router_label:30s} {snr:4d} "
                  f"{cell['pooled_map50_mean']:11.4f} {cell['per_image_ap_mean']:11.4f} "
                  f"{cell['frame_loss_rate']*100:10.2f}% {cell['lost_mean']:8.4f} {cell['alive_mean']:8.4f}")

    out_path = RUNS / "aggregate_results.json"
    RUNS.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
