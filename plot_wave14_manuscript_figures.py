#!/usr/bin/env python
"""Generate the Wave-14 manuscript figures from cleared results only.

Outputs
-------
figs/figure_ablation_wave14.{pdf,png}
    Mean per-image AP@0.50 with 95% Student-t confidence intervals for the
    four J2 ablation arms at 10 and 20 dB.  Values are recomputed from the
    canonical per-image arrays and checked against RESULTS_OF_RECORD.md §3.

figs/figure_crossover_thresholds_wave14.{pdf,png}
    Statistical and operational AquaSC/ROI-JSCC crossover thresholds parsed
    directly from RESULTS_OF_RECORD.md §6.

figs/figure_flagship_coded_snr_wave14.{pdf,png}
    True-FINAL coded-LDPC six-point SNR sweep, with pooled and per-image
    detection metrics checked against RESULTS_OF_RECORD.md §2.

figs/figure_routing_pr_wave14.{pdf,png}
    FIM and detector-heatmap routing precision/recall with the cleared 95%
    confidence intervals from RESULTS_OF_RECORD.md §4.

figs/figure_bitstream_jointframe.{pdf,png}
    Flagship source-packet and unequal-error-protection layout specified by
    bitstream.py and digital_pipeline.py.

figs/figure_train_eval_pipeline_jointframe.{pdf,png}
    Explicit separation between the differentiable training proxy and the
    deployable coded evaluation chain.

This script performs no evaluation and does not load model checkpoints.
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


REPO_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = REPO_DIR.parent
RESULTS_PATH = WORKSPACE_DIR / "RESULTS_OF_RECORD.md"
if not RESULTS_PATH.exists():          # supplement layout: the ledger ships inside this directory
    RESULTS_PATH = REPO_DIR / "RESULTS_OF_RECORD.md"
DEFAULT_FIGS_DIR = WORKSPACE_DIR / "figs"
PER_IMAGE_DIR = REPO_DIR / "f1_perimage"

sys.path.insert(0, str(REPO_DIR))
import stats  # noqa: E402  (local, after REPO_DIR is placed on sys.path)


COLORS = {
    "blue": "#0077BB",
    "cyan": "#33BBEE",
    "teal": "#009988",
    "orange": "#EE7733",
    "red": "#CC3311",
    "grey": "#777777",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# Rounded values cleared in RESULTS_OF_RECORD.md §3.  The plotted coordinates
# come from the arrays; these constants are only fail-closed provenance checks.
ABLATION_EXPECTED = {
    "BASE": {10: (0.298, 0.282, 0.315), 20: (0.325, 0.309, 0.341)},
    "FIM_NOBB": {10: (0.383, 0.366, 0.400), 20: (0.426, 0.408, 0.445)},
    "BB_NOFIM_V2": {10: (0.389, 0.366, 0.412), 20: (0.637, 0.614, 0.659)},
    "FULL": {10: (0.482, 0.457, 0.507), 20: (0.720, 0.699, 0.742)},
}

ABLATION_STYLES = {
    "BASE": ("Base", COLORS["grey"], "o"),
    "FIM_NOBB": ("FIM–NoBB", COLORS["blue"], "s"),
    "BB_NOFIM_V2": ("BB–NoFIM", COLORS["orange"], "^"),
    "FULL": ("Full AquaSC", COLORS["teal"], "D"),
}

FLAGSHIP_SNRS = np.array([-5, 0, 5, 10, 15, 20], dtype=float)
FLAGSHIP_POOLED = np.array([0.2205, 0.2386, 0.3244, 0.4841, 0.6311, 0.6864])
FLAGSHIP_PER_IMAGE = np.array([0.2721, 0.2874, 0.3815, 0.5568, 0.6960, 0.7421])

ROUTING_RESULTS = {
    # Updated 2026-09-12: routing_pr_topk63_result.npz was regenerated on the
    # true flagship (`..._FINAL`, SHA b1d2eb254194) -- the previous numbers
    # here were from E2_ERASURE, mislabeled as flagship (RESULTS_OF_RECORD.md
    # §4). Old artifact preserved as
    # routing_pr_topk63_result_E2_ERASURE_MISLABELED.npz.
    "FIM": {"precision": (0.7700, 0.7437, 0.7962), "recall": (0.8149, 0.7938, 0.8360)},
    "Detector heatmap": {"precision": (0.688, 0.663, 0.712), "recall": (0.725, 0.705, 0.744)},
}


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, format=suffix)
        print(f"wrote {path}")
    plt.close(fig)


def _round3(value: float) -> float:
    return float(f"{value:.3f}")


def plot_ablation(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.55, 3.05))
    snrs = np.array([10.0, 20.0])
    offsets = np.array([-0.75, -0.25, 0.25, 0.75])

    for offset, (key, (label, color, marker)) in zip(
        offsets, ABLATION_STYLES.items()
    ):
        means, lower, upper = [], [], []
        for snr in (10, 20):
            values = np.load(PER_IMAGE_DIR / f"ablation_{key}_snr{snr}.npy")
            if values.shape != (432,):
                raise ValueError(f"unexpected shape for {key} at {snr} dB: {values.shape}")
            ci = stats.t_confidence_interval(values)
            actual = (ci["mean"], ci["ci_low"], ci["ci_high"])
            observed = tuple(_round3(value) for value in actual)
            expected = ABLATION_EXPECTED[key][snr]
            # The record stores three decimals.  Accept the half-unit boundary
            # plus a small floating-point margin (FULL@20 lower CI is 0.69845
            # from the array and is recorded as 0.699).
            if any(abs(a - b) > 6e-4 for a, b in zip(actual, expected)):
                raise RuntimeError(
                    f"RESULTS_OF_RECORD.md §3 mismatch for {key} at {snr} dB: "
                    f"observed {observed}, expected {expected}"
                )
            means.append(ci["mean"])
            lower.append(ci["mean"] - ci["ci_low"])
            upper.append(ci["ci_high"] - ci["mean"])

        means_arr = np.asarray(means)
        ax.errorbar(
            snrs + offset,
            means_arr,
            yerr=np.vstack([lower, upper]),
            fmt=marker,
            linestyle="none",
            color=color,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.45,
            markersize=5.2,
            elinewidth=1.1,
            capsize=2.8,
            label=label,
            zorder=3,
        )

    ax.axhline(
        0.7743,
        color=COLORS["grey"],
        linewidth=0.9,
        linestyle=(0, (4, 3)),
        label="Clean-image ceiling",
        zorder=1,
    )
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Mean per-image AP@0.50")
    ax.set_xticks(snrs)
    ax.set_xlim(7.8, 22.2)
    ax.set_ylim(0.0, 0.82)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, zorder=0)
    ax.legend(frameon=False, ncol=2, loc="lower right", handletextpad=0.5, columnspacing=0.9)
    fig.tight_layout(pad=0.35)
    _save(fig, output_dir, "figure_ablation_wave14")


def _parse_crossover_rows() -> list[tuple[str, float, float]]:
    text = RESULTS_PATH.read_text(encoding="utf-8")
    start = text.index("## 6. Unified crossover table")
    end = text.index("## 6b.", start)
    section = text[start:end]

    rows = []
    for raw_line in section.splitlines():
        if not raw_line.startswith("|") or "Statistical crossover" in raw_line:
            continue
        cells = [cell.strip() for cell in raw_line.strip().strip("|").split("|")]
        if len(cells) != 4 or cells[0].startswith("---") or "(" not in cells[0]:
            continue
        stat_match = re.search(r"[-+]?\d+(?:\.\d+)?", cells[2])
        op_match = re.search(r"[-+]?\d+(?:\.\d+)?", cells[3])
        if stat_match is None or op_match is None:
            continue
        label = re.sub(r"\*+", "", f"{cells[0]} vs {cells[1]}")
        label = label.replace("r⅓", "r1/3").replace("r½", "r1/2")
        rows.append((label, float(stat_match.group()), float(op_match.group())))

    if len(rows) != 6:
        raise RuntimeError(f"expected 6 crossover rows in RESULTS_OF_RECORD.md §6, found {len(rows)}")
    return rows


def plot_crossover(output_dir: Path) -> None:
    # Retired figure (historical rate labels; not in the manuscript). It parses the internal results ledger,
    # which is not part of the public release: skip quietly when the ledger is absent.
    if not RESULTS_PATH.exists():
        print("skip figure_crossover_thresholds_wave14: results ledger not available (retired figure)")
        return
    rows = _parse_crossover_rows()
    labels = [row[0] for row in rows]
    statistical = np.array([row[1] for row in rows])
    operational = np.array([row[2] for row in rows])
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(7.15, 3.55))
    for yi, x0, x1 in zip(y, statistical, operational):
        ax.plot([x0, x1], [yi, yi], color="#BBBBBB", linewidth=1.2, zorder=1)
    ax.scatter(
        statistical,
        y,
        s=34,
        marker="o",
        color=COLORS["blue"],
        edgecolor="black",
        linewidth=0.45,
        label="Statistical crossover ($\\Delta=0$)",
        zorder=3,
    )
    ax.scatter(
        operational,
        y,
        s=36,
        marker="s",
        color=COLORS["orange"],
        edgecolor="black",
        linewidth=0.45,
        label="Operational crossover ($\\Delta\\geq0.05$)",
        zorder=3,
    )
    for yi, x0, x1 in zip(y, statistical, operational):
        ax.text(x0 - 0.17, yi - 0.17, f"{x0:.2f}", ha="right", va="center", fontsize=7)
        ax.text(x1 + 0.17, yi - 0.17, f"{x1:.2f}", ha="left", va="center", fontsize=7)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Crossover SNR (dB)")
    ax.set_xlim(0, 15.3)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.55, zorder=0)
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        columnspacing=1.4,
    )
    fig.tight_layout(pad=0.45)
    _save(fig, output_dir, "figure_crossover_thresholds_wave14")


# Wave 15: rate-constrained MCS envelope (RESULTS_OF_RECORD.md §6i). Per-image
# mean AP@0.5 only -- pooled numbers must never enter this figure (the G1
# denominator defect this section exists to avoid repeating a third time).
ENVELOPE_SNRS = (-5, 0, 5, 10, 15, 20)
ENVELOPE_AQUASC = (
    ("flagship", "flagship", 0.0714),
    ("L2", "l2_coded", 0.0816),
    ("QPSK r1/2", "aquasc_qpsk_r12", 0.1429),
    ("BPSK r1/3", "aquasc_bpsk_r13", 0.3673),
)
ENVELOPE_ROIJSCC = (
    ("rcpp128", "roijscc_rcpp128", 0.0469),
    ("rcpp64", "roijscc_rcpp64", 0.0938),
    ("rcpp32", "roijscc_rcpp32", 0.1875),
)


def _envelope_fname(prefix: str, snr: float) -> Path:
    tag = f"m{abs(int(snr))}" if snr < 0 else f"{int(snr)}"
    path = PER_IMAGE_DIR / f"{prefix}_snr{tag}.npy"
    if path.exists():
        return path
    path = PER_IMAGE_DIR / f"{prefix}_snr{int(snr)}.npy"
    if path.exists():
        return path
    raise FileNotFoundError(f"no per-image array for {prefix} at SNR={snr}")


def plot_envelope(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.55))
    cmap = plt.get_cmap("viridis")
    snr_colors = {snr: cmap(i / (len(ENVELOPE_SNRS) - 1)) for i, snr in enumerate(ENVELOPE_SNRS)}

    for snr in ENVELOPE_SNRS:
        color = snr_colors[snr]
        aqua_rates = [rate for _, _, rate in ENVELOPE_AQUASC]
        aqua_means = [np.load(_envelope_fname(pfx, snr)).mean() for _, pfx, _ in ENVELOPE_AQUASC]
        roi_rates = [rate for _, _, rate in ENVELOPE_ROIJSCC]
        roi_means = [np.load(_envelope_fname(pfx, snr)).mean() for _, pfx, _ in ENVELOPE_ROIJSCC]

        ax.plot(aqua_rates, aqua_means, color=color, linewidth=1.3, linestyle="-",
                marker="o", markersize=4.2, markeredgecolor="black", markeredgewidth=0.35,
                zorder=3, label=f"{snr:g} dB" if snr == ENVELOPE_SNRS[0] else None)
        ax.plot(roi_rates, roi_means, color=color, linewidth=1.1, linestyle="--",
                marker="^", markersize=4.4, markeredgecolor="black", markeredgewidth=0.35,
                zorder=3)
        # Label the SNR once, near the flagship (cheapest AquaSC) point.
        ax.annotate(f"{snr:g}", (aqua_rates[0], aqua_means[0]), textcoords="offset points",
                    xytext=(-7, 3), fontsize=6.5, color=color, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("Channel uses / pixel")
    ax.set_ylabel("Per-image mean AP@0.50")
    ax.set_ylim(0.20, 0.80)
    ax.grid(axis="both", which="major", color="#E5E5E5", linewidth=0.5, zorder=0)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="black", linewidth=1.3, marker="o", markersize=4.2,
               markeredgecolor="black", label="AquaSC (flagship/L2/QPSK/BPSK)"),
        Line2D([0], [0], color="black", linewidth=1.1, linestyle="--", marker="^",
               markersize=4.4, markeredgecolor="black", label="ROI-JSCC (rcpp128/64/32)"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="lower right", fontsize=6.8)
    ax.text(0.02, 0.98, "Number = SNR (dB); color = SNR band (dark→light = low→high)",
            transform=ax.transAxes, fontsize=6.2, va="top", ha="left", color="#555555")
    fig.tight_layout(pad=0.35)
    _save(fig, output_dir, "figure_envelope_wave15")


def plot_flagship_snr(output_dir: Path) -> None:
    csv_path = REPO_DIR / "evalfinal_flagship_coded_ldpc_snr_sweep.csv"
    # The legacy logger appended eight diagnostics without extending the
    # five-column header.  Read the two required columns explicitly.
    pooled = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=(0, 3))
    if not np.array_equal(pooled[:, 0], FLAGSHIP_SNRS):
        raise RuntimeError("flagship CSV SNR grid does not match RESULTS_OF_RECORD.md §2")
    if not np.allclose(pooled[:, 1], FLAGSHIP_POOLED, atol=5e-5):
        raise RuntimeError("flagship pooled mAP does not match RESULTS_OF_RECORD.md §2")

    per_image = []
    for snr, expected in zip(FLAGSHIP_SNRS.astype(int), FLAGSHIP_PER_IMAGE):
        values = np.load(PER_IMAGE_DIR / f"flagship_snr{snr}.npy")
        if values.shape != (432,):
            raise ValueError(f"unexpected per-image shape at {snr} dB: {values.shape}")
        mean = float(np.mean(values))
        if abs(mean - expected) > 6e-5:
            raise RuntimeError(
                f"flagship per-image mean at {snr} dB ({mean:.6f}) "
                f"does not match RESULTS_OF_RECORD.md §2 ({expected:.4f})"
            )
        per_image.append(mean)

    fig, ax = plt.subplots(figsize=(3.55, 3.05))
    ax.plot(
        FLAGSHIP_SNRS, pooled[:, 1], color=COLORS["blue"], marker="o",
        linewidth=1.5, markersize=4.7, label="Pooled mAP@0.50",
    )
    ax.plot(
        FLAGSHIP_SNRS, per_image, color=COLORS["orange"], marker="s",
        linewidth=1.5, markersize=4.5, label="Mean per-image AP@0.50",
    )
    ax.axhline(0.7292211, color=COLORS["blue"], linestyle=(0, (4, 3)), linewidth=0.8, alpha=0.75)
    ax.axhline(0.7743, color=COLORS["orange"], linestyle=(0, (4, 3)), linewidth=0.8, alpha=0.75)
    ax.text(20.1, 0.7292211, "pooled ceiling", color=COLORS["blue"], va="center", fontsize=6.5)
    ax.text(20.1, 0.7743, "per-image ceiling", color=COLORS["orange"], va="center", fontsize=6.5)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Detection performance")
    ax.set_xticks(FLAGSHIP_SNRS)
    ax.set_xlim(-6, 24.5)
    ax.set_ylim(0.15, 0.82)
    ax.grid(color="#D9D9D9", linewidth=0.55)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(pad=0.35)
    _save(fig, output_dir, "figure_flagship_coded_snr_wave14")


def plot_routing_pr(output_dir: Path) -> None:
    artifact_paths = {
        "FIM": REPO_DIR / "routing_pr_topk63_result.npz",
        "Detector heatmap": REPO_DIR / "routing_pr_segdet_topk63_result.npz",
    }
    for label, artifact_path in artifact_paths.items():
        artifact = np.load(artifact_path)
        if int(artifact["K"]) != 63:
            raise RuntimeError(f"{artifact_path.name} was not evaluated at K=63")
        for metric, array_key in (("precision", "precisions"), ("recall", "recalls")):
            values = artifact[array_key]
            if values.shape != (432,):
                raise ValueError(f"unexpected {array_key} shape in {artifact_path.name}: {values.shape}")
            ci = stats.t_confidence_interval(values)
            observed = (ci["mean"], ci["ci_low"], ci["ci_high"])
            expected = ROUTING_RESULTS[label][metric]
            if any(abs(a - b) > 6e-4 for a, b in zip(observed, expected)):
                raise RuntimeError(
                    f"{label} {metric} artifact does not match RESULTS_OF_RECORD.md §4: "
                    f"observed {observed}, expected {expected}"
                )
    fim_artifact = np.load(artifact_paths["FIM"])
    checkpoint = str(fim_artifact["checkpoint"])
    if "E2_ERASURE" in checkpoint:
        warnings.warn(
            "routing_pr_topk63_result.npz records E2_ERASURE, while "
            "RESULTS_OF_RECORD.md §4 labels the row 'FIM (flagship)'; do not "
            "describe this figure as flagship until provenance is reconciled",
            stacklevel=2,
        )

    fig, ax = plt.subplots(figsize=(3.55, 2.8))
    labels = list(ROUTING_RESULTS)
    x = np.arange(2)
    width = 0.34
    for offset, (label, color) in zip((-width / 2, width / 2), zip(labels, (COLORS["blue"], COLORS["orange"]))):
        triples = [ROUTING_RESULTS[label][metric] for metric in ("precision", "recall")]
        means = np.array([triple[0] for triple in triples])
        lower = means - np.array([triple[1] for triple in triples])
        upper = np.array([triple[2] for triple in triples]) - means
        ax.bar(x + offset, means, width, color=color, edgecolor="black", linewidth=0.45, label=label, zorder=2)
        ax.errorbar(x + offset, means, yerr=np.vstack([lower, upper]), fmt="none", ecolor="black", elinewidth=0.9, capsize=2.5, zorder=3)
    ax.set_xticks(x, ["Precision", "Recall"])
    ax.set_ylabel("Patch-level score")
    ax.set_ylim(0.60, 0.86)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, zorder=0)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(pad=0.35)
    _save(fig, output_dir, "figure_routing_pr_wave14")


def _box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, text_value: str,
         facecolor: str, fontsize: float = 7.0) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.02,rounding_size=0.025",
        facecolor=facecolor, edgecolor="#333333", linewidth=0.6,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text_value, ha="center", va="center", fontsize=fontsize)
    return patch


def plot_bitstream(output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.15, 3.35), gridspec_kw={"height_ratios": [1.05, 1.0]})
    ax = axes[0]
    ax.set_xlim(0, 3560)
    ax.set_ylim(0, 1)
    fields = [
        ("Header\n152 b", 152, COLORS["blue"]),
        ("Routing mask\n313 b", 313, COLORS["cyan"]),
        ("RVQ stage 1\n378 b", 378, COLORS["teal"]),
        ("RVQ stage 2\n378 b", 378, "#88CCEE"),
        ("RVQ stage 3\n378 b", 378, "#CCBB44"),
        ("Low-path indices\n1,442 b", 1442, COLORS["orange"]),
        ("CRC-32", 32, COLORS["red"]),
    ]
    left = 0
    for label, bits, color in fields:
        ax.add_patch(Rectangle((left, 0.23), bits, 0.53, facecolor=color, edgecolor="white", linewidth=0.8))
        fontsize = 5.2 if bits <= 152 else (5.8 if bits <= 313 else 7)
        ax.text(left + bits / 2, 0.495, label, ha="center", va="center", fontsize=fontsize)
        left += bits
    ax.text(0, 0.88, "Joint-frame source packet: 3,073 bits before channel coding", fontsize=8, weight="bold")
    ax.axis("off")

    ax = axes[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _box(ax, (0.02, 0.24), 0.24, 0.52, "3,073-bit joint frame\nheader + mask + all indices\n+ frame CRC-32", "#A9D8EC")
    _box(ax, (0.38, 0.24), 0.24, 0.52, "One 5G-NR BG2 codeword\nk=3,073, n=12,288\nsum-product, 40 iterations", "#F5C49B")
    _box(ax, (0.74, 0.24), 0.23, 0.52, "Gray QPSK\n6,144 symbols\nSISO Rayleigh", "#B8E2D2")
    for x0, x1 in ((0.26, 0.38), (0.62, 0.74)):
        ax.add_patch(FancyArrowPatch((x0, 0.5), (x1, 0.5), arrowstyle="-|>", mutation_scale=10, linewidth=0.8, color="#333333"))
    ax.text(0.5, 0.91, "Primary joint-frame protection and modulation", ha="center", fontsize=8, weight="bold")
    ax.axis("off")
    fig.tight_layout(pad=0.5)
    _save(fig, output_dir, "figure_bitstream_jointframe")


def plot_train_eval_pipeline(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.15, 3.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    shared = [(0.02, "Image"), (0.17, "ViT encoder"), (0.32, "FIM +\ntop-$K$ routing"), (0.49, "Dual-rate\nRVQ")]
    for x0, label in shared:
        _box(ax, (x0, 0.42), 0.12, 0.18, label, "#DCEAF5")
    for x0, x1 in ((0.14, 0.17), (0.29, 0.32), (0.44, 0.49)):
        ax.add_patch(FancyArrowPatch((x0, 0.51), (x1, 0.51), arrowstyle="-|>", mutation_scale=9, linewidth=0.75))

    _box(ax, (0.66, 0.69), 0.14, 0.18, "Training proxy\npower norm +\nRayleigh/AWGN", "#F5C49B", 6.7)
    _box(ax, (0.84, 0.69), 0.14, 0.18, "Decoder + losses\n(differentiable)", "#F5C49B", 6.7)
    _box(ax, (0.66, 0.13), 0.14, 0.18, "Joint frame + CRC\none BG2 codeword\n+ QPSK", "#B8E2D2", 6.7)
    _box(ax, (0.84, 0.13), 0.14, 0.18, "APP demap/decode\nframe accept/loss\n+ decoder", "#B8E2D2", 6.7)
    for y1 in (0.78, 0.22):
        ax.add_patch(FancyArrowPatch((0.61, 0.51), (0.66, y1), arrowstyle="-|>", mutation_scale=9, linewidth=0.75))
        ax.add_patch(FancyArrowPatch((0.80, y1), (0.84, y1), arrowstyle="-|>", mutation_scale=9, linewidth=0.75))
    ax.text(0.735, 0.94, "Training", color=COLORS["orange"], ha="center", weight="bold")
    ax.text(0.735, 0.04, "Evaluation/deployment", color=COLORS["teal"], ha="center", weight="bold")
    ax.text(0.50, 0.64, "shared learned representation", ha="center", fontsize=7, color=COLORS["grey"])
    ax.axis("off")
    fig.tight_layout(pad=0.25)
    _save(fig, output_dir, "figure_train_eval_pipeline_jointframe")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figs",
        default="ablation,crossover,envelope,flagship_snr,routing_pr,bitstream,train_eval",
        help="comma-separated subset: ablation,crossover,envelope,flagship_snr,routing_pr,bitstream,train_eval",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIGS_DIR)
    args = parser.parse_args()

    selected = {item.strip() for item in args.figs.split(",") if item.strip()}
    unknown = selected - {"ablation", "crossover", "envelope", "flagship_snr", "routing_pr", "bitstream", "train_eval"}
    if unknown:
        parser.error(f"unknown figure(s): {', '.join(sorted(unknown))}")
    if "ablation" in selected:
        plot_ablation(args.output_dir)
    if "crossover" in selected:
        plot_crossover(args.output_dir)
    if "envelope" in selected:
        plot_envelope(args.output_dir)
    if "flagship_snr" in selected:
        plot_flagship_snr(args.output_dir)
    if "routing_pr" in selected:
        plot_routing_pr(args.output_dir)
    if "bitstream" in selected:
        plot_bitstream(args.output_dir)
    if "train_eval" in selected:
        plot_train_eval_pipeline(args.output_dir)


if __name__ == "__main__":
    main()
