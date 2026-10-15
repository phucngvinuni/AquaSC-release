"""Shared Part-B-compliant matplotlib style (Wave 21 dispatch, 2026-09-15):
Liberation Serif throughout including mathtext, no in-image titles (panel
labels only), vector PDF + PNG, IEEE column widths. Import and call
apply_style() before building any figure; use COL_SINGLE/COL_DOUBLE for
figsize width in inches.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL_SINGLE = 3.5   # IEEE single-column width, inches
COL_DOUBLE = 7.16  # IEEE double-column width, inches

COLORS = {
    "blue": "#0077BB",
    "orange": "#EE7733",
    "teal": "#009988",
    "purple": "#AA3377",
    "grey": "#777777",
    "cyan": "#33BBEE",
    "red": "#CC3311",
    "magenta": "#EE3377",
}

CKPT_SHA_PREFIX = "b1d2eb25"
CEILING_PER_IMAGE = 0.7743
CEILING_POOLED = 0.7292211
PRIOR_FLOOR = 0.2711  # Site A, per RESULTS_OF_RECORD.md section 0/6f (labeled
                       # "per-image" in section 6f's prose but "(pooled)" in
                       # section 0's own table -- an inconsistency in the
                       # record itself, not resolved here; used as-is, flagged
                       # in every caption that cites it.


def apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Liberation Serif",
        "mathtext.it": "Liberation Serif:italic",
        "mathtext.bf": "Liberation Serif:bold",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 6.8,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def panel_label(ax, text, fontsize=9):
    ax.text(0.03, 0.95, text, transform=ax.transAxes, fontsize=fontsize,
             fontweight="bold", va="top")
