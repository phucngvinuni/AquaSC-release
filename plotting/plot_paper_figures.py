#!/usr/bin/env python3
"""Draw the six quantitative figures in the revised AquaSC manuscript.

Inputs are frozen aggregate plot values, not model inference. Run from any
working directory; no dataset, checkpoint, GPU, Torch, or Sionna is needed.
See docs/PAPER_FIGURES.md for the figure map and experiment protocols.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "figs"
BASELINE_INPUT = ROOT / "artifacts/figure_inputs/matched_digital_baselines_20261015.json"

def configure_ieee_style():
    """Configure matplotlib rcParams to match strict IEEE publication standards."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times'],
        'mathtext.fontset': 'stix',
        'font.size': 9.5,
        'axes.titlesize': 10,
        'axes.titleweight': 'normal',
        'axes.labelsize': 10,
        'axes.labelweight': 'normal',
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'legend.fontsize': 8,
        'figure.titlesize': 10.5,
        'lines.linewidth': 1.2,
        'lines.markersize': 5.5,
        'grid.color': '#E0E0E0',
        'grid.linestyle': ':',
        'grid.linewidth': 0.8,
        'axes.grid': True,
        'axes.axisbelow': True,
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })

def _plain_axis_label(text):
    """Use the same text face for axis labels, including the CI symbol."""
    return (text.replace(r'$\pm 95\%$', '±95%')
                .replace(r'$\\pm 95\\%$', '±95%')
                .replace(r'$\Delta$', 'Δ')
                .replace(r'$\\Delta$', 'Δ'))

def _wrap_panel_caption(text, font, renderer, max_width):
    words = text.split()
    lines, current = [], ''
    for word in words:
        candidate = (current + ' ' + word).strip()
        width = renderer.get_text_width_height_descent(candidate, font, ismath=False)[0]
        if current and width > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return '\n'.join(lines)

def _thin_artist_line(line):
    line.set_linewidth(max(0.75, min(1.25, line.get_linewidth() * 0.65)))
    if hasattr(line, 'get_markeredgewidth'):
        line.set_markeredgewidth(min(1.0, line.get_markeredgewidth() * 0.7))

def _finalize_paper_style(fig):
    """Apply typography and panel-caption placement without changing plot data."""
    if getattr(fig, '_paper_style_finalized', False):
        return
    x_sizes = [ax.xaxis.label.get_fontsize() for ax in fig.axes
               if ax.get_xlabel() and not re.match(r'^\([a-z]\)$', ax.get_xlabel())]
    y_sizes = [ax.yaxis.label.get_fontsize() for ax in fig.axes if ax.get_ylabel()]
    label_size = min(x_sizes or y_sizes or [plt.rcParams['axes.labelsize']])
    font_path = font_manager.findfont(font_manager.FontProperties(family=['Times New Roman', 'DejaVu Serif', 'Liberation Serif']))
    label_font = font_manager.FontProperties(fname=font_path, size=label_size,
                                             style='normal', weight='normal')
    panel_titles = []
    for ax in fig.axes:
        ax.set_xlabel(_plain_axis_label(ax.get_xlabel()))
        ax.set_ylabel(_plain_axis_label(ax.get_ylabel()))
        for label in [ax.xaxis.label, ax.yaxis.label]:
            label.set_fontproperties(label_font)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontproperties(font_manager.FontProperties(
                fname=font_path, size=tick.get_fontsize(), style='normal', weight='normal'))
        title = ax.get_title()
        if re.match(r'^\([a-z]\)', title):
            panel_titles.append((ax, title, ax.title.get_fontsize()))
            ax.set_title('')
        elif title:
            ax.title.set_fontproperties(font_manager.FontProperties(
                fname=font_path, size=ax.title.get_fontsize(), style='normal', weight='normal'))
        for line in ax.lines:
            _thin_artist_line(line)
        for collection in ax.collections:
            if hasattr(collection, 'get_linewidths'):
                widths = collection.get_linewidths()
                collection.set_linewidths([min(1.2, max(0.65, width * 0.65)) for width in widths])
        legend = ax.get_legend()
        if legend:
            for line in legend.get_lines():
                _thin_artist_line(line)
    for legend in fig.legends:
        for line in legend.get_lines():
            _thin_artist_line(line)

    # Qualitative panels already carry their letter labels below the images.
    if panel_titles:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax, title, size in panel_titles:
            font = font_manager.FontProperties(fname=font_path, size=size,
                                               style='normal', weight='normal')
            title = _wrap_panel_caption(title, font, renderer, ax.bbox.width * 0.98)
            decorations = [ax.xaxis.label] + ax.get_xticklabels()
            bottom = ax.bbox.y0
            visible_boxes = [item.get_window_extent(renderer) for item in decorations
                             if item.get_visible() and item.get_text()]
            clearance = max([(bottom-box.y0)*72/fig.dpi for box in visible_boxes] or [5]) + 8
            caption = ax.annotate(title, xy=(0.5, 0), xycoords='axes fraction',
                                  xytext=(0, -clearance), textcoords='offset points',
                                  ha='center', va='top', fontproperties=font,
                                  annotation_clip=False)
            caption.set_in_layout(True)
            ax._paper_panel_caption = caption
        legend_top = min([legend.get_window_extent(renderer).y0/fig.bbox.height-0.025
                          for legend in fig.legends] or [0.98])
        fig.tight_layout(pad=0.6, h_pad=1.4, w_pad=0.7,
                         rect=(0.005, 0.005, 0.995, legend_top))
    fig._paper_style_finalized = True

def save_paper_figure(filename, **kwargs):
    """Keep the original canvas size so LaTeX page layout remains stable."""
    fig = plt.gcf()
    _finalize_paper_style(fig)
    kwargs.setdefault('bbox_inches', None)
    target = OUTPUT_DIR / Path(filename).name
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, **kwargs)

COLOR_AQUASC_DARK  = '#08519C'

COLOR_AQUASC_MID   = '#3182BD'

COLOR_AQUASC_LIGHT = '#6BAED6'

COLOR_ROIJSCC      = '#D94801'

COLOR_SWINJSCC     = '#238B45'

COLOR_JSCC_BBW     = '#756BB1'

COLOR_JSCC_STD     = '#C2185B'

COLOR_WEBP         = '#4D4D4D'

COLOR_JPEG         = '#8C510A'

COLOR_CEILING      = '#252525'

COLOR_FLOOR        = '#CB181D'

SNR = np.array([-5, 0, 5, 10, 15, 20])

MATCHED_DIGITAL = json.loads(BASELINE_INPUT.read_text(encoding='utf-8'))

MATCHED_BUDGET_ORDER = MATCHED_DIGITAL['budget_order']

MATCHED_RATES = [MATCHED_DIGITAL['budgets'][key]['rate'] for key in MATCHED_BUDGET_ORDER]

def _plot_matched_digital(ax, codec, budget, metric, label=None):
    values = MATCHED_DIGITAL['codecs'][codec][budget][metric]
    style, color = ('v--', COLOR_WEBP) if codec == 'webp' else ('P:', COLOR_JPEG)
    line, = ax.plot(SNR, values, style, color=color, lw=1.4, ms=4.5, label=label)
    return line

def plot_figure_1_main_comparison():
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.6), dpi=300, sharex=True)

    # -------------------------------------------------------------
    # Row 0: Task Performance (mAP@0.50 pooled) across 3 rates
    # -------------------------------------------------------------
    # (a) High Rate (~0.122 uses/px)
    ax_a = axes[0, 0]
    h_aquasc, = ax_a.plot(SNR, [0.3287, 0.5291, 0.6405, 0.6838, 0.7042, 0.7070],
                          'o-', color=COLOR_AQUASC_DARK, lw=2.0, ms=5.5, label='AquaSC (Joint-Frame)')
    h_swin, = ax_a.plot(SNR, [0.2969, 0.4698, 0.5953, 0.6579, 0.6823, 0.6849],
                        '^-.', color=COLOR_SWINJSCC, lw=1.6, ms=5.0, label='SwinJSCC')
    h_roi, = ax_a.plot(SNR, [0.4221, 0.5199, 0.5764, 0.5990, 0.6124, 0.6169],
                       's--', color=COLOR_ROIJSCC, lw=1.6, ms=5.0, label='ROI-JSCC')
    h_bbw, = ax_a.plot(SNR, [0.2027, 0.3127, 0.3808, 0.4118, 0.4296, 0.4361],
                       'D:', color=COLOR_JSCC_BBW, lw=1.4, ms=4.5, label='JSCC-CNN (BBW)')
    h_std, = ax_a.plot(SNR, [0.01574, 0.02801, 0.04034, 0.04464, 0.04754, 0.05037],
                       'x:', color=COLOR_JSCC_STD, lw=1.8, ms=5.8, mew=1.6, label='JSCC-CNN (Std-L1)')
    h_webp = _plot_matched_digital(ax_a, 'webp', 'high', 'pooled_map50_mean', 'WebP+LDPC')
    h_jpeg = _plot_matched_digital(ax_a, 'jpeg', 'high', 'pooled_map50_mean', 'JPEG+LDPC')
    h_ceil = ax_a.axhline(0.7292, color=COLOR_CEILING, ls='--', lw=1.0, alpha=0.75, label='Clean Ceiling (0.729)')

    ax_a.set_ylabel('mAP@0.50 (pooled)', fontsize=8.8)
    ax_a.set_ylim(-0.04, 0.78)
    ax_a.set_xlim(-6, 21)
    ax_a.xaxis.set_major_locator(MultipleLocator(5))
    ax_a.set_title('(a) High Rate (~0.122 u/px)', pad=5, fontsize=8.8)

    # (b) Mid Rate (~0.061 uses/px) - Retrained r061
    ax_b = axes[0, 1]
    ax_b.plot(SNR, [0.3211, 0.4856, 0.5710, 0.6032, 0.6171, 0.6194],
              'o-', color=COLOR_AQUASC_DARK, lw=2.0, ms=5.5)
    ax_b.plot(SNR, [0.1245, 0.2538, 0.4685, 0.5684, 0.6012, 0.6120],
              '^-.', color=COLOR_SWINJSCC, lw=1.6, ms=5.0)
    ax_b.plot(SNR, [0.3864, 0.4611, 0.5085, 0.5332, 0.5477, 0.5519],
              's--', color=COLOR_ROIJSCC, lw=1.6, ms=5.0)
    ax_b.plot(SNR, [0.1778, 0.2966, 0.3778, 0.4106, 0.4337, 0.4397],
              'D:', color=COLOR_JSCC_BBW, lw=1.4, ms=4.5)
    ax_b.plot(SNR, [0.00270, 0.00535, 0.00989, 0.00958, 0.01069, 0.01103],
              'x:', color=COLOR_JSCC_STD, lw=1.8, ms=5.8, mew=1.6)
    _plot_matched_digital(ax_b, 'webp', 'mid', 'pooled_map50_mean')
    _plot_matched_digital(ax_b, 'jpeg', 'mid', 'pooled_map50_mean')
    ax_b.axhline(0.7292, color=COLOR_CEILING, ls='--', lw=1.0, alpha=0.75)
    ax_b.set_ylim(-0.04, 0.78)
    ax_b.set_title('(b) Mid Rate (~0.061 u/px)', pad=5, fontsize=8.8)

    # (c) Low Rate (~0.031 uses/px) - Retrained r030 (buffed)
    ax_c = axes[0, 2]
    ax_c.plot(SNR, [0.2785, 0.3620, 0.4218, 0.4892, 0.5385, 0.5580],
              'o-', color=COLOR_AQUASC_DARK, lw=2.0, ms=5.5)
    ax_c.plot(SNR, [0.0393, 0.1265, 0.3279, 0.4710, 0.5506, 0.5819],
              '^-.', color=COLOR_SWINJSCC, lw=1.6, ms=5.0)
    ax_c.plot(SNR, [0.3244, 0.3995, 0.4448, 0.4686, 0.4832, 0.4885],
              's--', color=COLOR_ROIJSCC, lw=1.6, ms=5.0)
    _plot_matched_digital(ax_c, 'webp', 'low', 'pooled_map50_mean')
    _plot_matched_digital(ax_c, 'jpeg', 'low', 'pooled_map50_mean')
    ax_c.axhline(0.7292, color=COLOR_CEILING, ls='--', lw=1.0, alpha=0.75)
    ax_c.set_ylim(-0.04, 0.78)
    ax_c.set_title('(c) Low Rate (~0.031 u/px)', pad=5, fontsize=8.8)

    # -------------------------------------------------------------
    # Row 1: Reconstruction Quality (PSNR dB) across 3 rates
    # -------------------------------------------------------------
    # (d) High Rate PSNR
    ax_d = axes[1, 0]
    ax_d.plot(SNR, [32.23, 36.87, 39.40, 40.44, 40.88, 40.95],
              'o-', color=COLOR_AQUASC_DARK, lw=2.0, ms=5.5)
    ax_d.plot(SNR, [25.73, 32.33, 39.39, 42.77, 44.37, 45.09],
              '^-.', color=COLOR_SWINJSCC, lw=1.6, ms=5.0)
    ax_d.plot(SNR, [36.51, 39.51, 41.90, 43.51, 44.56, 45.00],
              's--', color=COLOR_ROIJSCC, lw=1.6, ms=5.0)
    ax_d.plot(SNR, [32.99, 36.17, 38.44, 39.79, 40.48, 40.79],
              'D:', color=COLOR_JSCC_BBW, lw=1.4, ms=4.5)
    ax_d.plot(SNR, [33.62, 36.29, 38.02, 38.95, 39.39, 39.57],
              'x:', color=COLOR_JSCC_STD, lw=1.8, ms=5.8, mew=1.6)
    _plot_matched_digital(ax_d, 'webp', 'high', 'psnr_db')
    _plot_matched_digital(ax_d, 'jpeg', 'high', 'psnr_db')

    ax_d.set_xlabel('Channel SNR (dB)', fontsize=8.8)
    ax_d.set_ylabel('PSNR (dB)', fontsize=8.8)
    ax_d.set_ylim(0, 48)
    ax_d.set_xlim(-6, 21)
    ax_d.xaxis.set_major_locator(MultipleLocator(5))
    ax_d.yaxis.set_major_locator(MultipleLocator(10))
    ax_d.set_title('(d) High Rate (~0.122 u/px)', pad=5, fontsize=8.8)

    # (e) Mid Rate PSNR
    ax_e = axes[1, 1]
    ax_e.plot(SNR, [32.07, 36.25, 38.52, 39.46, 39.84, 39.91],
              'o-', color=COLOR_AQUASC_DARK, lw=2.0, ms=5.5)
    ax_e.plot(SNR, [18.70, 23.03, 32.02, 38.87, 42.48, 44.09],
              '^-.', color=COLOR_SWINJSCC, lw=1.6, ms=5.0)
    ax_e.plot(SNR, [35.37, 37.62, 39.38, 40.50, 41.22, 41.51],
              's--', color=COLOR_ROIJSCC, lw=1.6, ms=5.0)
    ax_e.plot(SNR, [31.21, 34.58, 37.08, 38.66, 39.50, 39.90],
              'D:', color=COLOR_JSCC_BBW, lw=1.4, ms=4.5)
    ax_e.plot(SNR, [31.36, 33.99, 35.73, 36.73, 37.24, 37.46],
              'x:', color=COLOR_JSCC_STD, lw=1.8, ms=5.8, mew=1.6)
    _plot_matched_digital(ax_e, 'webp', 'mid', 'psnr_db')
    _plot_matched_digital(ax_e, 'jpeg', 'mid', 'psnr_db')

    ax_e.set_xlabel('Channel SNR (dB)', fontsize=8.8)
    ax_e.set_ylim(0, 48)
    ax_e.set_xlim(-6, 21)
    ax_e.xaxis.set_major_locator(MultipleLocator(5))
    ax_e.yaxis.set_major_locator(MultipleLocator(10))
    ax_e.set_title('(e) Mid Rate (~0.061 u/px)', pad=5, fontsize=8.8)

    # (f) Low Rate PSNR
    ax_f = axes[1, 2]
    ax_f.plot(SNR, [29.47, 32.79, 34.63, 36.18, 37.05, 37.40],
              'o-', color=COLOR_AQUASC_DARK, lw=2.0, ms=5.5)
    ax_f.plot(SNR, [15.92, 19.97, 28.86, 35.96, 39.66, 41.36],
              '^-.', color=COLOR_SWINJSCC, lw=1.6, ms=5.0)
    ax_f.plot(SNR, [34.07, 36.59, 38.58, 39.87, 40.71, 41.05],
              's--', color=COLOR_ROIJSCC, lw=1.6, ms=5.0)
    _plot_matched_digital(ax_f, 'webp', 'low', 'psnr_db')
    _plot_matched_digital(ax_f, 'jpeg', 'low', 'psnr_db')

    ax_f.set_xlabel('Channel SNR (dB)', fontsize=8.8)
    ax_f.set_ylim(0, 48)
    ax_f.set_xlim(-6, 21)
    ax_f.xaxis.set_major_locator(MultipleLocator(5))
    ax_f.yaxis.set_major_locator(MultipleLocator(10))
    ax_f.set_title('(f) Low Rate (~0.031 u/px)', pad=5, fontsize=8.8)

    # Single Unified Shared Legend at top with comfortable clearance from subplots
    legend_handles = [h_aquasc, h_swin, h_roi, h_bbw, h_std, h_webp, h_jpeg, h_ceil]
    legend_labels = ['AquaSC (Joint-Frame)', 'SwinJSCC', 'ROI-JSCC', 'JSCC-CNN (BBW)',
                     'JSCC-CNN (Std-L1)', 'WebP+LDPC', 'JPEG+LDPC', 'Clean Ceiling (0.729)']
    fig.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.992),
               ncol=4, framealpha=0.95, edgecolor='#D0D0D0', fontsize=7.2, columnspacing=1.0, handletextpad=0.3)

    plt.subplots_adjust(top=0.86, bottom=0.09, left=0.08, right=0.98, wspace=0.18, hspace=0.28)
    save_paper_figure('figs/figure_main_comparison.png', dpi=300)
    save_paper_figure('figs/figure_main_comparison.pdf')
    plt.close()
    print("[+] Saved figs/figure_main_comparison (png & pdf)")

def plot_figure_2_rate_performance():
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.6), dpi=300)

    # 2(a): mAP@0.50 Frontier at SNR = 5 dB (Retrained Codecs for AquaSC)
    ax1 = axes[0]
    ax1.plot([0.030612, 0.061224, 0.122449], [0.4218, 0.5710, 0.6405],
             'o-', color=COLOR_AQUASC_DARK, lw=2.2, ms=6, label='AquaSC (Retrained)')
    ax1.plot([0.030612, 0.061224, 0.122449], [0.4448, 0.5085, 0.5764],
             's--', color=COLOR_ROIJSCC, lw=1.8, ms=5.5, label='ROI-JSCC')
    ax1.plot([0.03125, 0.0625, 0.125], [0.3279, 0.4685, 0.5953],
             '^-.', color=COLOR_SWINJSCC, lw=1.8, ms=5.5, label='SwinJSCC')
    ax1.plot([0.0625, 0.125], [0.3778, 0.3808],
             'D:', color=COLOR_JSCC_BBW, lw=1.6, ms=5, label='JSCC-CNN (BBW)')
    for codec, style, color, label in [('webp', 'v--', COLOR_WEBP, 'WebP+LDPC'),
                                       ('jpeg', 'P:', COLOR_JPEG, 'JPEG+LDPC')]:
        values = [MATCHED_DIGITAL['codecs'][codec][key]['pooled_map50_mean'][2]
                  for key in MATCHED_BUDGET_ORDER]
        ax1.plot(MATCHED_RATES, values, style, color=color, lw=1.4, ms=5, label=label)

    ax1.axhline(0.7292, color=COLOR_CEILING, ls='--', lw=1.1, alpha=0.75, label='Ceiling (0.729)')

    ax1.set_xlabel('Channel Uses per Pixel (uses/px)')
    ax1.set_ylabel('Pooled mAP@0.50')
    ax1.set_title('(a) mAP@0.50 vs. Rate (SNR = 5 dB)', pad=8)
    ax1.set_xlim(0.015, 0.135)
    ax1.set_ylim(-0.03, 0.78)

    # 2(b): Reconstruction PSNR at SNR = 5 dB
    ax2 = axes[1]
    ax2.plot([0.030612, 0.061224, 0.122449], [34.63, 38.52, 39.40],
             'o-', color=COLOR_AQUASC_DARK, lw=2.2, ms=6, label='AquaSC (Retrained)')
    ax2.plot([0.030612, 0.061224, 0.122449], [38.58, 39.38, 41.90],
             's--', color=COLOR_ROIJSCC, lw=1.8, ms=5.5, label='ROI-JSCC')
    ax2.plot([0.03125, 0.0625, 0.125], [28.86, 32.02, 39.39],
             '^-.', color=COLOR_SWINJSCC, lw=1.8, ms=5.5, label='SwinJSCC')
    ax2.plot([0.0625, 0.125], [37.08, 38.44],
             'D:', color=COLOR_JSCC_BBW, lw=1.6, ms=5, label='JSCC-CNN (BBW)')
    for codec, style, color, label in [('webp', 'v--', COLOR_WEBP, 'WebP+LDPC'),
                                       ('jpeg', 'P:', COLOR_JPEG, 'JPEG+LDPC')]:
        values = [MATCHED_DIGITAL['codecs'][codec][key]['psnr_db'][2]
                  for key in MATCHED_BUDGET_ORDER]
        ax2.plot(MATCHED_RATES, values, style, color=color, lw=1.4, ms=5, label=label)

    ax2.set_xlabel('Channel Uses per Pixel (uses/px)')
    ax2.set_ylabel('Reconstruction PSNR (dB)')
    ax2.set_title('(b) PSNR vs. Rate (SNR = 5 dB)', pad=8)
    ax2.set_xlim(0.015, 0.135)
    ax2.set_ylim(0, 46)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.995),
               ncol=3, fontsize=7.0, framealpha=0.95, edgecolor='#D0D0D0', columnspacing=1.0, handletextpad=0.4)

    plt.tight_layout(rect=(0, 0, 1, 0.80), h_pad=1.4)
    save_paper_figure('figs/figure_rate_performance_snr_5db.png', dpi=300)
    save_paper_figure('figs/figure_rate_performance_snr_5db.pdf')
    plt.close()
    print("[+] Saved figs/figure_rate_performance_snr_5db (png & pdf)")

def plot_figure_4_ablation_study():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9), dpi=300)

    categories = ['Base\n(Neither)', 'FIM only\n(No BBW)', 'BBW only\n(No FIM)', 'AquaSC Full\n(FIM + BBW)']
    x = np.arange(len(categories))
    width = 0.35

    # 4(a): Pooled mAP@0.50 (Uncoded Channel)
    ax1 = axes[0]
    snr10_pooled = [0.2383, 0.3129, 0.3073, 0.3874]
    snr20_pooled = [0.2711, 0.3649, 0.5458, 0.6510]

    rects1 = ax1.bar(x - width/2, snr10_pooled, width, label='10 dB SNR',
                     color='#6BAED6', edgecolor='#2B5C8F', lw=0.9)
    rects2 = ax1.bar(x + width/2, snr20_pooled, width, label='20 dB SNR',
                     color='#08519C', edgecolor='#042B54', lw=0.9)

    for rect in list(rects1) + list(rects2):
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2., h + 0.015, f'{h:.3f}',
                 ha='center', va='bottom', fontsize=7.5)

    ax1.axhline(0.7292, color=COLOR_CEILING, ls='--', lw=1.0, alpha=0.75, label='Pooled Ceiling (0.729)')
    ax1.set_ylabel('Pooled mAP@0.50')
    ax1.set_title('(a) Pooled mAP@0.50', pad=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=8)
    ax1.set_ylim(0, 0.82)
    ax1.legend(loc='upper left', framealpha=0.9, edgecolor='#D0D0D0')

    # 4(b): Per-image AP@0.50 with 95% Confidence Interval
    ax2 = axes[1]
    snr10_mean = [0.298, 0.383, 0.389, 0.482]
    snr10_ci   = [0.016, 0.017, 0.023, 0.025]
    snr20_mean = [0.325, 0.426, 0.637, 0.720]
    snr20_ci   = [0.016, 0.018, 0.023, 0.021]

    rects3 = ax2.bar(x - width/2, snr10_mean, width, yerr=snr10_ci, capsize=3.5,
                     label='10 dB SNR', color='#FDAE6B', edgecolor='#A64B00', lw=0.9)
    rects4 = ax2.bar(x + width/2, snr20_mean, width, yerr=snr20_ci, capsize=3.5,
                     label='20 dB SNR', color='#D94801', edgecolor='#7A2900', lw=0.9)

    for rect in list(rects3) + list(rects4):
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., h + 0.035, f'{h:.3f}',
                 ha='center', va='bottom', fontsize=7.5)

    ax2.axhline(0.7743, color=COLOR_CEILING, ls='--', lw=1.0, alpha=0.75, label='Per-image Ceiling (0.774)')
    ax2.set_ylabel('Per-image Mean AP@0.50 ($\\pm 95\\%$ CI)')
    ax2.set_title('(b) Per-image AP@0.50 with 95% CI', pad=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=8)
    ax2.set_ylim(0, 0.88)
    ax2.legend(loc='upper left', framealpha=0.9, edgecolor='#D0D0D0')

    plt.tight_layout()
    save_paper_figure('figs/figure_ablation_study.png', dpi=300)
    save_paper_figure('figs/figure_ablation_study.pdf')
    plt.close()
    print("[+] Saved figs/figure_ablation_study (png & pdf)")

def plot_figure_6_domain_transfer():
    # AquaSC: validation-selected epoch 9, real LDPC-coded joint-frame pipeline.
    # JSCC baselines: Site B continuous-valued Rayleigh evaluations at
    # 0.122449--0.125 uses/px. AquaSC remains on its coded joint-frame path.
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.5), dpi=300)

    ax1 = axes[0]
    labels = ['Clean', 'AquaSC', 'SwinJSCC', 'ROI-JSCC', 'JSCC-CNN']
    pooled_map_10db = [0.6044, 0.2032, 0.2441, 0.1007, 0.1105]
    colors = ['#4D4D4D', COLOR_AQUASC_DARK,
              COLOR_SWINJSCC, COLOR_ROIJSCC, COLOR_JSCC_BBW]
    bars = ax1.bar(np.arange(len(labels)), pooled_map_10db, width=0.65,
                   color=colors, edgecolor='#333333', lw=0.6)
    for bar, value in zip(bars, pooled_map_10db):
        ax1.text(bar.get_x() + bar.get_width()/2, value + 0.012,
                 f'{value:.4f}', ha='center', va='bottom', fontsize=6.1)
    ax1.set_xticks(np.arange(len(labels)), labels, fontsize=6.1)
    ax1.set_ylim(0, 0.69)
    ax1.set_ylabel('Pooled mAP@0.50', fontsize=8.4)
    ax1.set_title('(a) Site B detection at 10 dB', pad=6, fontsize=8.8)

    ax2 = axes[1]
    snr_grid = np.array([0, 5, 10, 15, 20])
    aquasc_ret = np.array([22.36, 30.18, 33.61, 34.83, 34.95])
    swin_ret = np.array([16.62, 30.04, 40.38, 45.30, 48.39])
    baseline_snr = np.array([0, 10, 20])
    roi_ret = np.array([6.90, 16.65, 20.37])
    jscc_ret = np.array([7.14, 13.60, 18.28, 21.30, 22.17])
    ax2.plot(snr_grid, aquasc_ret, 'o-', color=COLOR_AQUASC_DARK,
             lw=2.0, ms=4.5, label='AquaSC')
    ax2.plot(snr_grid, swin_ret, '^-.', color=COLOR_SWINJSCC,
             lw=1.6, ms=4.2, label='SwinJSCC')
    ax2.plot(baseline_snr, roi_ret, 's--', color=COLOR_ROIJSCC,
             lw=1.5, ms=4.0, label='ROI-JSCC')
    ax2.plot(snr_grid, jscc_ret, 'D:', color=COLOR_JSCC_BBW,
             lw=1.5, ms=3.8, label='JSCC-CNN')
    ax2.set_xlabel('Channel SNR (dB)', fontsize=8.4)
    ax2.set_ylabel('Pooled retention on Site B (%)', fontsize=8.4)
    ax2.set_title('(b) Site B retention across SNR', pad=6, fontsize=8.8)
    ax2.set_xlim(-0.7, 20.7)
    ax2.set_xticks(snr_grid)
    ax2.set_ylim(0, 52)
    ax2.legend(loc='lower right', framealpha=0.95, edgecolor='#E0E0E0',
               fontsize=6.2, borderpad=0.3, handletextpad=0.3)

    plt.tight_layout(h_pad=1.7)
    save_paper_figure('figs/figure_domain_transfer.png', dpi=300)
    save_paper_figure('figs/figure_domain_transfer.pdf')
    plt.close()
    print("[+] Saved figs/figure_domain_transfer (png & pdf)")

def plot_figure_10_semantic_bandwidth():
    """
    Compares measured channel budgets and raw task efficiency:
    Sparse targets (5.44% fish pixels) allow unequal bit allocation between Semantic ROI
    and Background patches.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.1), dpi=300)

    # 10(a): Channel uses per pixel
    ax1 = axes[0]
    schemes = ['WebP+LDPC', 'AquaSC\nHigh Rate', 'AquaSC\nLow-Rate']
    channel_uses = [0.122449, 0.122449, 0.030612]
    mAP_10db     = [MATCHED_DIGITAL['codecs']['webp']['high']['pooled_map50_mean'][3], 0.6838, 0.4892]

    colors = [COLOR_WEBP, COLOR_AQUASC_DARK, COLOR_AQUASC_LIGHT]
    x = np.arange(len(schemes))

    bars = ax1.bar(x, channel_uses, width=0.55, color=colors, edgecolor='#333333', lw=0.9)
    for b in bars:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2., h + 0.012, f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    ax1.set_ylabel('Channel Uses per Pixel (uses/px)')
    ax1.set_title('(a) Transmission Cost per Image', pad=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(schemes, fontsize=8)
    ax1.set_ylim(0, 0.16)

    # 10(b): Semantic Efficiency (mAP per Channel Use)
    ax2 = axes[1]
    eff = np.array(mAP_10db) / np.array(channel_uses)

    bars2 = ax2.bar(x, eff, width=0.55, color=[COLOR_WEBP, COLOR_AQUASC_DARK, COLOR_AQUASC_LIGHT],
                    edgecolor='#333333', lw=0.9)
    for b in bars2:
        h = b.get_height()
        label = f'{h:.2f}' if h < 1 else f'{h:.1f}'
        ax2.text(b.get_x() + b.get_width()/2., h + 0.35, label, ha='center', va='bottom', fontsize=8)

    ax2.set_ylabel('Semantic Efficiency (mAP@0.50 / uses/px)')
    ax2.set_title('(b) Semantic Efficiency at SNR = 10 dB', pad=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(schemes, fontsize=8)
    ax2.set_ylim(0, 19)

    plt.tight_layout()
    save_paper_figure('figs/figure_semantic_bandwidth.png', dpi=300)
    save_paper_figure('figs/figure_semantic_bandwidth.pdf')
    plt.close()
    print("[+] Saved figs/figure_semantic_bandwidth (png & pdf)")

def plot_figure_aquasc_lite():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.25), dpi=300)

    task_maps = {
        'AquaSC-Large': [0.3287, 0.5291, 0.6405, 0.6838, 0.7042, 0.7070],
        'AquaSC-Small': [0.3308, 0.5247, 0.6316, 0.6716, 0.6901, 0.6926],
        'AquaSC-Tiny': [0.2847, 0.4748, 0.5882, 0.6309, 0.6497, 0.6530],
        'SwinJSCC': [0.2969, 0.4698, 0.5953, 0.6579, 0.6823, 0.6849],
        'ROI-JSCC': [0.4221, 0.5199, 0.5764, 0.5990, 0.6124, 0.6169],
    }

    # 11(a): Task Performance vs. SNR across Distilled Variants
    ax1 = axes[0]
    ax1.plot(SNR, task_maps['AquaSC-Large'],
             'o-', color=COLOR_AQUASC_DARK, lw=2.2, ms=6, label='AquaSC-Large (25.3M)')
    ax1.plot(SNR, task_maps['AquaSC-Small'],
             's--', color=COLOR_AQUASC_MID, lw=1.8, ms=5.5, label='AquaSC-Small (6.4M)')
    ax1.plot(SNR, task_maps['AquaSC-Tiny'],
             'd-.', color='#02818A', lw=1.8, ms=5.5, label='AquaSC-Tiny (0.93M)')
    ax1.plot(SNR, task_maps['SwinJSCC'],
             '^:', color=COLOR_SWINJSCC, lw=1.5, ms=5, label='SwinJSCC (19.0M)')
    ax1.plot(SNR, task_maps['ROI-JSCC'],
             'x--', color=COLOR_ROIJSCC, lw=1.5, ms=5, label='ROI-JSCC (1.46M)')

    ax1.axhline(0.7292, color=COLOR_CEILING, ls='--', lw=1.0, alpha=0.7, label='Clean Ceiling (0.729)')

    ax1.set_xlabel('Channel SNR (dB)')
    ax1.set_ylabel('Object Detection mAP@0.50')
    ax1.set_ylim(0.20, 0.78)
    ax1.set_xlim(-6, 21)
    ax1.xaxis.set_major_locator(MultipleLocator(5))
    ax1.set_title('(a) Task Performance vs. SNR', pad=8)
    ax1.legend(loc='lower right', framealpha=0.9, edgecolor='#E0E0E0', fontsize=7.2)

    # 11(b): Use the same measured series as panel (a), sampled at 5 dB.
    ax2 = axes[1]
    comparison_snr = 5
    snr_index = int(np.flatnonzero(SNR == comparison_snr)[0])
    comparison_maps = {name: values[snr_index] for name, values in task_maps.items()}

    enc_params = [0.927, 6.405, 25.319]
    encoder_maps = [comparison_maps[name] for name in ['AquaSC-Tiny', 'AquaSC-Small', 'AquaSC-Large']]
    ax2.plot(enc_params, encoder_maps, '-', color=COLOR_AQUASC_DARK, lw=1.8, alpha=0.6, zorder=2)

    points = [
        ('AquaSC-Large', 25.319, '25.3M', COLOR_AQUASC_DARK, 'o', 80, (0, 16), 'center', 'bottom'),
        ('AquaSC-Small', 6.405, '6.4M', COLOR_AQUASC_MID, 's', 75, (-12, 20), 'right', 'bottom'),
        ('AquaSC-Tiny', 0.927, '0.93M', '#02818A', 'd', 85, (9, 26), 'left', 'bottom'),
        ('SwinJSCC', 19.005, '19.0M', COLOR_SWINJSCC, '^', 65, (0, -14), 'center', 'top'),
        ('ROI-JSCC', 1.464, '1.46M', COLOR_ROIJSCC, 'x', 65, (10, -8), 'left', 'top'),
    ]
    for name, params, size_label, color, marker, size, offset, ha, va in points:
        value = comparison_maps[name]
        kwargs = {'linewidths': 2.0} if marker == 'x' else {}
        ax2.scatter([params], [value], color=color, s=size, marker=marker,
                    zorder=4, label=name, **kwargs)
        ax2.annotate(f'{name}\n({size_label})', xy=(params, value), xytext=offset,
                     textcoords='offset points', fontsize=7.5 if name.startswith('AquaSC') else 7.2,
                     color=color, fontweight='bold' if name.startswith('AquaSC') else 'normal',
                     ha=ha, va=va, zorder=5)

    ax2.set_xscale('log')
    ax2.set_xlim(0.5, 80)
    ax2.set_xticks([0.5, 1.0, 10.0, 30.0])
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.set_xlabel('Image Encoder Parameters (Millions, log scale)')
    ax2.set_ylabel(f'Object Detection mAP@0.50 (at {comparison_snr} dB)')
    ax2.set_ylim(0.54, 0.70)
    ax2.set_title(f'(b) Edge Complexity vs. Task Performance ({comparison_snr} dB)', pad=8)
    ax2.grid(True, which="both", ls=":", lw=0.6, alpha=0.7)

    plt.tight_layout()
    save_paper_figure('figs/figure_aquasc_lite.png', dpi=300)
    save_paper_figure('figs/figure_aquasc_lite.pdf')
    plt.close()
    print("[+] Saved figs/figure_aquasc_lite (png & pdf)")

PAPER_FIGURES = {
    "main": (3, "figure_main_comparison", plot_figure_1_main_comparison),
    "efficiency": (4, "figure_semantic_bandwidth", plot_figure_10_semantic_bandwidth),
    "ablation": (5, "figure_ablation_study", plot_figure_4_ablation_study),
    "rate": (6, "figure_rate_performance_snr_5db", plot_figure_2_rate_performance),
    "site-b": (7, "figure_domain_transfer", plot_figure_6_domain_transfer),
    "lite": (8, "figure_aquasc_lite", plot_figure_aquasc_lite),
}


def main():
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", nargs="+", choices=["all", *PAPER_FIGURES], default=["all"])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--list", action="store_true", help="List current manuscript figure names and exit.")
    args = parser.parse_args()
    if args.list:
        for key, (number, stem, _) in PAPER_FIGURES.items():
            print(f"{key}: Fig. {number}, {stem}.pdf/.png")
        return
    OUTPUT_DIR = args.output_dir.expanduser().resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = list(PAPER_FIGURES) if "all" in args.figures else list(dict.fromkeys(args.figures))
    configure_ieee_style()
    font_path = font_manager.findfont(font_manager.FontProperties(
        family=["Times New Roman", "DejaVu Serif", "Liberation Serif"]))
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Resolved label font: {font_path}")
    entries = []
    for key in selected:
        number, stem, draw = PAPER_FIGURES[key]
        draw()
        outputs = []
        for suffix in (".pdf", ".png"):
            path = OUTPUT_DIR / (stem + suffix)
            outputs.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        entries.append({"figure": number, "name": key, "outputs": outputs})
    manifest = {
        "source": "Frozen aggregate inputs for the revised AquaSC manuscript",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "matched_baseline_input_sha256": hashlib.sha256(BASELINE_INPUT.read_bytes()).hexdigest(),
        "resolved_font": Path(font_path).name,
        "figures": entries,
    }
    (OUTPUT_DIR / "paper_figures_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
