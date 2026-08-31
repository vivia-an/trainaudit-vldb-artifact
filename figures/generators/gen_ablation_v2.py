#!/usr/bin/env python3
"""Regenerate guard-ablation figures at the manuscript's half-column size.

Design: each figure targets 0.49\\columnwidth (~118pt = 1.64in) so fonts
render at their designed size (7pt) in the final PDF, matching
Auto-Validate-style compact vector panels. Data identical to v1 sources.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
    'font.size': 7,
    'axes.labelsize': 7,
    'axes.titlesize': 7,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'legend.fontsize': 6,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'pdf.fonttype': 42,
})

GRAY = '#9a9a9a'
DARK = '#3a3a3a'
ACCENT = '#b2182b'   # red accent for the failure/FP side
OK = '#2166ac'       # blue accent for the full/verified side

W = 1.64  # inches ~ 0.49 columnwidth


def predicate_figure(path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(W, 1.72),
                                   gridspec_kw={'hspace': 0.95})
    labels = ['Full', r'$-\pi_{\mathrm{topo}}$', r'$-\pi_{\mathrm{precond}}$']

    # (top) detection
    det = [6, 6, 4]
    colors = [OK, GRAY, GRAY]
    b = ax1.bar(range(3), det, width=0.58, color=colors, edgecolor='none')
    for i, v in enumerate(det):
        ax1.text(i, v + 0.15, f'{v}/6', ha='center', va='bottom', fontsize=6.5,
                 fontweight='bold' if i == 0 else 'normal')
    ax1.set_ylim(0, 7.6)
    ax1.set_yticks([0, 3, 6])
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Cases detected', labelpad=1)
    ax1.set_title('(a) Detection ($n{=}6$)', pad=2)

    # (bottom) FP per 1M
    fp = [25.8, 83.3, 31.7]
    colors = [OK, ACCENT, GRAY]
    ax2.bar(range(3), fp, width=0.58, color=colors, edgecolor='none')
    ann = ['25.8', '83.3', '31.7']
    for i, (v, a) in enumerate(zip(fp, ann)):
        ax2.text(i, v + 3, a, ha='center', va='bottom', fontsize=6.5,
                 fontweight='bold' if i == 1 else 'normal',
                 color=ACCENT if i == 1 else DARK)
    ax2.set_ylim(0, 130)
    ax2.set_yticks([0, 40, 80])
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('FPs / 1M evals', labelpad=1)
    ax2.set_title('(b) FP rate (504K evals)', pad=2)
    ax2.annotate(r'3.2$\times$', xy=(1.28, 83.3), xytext=(1.55, 96),
                 fontsize=6, color=ACCENT, ha='center',
                 arrowprops=dict(arrowstyle='->', lw=0.5, color=ACCENT))

    for ax in (ax1, ax2):
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(length=2, pad=1.5)
    fig.savefig(path, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)


def funnel_figure(path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(W, 1.90),
                                   gridspec_kw={'hspace': 1.05,
                                                'height_ratios': [1.25, 1.05]})
    # (top) funnel: horizontal bars, log-ish visual via raw counts
    stages = ['L1 hyps', 'L2 enum.', 'L3 healthy', 'L4 adv.', 'Deploy']
    counts = [420, 5334, 3436, 357, 45]
    shades = ['#c6c6c6', '#b0b0b0', '#8f8f8f', OK, OK]
    y = range(len(stages))[::-1]
    ax1.barh(list(y), counts, height=0.6, color=shades, edgecolor='none')
    lab = ['420', '5,334', '3,436', '357', '45']
    for yi, c, t in zip(y, counts, lab):
        ax1.text(c + 120, yi, t, va='center', ha='left', fontsize=6,
                 color=DARK)
    ax1.set_yticks(list(y))
    ax1.set_yticklabels(stages, fontsize=6)
    ax1.set_xlim(0, 6600)
    ax1.set_xticks([0, 2000, 4000, 6000])
    ax1.set_xticklabels(['0', '2k', '4k', '6k'])
    ax1.set_title('(a) Verification funnel', pad=2)

    # (bottom) skip-L3 stress — label RIGHT of bar (never under title)
    conf = ['Full (L1$\\to$L4)', 'Skip-L3']
    rate = [0.0, 28.5]
    ax2.bar(range(2), rate, width=0.5, color=[OK, ACCENT], edgecolor='none')
    ax2.text(0, 3.5, '0.0%\n(0/764)', ha='center', va='bottom', fontsize=5.5)
    ax2.text(1.38, 28.5, '28.5%\n(114/400)', ha='left', va='center',
             fontsize=5.5, color=ACCENT, fontweight='bold',
             linespacing=1.05, clip_on=False)
    # 95% CI whisker for skip-L3
    ax2.plot([1, 1], [24.3, 33.1], color=DARK, lw=0.7, zorder=3)
    ax2.plot([0.93, 1.07], [24.3, 24.3], color=DARK, lw=0.7, zorder=3)
    ax2.plot([0.93, 1.07], [33.1, 33.1], color=DARK, lw=0.7, zorder=3)
    ax2.set_ylim(0, 40)
    ax2.set_xlim(-0.55, 2.05)
    ax2.set_yticks([0, 20, 40])
    ax2.set_yticklabels(['0%', '20%', '40%'])
    ax2.set_xticks(range(2))
    ax2.set_xticklabels(conf, fontsize=6.5)
    ax2.set_ylabel('Clean-trace FP', labelpad=1)
    ax2.set_title('(b) Skip-L3 stress test', pad=8)

    for ax in (ax1, ax2):
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(length=2, pad=1.5)
    fig.savefig(path, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


if __name__ == '__main__':
    predicate_figure('fig_predicate_ablation_v2.pdf')
    funnel_figure('fig_funnel_ablation_v2.pdf')
    print('done')
