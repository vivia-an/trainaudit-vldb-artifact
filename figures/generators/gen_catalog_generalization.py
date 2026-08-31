#!/usr/bin/env python3
"""Catalog-generalization figure in the manuscript's single-column style.

Shows that a frozen Pattern Catalog generalizes to held-out bugs better than
free-form (no-catalog) baselines. Coverage vs. taxonomy-size line chart.

Data provenance: benchmark/eval/catalog_generalization/generalization_summary.csv
  - A_catalog:            83.1% held-out coverage at size 35
  - catalog_coverage_vs_size: (5,38.2)(10,58.2)(15,67.5)(20,74.3)(25,78.3)(35,83.1)
  - B1_freeform_frozen:   47.8% at size 15
  - B2_freeform_remine:   55.4% (per-bug re-mine, upper bound)

Style matches figures/generators/gen_ablation_v2.py (serif, stix, 7pt, pdf.fonttype 42).
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
ACCENT = '#b2182b'   # red: free-form baselines
OK = '#2166ac'       # blue: catalog


def catalog_generalization_figure(path):
    fig, ax = plt.subplots(figsize=(3.3, 1.55))

    # SERIES 1 -- catalog curve
    cat_x = [5, 10, 15, 20, 25, 35]
    cat_y = [38.2, 58.2, 67.5, 74.3, 78.3, 83.1]
    ax.plot(cat_x, cat_y, color=OK, lw=1.1, marker='o', ms=3.2,
            mfc=OK, mec='white', mew=0.4, zorder=6,
            label='catalog (frozen)')
    ax.annotate('catalog (35 templates)', xy=(35, 83.1), xytext=(34.6, 88),
                fontsize=5.8, color=OK, ha='right', va='center', linespacing=1.0)

    # SERIES 2 -- free-form frozen single point
    ax.plot([15], [47.8], marker='^', ms=5, mfc=ACCENT, mec='white',
            mew=0.4, ls='none', zorder=6, label='free-form frozen')
    ax.annotate('free-form frozen (15)', xy=(15, 47.8), xytext=(17, 40),
                fontsize=5.8, color=ACCENT, ha='left', va='center',
                arrowprops=dict(arrowstyle='-', lw=0.4, color=ACCENT))

    # SERIES 3 -- free-form re-mine upper bound
    ax.axhline(55.4, color=GRAY, lw=0.9, ls='--', dashes=(5, 3), zorder=3,
               label='free-form re-mine')
    ax.text(34.6, 55.4 + 1.2, 'free-form re-mine (upper bound)',
            fontsize=5.6, color=GRAY, ha='right', va='bottom')

    # annotation: +19.7 pts at equal size 15
    ax.plot([15, 15], [47.8, 67.5], color=DARK, lw=0.6, zorder=5)
    ax.plot([14.6, 15.4], [67.5, 67.5], color=DARK, lw=0.6, zorder=5)
    ax.plot([14.6, 15.4], [47.8, 47.8], color=DARK, lw=0.6, zorder=5)
    ax.annotate('+19.7 pts\nat equal size', xy=(15, 57.6), xytext=(10.1, 40),
                fontsize=5.8, color=DARK, ha='center', va='center',
                linespacing=1.0,
                arrowprops=dict(arrowstyle='->', lw=0.5, color=DARK))

    ax.set_xlim(3.5, 36.5)
    ax.set_ylim(0, 100)
    ax.set_xticks([5, 10, 15, 20, 25, 30, 35])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel('Taxonomy size (# invariant types)', labelpad=2)
    ax.set_ylabel('Held-out coverage (%)', labelpad=2)

    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=2, pad=1.5)
    ax.grid(axis='y', ls=':', lw=0.4, color='#dddddd', zorder=0)
    ax.set_axisbelow(True)

    ax.legend(loc='lower right', frameon=False, handlelength=1.5,
              borderpad=0.2, labelspacing=0.3, handletextpad=0.5)

    fig.savefig(path, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


if __name__ == '__main__':
    catalog_generalization_figure('fig_catalog_generalization.pdf')
    print('done')
