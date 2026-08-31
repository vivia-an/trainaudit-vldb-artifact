#!/usr/bin/env python3
"""Headline detection comparison as a single-column bar panel (replaces the
3-way table). GPTuner-style grouped bars: counts + on-figure multiplier and
Wilson-CI whisker; FP row stated as an annotation since all methods are 0/17.
Data identical to tab:summary_3way."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
    'font.size': 7,
    'axes.labelsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 6.5,
    'axes.linewidth': 0.6,
    'pdf.fonttype': 42,
})

GRAY = '#9a9a9a'
DARK = '#3a3a3a'
ACCENT = '#b2182b'
OK = '#2166ac'

fig, ax = plt.subplots(figsize=(3.3, 1.30))

methods = ['TrainAudit\n(this work)', 'TrainCheck', 'Naïve\nMonitoring']
det = [17, 5, 0]
colors = [OK, GRAY, GRAY]
bars = ax.bar(range(3), det, width=0.52, color=colors, edgecolor='none')

# Wilson 95% CI on 17/19 = [68.6%, 97.1%] -> counts of 19
ax.plot([0, 0], [0.686 * 19, 0.971 * 19], color=DARK, lw=0.8, zorder=4)
ax.plot([-0.06, 0.06], [0.686 * 19] * 2, color=DARK, lw=0.8, zorder=4)
ax.plot([-0.06, 0.06], [0.971 * 19] * 2, color=DARK, lw=0.8, zorder=4)

labels = ['17/19 (89.5%)', '5/19 (26.3%)', '0/19']
for i, (v, t) in enumerate(zip(det, labels)):
    ax.text(i, v + 0.9 if i else 18.9, t, ha='center', va='bottom',
            fontsize=7, fontweight='bold' if i == 0 else 'normal',
            color=OK if i == 0 else DARK)

ax.annotate('3.4×', xy=(0.30, 15.2), xytext=(0.72, 11.5),
            fontsize=7.5, color=ACCENT, fontweight='bold', ha='center',
            arrowprops=dict(arrowstyle='->', lw=0.7, color=ACCENT))

ax.text(1.62, 6.6, 'Fixed-replay FP:\n0/17 for all methods',
        fontsize=6.5, color=DARK, ha='left', va='center',
        bbox=dict(boxstyle='round,pad=0.35', fc='#f4f4f4', ec='#cccccc',
                  lw=0.5))

ax.set_ylim(0, 21.5)
ax.set_yticks([0, 5, 10, 15, 19])
ax.set_ylabel('Real-SE buggy cases detected', labelpad=2)
ax.set_xticks(range(3))
ax.set_xticklabels(methods, fontsize=6.8, linespacing=1.0)
ax.spines[['top', 'right']].set_visible(False)
ax.tick_params(length=2, pad=1.5)

fig.savefig('fig_detection_bars.pdf', bbox_inches='tight', pad_inches=0.01)
print('done')
