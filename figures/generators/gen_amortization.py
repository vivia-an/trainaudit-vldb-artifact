#!/usr/bin/env python3
"""Amortized collection overhead vs. dump period K (MegaScale-style trade-off
curve). overhead(K) = dump / (K * 732 ms); naive dump 192 s, optimized 25 s.
Work points from the text: 10% at K~380 (opt) vs K~2630 (naive); K=1000 gives
3.3% vs 26%."""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
    'font.size': 7,
    'axes.labelsize': 7,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'legend.fontsize': 6.5,
    'axes.linewidth': 0.6,
    'pdf.fonttype': 42,
})

GRAY = '#9a9a9a'
DARK = '#3a3a3a'
ACCENT = '#b2182b'
OK = '#2166ac'

STEP = 0.732
K = np.logspace(np.log10(150), np.log10(6000), 300)
ov_naive = 192.0 / (K * STEP) * 100
ov_opt = 25.0 / (K * STEP) * 100

fig, ax = plt.subplots(figsize=(3.3, 1.30))
ax.plot(K, ov_naive, color=GRAY, lw=1.1, label='Naive (192 s/dump)')
ax.plot(K, ov_opt, color=OK, lw=1.3, label='Optimized (25 s/dump)')

ax.axhline(10, color=DARK, lw=0.6, ls=':')
ax.text(158, 10.9, '10% budget', fontsize=6, color=DARK)

for k, c, yt in ((380, OK, 15.5), (2630, GRAY, 21.5)):
    ax.plot([k, k], [0, 10], color=c, lw=0.7, ls='--')
    ax.annotate(f'$K{{\\approx}}{k}$', xy=(k, 10), xytext=(k, yt),
                fontsize=6.2, color=c, ha='center',
                arrowprops=dict(arrowstyle='-', lw=0.5, color=c))

ax.plot([1000], [3.3], 'o', ms=3, color=OK)
ax.plot([1000], [26.2], 'o', ms=3, color=GRAY)
ax.annotate('26% vs. 3.3%\nat $K{=}1000$', xy=(1000, 26.2),
            xytext=(1350, 17), fontsize=6.2, color=DARK,
            arrowprops=dict(arrowstyle='->', lw=0.5, color=DARK))
ax.text(430, 4.4, '$\\sim$7× more frequent\nchecks at any budget',
        fontsize=6.2, color=OK, style='italic')

ax.set_xscale('log')
ax.set_xlim(150, 6000)
ax.set_ylim(0, 42)
ax.set_yticks([0, 10, 20, 30, 40])
ax.set_yticklabels(['0%', '10%', '20%', '30%', '40%'])
ax.set_xticks([200, 500, 1000, 2000, 5000])
ax.set_xticklabels(['200', '500', '1k', '2k', '5k'])
ax.set_xlabel('Dump period $K$ (steps between dumps)', labelpad=1.5)
ax.set_ylabel('Step overhead', labelpad=2)
ax.legend(frameon=False, loc='upper right', handlelength=1.4,
          borderaxespad=0.1)
ax.spines[['top', 'right']].set_visible(False)
ax.tick_params(length=2, pad=1.5)

fig.savefig('fig_amortization.pdf', bbox_inches='tight', pad_inches=0.01)
print('done')
