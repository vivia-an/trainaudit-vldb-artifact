#!/usr/bin/env python3
"""Amortized collection overhead vs. dump period K.

All annotations are derived from the measured 732 ms step, 192 s naive dump,
and 25 s optimized dump: overhead(K) = dump / (K * step).  We use the first
integer K strictly meeting each budget, rather than scaling one rounded
crossing by a rounded speedup.
"""
import math
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
DUMP_NAIVE = 192.0
DUMP_OPT = 25.0
K_OPT_10 = math.ceil(DUMP_OPT / (0.10 * STEP))
K_OPT_5 = math.ceil(DUMP_OPT / (0.05 * STEP))
K_NAIVE_10 = math.ceil(DUMP_NAIVE / (0.10 * STEP))
assert (K_OPT_10, K_OPT_5, K_NAIVE_10) == (342, 684, 2623)
K = np.logspace(np.log10(150), np.log10(6000), 300)
ov_naive = DUMP_NAIVE / (K * STEP) * 100
ov_opt = DUMP_OPT / (K * STEP) * 100

fig, ax = plt.subplots(figsize=(3.3, 1.30))
ax.plot(K, ov_naive, color=GRAY, lw=1.1, label='Naive (192 s/dump)')
ax.plot(K, ov_opt, color=OK, lw=1.3, label='Optimized (25 s/dump)')

ax.axhline(10, color=DARK, lw=0.6, ls=':')
ax.text(158, 9.1, '10% budget', fontsize=6, color=DARK, va='top')

for k, c, yt in ((K_OPT_10, OK, 15.5), (K_NAIVE_10, GRAY, 14.0)):
    ax.plot([k, k], [0, 10], color=c, lw=0.7, ls='--')
    ax.annotate(f'$K{{\\approx}}{k}$', xy=(k, 10), xytext=(k, yt),
                fontsize=6.2, color=c, ha='center',
                arrowprops=dict(arrowstyle='-', lw=0.5, color=c))

opt_1000 = DUMP_OPT / (1000 * STEP) * 100
naive_1000 = DUMP_NAIVE / (1000 * STEP) * 100
ax.plot([1000], [opt_1000], 'o', ms=3, color=OK)
ax.plot([1000], [naive_1000], 'o', ms=3, color=GRAY)
ax.annotate(f'{naive_1000:.1f}% vs. {opt_1000:.1f}%\nat $K{{=}}1000$', xy=(1000, naive_1000),
            xytext=(1350, 23.0), fontsize=6.2, color=DARK,
            arrowprops=dict(arrowstyle='->', lw=0.5, color=DARK))
ax.text(480, 7.7, '$\\sim$7.7× more frequent at any budget',
        fontsize=6.0, color=OK, style='italic', va='center')

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
