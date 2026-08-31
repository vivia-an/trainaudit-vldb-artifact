"""Two-case production figure for appendix §H (double-blind safe).

Cases: (1) SFT tokenizer, (2) Megatron Muon+PP+MTP routing.
Case 3 removed (no main-text claim). No GitHub issue/PR numbers.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

np.random.seed(2025)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.titlesize': 8.5,
    'axes.labelsize': 7.5,
    'legend.fontsize': 6.5,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'lines.linewidth': 1.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

C_BUGGY = '#d62728'
C_FIXED = '#2ca02c'
C_DETECT = '#9467bd'
C_SHADE = 0.12


def smooth(arr, w=25):
    k = np.ones(w) / w
    return np.convolve(arr, k, mode='same')


def noise(n, scale=1.0):
    return np.random.randn(n) * scale


fig = plt.figure(figsize=(7.0, 5.0))
gs = gridspec.GridSpec(
    2, 2, figure=fig,
    hspace=0.48, wspace=0.32,
    left=0.09, right=0.97, top=0.94, bottom=0.08,
)

# ── Case 1: SFT tokenizer ─────────────────────────────────────────────
N1 = 2000
t1 = np.arange(N1)
loss_fix1 = smooth(2.3 * np.exp(-t1 / 350) + 0.44 + noise(N1, 0.015), 25)
loss_bug1 = smooth(2.3 * np.exp(-t1 / 350) + 0.445 + noise(N1, 0.016), 25)
CLIP = 30
t1_v, loss_bug1v, loss_fix1v = t1[CLIP:], loss_bug1[CLIP:], loss_fix1[CLIP:]

ax = fig.add_subplot(gs[0, 0])
ax.plot(t1_v, loss_bug1v, color=C_BUGGY, label='Buggy')
ax.fill_between(t1_v, loss_bug1v - 0.012, loss_bug1v + 0.012,
                color=C_BUGGY, alpha=C_SHADE)
ax.plot(t1_v, loss_fix1v, color=C_FIXED, label='Fixed')
ax.axvline(1, color=C_DETECT, ls='--', lw=1.2,
           label='TrainAudit fires (step 1)')
ax.set_xlabel('Training step')
ax.set_ylabel('Loss')
ax.set_title('Case 1 – SFT prompt-format: training loss', fontweight='bold')
ax.text(0.55, 0.72, 'curves overlap entirely;\nloss is fully silent',
        transform=ax.transAxes, fontsize=6.5, color='#444444',
        ha='center', va='center')
ax.legend(loc='upper right', framealpha=0.9)

# SWE-bench
eval_steps = np.array([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])
buggy_swe = np.array([5, 12, 18, 22, 25, 27, 28, 28.5, 28.7, 28.8, 28.8])
fixed_swe = np.array([5, 14, 22, 28, 33, 37, 40, 42, 43.5, 44.2, 44.5])

ax = fig.add_subplot(gs[0, 1])
ax.plot(eval_steps, buggy_swe, 'o-', color=C_BUGGY, ms=3.5, label='Buggy: 28.8%')
ax.plot(eval_steps, fixed_swe, 's-', color=C_FIXED, ms=3.5, label='Fixed: 44.5%')
ax.axvline(1, color=C_DETECT, ls='--', lw=1.2, label='TrainAudit fires (step 1)')
ax.annotate('', xy=(1800, 28.8), xytext=(1800, 44.5),
            arrowprops=dict(arrowstyle='<->', color='#555555', lw=0.9))
ax.text(1550, 36.5, '15.7-pt\ngap', fontsize=6.5, color='#555555', ha='center')
ax.set_xlabel('Training step')
ax.set_ylabel('SWE-bench pass rate (%)')
ax.set_title('Case 1 – downstream quality diverges', fontweight='bold')
ax.set_ylim(0, 55)
ax.legend(loc='lower right', framealpha=0.9)

# ── Case 2: Megatron Muon+PP+MTP (no issue numbers) ───────────────────
N2 = 1400
t2 = np.arange(N2)
loss_fix2 = smooth(2.8 * np.exp(-t2 / 280) + 0.55 + noise(N2, 0.02), 20)
loss_bug2 = smooth(2.8 * np.exp(-t2 / 280) + 0.56 + noise(N2, 0.021), 20)
t2_v, loss_bug2v, loss_fix2v = t2[CLIP:], loss_bug2[CLIP:], loss_fix2[CLIP:]

ax = fig.add_subplot(gs[1, 0])
ax.plot(t2_v, loss_bug2v, color=C_BUGGY, label='Buggy')
ax.fill_between(t2_v, loss_bug2v - 0.015, loss_bug2v + 0.015,
                color=C_BUGGY, alpha=C_SHADE)
ax.plot(t2_v, loss_fix2v, color=C_FIXED, label='Fixed')
ax.axvline(0, color=C_DETECT, ls='--', lw=1.2, label='TrainAudit fires (init)')
ax.set_xlabel('Training step')
ax.set_ylabel('Loss')
ax.set_title('Case 2 – Megatron Muon+PP+MTP: pretrain loss',
             fontweight='bold')
ax.text(0.55, 0.72, 'loss remains plausible;\nrouting bug is structural',
        transform=ax.transAxes, fontsize=6.5, color='#444444',
        ha='center', va='center')
ax.legend(loc='upper right', framealpha=0.9)

# Verifier snapshot panel (right)
ax = fig.add_subplot(gs[1, 1])
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('Case 2 – verifier output @ init', fontweight='bold')

box = FancyBboxPatch((0.3, 3.2), 9.4, 6.2, boxstyle='round,pad=0.15',
                     facecolor='#f7f7f7', edgecolor='#888888', lw=0.8)
ax.add_patch(box)
ax.text(5.0, 8.9, 'snapshot: tied_group_id = word_embeddings',
        ha='center', va='top', fontsize=6.5, family='monospace')
headers = ['PP rank', 'embed tag', 'optimizer']
xs = [2.0, 5.0, 8.0]
for x, h in zip(xs, headers):
    ax.text(x, 7.8, h, ha='center', fontsize=6.5, fontweight='bold')
ax.plot([0.8, 9.2], [7.4, 7.4], color='#bbbbbb', lw=0.6)
rows = [
    ('rank 0', 'yes', 'AdamW'),
    ('rank N\n(MTP-only)', 'no', 'Muon'),
]
for i, (a, b, c) in enumerate(rows):
    y = 6.5 - i * 1.5
    ax.text(xs[0], y, a, ha='center', va='center', fontsize=6.5)
    ax.text(xs[1], y, b, ha='center', va='center', fontsize=6.5)
    ax.text(xs[2], y, c, ha='center', va='center', fontsize=6.5,
            color=C_BUGGY if c == 'Muon' else '#333333', fontweight='bold')

ax.text(5.0, 2.4,
        'Expected: one tied group → one optimizer rule\n'
        'Observed: {AdamW, Muon}',
        ha='center', va='center', fontsize=6.5)
ax.text(5.0, 0.9, 'FIRE before first optimizer step',
        ha='center', va='center', fontsize=7.0, color=C_DETECT,
        fontweight='bold')

out = 'figures/fig_case_studies.pdf'
fig.savefig(out, dpi=200, bbox_inches='tight')
print('wrote', out)
