#!/usr/bin/env python3
"""Figure 2 (architecture) redraw, panel copy taken verbatim from main.tex:
offline lane (Pattern Catalog + framework source -> Invariant Miner) feeds the
verified constraint library P; online lane (Data Collector -> DuckDB trace ->
Verifier) checks compiled SQL with no LLM. Blue = offline mining, orange =
online checking (matches the \\Description in main.tex); red = reject/violation.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Rectangle
from matplotlib import rcParams

rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,
})

BLUE = '#2166ac'
BLUE_BG = '#edf3fa'
ORANGE = '#c85a19'
ORANGE_BG = '#fdf3e7'
RED = '#b2182b'
DARK = '#2b2b2b'
GRAY = '#7a7a7a'
CARD = '#ffffff'
HUB_BG = '#f5f0e8'

W, H = 144, 96
fig = plt.figure(figsize=(7.2, 4.8))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis('off')


def box(x, y, w, h, fc=CARD, ec=DARK, lw=0.7, r=1.2, ls='-', z=2):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}',
                       fc=fc, ec=ec, lw=lw, ls=ls, zorder=z)
    ax.add_patch(p)
    return p


def txt(x, y, s, fs=5, w='normal', c=DARK, ha='center', va='center',
        style='normal', family=None, ls=1.15, z=5):
    kw = dict(fontsize=fs, fontweight=w, color=c, ha=ha, va=va,
              style=style, linespacing=ls, zorder=z)
    if family:
        kw['family'] = family
    ax.text(x, y, s, **kw)


def arrow(p0, p1, c=DARK, lw=0.9, ls='-', style='-|>', ms=7, cs=None, z=4):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                        color=c, lw=lw, linestyle=ls, zorder=z,
                        connectionstyle=cs or 'arc3,rad=0')
    ax.add_patch(a)


def cyl(x, y, w, h, fc='#dbe7f3', ec=BLUE, z=3):
    ax.add_patch(Rectangle((x, y + h * 0.12), w, h * 0.76, fc=fc, ec='none', zorder=z))
    ax.add_patch(Ellipse((x + w / 2, y + h * 0.12), w, h * 0.24, fc=fc, ec=ec, lw=0.7, zorder=z))
    ax.plot([x, x], [y + h * 0.12, y + h * 0.88], color=ec, lw=0.7, zorder=z + 1)
    ax.plot([x + w, x + w], [y + h * 0.12, y + h * 0.88], color=ec, lw=0.7, zorder=z + 1)
    ax.add_patch(Ellipse((x + w / 2, y + h * 0.88), w, h * 0.24, fc=fc, ec=ec, lw=0.7, zorder=z + 2))


def badge(x, y, n, c=DARK, r=2.0, fs=6):
    ax.add_patch(plt.Circle((x, y), r, fc=c, ec='none', zorder=6))
    txt(x, y - 0.05, n, fs=fs, w='bold', c='white', z=7)


# ---------------- swim lanes ----------------
box(1, 58, 106.5, 37, fc=BLUE_BG, ec=BLUE, lw=0.9, r=1.8, z=1)
txt(3, 92.6, 'OFFLINE  —  mine once per framework', fs=6.4, w='bold', c=BLUE, ha='left')

box(1, 21, 106.5, 34.5, fc=ORANGE_BG, ec=ORANGE, lw=0.9, r=1.8, z=1)
txt(3, 52.9, 'ONLINE  —  every training job · fully deterministic · no LLM',
    fs=6.4, w='bold', c=ORANGE, ha='left')

# hub column
box(110.5, 21, 32.5, 74, fc=HUB_BG, ec=DARK, lw=0.9, r=1.8, z=1)

# ---------------- offline inputs ----------------
box(3, 76, 25, 15, fc=CARD, ec=BLUE, lw=0.7)
txt(15.5, 88.6, 'Pattern Catalog', fs=5.8, w='bold', c=BLUE)
txt(15.5, 85.9, '16 framework-agnostic\nconstraint templates', fs=4.6)
txt(15.5, 82.2, 'dtype preservation · cross-rank\nreplication · scaling consistency ·\ncounter consistency · ...', fs=4.0, c=GRAY)
txt(15.5, 78.0, 'from 392 real silent errors · 13 fault classes', fs=3.8, style='italic', c=GRAY)

box(3, 66, 25, 8, fc=CARD, ec=BLUE, lw=0.7)
txt(15.5, 71.7, 'Framework source & docs', fs=5.4, w='bold', c=BLUE)
txt(15.5, 68.9, 'Megatron-LM · DeepSpeed ·\nOLMo · OLMo-core', fs=4.3)

# ---------------- miner panel ----------------
box(31, 60, 74.5, 33, fc=CARD, ec=DARK, lw=0.8)
badge(34.6, 90.2, '1', c=BLUE)
txt(37.5, 90.2, 'Invariant Miner', fs=6.6, w='bold', c=DARK, ha='left')
txt(70.5, 90.2, 'LLM-instantiated candidates, untrusted until verified',
    fs=4.6, style='italic', c=GRAY, ha='left')

steps = [
    (33.5, 'Scope', 'fix $\\pi_{topo} \\wedge \\pi_{precond}$\n(phase + param class)', DARK, 0.7),
    (50.5, 'Ground', 'evidence $\\mathcal{E}^+/\\mathcal{E}^-$\nfrom framework source', DARK, 0.7),
    (67.5, 'Construct', 'counterexample attack\n$\\geq$2 CEs; one CE holds\n$\\Rightarrow$ reject outright', RED, 1.2),
    (84.5, 'Accept', 'no CE holds $\\wedge$\nConf$(c) \\geq 0.8$', DARK, 0.7),
]
for x, name, sub, ec, lw in steps:
    box(x, 72.5, 15.5, 11.5, fc='#fbfbfb' if ec is DARK else '#fdeeee', ec=ec, lw=lw)
    txt(x + 7.75, 81.9, name, fs=5.6, w='bold', c=ec)
    txt(x + 7.75, 76.4, sub, fs=3.9)
for x0 in (49.0, 66.0, 83.0):
    arrow((x0, 78.3), (x0 + 1.6, 78.3), c=BLUE, lw=0.9)

# reject loop (decisive counterexample gate)
arrow((92.2, 84.2), (41.3, 84.4), c=RED, lw=0.8, ls=(0, (3, 2)),
      cs='arc3,rad=0.18')
txt(66, 87.0, 'reject $\\rightarrow$ next iteration', fs=4.2, c=RED, style='italic')

# healthy-run replay (verification-only input)
box(66, 61.5, 28, 6.5, fc='#f2f7f2', ec='#4a7c4a', lw=0.7)
txt(80, 64.8, 'healthy-run replay  (verification only)', fs=4.5, c='#2f5d2f')
arrow((80, 68.0), (79, 72.3), c='#4a7c4a', lw=0.8)

# FSM footnote
txt(33.5, 64.8, 'five-stage FSM:  $S_1$ gap analysis · $S_2$ evidence\ngathering · $S_3$ synthesis+attack · $S_4$ persist · $S_5$ report',
    fs=3.8, c=GRAY, ha='left')

# input arrows into miner
arrow((28.2, 83.5), (31.0, 80.5), c=BLUE, lw=1.0)
arrow((28.2, 70), (31.0, 74), c=BLUE, lw=1.0)

# ---------------- hub: verified constraint library ----------------
cyl(119, 84.5, 15, 8, fc='#e9e2d2', ec=DARK)
txt(126.7, 81.9, 'Verified Constraint Library $\\mathcal{P}$', fs=5.8, w='bold')
txt(126.7, 79.4, 'guarded relational constraints', fs=4.3, style='italic', c=GRAY)

box(113, 58.5, 27.5, 18.5, fc=CARD, ec=DARK, lw=0.7)
txt(126.7, 74.6, 'P3 · cross-rank replication', fs=4.8, w='bold')
txt(114.5, 69.5, '$\\pi_{topo}$:  TP > 1\n$\\pi_{precond}$:  tpl = False\n   (not tensor-parallel sharded)\n$\\pi_{schema}$:  replicas share\n   checksum',
    fs=4.3, ha='left', ls=1.35)

arrow((105.7, 78.3), (117.5, 85.5), c=BLUE, lw=1.1, cs='arc3,rad=-0.2')
txt(111, 84.9, 'accepted', fs=4.0, c=BLUE, style='italic')

# hub -> online arrows
arrow((126.7, 58.0), (90.5, 45.2), c=ORANGE, lw=1.0, cs='arc3,rad=0.22')
txt(126.5, 47.5, 'compiled to\nparameterized SQL\nat job start', fs=4.3, c=ORANGE, ha='center')
arrow((113, 60.5), (44, 51.3), c=ORANGE, lw=1.0, cs='arc3,rad=0.12')
txt(76, 56.5, 'accepted rules determine trace schema (S0–S6)', fs=4.0, c=ORANGE)

# ---------------- online: data collector ----------------
box(3, 23, 42, 27, fc=CARD, ec=DARK, lw=0.8)
badge(6.6, 47.3, '2', c=ORANGE)
txt(9.5, 47.3, 'Data Collector', fs=6.6, w='bold', ha='left')
txt(3.6, 44.3, 'no user code changes · adapter 30–150 LoC', fs=4.2,
    style='italic', c=GRAY, ha='left')

anchors = ['before-\nforward', 'after-\nforward', 'main-grad-\nin-backward',
           'after-\nbackward', 'before-\noptimizer']
for i, a in enumerate(anchors):
    x = 4.5 + i * 8.0
    box(x, 36.5, 7.2, 6, fc='#fdf8f0', ec=ORANGE, lw=0.6, r=0.8)
    txt(x + 3.6, 39.5, a, fs=3.6, family='monospace')
    if i:
        arrow((x - 0.9, 39.5), (x - 0.1, 39.5), c=ORANGE, lw=0.6, ms=4)
txt(24, 34.4, '+3 auxiliary taps (ckpt save/load · all_reduce · build snapshot) = 8 hookpoints',
    fs=3.8, c=GRAY)
txt(24, 31.6, 'GPU-side scalar reductions (checksums only when required)', fs=4.0)

box(9, 24.5, 30, 5.5, fc='#fdf8f0', ec=ORANGE, lw=0.7)
txt(24, 28.3, 'captured record (S0):', fs=3.9, c=GRAY)
txt(24, 26.2, 'cksum · param_name · rank · step · stage', fs=4.2, family='monospace')

# duckdb cylinder between panels
cyl(48.5, 30, 12, 12, fc='#fbe9d4', ec=ORANGE)
txt(54.5, 27.5, 'DuckDB trace DB', fs=4.8, w='bold', c=ORANGE)
txt(54.5, 24.9, 'two-step\nsliding window', fs=3.9, c=GRAY)
arrow((45.3, 36), (48.0, 36), c=ORANGE, lw=1.0)
arrow((61.0, 36), (63.7, 36), c=ORANGE, lw=1.0)

# ---------------- online: verifier ----------------
box(64, 23, 43.5, 27, fc=CARD, ec=DARK, lw=0.8)
badge(67.6, 47.3, '3', c=ORANGE)
txt(70.5, 47.3, 'Verifier', fs=6.6, w='bold', ha='left')
txt(79.5, 47.3, 'deterministic SQL checking — no LLM',
    fs=4.3, style='italic', c=GRAY, ha='left')

box(65.5, 31, 13, 13.5, fc='#fbfbfb', ec=DARK, lw=0.6)
txt(72, 42.6, 'at job start', fs=4.4, w='bold')
txt(72, 37.2, 'extract topology $\\tau$\ntopology-aware\npruning · lineage:\nreplicated vs.\nsharded', fs=3.8, ls=1.3)

box(79.5, 31, 14.5, 13.5, fc='#f4f7fb', ec=BLUE, lw=0.7)
txt(86.75, 42.6, 'per step, per rule', fs=4.4, w='bold', c=BLUE)
txt(80.3, 37.0, 'WHERE $\\pi_{topo}$\n  $\\wedge\\ \\pi_{precond}$\nGROUP BY param,step\nHAVING $\\pi_{schema}$\n  violated', fs=3.6, ha='left', family='monospace', ls=1.35)

box(95.2, 31, 11.3, 13.5, fc='#fdf0f0', ec=RED, lw=0.7)
txt(100.85, 42.6, 'violation report', fs=4.3, w='bold', c=RED)
txt(95.9, 36.8, 'violating constraint\naffected parameter\ndivergent ranks\nfirst violating step', fs=3.7, ha='left', ls=1.4)

arrow((78.6, 37.75), (79.3, 37.75), c=ORANGE, lw=0.8)
arrow((94.1, 37.75), (95.0, 37.75), c=RED, lw=0.8)

txt(85.5, 28.4, 'empty result = holds · non-empty result IS the violation set', fs=4.0, style='italic')
txt(85.5, 25.6, 'coarse$\\rightarrow$fine: checksum $\\rightarrow$ module $\\rightarrow$ parameter $\\rightarrow$ bitwise',
    fs=3.9, c=GRAY)

# ---------------- running example strip ----------------
box(1, 7.5, 142, 11, fc='#faf7f7', ec=GRAY, lw=0.6, r=1.5, z=1)
txt(2.8, 15.4, 'Running\nexample', fs=4.9, w='bold', c=RED, ha='left')
txt(2.8, 11.4, 'P3 · SwitchMLP\nrouter sync', fs=3.7, ha='left', c=GRAY)

ex = [
    ('Bug', 'Megatron-LM SwitchMLP router\nweights miss a sync $\\rightarrow$ replicas\nsilently diverge', DARK),
    ('Mined rule (offline)', 'P3 guarded: checked only when\nTP>1 $\\wedge$ not sharded', BLUE),
    ('Clean TP=2 run', '57 sharded query_key_value.weight\nexcluded by WHERE $\\rightarrow$ silent  $\\checkmark$', '#2f6b2f'),
    ('Buggy run', 'query returns divergent router.weight\n+ disagreeing ranks  $\\mathbf{\\times}$', RED),
    ('Production', 'same rule intercepted a 256$\\times$H200\nfault (~43,000 GPU-hours saved)', DARK),
]
x = 16.5
cw = 23.8
for i, (t, s, c) in enumerate(ex):
    box(x, 8.6, cw, 8.6, fc=CARD, ec=c, lw=0.7 if c in (RED,) else 0.6, r=0.9)
    txt(x + cw / 2, 15.5, t, fs=4.4, w='bold', c=c)
    txt(x + cw / 2, 11.7, s, fs=3.7)
    if i < len(ex) - 1:
        arrow((x + cw + 0.15, 12.9), (x + cw + 1.25, 12.9), c=GRAY, lw=0.8, ms=5)
    x += cw + 1.4

# ---------------- legend ----------------
y = 3.4
ax.plot([3, 9], [y, y], color=BLUE, lw=1.1)
txt(10, y, 'offline mining', fs=4.4, ha='left')
ax.plot([26, 32], [y, y], color=ORANGE, lw=1.1)
txt(33, y, 'online checking', fs=4.4, ha='left')
ax.plot([50, 56], [y, y], color=RED, lw=1.0, ls=(0, (3, 2)))
txt(57, y, 'reject / violation', fs=4.4, ha='left')
txt(78, y, 'LLM used only inside the offline Miner — the online path contains no LLM call',
    fs=4.4, style='italic', c=GRAY, ha='left')

fig.savefig('fig_overview_v3.pdf')
fig.savefig('/tmp/fig_overview_v3.png', dpi=220)
print('done')
