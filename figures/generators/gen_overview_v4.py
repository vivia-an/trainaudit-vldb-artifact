#!/usr/bin/env python3
"""Vector transcription of the approved Figure 2 redesign (user-approved raster,
2026-07-17 18:27). Every label is verbatim from main.tex: offline lane (Pattern
Catalog + framework source -> Invariant Miner with decisive counterexample
gate) feeds the verified constraint library P; online lane (Data Collector ->
DuckDB trace DB -> Verifier) runs compiled SQL with no LLM. Blue = offline
mining, orange = online checking (matches \\Description), red = reject/violation,
green = healthy-run replay (verification only).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Rectangle, Circle, Polygon
from matplotlib import rcParams

rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,
})

BLUE = '#2166ac'
BLUE_BG = '#eef3fa'
ORANGE = '#c85a19'
ORANGE_BG = '#fdf3e7'
RED = '#b2182b'
GREEN = '#3d7a3d'
GREEN_BG = '#f0f6f0'
DARK = '#2b2b2b'
GRAY = '#7a7a7a'
CARD = '#ffffff'
HUB_BG = '#f5f1e9'

W, H = 144, 102
fig = plt.figure(figsize=(7.2, 5.1))
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
    ax.add_patch(Circle((x, y), r, fc=c, ec='none', zorder=6))
    txt(x, y - 0.05, n, fs=fs, w='bold', c='white', z=7)


# ---------- line icons (single stroke style, panel-accent colors) ----------
def icon_book(cx, cy, s, c, lw=0.7):
    ax.plot([cx, cx], [cy - 0.55 * s, cy + 0.75 * s], color=c, lw=lw, zorder=6)
    for sgn in (-1, 1):
        xs = [cx, cx + sgn * 1.05 * s, cx + sgn * 1.05 * s, cx]
        ys = [cy + 0.75 * s, cy + 0.55 * s, cy - 0.7 * s, cy - 0.55 * s]
        ax.plot(xs + [xs[0]], ys + [ys[0]], color=c, lw=lw, zorder=6)

def icon_doc(cx, cy, s, c, lw=0.7):
    w, h, f = 0.85 * s, 1.15 * s, 0.35 * s
    xs = [cx - w, cx + w - f, cx + w, cx + w, cx - w, cx - w]
    ys = [cy + h, cy + h, cy + h - f, cy - h, cy - h, cy + h]
    ax.plot(xs, ys, color=c, lw=lw, zorder=6)
    ax.plot([cx + w - f, cx + w - f, cx + w], [cy + h, cy + h - f, cy + h - f],
            color=c, lw=lw * 0.8, zorder=6)
    for k, dy in enumerate((0.35, -0.05, -0.45)):
        ax.plot([cx - 0.5 * w, cx + (0.5 - 0.15 * k) * w], [cy + dy * s] * 2,
                color=c, lw=lw * 0.8, zorder=6)

def icon_gear(cx, cy, s, c, lw=0.7):
    import numpy as _np
    ax.add_patch(Circle((cx, cy), 0.62 * s, fc='none', ec=c, lw=lw, zorder=6))
    ax.add_patch(Circle((cx, cy), 0.26 * s, fc='none', ec=c, lw=lw * 0.85, zorder=6))
    for th in _np.arange(0, 360, 45):
        r = _np.deg2rad(th)
        ax.plot([cx + 0.62 * s * _np.cos(r), cx + 0.95 * s * _np.cos(r)],
                [cy + 0.62 * s * _np.sin(r), cy + 0.95 * s * _np.sin(r)],
                color=c, lw=lw, zorder=6)

def icon_shield(cx, cy, s, c, lw=0.75):
    xs = [cx - 0.75 * s, cx + 0.75 * s, cx + 0.75 * s, cx, cx - 0.75 * s]
    ys = [cy + 0.75 * s, cy + 0.75 * s, cy - 0.05 * s, cy - 0.95 * s, cy - 0.05 * s]
    ax.plot(xs + [xs[0]], ys + [ys[0]], color=c, lw=lw, zorder=6)
    ax.plot([cx - 0.32 * s, cx - 0.05 * s, cx + 0.42 * s],
            [cy + 0.02 * s, cy - 0.3 * s, cy + 0.38 * s], color=c, lw=lw, zorder=6)

def icon_replay(cx, cy, s, c, lw=0.75):
    from matplotlib.patches import Arc as _Arc
    ax.add_patch(_Arc((cx, cy), 1.7 * s, 1.7 * s, angle=0, theta1=300, theta2=210,
                      ec=c, lw=lw, zorder=6))
    import numpy as _np
    r = _np.deg2rad(210)
    tipx, tipy = cx + 0.85 * s * _np.cos(r), cy + 0.85 * s * _np.sin(r)
    ax.add_patch(Polygon([(tipx + 0.32 * s, tipy + 0.1 * s),
                          (tipx - 0.22 * s, tipy + 0.28 * s),
                          (tipx + 0.05 * s, tipy - 0.38 * s)],
                         closed=True, fc=c, ec='none', zorder=6))

def icon_search(cx, cy, s, c, lw=0.75):
    ax.add_patch(Circle((cx - 0.25 * s, cy + 0.25 * s), 0.62 * s, fc='none', ec=c,
                        lw=lw, zorder=6))
    ax.plot([cx + 0.22 * s, cx + 0.85 * s], [cy - 0.22 * s, cy - 0.85 * s],
            color=c, lw=lw * 1.2, zorder=6)
    ax.plot([cx - 0.5 * s, cx - 0.28 * s, cx + 0.08 * s],
            [cy + 0.22 * s, cy - 0.02 * s, cy + 0.52 * s], color=c, lw=lw * 0.85, zorder=6)

def icon_probe(cx, cy, s, c, lw=0.75):
    ax.plot([cx - 1.0 * s, cx + 1.0 * s], [cy + 0.55 * s] * 2, color=c, lw=lw, zorder=6)
    ax.plot([cx, cx], [cy + 0.55 * s, cy - 0.35 * s], color=c, lw=lw, zorder=6)
    ax.add_patch(Circle((cx, cy - 0.6 * s), 0.28 * s, fc=c, ec='none', zorder=6))
    ax.add_patch(Circle((cx, cy + 0.55 * s), 0.16 * s, fc=c, ec='none', zorder=6))


# ================= swim lanes =================
box(1, 62, 106.5, 39, fc=BLUE_BG, ec=BLUE, lw=0.9, r=1.8, z=1)
txt(3, 98.6, 'OFFLINE  —  mine once per framework', fs=6.4, w='bold', c=BLUE, ha='left')

box(1, 26, 106.5, 33.5, fc=ORANGE_BG, ec=ORANGE, lw=0.9, r=1.8, z=1)
txt(3, 57.4, 'ONLINE  —  every training job · fully deterministic · no LLM',
    fs=6.4, w='bold', c=ORANGE, ha='left')

box(110.5, 26, 32.5, 75, fc=HUB_BG, ec=DARK, lw=0.9, r=1.8, z=1)

# ================= offline inputs =================
box(3, 74.5, 25, 21.5, fc=CARD, ec=BLUE, lw=0.7)
icon_book(6.4, 93.7, 1.4, BLUE)
txt(16.9, 93.7, 'Pattern Catalog', fs=5.8, w='bold', c=BLUE)
txt(4.2, 90.6, '16 templates · from 392 real\nsilent errors · 13 fault classes',
    fs=4.4, ha='left')
txt(4.2, 87.4, 'Template families:', fs=4.2, c=GRAY, ha='left')
txt(4.2, 81.4, '· dtype preservation\n· cross-rank replication\n· scaling consistency\n· invocation frequency\n· state restoration\n· counter consistency',
    fs=4.2, ha='left', ls=1.3)

box(3, 63.5, 25, 9.5, fc=CARD, ec=BLUE, lw=0.7)
icon_doc(5.9, 70.9, 1.3, BLUE)
txt(17.2, 70.9, 'Framework source & docs', fs=5.2, w='bold', c=BLUE)
txt(15.5, 67.0, 'Megatron-LM · DeepSpeed ·\nOLMo · OLMo-core', fs=4.6)

# ================= miner panel =================
box(30.5, 63.5, 76, 33, fc=CARD, ec=DARK, lw=0.8)
badge(34.1, 93.7, '1', c=BLUE)
txt(37, 93.7, 'Invariant Miner', fs=6.6, w='bold', ha='left')
icon_gear(55.2, 93.7, 1.5, BLUE)
txt(105, 93.7, 'LLM-instantiated candidates, untrusted until verified',
    fs=4.6, style='italic', c=GRAY, ha='right')

steps = [
    (32.5, 'Scope', 'fix training phase\n+ parameter class', DARK, 0.7, '#fbfbfb'),
    (50.5, 'Ground', 'evidence $\\mathcal{E}^{+}/\\mathcal{E}^{-}$\nfrom framework\nsource & docs', DARK, 0.7, '#fbfbfb'),
    (68.5, 'Construct', 'counterexample attack\n$\\geq$2 CEs; one CE holds\n$\\Rightarrow$ reject outright', RED, 1.1, '#fdeeee'),
    (86.5, 'Accept', 'no CE holds $\\wedge$\nConf$(c) \\geq 0.8$\n$(\\theta = 0.8)$', DARK, 0.7, '#fbfbfb'),
]
for x, name, sub, ec, lw, fc in steps:
    box(x, 74.5, 16.5, 12.5, fc=fc, ec=ec, lw=lw)
    txt(x + 8.25, 84.6, name, fs=5.6, w='bold', c=ec)
    if name == 'Construct':
        icon_shield(70.6, 84.6, 1.35, RED)
    txt(x + 8.25, 79.2, sub, fs=4.3, ls=1.3)
for x0 in (49.2, 67.2, 85.2):
    arrow((x0, 80.75), (x0 + 1.1, 80.75), c=BLUE, lw=0.9)

# reject loop (decisive counterexample gate)
arrow((94.75, 87.3), (40.75, 87.4), c=RED, lw=0.8, ls=(0, (3, 2)), cs='arc3,rad=0.18')
txt(67.5, 90.6, 'reject $\\rightarrow$ next iteration', fs=4.5, c=RED, style='italic')

# healthy-run replay (verification only)
box(66, 64.5, 32, 5.8, fc=GREEN_BG, ec=GREEN, lw=0.7)
icon_replay(69.3, 67.4, 1.15, GREEN)
txt(83, 67.4, 'healthy-run replay  (verification only)', fs=4.7, c=GREEN)
arrow((92, 70.4), (93.5, 74.3), c=GREEN, lw=0.8)

# FSM footnote
txt(32.5, 68.0, 'five-stage FSM:  $S_1$ gap analysis · $S_2$ evidence\ngathering · $S_3$ synthesis+attack · $S_4$ persist · $S_5$ report',
    fs=4.3, c=GRAY, ha='left', ls=1.35)

# input arrows into miner
arrow((28.2, 88), (30.4, 84), c=BLUE, lw=1.0)
arrow((28.2, 68.5), (30.4, 76), c=BLUE, lw=1.0)

# ================= hub: verified constraint library =================
cyl(119.5, 91.5, 14, 7.5, fc='#e9e2d2', ec=DARK)
icon_shield(113.9, 88.9, 1.3, DARK)
txt(128.0, 88.9, 'Verified Constraint Library $\\mathcal{P}$', fs=5.8, w='bold')
txt(126.7, 86.4, 'guarded relational constraints', fs=4.5, style='italic', c=GRAY)

box(112.5, 58, 28.5, 26, fc=CARD, ec=DARK, lw=0.7)
txt(126.7, 81.4, 'P3 · cross-rank replication', fs=5.0, w='bold')
txt(126.7, 78.9, '(SwitchMLP router sync)', fs=4.4, c=GRAY)
txt(114, 69.5, '$\\pi_{topo}$:  TP > 1\n\n$\\pi_{precond}$:  tpl = False\n   (not tensor-parallel sharded)\n\n$\\pi_{schema}$:  replicas share\n   checksum',
    fs=4.6, ha='left', ls=1.35)

arrow((103.2, 80.75), (117.9, 92.5), c=BLUE, lw=1.1, cs='arc3,rad=-0.25')
txt(106.2, 90.2, 'accepted', fs=4.5, c=BLUE, style='italic', ha='right')

# hub -> online arrows
txt(126.7, 44.0, 'compiled to\nparameterized SQL\nat job start', fs=4.8, c=ORANGE, w='bold')
arrow((122, 57.7), (107.2, 46.5), c=ORANGE, lw=1.0, cs='arc3,rad=0.2')
arrow((112.5, 60.5), (42, 59.7), c=ORANGE, lw=1.0, cs='arc3,rad=0.1')
txt(78, 61.0, 'accepted rules determine trace schema (S0–S6)', fs=4.4, c=ORANGE)

# ================= online: data collector =================
box(3, 27.5, 42.5, 29, fc=CARD, ec=DARK, lw=0.8)
badge(6.6, 53.7, '2', c=ORANGE)
txt(9.5, 53.7, 'Data Collector', fs=6.6, w='bold', ha='left')
icon_probe(26.0, 53.9, 1.3, ORANGE)
txt(3.8, 50.9, 'no user code changes · adapter 30–150 LoC', fs=4.5,
    style='italic', c=GRAY, ha='left')

anchors = ['before-\nforward', 'after-\nforward', 'main-grad-\nin-backward',
           'after-\nbackward', 'before-\noptimizer']
for i, a in enumerate(anchors):
    x = 4.5 + i * 8.2
    box(x, 43.5, 7.4, 6, fc='#fdf8f0', ec=ORANGE, lw=0.6, r=0.8)
    txt(x + 3.7, 46.5, a, fs=4.0, family='monospace')
    if i:
        arrow((x - 0.9, 46.5), (x - 0.1, 46.5), c=ORANGE, lw=0.6, ms=4)
txt(24.2, 41.3, '+ auxiliary taps (ckpt save/load · all_reduce · snapshot) = 8 hookpoints',
    fs=4.1, c=GRAY)
txt(24.2, 38.7, 'GPU-side scalar reductions (checksums only when required)', fs=4.4)

box(6.5, 29.5, 35.5, 7, fc='#fdf8f0', ec=ORANGE, lw=0.7)
txt(24.25, 34.4, 'captured record (S0):', fs=4.3, c=GRAY)
txt(24.25, 31.8, 'cksum · param_name · rank · step · stage', fs=4.6, family='monospace')

# duckdb cylinder
cyl(48.5, 37, 12, 13, fc='#fbe9d4', ec=ORANGE)
txt(54.5, 34.3, 'DuckDB trace DB', fs=4.8, w='bold', c=ORANGE)
txt(54.5, 31.6, 'two-step\nsliding window', fs=4.3, c=GRAY)
arrow((45.8, 43), (48.2, 43), c=ORANGE, lw=1.0)
arrow((60.9, 43), (63.0, 43), c=ORANGE, lw=1.0)

# ================= online: verifier =================
box(63.5, 27.5, 43, 29, fc=CARD, ec=DARK, lw=0.8)
badge(67.1, 53.7, '3', c=ORANGE)
txt(70, 53.7, 'Verifier', fs=6.6, w='bold', ha='left')
icon_search(79.3, 53.9, 1.25, ORANGE)
txt(82.5, 53.7, 'deterministic SQL checking — no LLM',
    fs=4.6, style='italic', c=GRAY, ha='left')

box(65, 36.5, 13, 14.5, fc='#fbfbfb', ec=DARK, lw=0.6)
txt(71.5, 49.4, 'at job start', fs=4.7, w='bold')
txt(71.5, 42.6, 'extract topology $\\tau$\ntopology-aware\npruning · lineage:\nreplicated vs.\nsharded', fs=4.1, ls=1.35)

box(79.5, 36.5, 14.5, 14.5, fc='#f4f7fb', ec=BLUE, lw=0.7)
txt(86.75, 49.4, 'per step, per rule', fs=4.7, w='bold', c=BLUE)
txt(80.3, 42.4, 'WHERE $\\pi_{topo}$\n  $\\wedge\\ \\pi_{precond}$\nGROUP BY param,step\nHAVING $\\pi_{schema}$\n  violated', fs=4.0, ha='left', family='monospace', ls=1.4)

box(95.3, 36.5, 11.2, 14.5, fc='#fdf0f0', ec=RED, lw=0.7)
txt(100.9, 49.4, 'violation report', fs=4.5, w='bold', c=RED)
txt(96.0, 42.4, 'violating constraint\naffected parameter\ndivergent ranks\nfirst violating step', fs=4.0, ha='left', ls=1.4)

arrow((78.2, 43.75), (79.3, 43.75), c=ORANGE, lw=0.8)
arrow((94.2, 43.75), (95.3, 43.75), c=RED, lw=0.8)

txt(85, 33.6, 'empty result = holds · non-empty result IS the violation set',
    fs=4.4, style='italic')
txt(85, 30.8, 'coarse$\\rightarrow$fine: checksum $\\rightarrow$ module $\\rightarrow$ parameter $\\rightarrow$ bitwise',
    fs=4.3, c=GRAY)

# ================= running example strip =================
box(1, 13, 142, 11.5, fc='#faf8f7', ec=GRAY, lw=0.6, r=1.5, z=1)
txt(2.8, 21.3, 'Running\nexample', fs=5.0, w='bold', c=RED, ha='left')
txt(2.8, 16.3, '(end-to-end)', fs=4.3, ha='left', c=GRAY)

ex = [
    ('Bug in training',
     'Megatron-LM SwitchMLP missing sync\nleaves router.weight replicas\nsilently divergent', DARK),
    ('Mined rule (offline)',
     'P3 cross-rank replication guards with\n$\\pi_{topo} \\wedge \\pi_{precond} \\wedge \\pi_{schema}$\n(router weight checksums)', BLUE),
    ('Clean TP=2 run',
     '57 sharded query_key_value.weight\nexcluded by WHERE $\\Rightarrow$ silent  $\\checkmark$', GREEN),
    ('Buggy run (same job)',
     'query returns divergent router.weight\n(+ disagreeing ranks)  $\\mathbf{\\times}$', RED),
    ('Production blocked',
     'same rule intercepted a 256$\\times$H200 fault\n($\\approx$43,000 GPU-hours saved)', DARK),
]
x = 16
cw = 24.0
for i, (t, s, c) in enumerate(ex):
    box(x, 14.1, cw, 9.2, fc=CARD, ec=c, lw=0.7, r=0.9)
    badge(x + 2.4, 21.4, str(i + 1), c=c, r=1.4, fs=4.6)
    txt(x + cw / 2 + 1.2, 21.4, t, fs=4.6, w='bold', c=c)
    txt(x + cw / 2, 17.2, s, fs=4.1, ls=1.3)
    if i < len(ex) - 1:
        arrow((x + cw + 0.15, 18.7), (x + cw + 1.25, 18.7), c=GRAY, lw=0.8, ms=5)
    x += cw + 1.4

# ================= legend =================
box(1, 4.5, 142, 6.5, fc='#fcfcfc', ec=GRAY, lw=0.5, r=1.2, z=1)
ay = 7.75
txt(3, ay, 'Legend', fs=4.7, w='bold', c=DARK, ha='left')
ax.plot([14, 20], [ay, ay], color=BLUE, lw=1.1)
txt(21.5, ay, 'Offline Flow (Mining)', fs=4.6, ha='left')
ax.plot([48, 54], [ay, ay], color=ORANGE, lw=1.1)
txt(55.5, ay, 'Online Flow (Checking)', fs=4.6, ha='left')
ax.plot([82, 88], [ay, ay], color=GREEN, lw=1.0)
txt(89.5, ay, 'Replay / Verification', fs=4.6, ha='left')
ax.plot([112, 118], [ay, ay], color=RED, lw=1.0, ls=(0, (3, 2)))
txt(119.5, ay, 'Violation / Reject (Abnormal Path)', fs=4.6, ha='left')

fig.savefig('fig_overview_v4.pdf')
fig.savefig('/tmp/fig_overview_v4.png', dpi=220)
print('done')
