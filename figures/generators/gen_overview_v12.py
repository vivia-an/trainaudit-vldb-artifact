#!/usr/bin/env python3
"""v12 = v11 bloodline-safe + QUITE pixel palette/spacing (no new content).
Same labels as v11; only colors, stroke weights, alignment, icon polish.
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

BLUE = '#306092'          # QUITE navy fill
BLUE_SOFT = '#6C8EBF'     # QUITE mid blue
BLUE_BG = '#EEF2F8'
ORANGE = '#B85A2A'        # warm online accent (not QUITE teal; keep offline/online split)
ORANGE_BG = '#FBF3EA'
RED = '#B85450'           # QUITE red-ish
GREEN = '#1AABA8'         # QUITE teal checkmarks
GREEN_BG = '#EAF7F6'
DARK = '#2C2C2C'
GRAY = '#7A7A7A'
CARD = '#FFFFFF'
HUB_BG = '#F4F1EA'
LAVENDER = '#E1D5E7'      # QUITE lavender
MINT = '#D5E8D4'
PEACH = '#F8CECC'

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


import matplotlib.patheffects as _pe


def arrow(p0, p1, c=DARK, lw=0.9, ls='-', style='-|>', ms=7, cs=None, z=4,
          halo=False):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                        color=c, lw=lw, linestyle=ls, zorder=z,
                        connectionstyle=cs or 'arc3,rad=0')
    if halo:
        a.set_path_effects([_pe.Stroke(linewidth=lw + 1.8, foreground='white'),
                            _pe.Normal()])
    ax.add_patch(a)


def cyl(x, y, w, h, fc='#dbe7f3', ec=BLUE, z=3):
    ax.add_patch(Rectangle((x, y + h * 0.12), w, h * 0.76, fc=fc, ec='none', zorder=z))
    ax.add_patch(Ellipse((x + w / 2, y + h * 0.12), w, h * 0.24, fc=fc, ec=ec, lw=0.7, zorder=z))
    ax.plot([x, x], [y + h * 0.12, y + h * 0.88], color=ec, lw=0.7, zorder=z + 1)
    ax.plot([x + w, x + w], [y + h * 0.12, y + h * 0.88], color=ec, lw=0.7, zorder=z + 1)
    ax.add_patch(Ellipse((x + w / 2, y + h * 0.88), w, h * 0.24, fc=fc, ec=ec, lw=0.7, zorder=z + 2))


def badge(x, y, n, c='#1A1A1A', r=1.85, fs=5.8):
    ax.add_patch(Circle((x, y), r, fc=c, ec='none', zorder=6))
    # hairline white ring (QUITE numbered-marker crispness)
    ax.add_patch(Circle((x, y), r + 0.15, fc='none', ec='white', lw=0.45, zorder=5.8))
    txt(x, y - 0.05, n, fs=fs, w='bold', c='white', z=7)


def shadow(x, y, w, h, r=1.2):
    ax.add_patch(FancyBboxPatch((x + 0.35, y - 0.4), w, h,
                 boxstyle=f'round,pad=0,rounding_size={r}',
                 fc='#6A655C', ec='none', alpha=0.12, zorder=1.6))


def chip(cx, cy, s, c):
    ax.add_patch(FancyBboxPatch((cx - s, cy - s), 2 * s, 2 * s,
                 boxstyle='round,pad=0,rounding_size=0.55',
                 fc=c, ec='none', zorder=5.5))
    # 1px highlight for print depth
    ax.add_patch(FancyBboxPatch((cx - s + 0.12, cy - s + 0.12), 2 * s - 0.24, 0.55,
                 boxstyle='round,pad=0,rounding_size=0.2',
                 fc='white', ec='none', alpha=0.22, zorder=5.6))


def pill(x, y, w, h, t, fc, ec, tc=DARK, fs=3.8):
    box(x, y, w, h, fc=fc, ec=ec, lw=0.45, r=min(h / 2, 1.1), z=4)
    txt(x + w / 2, y + h / 2, t, fs=fs, family='monospace', c=tc, z=5)


def checkbox(x, y, sz, c):
    ax.add_patch(FancyBboxPatch((x, y), sz, sz, boxstyle='round,pad=0,rounding_size=0.15',
                                fc='white', ec=c, lw=0.55, zorder=5))
    ax.plot([x + 0.18 * sz, x + 0.42 * sz, x + 0.85 * sz],
            [y + 0.48 * sz, y + 0.12 * sz, y + 0.82 * sz], color=c, lw=0.85, zorder=6,
            solid_capstyle='round')


def icon_target(cx, cy, s, c, lw=0.7):
    ax.add_patch(Circle((cx, cy), 0.55 * s, fc='none', ec=c, lw=lw, zorder=6))
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ax.plot([cx + 0.55 * s * dx, cx + 0.95 * s * dx],
                [cy + 0.55 * s * dy, cy + 0.95 * s * dy], color=c, lw=lw, zorder=6)


def icon_layers(cx, cy, s, c, lw=0.7):
    for k, dy in enumerate((0.55, 0.0, -0.55)):
        w = (1.0 - 0.18 * k) * s
        ax.plot([cx - w, cx + w], [cy + dy * s] * 2, color=c, lw=lw, zorder=6)


def icon_checkc(cx, cy, s, c, lw=0.7):
    ax.add_patch(Circle((cx, cy), 0.75 * s, fc='none', ec=c, lw=lw, zorder=6))
    ax.plot([cx - 0.35 * s, cx - 0.08 * s, cx + 0.42 * s],
            [cy + 0.0 * s, cy - 0.32 * s, cy + 0.32 * s], color=c, lw=lw, zorder=6)


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



def icon_person(cx, cy, s, c):
    ax.add_patch(Circle((cx, cy + 0.5 * s), 0.42 * s, fc=c, ec='none', zorder=6))
    ax.add_patch(Ellipse((cx, cy - 0.45 * s), 1.5 * s, 1.05 * s, fc=c, ec='none', zorder=6))


# ================= outer container (QUITE genome) =================
shadow(25, 7.5, 118, 86, r=2.0)
box(25, 7.5, 118, 86, fc='#FFFEFB', ec='#1A1A1A', lw=1.85, r=1.6, z=1.8)
txt(84, 91.35, 'TrainAudit', fs=8.0, w='bold', c=BLUE)
ax.plot([38, 74.5], [91.35, 91.35], color=BLUE, lw=0.55, zorder=4)
ax.plot([93.5, 130], [91.35, 91.35], color=BLUE, lw=0.55, zorder=4)

# lane header bars (QUITE subsection cue)
ax.add_patch(Rectangle((27, 87.4), 79, 2.0, fc=BLUE_BG, ec='none', zorder=2))
txt(27.5, 88.4, 'OFFLINE — mine once per framework', fs=5.7, w='bold', c=BLUE, ha='left')
ax.add_patch(Rectangle((27, 49.8), 79, 2.0, fc=ORANGE_BG, ec='none', zorder=2))
txt(27.5, 50.8, 'ONLINE — every training job · fully deterministic · no LLM',
    fs=5.7, w='bold', c=ORANGE, ha='left')


def folded_doc(x, y, w, h, lines, title=None, title_c=DARK, ec=DARK, fc='#fafafa', hl=None):
    """QUITE-style folded-corner document with monospace code lines."""
    fold = min(2.2, w * 0.18)
    shadow(x, y, w, h, r=0.6)
    # body (clip fold by overlay)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.5',
                                fc=fc, ec=ec, lw=0.7, zorder=3))
    # fold triangle
    ax.add_patch(Polygon([(x + w - fold, y + h), (x + w, y + h - fold), (x + w - fold, y + h - fold)],
                         closed=True, fc='#E6E6E6', ec=ec, lw=0.55, zorder=4))
    ax.plot([x + w - fold, x + w - fold, x + w], [y + h, y + h - fold, y + h - fold],
            color=ec, lw=0.5, zorder=5)
    if title:
        txt(x + 0.8, y + h - 1.5, title, fs=3.6, w='bold', c=title_c, ha='left', z=5)
        y0 = y + h - 3.4
    else:
        y0 = y + h - 1.8
    for i, ln in enumerate(lines):
        fam = None if ('$' in ln or '\\' in ln) else 'monospace'
        cc = (hl or {}).get(i, '#333333')
        txt(x + 0.9, y0 - i * 1.85, ln, fs=3.15, family=fam, ha='left',
            c=cc, w='bold' if i in (hl or {}) else 'normal', z=5)


def dashed_group(x, y, w, h, fc, label=None, lc=GRAY):
    """QUITE soft pastel subsection with dashed border."""
    box(x, y, w, h, fc=fc, ec=lc, lw=0.7, r=1.0, ls=(0, (3, 2)), z=2)
    if label:
        txt(x + 1.2, y + h - 1.3, label, fs=3.6, w='bold', c=lc, ha='left', z=5)


def icon_agent(cx, cy, s, tool='bulb', c='#39404d'):
    """QUITE person+tool agent glyph (offline roles only)."""
    import numpy as _np
    ax.add_patch(Circle((cx, cy + 0.55 * s), 0.38 * s, fc=c, ec='none', zorder=6))
    ax.add_patch(Ellipse((cx, cy - 0.35 * s), 1.1 * s, 0.95 * s, fc=c, ec='none', zorder=6))
    if tool == 'bulb':
        ax.add_patch(Circle((cx + 0.85 * s, cy + 0.7 * s), 0.32 * s, fc='none', ec=c, lw=0.55, zorder=6))
        ax.plot([cx + 0.85 * s, cx + 0.85 * s], [cy + 0.38 * s, cy + 0.15 * s], color=c, lw=0.55, zorder=6)
    elif tool == 'pen':
        ax.plot([cx + 0.55 * s, cx + 1.15 * s], [cy + 0.15 * s, cy + 0.85 * s], color=c, lw=0.7, zorder=6)
        ax.plot([cx + 0.55 * s, cx + 0.7 * s], [cy + 0.15 * s, cy + 0.0 * s], color=c, lw=0.55, zorder=6)
    elif tool == 'shield':
        icon_shield(cx + 0.9 * s, cy + 0.55 * s, 0.55 * s, c, lw=0.55)
    elif tool == 'disk':
        ax.add_patch(Ellipse((cx + 0.9 * s, cy + 0.55 * s), 0.7 * s, 0.45 * s, fc='none', ec=c, lw=0.55, zorder=6))
        ax.plot([cx + 0.55 * s, cx + 1.25 * s], [cy + 0.55 * s, cy + 0.55 * s], color=c, lw=0.45, zorder=6)
    elif tool == 'report':
        icon_doc(cx + 0.85 * s, cy + 0.5 * s, 0.55 * s, c, lw=0.5)


def icon_brain(cx, cy, s, c, lw=0.6):
    from matplotlib.patches import Arc as _Arc
    ax.add_patch(Ellipse((cx, cy), 1.9 * s, 1.5 * s, fc='none', ec=c, lw=lw, zorder=6))
    ax.plot([cx, cx], [cy - 0.72 * s, cy + 0.72 * s], color=c, lw=lw * 0.8, zorder=6)
    ax.add_patch(_Arc((cx - 0.45 * s, cy + 0.12 * s), 0.7 * s, 0.5 * s,
                      theta1=0, theta2=180, ec=c, lw=lw * 0.7, zorder=6))
    ax.add_patch(_Arc((cx + 0.45 * s, cy - 0.12 * s), 0.7 * s, 0.5 * s,
                      theta1=180, theta2=360, ec=c, lw=lw * 0.7, zorder=6))


def red_x(cx, cy, s=1.0, c=RED):
    ax.plot([cx - s, cx + s], [cy - s, cy + s], color=c, lw=1.1, zorder=7)
    ax.plot([cx - s, cx + s], [cy + s, cy - s], color=c, lw=1.1, zorder=7)

# ================= left bay: catalog + source (not a numbered component) =================
txt(2.0, 91.5, 'offline inputs', fs=5.0, w='bold', c=BLUE, ha='left')
folded_doc(1.5, 70.5, 21, 18.5,
           lines=['template P3:',
                  '  pi_topo: TP > 1',
                  '  pi_precond: tpl=False',
                  '  pi_schema: share cksum',
                  '  (cross-rank replicas)',
                  'from 392 silent errors'],
           title='Pattern Catalog', title_c=BLUE, ec=BLUE, fc='#f7faff')
txt(12, 68.5, '16 templates · 13 fault classes', fs=3.3, c=GRAY)
folded_doc(1.5, 52.0, 21, 15.2,
           lines=['Megatron-LM',
                  'DeepSpeed',
                  'OLMo / OLMo-core',
                  'silent-vs-loud filter',
                  'no raise / assert'],
           title='Framework source', title_c=BLUE, ec='#5a7a9a', fc='#fafafa')
txt(12, 50.0, 'once per framework', fs=3.3, style='italic', c=GRAY)
arrow((23.0, 78), (28.6, 78), c=BLUE, lw=1.6, ms=9, halo=True)

# ================= Invariant Miner (offline row) =================
shadow(27, 53.5, 79, 33)
box(27, 53.5, 79, 33, fc='#f9fbfe', ec=DARK, lw=1.0)
badge(30.6, 83.9, '1', c='#1a1a1a')
txt(33.4, 83.9, 'Invariant Miner', fs=6.2, w='bold', ha='left')
chip(58.0, 83.9, 1.7, BLUE)
icon_gear(58.0, 83.9, 0.95, 'white')
txt(105, 83.9, 'LLM-instantiated candidates, untrusted until verified',
    fs=4.0, style='italic', c=GRAY, ha='right')

# plus-composition row (QUITE stage-2 genome)
plus_cards = [
    (29.0, icon_book, 'Pattern Catalog\ntemplate'),
    (48.0, icon_doc, 'framework\nsource & docs'),
    (67.0, icon_layers, 'evidence\n$\\mathcal{E}^{+}/\\mathcal{E}^{-}$'),
    (86.0, icon_shield, 'coverage gap\nin $\\mathcal{P}$'),
]
for px, ic, lb in plus_cards:
    box(px, 77.2, 15.5, 5.0, fc='white', ec=BLUE_SOFT, lw=0.55, r=0.7)
    ic(px + 2.6, 79.7, 0.85, BLUE)
    txt(px + 9.4, 79.7, lb, fs=3.3, ls=1.15)
for gx in (46.2, 65.2, 84.2):
    txt(gx, 79.7, '+', fs=6.5, w='bold', c=GRAY)
arrow((65.5, 77.0), (65.5, 76.3), c=BLUE, lw=0.9)

# pastel dashed groups (QUITE lavender/mint genome)
dashed_group(28.2, 64.8, 38.2, 12.2, LAVENDER, label='instantiate', lc='#7B6B95')
icon_brain(36.8, 75.6, 0.85, '#7B6B95')
dashed_group(66.8, 64.8, 37.8, 12.2, MINT, label='verify (decisive)', lc='#5B8A5B')
# four-step funnel
steps = [
    (29.0, 'Scope', 'fix training phase\n+ parameter class', DARK, 0.7, 'white'),
    (48.25, 'Ground', 'evidence $\\mathcal{E}^{+}/\\mathcal{E}^{-}$ from\nframework source & docs', DARK, 0.7, 'white'),
    (67.5, 'Construct', 'counterexample attack\n$\\geq$2 CEs; one CE holds\n$\\Rightarrow$ reject outright', RED, 1.15, PEACH),
    (86.75, 'Accept', 'no CE holds $\\wedge$\nConf$(c) \\geq 0.8$\n$(\\theta = 0.8)$', DARK, 0.7, 'white'),
]
for x, name, sub, ec, lw, fc in steps:
    box(x, 65.5, 16.5, 10.3, fc=fc, ec=ec, lw=lw)
    txt(x + (9.3 if name == 'Construct' else 8.9), 73.5, name, fs=5.2, w='bold', c=ec)
    ax.plot([x + 2.2, x + 14.3], [72.3, 72.3], color=ec, lw=0.45, alpha=0.65, zorder=5)
    txt(x + 8.25, 68.7, sub, fs=3.6, ls=1.3)
    if name == 'Scope':
        icon_target(x + 3.2, 73.5, 0.95, DARK)
    elif name == 'Ground':
        icon_layers(x + 3.2, 73.5, 0.9, DARK)
    elif name == 'Accept':
        icon_checkc(x + 3.4, 73.5, 0.95, DARK)
    elif name == 'Construct':
        chip(x + 3.0, 73.5, 1.35, RED)
        icon_shield(x + 3.0, 73.55, 0.8, 'white')
for x0 in (45.8, 65.05, 84.3):
    arrow((x0, 70.65), (x0 + 2.2, 70.65), c=BLUE, lw=0.9)

# reject loop under the funnel
arrow((95.0, 65.3), (95.0, 63.6), c=RED, lw=0.8, ls=(0, (3, 2)), style='-')
arrow((95.0, 63.6), (37.3, 63.6), c=RED, lw=0.8, ls=(0, (3, 2)), style='-')
arrow((37.3, 63.6), (37.3, 65.2), c=RED, lw=0.8, ls=(0, (3, 2)))
red_x(99.2, 63.55, 0.7)
txt(66, 62.4, 'reject $\\rightarrow$ next iteration', fs=3.8, c=RED, style='italic')

# FSM chain
txt(29.3, 60.6, 'five-stage\nFSM', fs=3.4, c=GRAY, ha='left', ls=1.2)
fsm = [('$S_1$', 'gap analysis', '#6f8fc4'), ('$S_2$', 'evidence retrieval', '#7fae9e'),
       ('$S_3$', 'synthesis+attack', '#c48a8a'), ('$S_4$', 'persistence', '#a795c2'),
       ('$S_5$', 'reporting', '#c2a370')]
for k, (sn, lb, cc) in enumerate(fsm):
    cx = 40.5 + k * 8.6
    ax.add_patch(Circle((cx, 59.9), 1.5, fc=cc, ec='none', zorder=5))
    txt(cx, 59.9, sn, fs=3.5, c='white', w='bold', z=6)
    txt(cx, 57.2, lb, fs=2.9, c=GRAY)
    if k < 4:
        arrow((cx + 1.75, 59.9), (cx + 6.6, 59.9), c='#9aa5b5', lw=0.55, ms=4)

# QUITE-style credential chips
cred = [(30.0, 'decisive counterexample gate'),
        (56.0, 'healthy-run replay (verification only)'),
        (85.0, 'Conf$(c) \\geq 0.8$')]
for cx0, tt in cred:
    checkbox(cx0, 54.3, 1.3, GREEN)
    txt(cx0 + 2.0, 54.95, tt, fs=3.6, c='#0E6E6C', ha='left')

# accepted -> library
arrow((103.3, 70.65), (110.5, 74.5), c=BLUE, lw=1.6, ms=9, cs='arc3,rad=-0.2', halo=True)
txt(107.5, 77.6, 'accepted', fs=3.9, c=BLUE, style='italic', ha='right')

# ================= Verified Constraint Library (right column) =================
shadow(108.5, 44, 32.5, 42.5)
box(108.5, 44, 32.5, 42.5, fc=HUB_BG, ec=DARK, lw=1.0)
cyl(120.5, 80.2, 8.5, 5.2, fc='#e9e2d2', ec=DARK)
txt(124.7, 77.4, 'Verified Constraint Library $\\mathcal{P}$', fs=4.8, w='bold')
txt(124.7, 75.2, 'guarded relational constraints', fs=3.8, style='italic', c=GRAY)
shadow(110.5, 52, 28.5, 21.3, r=1.0)
box(110.5, 52, 28.5, 21.3, fc='white', ec=DARK, lw=0.7, r=1.0)
box(110.5, 68.8, 28.5, 4.5, fc='#39404d', ec='none', r=1.0, z=3)
ax.add_patch(Rectangle((110.5, 68.8), 28.5, 2.2, fc='#39404d', ec='none', zorder=3))
txt(124.7, 71.9, 'P3 · cross-rank replication', fs=4.6, w='bold', c='white', z=5)
txt(124.7, 69.8, '(SwitchMLP router sync)', fs=3.6, c='#c9cedb', z=5)
txt(112, 60.6, '$\\pi_{topo}$:  TP > 1\n$\\pi_{precond}$:  tpl = False\n   (not tensor-parallel sharded)\n$\\pi_{schema}$:  replicas share\n   checksum',
    fs=4.0, ha='left', ls=1.35)
txt(124.7, 50.7, 'compiled to parameterized SQL at job start', fs=3.5, c=ORANGE, w='bold')
pill(110.5, 46.3, 28.5, 3.3, '', GREEN_BG, GREEN)
txt(124.7, 47.95, '$\\checkmark$  0 false positives over 764 clean evals', fs=3.7, c='#0E6E6C', z=6)
arrow((114, 43.8), (100.5, 41.5), c=ORANGE, lw=1.6, ms=9, cs='arc3,rad=0.15', halo=True)
arrow((108.5, 55), (46, 48.9), c=ORANGE, lw=0.8, cs='arc3,rad=0.12')
txt(105.5, 52.6, 'accepted rules determine trace schema (S0–S6)', fs=3.6, c=ORANGE, ha='right')

# ================= Data Collector (online row) =================
shadow(27, 15.5, 37, 33)
box(27, 15.5, 37, 33, fc='#fffdf9', ec=DARK, lw=1.0)
badge(30.6, 45.9, '2', c='#1a1a1a')
txt(33.4, 45.9, 'Data Collector', fs=6.2, w='bold', ha='left')
chip(56.5, 45.9, 1.6, ORANGE)
icon_probe(56.5, 45.9, 0.9, 'white')
txt(28, 43.5, 'no user code changes · adapter 30–150 LoC', fs=3.8,
    style='italic', c=GRAY, ha='left')

anchors = ['before-\nforward', 'after-\nforward', 'main-grad-\nin-backward',
           'after-\nbackward', 'before-\noptimizer']
tints = ['#fbe3c9', '#f8d5ac', '#f4c48c', '#efb26d', '#e99f4e']
for i, a in enumerate(anchors):
    x = 28.5 + i * 6.9
    w, n = 7.6, 1.3
    pts = [(x, 36.2), (x + w - n, 36.2), (x + w, 39.0), (x + w - n, 41.8),
           (x, 41.8), (x + n, 39.0)]
    ax.add_patch(Polygon(pts, closed=True, fc=tints[i], ec='none', zorder=3))
    txt(x + w / 2 + 0.2, 39.0, a, fs=3.3, family='monospace', z=5)
txt(45.5, 33.7, '+ auxiliary taps (ckpt save/load · all_reduce ·\nbuild snapshot) = 8 hookpoints', fs=3.5, c=GRAY, ls=1.25)
txt(45.5, 30.3, 'GPU-side scalar reductions\n(checksums only when required)', fs=3.6, ls=1.25)
box(29, 17.5, 33, 7.2, fc='#fdf8f0', ec=ORANGE, lw=0.7)
txt(45.5, 23.2, 'captured record (S0):', fs=3.8, c=GRAY)
_pw = [('cksum', 4.9), ('param_name', 7.7), ('rank', 4.1), ('step', 4.1), ('stage', 4.9)]
_x = 29 + (33 - (sum(w for _, w in _pw) + 0.7 * 4)) / 2
for _t, _w in _pw:
    pill(_x, 18.6, _w, 2.8, _t, 'white', ORANGE, fs=3.3)
    _x += _w + 0.7

# DuckDB
cyl(65.5, 26, 8.5, 11, fc='#fbe9d4', ec=ORANGE)
for _yy in (29.5, 32.5):
    ax.plot([66.3, 73.2], [_yy, _yy], color=ORANGE, lw=0.4, alpha=0.5, zorder=4)
txt(69.6, 23.6, 'DuckDB\ntrace DB', fs=4.0, w='bold', c=ORANGE, ls=1.15)
txt(69.6, 19.6, 'two-step\nsliding window', fs=3.3, c=GRAY, ls=1.15)
arrow((64.2, 31.5), (65.3, 31.5), c=ORANGE, lw=1.0)
arrow((74.2, 31.5), (75.3, 31.5), c=ORANGE, lw=1.0)

# ================= Verifier =================
shadow(75.5, 15.5, 30.5, 33)
box(75.5, 15.5, 30.5, 33, fc='#fffdf9', ec=DARK, lw=1.0)
badge(79.1, 45.9, '3', c='#1a1a1a')
txt(81.9, 45.9, 'Verifier', fs=6.2, w='bold', ha='left')
chip(92.5, 45.9, 1.6, ORANGE)
icon_search(92.5, 45.9, 0.9, 'white')
txt(76.5, 43.5, 'deterministic SQL checking — no LLM', fs=3.8,
    style='italic', c=GRAY, ha='left')
box(77, 27, 13, 14, fc=MINT, ec='#5B8A5B', lw=0.65, ls=(0, (2.5, 1.8)))
txt(83.5, 39.5, 'at job start', fs=4.2, w='bold')
ax.plot([78.2, 88.8], [38.4, 38.4], color='#555b63', lw=0.5, zorder=5)
txt(83.5, 32.6, 'extract topology $\\tau$\ntopology-aware\npruning · lineage:\nreplicated vs.\nsharded', fs=3.5, ls=1.3)
box(91.5, 27, 13, 14, fc='#2b313d', ec='#1f242e', lw=0.7)
txt(98, 39.5, 'per step, per rule', fs=4.1, w='bold', c='#9db8e8')
ax.plot([92.7, 103.3], [38.4, 38.4], color='#4a5568', lw=0.5, zorder=5)
_KW, _TX, _PI = '#61afef', '#d7dae0', '#e5c07b'
txt(92.3, 36.6, 'WHERE', fs=3.4, ha='left', family='monospace', c=_KW)
txt(96.0, 36.6, '$\\pi_{topo} \\wedge \\pi_{precond}$', fs=3.4, ha='left', c=_PI)
txt(92.3, 34.2, 'GROUP BY', fs=3.4, ha='left', family='monospace', c=_KW)
txt(92.9, 32.2, 'param, step', fs=3.4, ha='left', family='monospace', c=_TX)
txt(92.3, 30.0, 'HAVING', fs=3.4, ha='left', family='monospace', c=_KW)
txt(96.3, 30.0, '$\\pi_{schema}$', fs=3.4, ha='left', c=_PI)
txt(92.9, 28.2, 'violated', fs=3.4, ha='left', family='monospace', c=_TX)
txt(90.75, 23.9, 'empty result = holds\nnon-empty result IS the violation set', fs=3.7, style='italic', ls=1.3)
txt(90.75, 19.4, 'coarse$\\rightarrow$fine: checksum $\\rightarrow$ module\n$\\rightarrow$ parameter $\\rightarrow$ bitwise', fs=3.4, c=GRAY, ls=1.25)
arrow((106.2, 29), (108.3, 29), c=RED, lw=1.5, ms=9, halo=True)

# ================= Violation panel (dashed, QUITE termination genome) =================
shadow(108.5, 10.5, 32.5, 31.5)
box(108.5, 10.5, 32.5, 31.5, fc='#FDF0EF', ec=RED, lw=1.0, ls=(0, (3.5, 2)))
txt(124.7, 39.3, 'violation report', fs=5.0, w='bold', c=RED)
ax.plot([117, 132.4], [38.0, 38.0], color=RED, lw=0.5, zorder=5)
_tx, _cols = 111.0, [('param', 8.4), ('rank', 3.2), ('cksum', 7.0)]
_rows = [('router.weight', '0', 'cksum$_A$', False), ('router.weight', '1', 'cksum$_B$', True)]
_ty, _rh = 36.6, 2.15
_cx = _tx
for _h, _wd in _cols:
    ax.add_patch(Rectangle((_cx, _ty - _rh), _wd, _rh, fc='#f0f0f0', ec='#c8c8c8', lw=0.35, zorder=4))
    txt(_cx + _wd / 2, _ty - _rh / 2, _h, fs=3.2, c=GRAY, z=5)
    _cx += _wd
for _r, (_p, _rk, _ck, _bad) in enumerate(_rows):
    _cx = _tx
    _yy = _ty - _rh * (_r + 2)
    for _v, (_h, _wd) in zip((_p, _rk, _ck), _cols):
        ax.add_patch(Rectangle((_cx, _yy), _wd, _rh, fc='#fdecec' if _bad else 'white',
                               ec='#c8c8c8', lw=0.35, zorder=4))
        txt(_cx + _wd / 2, _yy + _rh / 2, _v, fs=3.2, family='monospace',
            c=RED if _bad else DARK, z=5)
        _cx += _wd
txt(135.5, 31.4, '$\\mathbf{\\times}$ divergent', fs=3.7, c=RED)
for _k, _it in enumerate(['violating constraint', 'affected parameter',
                          'divergent ranks', 'first violating step']):
    _yy = 26.6 - _k * 2.6
    checkbox(111.8, _yy - 0.6, 1.2, RED)
    txt(113.8, _yy, _it, fs=3.8, ha='left')
arrow((108.5, 12.6), (23.6, 17.5), c=DARK, lw=1.6, ms=9, cs='arc3,rad=-0.06', halo=True)

# ================= left bay: P3 violation report artifact (not a numbered component) =================
txt(2.0, 40.5, 'online output', fs=5.0, w='bold', c=ORANGE, ha='left')
folded_doc(1.5, 18.5, 21, 19.5,
           lines=['violating constraint: P3',
                  'affected parameter:',
                  '  router.weight',
                  'divergent ranks: 0 vs 1',
                  '(SwitchMLP router sync)',
                  'non-empty SQL result'],
           title='Violation Report', title_c=RED, ec=RED, fc='#fff8f8',
           hl={0: RED, 2: RED})
txt(12, 15.5, 'empty result = holds', fs=3.8, style='italic', c=GRAY)
txt(12, 12.8, 'non-empty = violation set', fs=3.8, style='italic', c=RED)

# ================= legend (arrow types only; bloodline-safe) =================
ax.add_patch(Rectangle((25, 2.2), 117.5, 3.6, fc='#FAFAFA', ec='#D0D0D0', lw=0.4, zorder=2))
txt(26.5, 4.0, 'Legend', fs=4.2, w='bold', ha='left')
ax.plot([36, 43], [4.0, 4.0], color=BLUE, lw=1.55)
txt(44.5, 4.0, 'offline mining', fs=3.9, ha='left')
ax.plot([62, 69], [4.0, 4.0], color=ORANGE, lw=1.55)
txt(70.5, 4.0, 'online checking', fs=3.9, ha='left')
ax.plot([92, 99], [4.0, 4.0], color=RED, lw=1.25, ls=(0, (3, 2)))
txt(100.5, 4.0, 'reject / violation', fs=3.9, ha='left')
txt(141.5, 4.0, 'LLM only inside offline Miner · online path: no LLM',
    fs=3.7, style='italic', c=GRAY, ha='right')

fig.savefig('fig_overview_v12.pdf')
fig.savefig('/tmp/fig_overview_v12.png', dpi=220)
print('done')
