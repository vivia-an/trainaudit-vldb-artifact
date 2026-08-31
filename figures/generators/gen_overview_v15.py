#!/usr/bin/env python3
"""v15 = v14 bloodline-safe + QUITE premium feel (wash / soft icons / sans / chrome).
Labels identical to v14 (=v13). No Agents row, no 0/4, no 43k, no fake hex.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Rectangle, Circle, Polygon
from matplotlib import rcParams

rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'],
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,
})

# QUITE pixel DNA + premium wash ladder
BLUE = '#306092'
BLUE_SOFT = '#6C8EBF'
BLUE_DEEP = '#23445D'
BLUE_BG = '#EEF3F9'       # offline lane wash (QUITE soft blue)
ORANGE = '#C05621'
ORANGE_BG = '#FBF6F0'     # online lane wash
ORANGE_SOFT = '#E8A87C'
RED = '#B85450'
RED_DEEP = '#AE4132'
GREEN = '#1AABA8'
GREEN_BG = '#E8F7F6'
DARK = '#2C2C2C'
GRAY = '#666666'
GRAY_LINE = '#9A9A9A'
CARD = '#FFFEFB'          # warm white (PPT card)
HUB_BG = '#F7F4EF'        # beige hub (QUITE knowledge column)
VIOL_BG = '#FDF0EE'
LAVENDER = '#E1D5E7'
PURPLE = '#9673A6'
MINT = '#D5E8D4'
PEACH = '#FAD9D5'
CREAM = '#FFF8E7'         # QUITE yellow-ish inner zone
# stroke ladder (QUITE: 0.31 / 0.63 / 0.94 / 1.57)
LW_HAIR, LW_THIN, LW_MID, LW_THICK = 0.31, 0.63, 0.94, 1.57

W, H = 144, 100
fig = plt.figure(figsize=(7.2, 5.0))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis('off')


def box(x, y, w, h, fc=CARD, ec=DARK, lw=None, r=1.0, ls='-', z=2):
    if lw is None:
        lw = LW_THIN
    p = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}',
                       fc=fc, ec=ec, lw=lw, ls=ls, zorder=z,
                       joinstyle='round', capstyle='round')
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


def arrow(p0, p1, c=DARK, lw=None, ls='-', style='-|>', ms=8, cs=None, z=4,
          halo=False):
    if lw is None:
        lw = LW_MID
    # QUITE arrows are filled triangular heads on ~0.94pt stems
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                        color=c, lw=lw, linestyle=ls, zorder=z,
                        connectionstyle=cs or 'arc3,rad=0',
                        shrinkA=0.5, shrinkB=0.5,
                        joinstyle='miter', capstyle='butt')
    if halo:
        a.set_path_effects([_pe.Stroke(linewidth=lw + 1.6, foreground='white'),
                            _pe.Normal()])
    ax.add_patch(a)


def cyl(x, y, w, h, fc='#dbe7f3', ec=BLUE, z=3):
    # soft-banded cylinder (QUITE DB chrome)
    ax.add_patch(Rectangle((x, y + h * 0.12), w, h * 0.76, fc=fc, ec='none', zorder=z))
    ax.add_patch(Ellipse((x + w / 2, y + h * 0.12), w, h * 0.24, fc=fc, ec=ec, lw=LW_THIN, zorder=z))
    ax.plot([x, x], [y + h * 0.12, y + h * 0.88], color=ec, lw=LW_THIN, zorder=z + 1)
    ax.plot([x + w, x + w], [y + h * 0.12, y + h * 0.88], color=ec, lw=LW_THIN, zorder=z + 1)
    ax.add_patch(Ellipse((x + w / 2, y + h * 0.88), w, h * 0.24, fc='#FFFFFF', ec=ec, lw=LW_THIN, zorder=z + 2))
    ax.add_patch(Ellipse((x + w / 2, y + h * 0.55), w * 0.92, h * 0.14,
                         fc='white', ec='none', alpha=0.22, zorder=z + 1))


def badge(x, y, n, c='#000000', r=1.75, fs=5.6):
    ax.add_patch(Circle((x + 0.08, y - 0.1), r, fc='#000000', ec='none', alpha=0.12, zorder=5.8))
    ax.add_patch(Circle((x, y), r, fc=c, ec='none', zorder=6))
    txt(x, y - 0.04, n, fs=fs, w='bold', c='white', z=7)


def shadow(x, y, w, h, r=1.0):
    ax.add_patch(FancyBboxPatch((x + 0.55, y - 0.65), w, h,
                 boxstyle=f'round,pad=0,rounding_size={r}',
                 fc='#000000', ec='none', alpha=0.05, zorder=1.55))
    ax.add_patch(FancyBboxPatch((x + 0.22, y - 0.28), w, h,
                 boxstyle=f'round,pad=0,rounding_size={r}',
                 fc='#000000', ec='none', alpha=0.035, zorder=1.6))


def chip(cx, cy, s, c):
    ax.add_patch(FancyBboxPatch((cx - s, cy - s), 2 * s, 2 * s,
                 boxstyle='round,pad=0,rounding_size=0.5',
                 fc=c, ec=c, lw=LW_HAIR, zorder=5.5))
    ax.add_patch(FancyBboxPatch((cx - s + 0.12, cy - s + 0.12), 2 * s - 0.24, 0.55,
                 boxstyle='round,pad=0,rounding_size=0.2',
                 fc='white', ec='none', alpha=0.32, zorder=5.6))


def pill(x, y, w, h, t, fc, ec, tc=DARK, fs=3.8):
    box(x, y, w, h, fc=fc, ec=ec, lw=LW_HAIR, r=min(h / 2, 1.05), z=4)
    txt(x + w / 2, y + h / 2, t, fs=fs, family='monospace', c=tc, z=5)


def checkbox(x, y, sz, c):
    # QUITE-style solid teal/red check disc
    ax.add_patch(Circle((x + sz / 2, y + sz / 2), sz * 0.55, fc=c, ec='none', zorder=5))
    ax.plot([x + 0.22 * sz, x + 0.42 * sz, x + 0.82 * sz],
            [y + 0.48 * sz, y + 0.22 * sz, y + 0.78 * sz], color='white', lw=LW_MID, zorder=6,
            solid_capstyle='round', solid_joinstyle='round')


def icon_target(cx, cy, s, c, lw=None):
    if lw is None:
        lw = LW_THIN
    ax.add_patch(Circle((cx, cy), 0.55 * s, fc='#EEF3F9', ec=c, lw=lw, zorder=6))
    ax.add_patch(Circle((cx, cy), 0.22 * s, fc=c, ec='none', zorder=6))
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ax.plot([cx + 0.55 * s * dx, cx + 0.95 * s * dx],
                [cy + 0.55 * s * dy, cy + 0.95 * s * dy], color=c, lw=lw, zorder=6)


def icon_layers(cx, cy, s, c, lw=None):
    if lw is None:
        lw = LW_THIN
    for k, (dy, alpha) in enumerate(((0.55, 0.35), (0.0, 0.55), (-0.55, 0.8))):
        w = (1.0 - 0.18 * k) * s
        ax.add_patch(FancyBboxPatch((cx - w, cy + dy * s - 0.22 * s), 2 * w, 0.44 * s,
                     boxstyle='round,pad=0,rounding_size=0.12',
                     fc=c, ec='none', alpha=alpha, zorder=6))


def icon_checkc(cx, cy, s, c, lw=None):
    if lw is None:
        lw = LW_THIN
    ax.add_patch(Circle((cx, cy), 0.75 * s, fc=GREEN, ec='none', zorder=6))
    ax.plot([cx - 0.35 * s, cx - 0.08 * s, cx + 0.42 * s],
            [cy + 0.0 * s, cy - 0.32 * s, cy + 0.32 * s], color='white', lw=lw * 1.2, zorder=6)


# ---------- soft-fill icons (QUITE PPT chrome) ----------
def icon_book(cx, cy, s, c, lw=None):
    if lw is None:
        lw = LW_THIN
    for sgn, fc in ((-1, '#D6E4F5'), (1, '#EAF1FA')):
        xs = [cx, cx + sgn * 1.05 * s, cx + sgn * 1.05 * s, cx]
        ys = [cy + 0.75 * s, cy + 0.55 * s, cy - 0.7 * s, cy - 0.55 * s]
        ax.add_patch(Polygon(list(zip(xs, ys)), closed=True, fc=fc, ec=c, lw=lw, zorder=6))
    ax.plot([cx, cx], [cy - 0.55 * s, cy + 0.75 * s], color=c, lw=lw, zorder=7)

def icon_doc(cx, cy, s, c, lw=None):
    if lw is None:
        lw = LW_THIN
    w, h, f = 0.85 * s, 1.15 * s, 0.35 * s
    xs = [cx - w, cx + w - f, cx + w, cx + w, cx - w, cx - w]
    ys = [cy + h, cy + h, cy + h - f, cy - h, cy - h, cy + h]
    ax.add_patch(Polygon(list(zip(xs, ys)), closed=True, fc='#F4F8FC', ec=c, lw=lw, zorder=6))
    ax.add_patch(Polygon([(cx + w - f, cy + h), (cx + w, cy + h - f), (cx + w - f, cy + h - f)],
                         closed=True, fc='#D0D8E4', ec=c, lw=LW_HAIR, zorder=7))
    # header band
    ax.add_patch(Rectangle((cx - w + 0.05 * s, cy + 0.35 * s), 2 * w - f - 0.1 * s, 0.55 * s,
                           fc=c, ec='none', alpha=0.18, zorder=6.5))
    for k, dy in enumerate((0.15, -0.25, -0.55)):
        ax.plot([cx - 0.5 * w, cx + (0.5 - 0.15 * k) * w], [cy + dy * s] * 2,
                color=c, lw=lw * 0.75, alpha=0.85, zorder=7)

def icon_gear(cx, cy, s, c, lw=None):
    if lw is None:
        lw = LW_THIN
    import numpy as _np
    ax.add_patch(Circle((cx, cy), 0.62 * s, fc='#EAF1FA' if c == 'white' else '#D6E4F5',
                        ec=c if c != 'white' else BLUE, lw=lw, zorder=6))
    ax.add_patch(Circle((cx, cy), 0.26 * s, fc='white', ec=c if c != 'white' else BLUE,
                        lw=lw * 0.85, zorder=6))
    ec = c if c != 'white' else 'white'
    for th in _np.arange(0, 360, 45):
        r = _np.deg2rad(th)
        ax.plot([cx + 0.62 * s * _np.cos(r), cx + 0.95 * s * _np.cos(r)],
                [cy + 0.62 * s * _np.sin(r), cy + 0.95 * s * _np.sin(r)],
                color=ec, lw=lw * 1.15, zorder=6, solid_capstyle='round')

def icon_shield(cx, cy, s, c, lw=None):
    if lw is None:
        lw = LW_THIN
    xs = [cx - 0.75 * s, cx + 0.75 * s, cx + 0.75 * s, cx, cx - 0.75 * s]
    ys = [cy + 0.75 * s, cy + 0.75 * s, cy - 0.05 * s, cy - 0.95 * s, cy - 0.05 * s]
    fc = 'white' if c == 'white' else '#E8F7F6'
    ec = c if c != 'white' else '#FFFFFF'
    ax.add_patch(Polygon(list(zip(xs, ys)), closed=True, fc=fc, ec=ec, lw=lw, zorder=6))
    ax.plot([cx - 0.32 * s, cx - 0.05 * s, cx + 0.42 * s],
            [cy + 0.02 * s, cy - 0.3 * s, cy + 0.38 * s],
            color=GREEN if c != 'white' else 'white', lw=lw, zorder=7)

def icon_replay(cx, cy, s, c, lw=None):
    if lw is None:
        lw = LW_THIN
    from matplotlib.patches import Arc as _Arc
    ax.add_patch(Circle((cx, cy), 0.85 * s, fc='#EEF3F9', ec='none', zorder=5.5))
    ax.add_patch(_Arc((cx, cy), 1.7 * s, 1.7 * s, angle=0, theta1=300, theta2=210,
                      ec=c, lw=lw, zorder=6))
    import numpy as _np
    r = _np.deg2rad(210)
    tipx, tipy = cx + 0.85 * s * _np.cos(r), cy + 0.85 * s * _np.sin(r)
    ax.add_patch(Polygon([(tipx + 0.32 * s, tipy + 0.1 * s),
                          (tipx - 0.22 * s, tipy + 0.28 * s),
                          (tipx + 0.05 * s, tipy - 0.38 * s)],
                         closed=True, fc=c, ec='none', zorder=6))

def icon_search(cx, cy, s, c, lw=None):
    if lw is None:
        lw = LW_THIN
    ec = c if c != 'white' else 'white'
    ax.add_patch(Circle((cx - 0.25 * s, cy + 0.25 * s), 0.62 * s,
                        fc='#EAF1FA' if c != 'white' else 'none',
                        ec=ec, lw=lw, zorder=6, alpha=0.95 if c != 'white' else 1))
    ax.plot([cx + 0.22 * s, cx + 0.85 * s], [cy - 0.22 * s, cy - 0.85 * s],
            color=ec, lw=lw * 1.25, zorder=6, solid_capstyle='round')
    if c != 'white':
        ax.plot([cx - 0.5 * s, cx - 0.28 * s, cx + 0.08 * s],
                [cy + 0.22 * s, cy - 0.02 * s, cy + 0.52 * s], color=GREEN, lw=lw * 0.9, zorder=6)

def icon_probe(cx, cy, s, c, lw=None):
    if lw is None:
        lw = LW_THIN
    ec = c if c != 'white' else 'white'
    ax.add_patch(FancyBboxPatch((cx - 1.05 * s, cy + 0.35 * s), 2.1 * s, 0.4 * s,
                 boxstyle='round,pad=0,rounding_size=0.12',
                 fc='#F8EDE0' if c != 'white' else 'none', ec=ec, lw=lw, zorder=6))
    ax.plot([cx, cx], [cy + 0.55 * s, cy - 0.35 * s], color=ec, lw=lw, zorder=6)
    ax.add_patch(Circle((cx, cy - 0.6 * s), 0.28 * s, fc=ec, ec='none', zorder=6))
    ax.add_patch(Circle((cx, cy + 0.55 * s), 0.16 * s, fc=ec, ec='none', zorder=6))


def icon_person(cx, cy, s, c):
    ax.add_patch(Circle((cx, cy + 0.5 * s), 0.42 * s, fc=c, ec='none', zorder=6))
    ax.add_patch(Ellipse((cx, cy - 0.45 * s), 1.5 * s, 1.05 * s, fc=c, ec='none', zorder=6))


# ================= outer container (QUITE genome) =================
shadow(25, 8.0, 118, 89, r=1.6)
box(25, 8.0, 118, 89, fc='#FFFFFF', ec='#000000', lw=LW_THICK, r=1.35, z=1.8)
# lane washes inside outer (premium air — QUITE pastel zones)
box(26.2, 52.2, 80.6, 40.4, fc=BLUE_BG, ec='none', r=1.1, z=1.85)
box(26.2, 10.2, 80.6, 41.0, fc=ORANGE_BG, ec='none', r=1.1, z=1.85)
box(108.0, 10.5, 33.8, 79.5, fc=HUB_BG, ec='none', r=1.1, z=1.85)
txt(84, 94.8, 'TrainAudit', fs=8.2, w='bold', c=BLUE)
ax.plot([38, 74.5], [94.8, 94.8], color=BLUE_SOFT, lw=LW_HAIR, zorder=4)
ax.plot([93.5, 130], [94.8, 94.8], color=BLUE_SOFT, lw=LW_HAIR, zorder=4)

# lane header bars (module titles nest in wash)
ax.add_patch(FancyBboxPatch((27, 90.5), 79, 2.3,
             boxstyle='round,pad=0,rounding_size=0.45',
             fc='#DCE6F2', ec='none', zorder=2))
txt(27.5, 91.65, 'OFFLINE — mine once per framework', fs=5.7, w='bold', c=BLUE, ha='left')
ax.add_patch(FancyBboxPatch((27, 49.5), 79, 2.3,
             boxstyle='round,pad=0,rounding_size=0.45',
             fc='#F3E4D4', ec='none', zorder=2))
txt(27.5, 50.65, 'ONLINE — every training job · fully deterministic · no LLM',
    fs=5.7, w='bold', c=ORANGE, ha='left')


def folded_doc(x, y, w, h, lines, title=None, title_c=DARK, ec=DARK, fc='#fafafa', hl=None):
    """QUITE-style folded-corner document with soft header band."""
    fold = min(2.2, w * 0.18)
    shadow(x, y, w, h, r=0.7)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.65',
                                fc=fc, ec=ec, lw=LW_THIN, zorder=3,
                                joinstyle='round'))
    if title:
        ax.add_patch(FancyBboxPatch((x + 0.15, y + h - 2.7), w - 0.3, 2.4,
                     boxstyle='round,pad=0,rounding_size=0.4',
                     fc=title_c, ec='none', alpha=0.12, zorder=3.5))
        txt(x + 0.8, y + h - 1.45, title, fs=3.7, w='bold', c=title_c, ha='left', z=5)
        y0 = y + h - 3.55
    else:
        y0 = y + h - 1.8
    ax.add_patch(Polygon([(x + w - fold, y + h), (x + w, y + h - fold), (x + w - fold, y + h - fold)],
                         closed=True, fc='#E6E6E6', ec=ec, lw=LW_HAIR, zorder=4))
    ax.plot([x + w - fold, x + w - fold, x + w], [y + h, y + h - fold, y + h - fold],
            color=ec, lw=LW_HAIR, zorder=5)
    for i, ln in enumerate(lines):
        fam = None if ('$' in ln or '\\' in ln) else 'monospace'
        cc = (hl or {}).get(i, '#333333')
        txt(x + 0.9, y0 - i * 1.85, ln, fs=3.15, family=fam, ha='left',
            c=cc, w='bold' if i in (hl or {}) else 'normal', z=5)


def dashed_group(x, y, w, h, fc, label=None, lc=GRAY):
    """QUITE soft pastel subsection — thin dashed border (0.63pt)."""
    box(x, y, w, h, fc=fc, ec=lc, lw=LW_THIN, r=0.85, ls=(0, (2.2, 1.6)), z=2)
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


def icon_brain(cx, cy, s, c, lw=None):
    if lw is None:
        lw = LW_THIN
    from matplotlib.patches import Arc as _Arc
    ax.add_patch(Ellipse((cx, cy), 1.9 * s, 1.5 * s, fc=LAVENDER, ec=c, lw=lw, zorder=6))
    ax.plot([cx, cx], [cy - 0.72 * s, cy + 0.72 * s], color=c, lw=lw * 0.8, zorder=6)
    ax.add_patch(_Arc((cx - 0.45 * s, cy + 0.12 * s), 0.7 * s, 0.5 * s,
                      theta1=0, theta2=180, ec=c, lw=lw * 0.7, zorder=6))
    ax.add_patch(_Arc((cx + 0.45 * s, cy - 0.12 * s), 0.7 * s, 0.5 * s,
                      theta1=180, theta2=360, ec=c, lw=lw * 0.7, zorder=6))


def red_x(cx, cy, s=1.0, c=None):
    if c is None:
        c = RED_DEEP
    ax.plot([cx - s, cx + s], [cy - s, cy + s], color=c, lw=LW_MID, zorder=7,
            solid_capstyle='round')
    ax.plot([cx - s, cx + s], [cy + s, cy - s], color=c, lw=LW_MID, zorder=7,
            solid_capstyle='round')

# ================= left bay: catalog + source (not a numbered component) =================
txt(2.0, 94.8, 'offline inputs', fs=5.0, w='bold', c=BLUE, ha='left')
folded_doc(1.5, 73.0, 21, 19.0,
           lines=['template P3:',
                  '  pi_topo: TP > 1',
                  '  pi_precond: tpl=False',
                  '  pi_schema: share cksum',
                  '  (cross-rank replicas)',
                  'from 392 silent errors'],
           title='Pattern Catalog', title_c=BLUE, ec=BLUE, fc='#f7faff')
txt(12, 70.8, '16 templates · 13 fault classes', fs=3.3, c=GRAY)
folded_doc(1.5, 53.5, 21, 15.5,
           lines=['Megatron-LM',
                  'DeepSpeed',
                  'OLMo / OLMo-core',
                  'silent-vs-loud filter',
                  'no raise / assert'],
           title='Framework source', title_c=BLUE, ec='#5a7a9a', fc='#fafafa')
txt(12, 51.5, 'once per framework', fs=3.3, style='italic', c=GRAY)
arrow((23.0, 81), (28.6, 81), c=BLUE_SOFT, lw=LW_THICK, ms=10, halo=True)

# ================= Invariant Miner (offline row) =================
shadow(27, 54.5, 79, 35.5)
box(27, 54.5, 79, 35.5, fc=CARD, ec=DARK, lw=LW_MID, r=1.15)
badge(30.6, 87.2, '1')
txt(33.4, 87.2, 'Invariant Miner', fs=6.2, w='bold', c=BLUE_DEEP, ha='left')
chip(52.8, 87.2, 1.55, BLUE)
icon_gear(52.8, 87.2, 0.9, 'white')
txt(104.5, 87.2, 'LLM-instantiated candidates, untrusted until verified',
    fs=3.8, style='italic', c=GRAY, ha='right')

plus_cards = [
    (29.0, icon_book, 'Pattern Catalog\ntemplate'),
    (48.0, icon_doc, 'framework\nsource & docs'),
    (67.0, icon_layers, 'evidence\n$\\mathcal{E}^{+}/\\mathcal{E}^{-}$'),
    (86.0, icon_shield, 'coverage gap\nin $\\mathcal{P}$'),
]
for px, ic, lb in plus_cards:
    box(px, 80.6, 15.5, 4.6, fc='#FFFFFF', ec=BLUE_SOFT, lw=LW_THIN, r=0.65)
    ic(px + 2.6, 82.9, 0.8, BLUE)
    txt(px + 9.4, 82.9, lb, fs=3.2, ls=1.12)
for gx in (46.2, 65.2, 84.2):
    txt(gx, 82.9, '+', fs=6.2, w='bold', c=GRAY)
arrow((65.5, 80.4), (65.5, 79.5), c=BLUE_SOFT, lw=LW_MID, ms=6)

dashed_group(28.2, 66.0, 38.2, 13.2, LAVENDER, label=None, lc=PURPLE)
dashed_group(66.8, 66.0, 37.8, 13.2, MINT, label=None, lc='#82B366')
box(28.8, 76.6, 12.5, 2.0, fc=LAVENDER, ec=PURPLE, lw=LW_HAIR, r=0.35, ls=(0, (1.8, 1.2)))
txt(35.05, 77.6, 'instantiate', fs=3.5, w='bold', c=PURPLE)
box(67.4, 76.6, 16.5, 2.0, fc=MINT, ec='#82B366', lw=LW_HAIR, r=0.35, ls=(0, (1.8, 1.2)))
txt(75.65, 77.6, 'verify (decisive)', fs=3.5, w='bold', c='#5B8A5B')
icon_brain(42.2, 77.6, 0.7, PURPLE)

steps = [
    (29.0, 'Scope', 'fix training phase\n+ parameter class', DARK, 0.7, '#F7F9FC'),
    (48.25, 'Ground', 'evidence $\\mathcal{E}^{+}/\\mathcal{E}^{-}$ from\nframework source & docs', DARK, 0.7, CREAM),
    (67.5, 'Construct', 'counterexample attack\n$\\geq$2 CEs; one CE holds\n$\\Rightarrow$ reject outright', RED, 1.1, PEACH),
    (86.75, 'Accept', 'no CE holds $\\wedge$\nConf$(c) \\geq 0.8$\n$(\\theta = 0.8)$', DARK, 0.7, '#F3FAF3'),
]
for x, name, sub, ec, lw, fc in steps:
    box(x, 66.6, 16.5, 9.5, fc=fc, ec=ec, lw=(LW_MID if ec is RED else LW_THIN), r=0.8)
    txt(x + (9.3 if name == 'Construct' else 8.9), 73.9, name, fs=5.1, w='bold', c=ec)
    ax.plot([x + 2.2, x + 14.3], [72.8, 72.8], color=ec, lw=LW_HAIR, alpha=0.75, zorder=5)
    txt(x + 8.25, 69.4, sub, fs=3.5, ls=1.25)
    if name == 'Scope':
        icon_target(x + 3.2, 73.9, 0.9, DARK)
    elif name == 'Ground':
        icon_layers(x + 3.2, 73.9, 0.85, DARK)
    elif name == 'Accept':
        icon_checkc(x + 3.4, 73.9, 0.9, DARK)
    elif name == 'Construct':
        chip(x + 3.0, 73.9, 1.3, RED)
        icon_shield(x + 3.0, 73.95, 0.75, 'white')
for x0 in (45.8, 65.05, 84.3):
    arrow((x0, 71.35), (x0 + 2.2, 71.35), c=BLUE_SOFT, lw=LW_MID, ms=7)

# reject loop tight under step cards; FSM band fully below
arrow((95.0, 66.45), (95.0, 65.15), c=RED_DEEP, lw=LW_THIN, ls=(0, (2.4, 1.5)), style='-')
arrow((95.0, 65.15), (37.3, 65.15), c=RED_DEEP, lw=LW_THIN, ls=(0, (2.4, 1.5)), style='-')
arrow((37.3, 65.15), (37.3, 66.35), c=RED_DEEP, lw=LW_THIN, ls=(0, (2.4, 1.5)))
red_x(99.2, 65.15, 0.6)
txt(66.0, 64.15, 'reject $\\rightarrow$ next iteration', fs=3.5, c=RED, style='italic')

txt(28.8, 60.0, 'five-stage FSM', fs=3.5, c=GRAY, ha='left')
fsm = [('$S_1$', 'gap analysis', '#6f8fc4'), ('$S_2$', 'evidence retrieval', '#7fae9e'),
       ('$S_3$', 'synthesis+attack', '#c48a8a'), ('$S_4$', 'persistence', '#a795c2'),
       ('$S_5$', 'reporting', '#c2a370')]
for k, (sn, lb, cc) in enumerate(fsm):
    cx = 42.0 + k * 8.4
    ax.add_patch(Circle((cx, 59.7), 1.4, fc=cc, ec='none', zorder=5))
    txt(cx, 59.7, sn, fs=3.35, c='white', w='bold', z=6)
    txt(cx, 57.2, lb, fs=2.8, c=GRAY)
    if k < 4:
        arrow((cx + 1.65, 59.7), (cx + 6.55, 59.7), c=GRAY_LINE, lw=LW_HAIR, ms=5)

cred = [(29.5, 'decisive counterexample gate'),
        (53.5, 'healthy-run replay (verification only)'),
        (82.0, 'Conf$(c) \\geq 0.8$')]
for cx0, tt in cred:
    checkbox(cx0, 55.6, 1.15, GREEN)
    txt(cx0 + 1.8, 56.15, tt, fs=3.4, c='#0E6E6C', ha='left')

arrow((103.3, 71.35), (110.5, 78.0), c=BLUE_SOFT, lw=LW_THICK, ms=10, cs='arc3,rad=-0.18', halo=True)
txt(107.2, 80.6, 'accepted', fs=3.8, c=BLUE, style='italic', ha='right')

# ================= Verified Constraint Library (right column) =================
shadow(108.5, 46.5, 32.5, 43.5)
box(108.5, 46.5, 32.5, 43.5, fc=CARD, ec=DARK, lw=LW_MID, r=1.15)
cyl(120.5, 83.5, 8.5, 5.0, fc='#e9e2d2', ec=DARK)
txt(124.7, 80.6, 'Verified Constraint Library $\\mathcal{P}$', fs=4.8, w='bold', c=BLUE_DEEP)
txt(124.7, 78.4, 'guarded relational constraints', fs=3.8, style='italic', c=GRAY)
shadow(110.5, 55.5, 28.5, 21.0, r=1.0)
box(110.5, 55.5, 28.5, 21.0, fc='#FFFFFF', ec=DARK, lw=LW_THIN, r=0.95)
box(110.5, 71.8, 28.5, 4.7, fc='#39404d', ec='none', r=1.0, z=3)
ax.add_patch(Rectangle((110.5, 71.8), 28.5, 2.2, fc='#39404d', ec='none', zorder=3))
txt(124.7, 75.0, 'P3 · cross-rank replication', fs=4.6, w='bold', c='white', z=5)
txt(124.7, 72.8, '(SwitchMLP router sync)', fs=3.6, c='#c9cedb', z=5)
txt(112, 63.5, '$\\pi_{topo}$:  TP > 1\n$\\pi_{precond}$:  tpl = False\n   (not tensor-parallel sharded)\n$\\pi_{schema}$:  replicas share\n   checksum',
    fs=4.0, ha='left', ls=1.35)
txt(124.7, 53.5, 'compiled to parameterized SQL at job start', fs=3.5, c=ORANGE, w='bold')
pill(110.5, 48.8, 28.5, 3.4, '', GREEN_BG, GREEN)
txt(124.7, 50.5, '$\\checkmark$  0 false positives over 764 clean evals', fs=3.7, c='#0E6E6C', z=6)
# schema cue in gap between Miner (bottom 54.5) and ONLINE banner (51.2)
txt(105.5, 53.0, 'accepted rules determine trace schema (S0–S6)',
    fs=3.4, c=ORANGE, ha='right')
arrow((108.5, 56.5), (63.5, 48.5), c=ORANGE, lw=LW_MID, ms=8, cs='arc3,rad=0.05')
# compile arrow into Verifier right edge (not over title bar)
arrow((124.7, 46.3), (106.4, 34.5), c=ORANGE, lw=LW_THICK, ms=10, cs='arc3,rad=0.18', halo=True)

# ================= Data Collector (online row) =================
shadow(27, 16.0, 36.5, 33.5)
box(27, 16.0, 36.5, 33.5, fc=CARD, ec=DARK, lw=LW_MID, r=1.15)
badge(30.6, 46.8, '2')
txt(33.4, 46.8, 'Data Collector', fs=6.0, w='bold', c=ORANGE, ha='left')
chip(54.8, 46.8, 1.45, ORANGE)
icon_probe(54.8, 46.8, 0.85, 'white')
txt(28.2, 44.2, 'no user code changes · adapter 30–150 LoC', fs=3.6,
    style='italic', c=GRAY, ha='left')

anchors = ['before-\nforward', 'after-\nforward', 'main-grad-\nin-backward',
           'after-\nbackward', 'before-\noptimizer']
tints = ['#fbe3c9', '#f8d5ac', '#f4c48c', '#efb26d', '#e99f4e']
for i, a in enumerate(anchors):
    x = 28.3 + i * 6.85
    w, n = 7.4, 1.2
    pts = [(x, 37.0), (x + w - n, 37.0), (x + w, 39.7), (x + w - n, 42.4),
           (x, 42.4), (x + n, 39.7)]
    ax.add_patch(Polygon(pts, closed=True, fc=tints[i], ec='none', zorder=3))
    txt(x + w / 2 + 0.15, 39.7, a, fs=3.2, family='monospace', z=5)
txt(45.2, 34.6, '+ auxiliary taps (ckpt · all_reduce · snapshot) = 8 hookpoints',
    fs=3.4, c=GRAY)
txt(45.2, 32.0, 'GPU-side scalar reductions (checksums only when required)', fs=3.5)
box(28.8, 18.6, 33, 6.2, fc='#FDF5EC', ec=ORANGE, lw=LW_THIN, r=0.7)
txt(45.3, 23.4, 'captured record (S0):', fs=3.7, c=GRAY)
_pw = [('cksum', 4.9), ('param_name', 7.7), ('rank', 4.1), ('step', 4.1), ('stage', 4.9)]
_x = 28.8 + (33 - (sum(w for _, w in _pw) + 0.65 * 4)) / 2
for _t, _w in _pw:
    pill(_x, 19.3, _w, 2.7, _t, 'white', ORANGE, fs=3.3)
    _x += _w + 0.65

# DuckDB bridge
cyl(65.0, 27.5, 8.8, 11.5, fc='#fbe9d4', ec=ORANGE)
for _yy in (31.2, 34.2):
    ax.plot([65.9, 73.0], [_yy, _yy], color=ORANGE, lw=LW_HAIR, alpha=0.55, zorder=4)
txt(69.4, 25.0, 'DuckDB\ntrace DB', fs=4.0, w='bold', c=ORANGE, ls=1.15)
txt(69.4, 20.8, 'two-step\nsliding window', fs=3.3, c=GRAY, ls=1.15)
arrow((63.7, 33.0), (64.8, 33.0), c=ORANGE, lw=LW_MID, ms=7)
arrow((74.0, 33.0), (75.2, 33.0), c=ORANGE, lw=LW_MID, ms=7)

# ================= Verifier =================
shadow(75.5, 16.0, 31.0, 33.5)
box(75.5, 16.0, 31.0, 33.5, fc=CARD, ec=DARK, lw=LW_MID, r=1.15)
badge(79.1, 46.8, '3')
txt(81.9, 46.8, 'Verifier', fs=6.0, w='bold', c=ORANGE, ha='left')
chip(92.0, 46.8, 1.45, ORANGE)
icon_search(92.0, 46.8, 0.85, 'white')
txt(76.5, 44.2, 'deterministic SQL checking — no LLM', fs=3.6,
    style='italic', c=GRAY, ha='left')
box(76.8, 27.2, 13.6, 14.6, fc=MINT, ec='#82B366', lw=LW_THIN, r=0.8, ls=(0, (2.2, 1.6)))
txt(83.6, 40.2, 'at job start', fs=4.1, w='bold')
ax.plot([78.0, 89.2], [38.9, 38.9], color=GRAY, lw=LW_HAIR, zorder=5)
txt(83.6, 33.0, 'extract topology $\\tau$\ntopology-aware\npruning · lineage:\nreplicated vs.\nsharded', fs=3.45, ls=1.35)
box(91.4, 27.2, 14.0, 14.6, fc='#36393D', ec='#2C2C2C', lw=LW_THIN, r=0.8)
txt(98.4, 40.2, 'per step, per rule', fs=4.0, w='bold', c='#9db8e8')
ax.plot([92.6, 104.2], [38.9, 38.9], color='#7A8A9A', lw=LW_HAIR, zorder=5)
_KW, _TX, _PI = '#61afef', '#d7dae0', '#e5c07b'
txt(92.6, 37.0, 'WHERE', fs=3.35, ha='left', family='monospace', c=_KW)
txt(96.4, 37.0, '$\\pi_{topo}\\wedge\\pi_{precond}$', fs=3.25, ha='left', c=_PI)
txt(92.6, 34.4, 'GROUP BY', fs=3.35, ha='left', family='monospace', c=_KW)
txt(93.2, 32.2, 'param, step', fs=3.35, ha='left', family='monospace', c=_TX)
txt(92.6, 29.8, 'HAVING', fs=3.35, ha='left', family='monospace', c=_KW)
txt(96.6, 29.8, '$\\pi_{schema}$', fs=3.25, ha='left', c=_PI)
txt(93.2, 28.0, 'violated', fs=3.35, ha='left', family='monospace', c=_TX)
txt(91.0, 24.2, 'empty result = holds · non-empty IS the violation set',
    fs=3.5, style='italic')
txt(91.0, 20.6, 'coarse$\\rightarrow$fine: checksum$\\rightarrow$module$\\rightarrow$param$\\rightarrow$bitwise',
    fs=3.25, c=GRAY)
arrow((106.6, 34.5), (108.2, 34.5), c=RED_DEEP, lw=LW_THICK, ms=10, halo=True)

# ================= Violation panel =================
shadow(108.5, 11.0, 32.5, 33.0)
box(108.5, 11.0, 32.5, 33.0, fc=VIOL_BG, ec=RED, lw=LW_MID, r=1.05, ls=(0, (2.8, 1.8)))
txt(124.7, 41.2, 'violation report', fs=5.0, w='bold', c=RED)
ax.plot([116.5, 132.8], [39.8, 39.8], color=RED, lw=LW_HAIR, zorder=5)
_tx, _cols = 111.2, [('param', 8.6), ('rank', 3.4), ('cksum', 7.2)]
_rows = [('router.weight', '0', 'cksum$_A$', False), ('router.weight', '1', 'cksum$_B$', True)]
_ty, _rh = 38.4, 2.35
_cx = _tx
for _h, _wd in _cols:
    ax.add_patch(Rectangle((_cx, _ty - _rh), _wd, _rh, fc='#F5F5F5', ec='#B3B3B3', lw=LW_HAIR, zorder=4))
    txt(_cx + _wd / 2, _ty - _rh / 2, _h, fs=3.3, c=GRAY, z=5)
    _cx += _wd
for _r, (_p, _rk, _ck, _bad) in enumerate(_rows):
    _cx = _tx
    _yy = _ty - _rh * (_r + 2)
    for _v, (_h, _wd) in zip((_p, _rk, _ck), _cols):
        ax.add_patch(Rectangle((_cx, _yy), _wd, _rh, fc='#F8CECC' if _bad else 'white',
                               ec='#B3B3B3', lw=LW_HAIR, zorder=4))
        txt(_cx + _wd / 2, _yy + _rh / 2, _v, fs=3.3, family='monospace',
            c=RED if _bad else DARK, z=5)
        _cx += _wd
txt(124.7, 29.6, '$\\mathbf{\\times}$ divergent checksums', fs=3.6, c=RED)
for _k, _it in enumerate(['violating constraint', 'affected parameter',
                          'divergent ranks', 'first violating step']):
    _yy = 26.8 - _k * 2.85
    checkbox(111.6, _yy - 0.55, 1.15, RED)
    txt(113.6, _yy, _it, fs=3.7, ha='left')
# route output arrow under the panel (no border cross through content)
arrow((108.5, 13.2), (23.5, 18.5), c=DARK, lw=LW_THICK, ms=10, cs='arc3,rad=-0.05', halo=True)

# ================= left bay: P3 violation report artifact =================
txt(2.0, 42.0, 'online output', fs=5.0, w='bold', c=ORANGE, ha='left')
folded_doc(1.5, 19.5, 21, 20.0,
           lines=['violating constraint: P3',
                  'affected parameter:',
                  '  router.weight',
                  'divergent ranks: 0 vs 1',
                  '(SwitchMLP router sync)',
                  'non-empty SQL result'],
           title='Violation Report', title_c=RED, ec=RED, fc='#fff8f8',
           hl={0: RED, 2: RED})
txt(12, 16.5, 'empty result = holds', fs=3.8, style='italic', c=GRAY)
txt(12, 13.6, 'non-empty = violation set', fs=3.8, style='italic', c=RED)

# ================= legend =================
ax.add_patch(FancyBboxPatch((25, 2.4), 117.5, 3.8,
             boxstyle='round,pad=0,rounding_size=0.5',
             fc='#F3F4F6', ec='#B3B3B3', lw=LW_HAIR, zorder=2))
txt(26.5, 4.3, 'Legend', fs=4.2, w='bold', ha='left')
ax.plot([36, 43], [4.3, 4.3], color=BLUE_SOFT, lw=LW_THICK)
txt(44.5, 4.3, 'offline mining', fs=3.9, ha='left')
ax.plot([62, 69], [4.3, 4.3], color=ORANGE, lw=LW_THICK)
txt(70.5, 4.3, 'online checking', fs=3.9, ha='left')
ax.plot([92, 99], [4.3, 4.3], color=RED_DEEP, lw=LW_MID, ls=(0, (2.4, 1.5)))
txt(100.5, 4.3, 'reject / violation', fs=3.9, ha='left')
txt(141.5, 4.3, 'LLM only inside offline Miner · online path: no LLM',
    fs=3.7, style='italic', c=GRAY, ha='right')

fig.savefig('fig_overview_v15.pdf')
fig.savefig('/tmp/fig_overview_v15.png', dpi=220)
print('done v15')
