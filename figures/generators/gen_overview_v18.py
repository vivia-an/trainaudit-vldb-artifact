#!/usr/bin/env python3
"""v18 = v17 + compress height, shaft arrows, deeper/larger FSM, bolder type.
Labels identical to v17. No Agents / 0-4 / 43k / fake hex.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as _np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Rectangle, Circle, Polygon
from matplotlib import rcParams

rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'],
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,
    'text.antialiased': True,
    'axes.linewidth': 0.7,
    'font.weight': 'medium',
})

# QUITE deeper pastels (Fig.3 path fills) + navy impact
BLUE = '#306092'          # QUITE navy
BLUE_SOFT = '#6C8EBF'
BLUE_DEEP = '#1A3348'
BLUE_BG = '#DCE6F2'       # deeper offline wash (was near-white)
ORANGE = '#C05621'
ORANGE_BG = '#F3E4D0'     # deeper online wash
ORANGE_SOFT = '#E8A87C'
RED = '#B85450'
RED_DEEP = '#AE4132'
GREEN = '#1AABA8'
GREEN_BG = '#D8EFEA'
DARK = '#111111'          # high-contrast ink
GRAY = '#3D3D3D'          # darker secondary (more impact)
GRAY_LINE = '#8A8A8A'
CARD = '#FFFFFF'
HUB_BG = '#EDE6D8'        # deeper beige hub
VIOL_BG = '#F5DDD8'       # deeper rose
LAVENDER = '#E1D5E7'      # QUITE lavender
PURPLE = '#9673A6'
MINT = '#D5E8D4'          # QUITE mint
PEACH = '#FAD9D5'         # QUITE peach
CREAM = '#FFF2CC'         # deeper cream
# stroke ladder — slightly heavier for impact
LW_HAIR, LW_THIN, LW_MID, LW_THICK = 0.32, 0.62, 0.95, 1.65

W, H = 144, 100
fig = plt.figure(figsize=(7.35, 4.85))  # slightly shorter print frame
ax = fig.add_axes([0.002, 0.002, 0.996, 0.996])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis('off')


def box(x, y, w, h, fc=CARD, ec=DARK, lw=None, r=1.05, ls='-', z=2):
    if lw is None:
        lw = LW_THIN
    p = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}',
                       fc=fc, ec=ec, lw=lw, ls=ls, zorder=z,
                       joinstyle='round', capstyle='round')
    ax.add_patch(p)
    return p


def txt(x, y, s, fs=5, w='bold', c=DARK, ha='center', va='center',
        style='normal', family=None, ls=1.28, z=5):
    # default bold = QUITE high-contrast type; pass w='normal' for dense mono
    kw = dict(fontsize=fs, fontweight=w, color=c, ha=ha, va=va,
              style=style, linespacing=ls, zorder=z)
    if family:
        kw['family'] = family
    ax.text(x, y, s, **kw)


import matplotlib.patheffects as _pe


def arrow(p0, p1, c=DARK, lw=None, ls='-', style='-|>', ms=11, cs=None, z=4,
          halo=False):
    """Shaft + triangular head (never head-only / never overlap). Curved → patch."""
    if lw is None:
        lw = LW_MID
    p0 = _np.asarray(p0, dtype=float)
    p1 = _np.asarray(p1, dtype=float)
    if cs is not None:
        a = FancyArrowPatch(tuple(p0), tuple(p1), arrowstyle=style, mutation_scale=ms,
                            color=c, lw=lw, linestyle=ls, zorder=z,
                            connectionstyle=cs,
                            shrinkA=0.4, shrinkB=0.4,
                            joinstyle='miter', capstyle='butt')
        if halo:
            a.set_path_effects([_pe.Stroke(linewidth=lw + 1.6, foreground='white'),
                                _pe.Normal()])
        ax.add_patch(a)
        return
    v = p1 - p0
    L = float(_np.linalg.norm(v))
    if L < 1e-6:
        return
    u = v / L
    # head ≤ 35% of span → visible shaft; style='-' = shaft only
    head = 0.0 if style == '-' else min(1.25, max(0.5, L * 0.30))
    if style != '-' and L - head < 0.7:
        head = max(0.35, L * 0.38)
    base = p1 - u * head if head > 0 else p1
    if halo:
        ax.plot([p0[0], base[0]], [p0[1], base[1]], color='white', lw=lw + 1.8,
                solid_capstyle='butt', zorder=z - 0.1)
    ax.plot([p0[0], base[0]], [p0[1], base[1]], color=c, lw=lw,
            solid_capstyle='butt', zorder=z, linestyle=ls)
    if head > 0:
        perp = _np.array([-u[1], u[0]])
        hw = head * 0.40
        tri = _np.vstack([p1, base + perp * hw, base - perp * hw])
        ax.add_patch(Polygon(tri, closed=True, fc=c, ec=c, lw=0.15, zorder=z + 0.2))


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
    # QUITE crops are flat — keep only a hairline lift (no muddy dual shadow)
    ax.add_patch(FancyBboxPatch((x + 0.18, y - 0.22), w, h,
                 boxstyle=f'round,pad=0,rounding_size={r}',
                 fc='#000000', ec='none', alpha=0.025, zorder=1.6))


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


# ================= outer container (QUITE genome) — tighter top/bottom air =================
shadow(25, 6.6, 118, 90.2, r=1.6)
box(25, 6.6, 118, 90.2, fc='#FFFFFF', ec='#111111', lw=LW_THICK, r=1.4, z=1.8)
box(26.4, 51.0, 80.2, 41.0, fc=BLUE_BG, ec='none', r=1.15, z=1.85)
box(26.4, 9.0, 80.2, 41.0, fc=ORANGE_BG, ec='none', r=1.15, z=1.85)
box(108.2, 9.2, 33.4, 80.5, fc=HUB_BG, ec='none', r=1.15, z=1.85)
txt(84, 94.6, 'TrainAudit', fs=8.8, w='bold', c=BLUE_DEEP)
ax.plot([37.2, 73.8], [94.6, 94.6], color=BLUE_SOFT, lw=LW_HAIR, zorder=4)
ax.plot([94.2, 130.8], [94.6, 94.6], color=BLUE_SOFT, lw=LW_HAIR, zorder=4)

ax.add_patch(FancyBboxPatch((27.2, 90.2), 78.6, 2.45,
             boxstyle='round,pad=0,rounding_size=0.55',
             fc='#C5D4E8', ec='none', zorder=2))
txt(28.0, 91.4, 'OFFLINE — mine once per framework', fs=6.15, w='bold', c=BLUE_DEEP, ha='left')
ax.add_patch(FancyBboxPatch((27.2, 50.55), 78.6, 2.35,
             boxstyle='round,pad=0,rounding_size=0.55',
             fc='#E8C9A8', ec='none', zorder=2))
txt(28.0, 51.7, 'ONLINE — every training job · fully deterministic · no LLM',
    fs=6.15, w='bold', c='#8A3A12', ha='left')


def folded_doc(x, y, w, h, lines, title=None, title_c=DARK, ec=DARK, fc='#fafafa', hl=None):
    """QUITE doc card: generous pad, sharp ink, soft header wash."""
    fold = min(2.2, w * 0.18)
    shadow(x, y, w, h, r=0.75)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.75',
                                fc=fc, ec=ec, lw=LW_THIN, zorder=3,
                                joinstyle='round'))
    if title:
        ax.add_patch(FancyBboxPatch((x + 0.25, y + h - 2.9), w - 0.5, 2.55,
                     boxstyle='round,pad=0,rounding_size=0.45',
                     fc=title_c, ec='none', alpha=0.10, zorder=3.5))
        txt(x + 1.05, y + h - 1.55, title, fs=3.9, w='bold', c=title_c, ha='left', z=5)
        y0 = y + h - 3.85
    else:
        y0 = y + h - 2.0
    ax.add_patch(Polygon([(x + w - fold, y + h), (x + w, y + h - fold), (x + w - fold, y + h - fold)],
                         closed=True, fc='#ECECEC', ec=ec, lw=LW_HAIR, zorder=4))
    ax.plot([x + w - fold, x + w - fold, x + w], [y + h, y + h - fold, y + h - fold],
            color=ec, lw=LW_HAIR, zorder=5)
    for i, ln in enumerate(lines):
        fam = None if ('$' in ln or '\\' in ln) else 'monospace'
        cc = (hl or {}).get(i, DARK)
        txt(x + 1.15, y0 - i * 1.95, ln, fs=3.3, family=fam, ha='left',
            c=cc, w='bold', z=5, ls=1.3)


def dashed_group(x, y, w, h, fc, label=None, lc=GRAY):
    """QUITE soft pastel subsection — pale fill + light dashed rim."""
    box(x, y, w, h, fc=fc, ec=lc, lw=LW_HAIR, r=1.0, ls=(0, (2.6, 1.8)), z=2)
    if label:
        txt(x + 1.4, y + h - 1.45, label, fs=3.7, w='bold', c=lc, ha='left', z=5)


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
txt(2.0, 94.8, 'offline inputs', fs=5.2, w='bold', c=BLUE, ha='left')
folded_doc(1.5, 73.0, 21, 19.0,
           lines=['template P3:',
                  '  pi_topo: TP > 1',
                  '  pi_precond: tpl=False',
                  '  pi_schema: share cksum',
                  '  (cross-rank replicas)',
                  'from 392 silent errors'],
           title='Pattern Catalog', title_c=BLUE, ec=BLUE, fc='#F7FAFF')
txt(12, 70.7, '16 templates · 13 fault classes', fs=3.4, c=GRAY)
folded_doc(1.5, 53.5, 21, 15.5,
           lines=['Megatron-LM',
                  'DeepSpeed',
                  'OLMo / OLMo-core',
                  'silent-vs-loud filter',
                  'no raise / assert'],
           title='Framework source', title_c=BLUE, ec='#5A7A9A', fc='#FAFAFA')
txt(12, 51.4, 'once per framework', fs=3.4, style='italic', c=GRAY)
arrow((23.0, 81), (28.6, 81), c=BLUE_SOFT, lw=LW_THICK, ms=10, halo=True)

# ================= Invariant Miner (offline row) — compressed FSM band =================
shadow(27, 53.2, 79, 36.5)
box(27, 53.2, 79, 36.5, fc=CARD, ec='#222222', lw=LW_MID, r=1.2)
badge(30.6, 86.9, '1')
txt(33.6, 86.9, 'Invariant Miner', fs=6.6, w='bold', c=BLUE_DEEP, ha='left')
chip(53.4, 86.9, 1.55, BLUE)
icon_gear(53.4, 86.9, 0.9, 'white')
txt(104.3, 86.9, 'LLM-instantiated candidates, untrusted until verified',
    fs=3.85, style='italic', c=GRAY, ha='right')

plus_cards = [
    (29.0, icon_book, 'Pattern Catalog\ntemplate'),
    (48.0, icon_doc, 'framework\nsource & docs'),
    (67.0, icon_layers, 'evidence\n$\\mathcal{E}^{+}/\\mathcal{E}^{-}$'),
    (86.0, icon_shield, 'coverage gap\nin $\\mathcal{P}$'),
]
for px, ic, lb in plus_cards:
    box(px, 80.55, 15.5, 4.7, fc='#FFFFFF', ec=BLUE_SOFT, lw=LW_THIN, r=0.75)
    ic(px + 2.7, 82.9, 0.78, BLUE)
    txt(px + 9.5, 82.9, lb, fs=3.55, ls=1.2)
for gx in (46.2, 65.2, 84.2):
    txt(gx, 82.9, '+', fs=6.5, w='bold', c=GRAY_LINE)
arrow((65.5, 80.4), (65.5, 79.55), c=BLUE_SOFT, lw=LW_MID, ms=6)

dashed_group(28.2, 66.4, 38.2, 12.8, LAVENDER, label=None, lc=PURPLE)
dashed_group(66.8, 66.4, 37.8, 12.8, MINT, label=None, lc='#7A9E72')
box(29.0, 76.85, 12.8, 2.1, fc='#D9C8E4', ec=PURPLE, lw=LW_HAIR, r=0.4, ls=(0, (1.8, 1.2)))
txt(35.4, 77.9, 'instantiate', fs=3.9, w='bold', c='#6B4F7A')
box(67.6, 76.85, 16.8, 2.1, fc='#BFD9B8', ec='#5B8A5B', lw=LW_HAIR, r=0.4, ls=(0, (1.8, 1.2)))
txt(76.0, 77.9, 'verify (decisive)', fs=3.9, w='bold', c='#3D6B3D')
icon_brain(42.4, 77.9, 0.68, PURPLE)

steps = [
    (29.2, 'Scope', 'fix training phase\n+ parameter class', DARK, 0.7, '#FFFFFF'),
    (48.35, 'Ground', 'evidence $\\mathcal{E}^{+}/\\mathcal{E}^{-}$ from\nframework source & docs', DARK, 0.7, '#FFFFFF'),
    (67.6, 'Construct', 'counterexample attack\n$\\geq$2 CEs; one CE holds\n$\\Rightarrow$ reject outright', RED, 1.1, PEACH),
    (86.85, 'Accept', 'no CE holds $\\wedge$\nConf$(c) \\geq 0.8$\n$(\\theta = 0.8)$', DARK, 0.7, '#FFFFFF'),
]
for x, name, sub, ec, lw, fc in steps:
    box(x, 66.9, 16.3, 9.55, fc=fc, ec=ec, lw=(LW_MID if ec is RED else LW_THIN), r=0.9)
    txt(x + (9.2 if name == 'Construct' else 8.85), 74.35, name, fs=5.55, w='bold', c=ec)
    ax.plot([x + 2.4, x + 14.0], [73.15, 73.15], color=ec, lw=LW_HAIR, alpha=0.55, zorder=5)
    txt(x + 8.15, 69.45, sub, fs=3.7, ls=1.32)
    if name == 'Scope':
        icon_target(x + 3.15, 74.35, 0.85, DARK)
    elif name == 'Ground':
        icon_layers(x + 3.15, 74.35, 0.8, DARK)
    elif name == 'Accept':
        icon_checkc(x + 3.35, 74.35, 0.85, DARK)
    elif name == 'Construct':
        chip(x + 2.95, 74.35, 1.25, RED)
        icon_shield(x + 2.95, 74.4, 0.72, 'white')
# step→step: longer shafts, no head-only / no overlap into cards
for x0 in (45.7, 64.95, 84.25):
    arrow((x0, 71.55), (x0 + 2.35, 71.55), c=BLUE_SOFT, lw=LW_MID, ms=7)

# reject loop — shaft segments only on elbows; head only on final up-leg
arrow((95.0, 66.75), (95.0, 65.35), c=RED_DEEP, lw=LW_THIN, ls=(0, (2.4, 1.5)), style='-')
arrow((95.0, 65.35), (37.3, 65.35), c=RED_DEEP, lw=LW_THIN, ls=(0, (2.4, 1.5)), style='-')
arrow((37.3, 65.35), (37.3, 66.7), c=RED_DEEP, lw=LW_THIN, ls=(0, (2.4, 1.5)))
red_x(99.3, 65.35, 0.55)
txt(66.0, 64.45, 'reject $\\rightarrow$ next iteration', fs=3.75, c=RED, style='italic')

# FSM: deeper fills, S_n fills ~55% of disc, shaft arrows between (no overlap)
txt(28.8, 61.85, 'five-stage FSM', fs=3.85, c=GRAY, ha='left')
fsm = [('$S_1$', 'gap analysis', '#3F6BA8'), ('$S_2$', 'evidence retrieval', '#3F8F72'),
       ('$S_3$', 'synthesis+attack', '#B04A4A'), ('$S_4$', 'persistence', '#6E4F9A'),
       ('$S_5$', 'reporting', '#A87830')]
_fsm_r, _fsm_y = 1.85, 59.85
for k, (sn, lb, cc) in enumerate(fsm):
    cx = 42.2 + k * 8.55
    ax.add_patch(Circle((cx, _fsm_y), _fsm_r, fc=cc, ec='#222222', lw=LW_HAIR, zorder=5))
    txt(cx, _fsm_y - 0.05, sn, fs=5.15, c='white', w='bold', z=6)  # ~55% of diameter
    txt(cx, _fsm_y - 2.65, lb, fs=3.25, c=DARK)
    if k < 4:
        # shaft from disc edge → next disc edge (gap clear, no overlap)
        arrow((cx + _fsm_r + 0.2, _fsm_y), (cx + 8.55 - _fsm_r - 0.2, _fsm_y),
              c='#555555', lw=LW_THIN, ms=7)

cred = [(29.5, 'decisive counterexample gate'),
        (53.5, 'healthy-run replay (verification only)'),
        (82.0, 'Conf$(c) \\geq 0.8$')]
for cx0, tt in cred:
    checkbox(cx0, 54.35, 1.25, GREEN)
    txt(cx0 + 1.95, 54.95, tt, fs=3.7, c='#0A6A68', ha='left')

arrow((103.3, 71.55), (110.5, 77.6), c=BLUE_SOFT, lw=LW_THICK, ms=10, cs='arc3,rad=-0.18', halo=True)
txt(107.2, 80.2, 'accepted', fs=4.0, c=BLUE, style='italic', ha='right')

# ================= Verified Constraint Library (right column) =================
shadow(108.5, 46.5, 32.5, 43.5)
box(108.5, 46.5, 32.5, 43.5, fc=CARD, ec='#222222', lw=LW_MID, r=1.2)
cyl(120.5, 83.5, 8.5, 5.0, fc='#EDE6D8', ec=DARK)
txt(124.7, 80.55, 'Verified Constraint Library $\\mathcal{P}$', fs=5.0, w='bold', c=BLUE_DEEP)
txt(124.7, 78.25, 'guarded relational constraints', fs=3.85, style='italic', c=GRAY)
shadow(110.5, 55.3, 28.5, 21.2, r=1.0)
box(110.5, 55.3, 28.5, 21.2, fc='#FFFFFF', ec='#222222', lw=LW_THIN, r=1.0)
box(110.5, 71.7, 28.5, 4.8, fc='#1E242C', ec='none', r=1.0, z=3)
ax.add_patch(Rectangle((110.5, 71.7), 28.5, 2.2, fc='#1E242C', ec='none', zorder=3))
txt(124.7, 75.0, 'P3 · cross-rank replication', fs=4.85, w='bold', c='white', z=5)
txt(124.7, 72.75, '(SwitchMLP router sync)', fs=3.75, w='bold', c='#B8C4D8', z=5)
txt(112.2, 63.35, '$\\pi_{topo}$:  TP > 1\n$\\pi_{precond}$:  tpl = False\n   (not tensor-parallel sharded)\n$\\pi_{schema}$:  replicas share\n   checksum',
    fs=4.1, ha='left', ls=1.42)
txt(124.7, 53.35, 'compiled to parameterized SQL at job start', fs=3.55, c=ORANGE, w='bold')
pill(110.5, 48.7, 28.5, 3.55, '', GREEN_BG, GREEN)
txt(124.7, 50.45, '$\\checkmark$  0 false positives over 764 clean evals', fs=3.8, c='#0A6A68', z=6)
txt(105.5, 52.55, 'accepted rules determine trace schema (S0–S6)',
    fs=3.55, c=ORANGE, ha='right')
arrow((108.5, 55.8), (63.5, 47.8), c=ORANGE, lw=LW_MID, ms=8, cs='arc3,rad=0.05')
arrow((124.7, 45.8), (106.4, 33.8), c=ORANGE, lw=LW_THICK, ms=10, cs='arc3,rad=0.18', halo=True)

# ================= Data Collector (online row) =================
shadow(27, 14.8, 36.5, 33.2)
box(27, 14.8, 36.5, 33.2, fc=CARD, ec='#222222', lw=LW_MID, r=1.2)
badge(30.6, 45.5, '2')
txt(33.6, 45.5, 'Data Collector', fs=6.45, w='bold', c=ORANGE, ha='left')
chip(55.2, 45.5, 1.45, ORANGE)
icon_probe(55.2, 45.5, 0.85, 'white')
txt(28.4, 42.9, 'no user code changes · adapter 30–150 LoC', fs=3.8,
    style='italic', c=GRAY, ha='left')

anchors = ['before-\nforward', 'after-\nforward', 'main-grad-\nin-backward',
           'after-\nbackward', 'before-\noptimizer']
# deeper chevron ladder (QUITE impact)
tints = ['#F0C89A', '#EBB886', '#E5A872', '#DF985E', '#D8884A']
for i, a in enumerate(anchors):
    x = 28.3 + i * 6.85
    w, n = 7.4, 1.2
    pts = [(x, 35.7), (x + w - n, 35.7), (x + w, 38.55), (x + w - n, 41.4),
           (x, 41.4), (x + n, 38.55)]
    ax.add_patch(Polygon(pts, closed=True, fc=tints[i], ec=ORANGE, lw=LW_HAIR, zorder=3))
    txt(x + w / 2 + 0.15, 38.55, a, fs=3.5, family='monospace', w='bold', z=5)
txt(45.2, 33.4, '+ auxiliary taps (ckpt · all_reduce · snapshot) = 8 hookpoints',
    fs=3.6, c=GRAY)
txt(45.2, 30.85, 'GPU-side scalar reductions (checksums only when required)', fs=3.7)
box(28.8, 17.2, 33, 6.5, fc='#FBF6EF', ec=ORANGE, lw=LW_THIN, r=0.85)
txt(45.3, 22.3, 'captured record (S0):', fs=3.95, c=GRAY)
_pw = [('cksum', 4.9), ('param_name', 7.7), ('rank', 4.1), ('step', 4.1), ('stage', 4.9)]
_x = 28.8 + (33 - (sum(w for _, w in _pw) + 0.65 * 4)) / 2
for _t, _w in _pw:
    pill(_x, 17.95, _w, 2.9, _t, 'white', ORANGE, fs=3.55)
    _x += _w + 0.65

# DuckDB bridge — longer shafts between cards
cyl(65.0, 26.2, 8.8, 11.5, fc='#F5E6D0', ec=ORANGE)
for _yy in (29.9, 32.9):
    ax.plot([65.9, 73.0], [_yy, _yy], color=ORANGE, lw=LW_HAIR, alpha=0.45, zorder=4)
txt(69.4, 23.7, 'DuckDB\ntrace DB', fs=4.3, w='bold', c=ORANGE, ls=1.22)
txt(69.4, 19.5, 'two-step\nsliding window', fs=3.55, c=GRAY, ls=1.22)
arrow((63.5, 31.8), (64.9, 31.8), c=ORANGE, lw=LW_MID, ms=7)
arrow((73.9, 31.8), (75.4, 31.8), c=ORANGE, lw=LW_MID, ms=7)

# ================= Verifier =================
shadow(75.5, 14.8, 31.0, 33.2)
box(75.5, 14.8, 31.0, 33.2, fc=CARD, ec='#222222', lw=LW_MID, r=1.2)
badge(79.1, 45.5, '3')
txt(82.1, 45.5, 'Verifier', fs=6.45, w='bold', c=ORANGE, ha='left')
chip(92.4, 45.5, 1.45, ORANGE)
icon_search(92.4, 45.5, 0.85, 'white')
txt(76.7, 42.9, 'deterministic SQL checking — no LLM', fs=3.8,
    style='italic', c=GRAY, ha='left')
box(76.8, 25.8, 13.6, 14.9, fc=MINT, ec='#7A9E72', lw=LW_HAIR, r=0.9, ls=(0, (2.6, 1.8)))
box(77.35, 26.35, 12.5, 13.8, fc='#FFFFFF', ec='#B8C8B4', lw=LW_HAIR, r=0.7)
txt(83.6, 39.15, 'at job start', fs=4.4, w='bold')
ax.plot([78.4, 88.8], [37.8, 37.8], color=GRAY_LINE, lw=LW_HAIR, zorder=5)
txt(83.6, 31.8, 'extract topology $\\tau$\ntopology-aware\npruning · lineage:\nreplicated vs.\nsharded', fs=3.7, ls=1.38)
box(91.4, 25.8, 14.0, 14.9, fc='#1E242C', ec='#111111', lw=LW_THIN, r=0.9)
txt(98.4, 39.15, 'per step, per rule', fs=4.35, w='bold', c='#9DB8E8')
ax.plot([92.7, 104.1], [37.8, 37.8], color='#7A8A9A', lw=LW_HAIR, zorder=5)
_KW, _TX, _PI = '#61AFEF', '#E8EBF0', '#E5C07B'
txt(92.7, 35.85, 'WHERE', fs=3.7, ha='left', family='monospace', c=_KW, w='bold')
txt(96.55, 35.85, '$\\pi_{topo}\\wedge\\pi_{precond}$', fs=3.55, ha='left', c=_PI, w='bold')
txt(92.7, 33.2, 'GROUP BY', fs=3.7, ha='left', family='monospace', c=_KW, w='bold')
txt(93.3, 30.95, 'param, step', fs=3.7, ha='left', family='monospace', c=_TX, w='bold')
txt(92.7, 28.55, 'HAVING', fs=3.7, ha='left', family='monospace', c=_KW, w='bold')
txt(96.7, 28.55, '$\\pi_{schema}$', fs=3.55, ha='left', c=_PI, w='bold')
txt(93.3, 26.7, 'violated', fs=3.7, ha='left', family='monospace', c=_TX, w='bold')
txt(91.0, 22.95, 'empty result = holds · non-empty IS the violation set',
    fs=3.7, style='italic')
txt(91.0, 19.35, 'coarse$\\rightarrow$fine: checksum$\\rightarrow$module$\\rightarrow$param$\\rightarrow$bitwise',
    fs=3.45, c=GRAY)
arrow((106.5, 33.3), (108.3, 33.3), c=RED_DEEP, lw=LW_THICK, ms=10, halo=True)

# ================= Violation panel =================
shadow(108.5, 9.5, 32.5, 33.5)
box(108.5, 9.5, 32.5, 33.5, fc=VIOL_BG, ec=RED, lw=LW_MID, r=1.1, ls=(0, (3.0, 2.0)))
txt(124.7, 40.2, 'violation report', fs=5.4, w='bold', c=RED)
ax.plot([116.0, 133.2], [38.75, 38.75], color=RED, lw=LW_HAIR, alpha=0.7, zorder=5)
_tx, _cols = 111.2, [('param', 8.6), ('rank', 3.4), ('cksum', 7.2)]
_rows = [('router.weight', '0', 'cksum$_A$', False), ('router.weight', '1', 'cksum$_B$', True)]
_ty, _rh = 37.5, 2.4
_cx = _tx
for _h, _wd in _cols:
    ax.add_patch(Rectangle((_cx, _ty - _rh), _wd, _rh, fc='#F5F5F5', ec='#B3B3B3', lw=LW_HAIR, zorder=4))
    txt(_cx + _wd / 2, _ty - _rh / 2, _h, fs=3.5, c=GRAY, z=5)
    _cx += _wd
for _r, (_p, _rk, _ck, _bad) in enumerate(_rows):
    _cx = _tx
    _yy = _ty - _rh * (_r + 2)
    for _v, (_h, _wd) in zip((_p, _rk, _ck), _cols):
        ax.add_patch(Rectangle((_cx, _yy), _wd, _rh, fc='#F8CECC' if _bad else 'white',
                               ec='#B3B3B3', lw=LW_HAIR, zorder=4))
        txt(_cx + _wd / 2, _yy + _rh / 2, _v, fs=3.5, family='monospace',
            c=RED if _bad else DARK, z=5)
        _cx += _wd
txt(124.7, 28.4, '$\\mathbf{\\times}$ divergent checksums', fs=3.9, c=RED)
for _k, _it in enumerate(['violating constraint', 'affected parameter',
                          'divergent ranks', 'first violating step']):
    _yy = 25.6 - _k * 2.85
    checkbox(111.6, _yy - 0.55, 1.25, RED)
    txt(113.7, _yy, _it, fs=3.95, ha='left')
arrow((108.5, 11.5), (23.5, 17.2), c=DARK, lw=LW_THICK, ms=10, cs='arc3,rad=-0.05', halo=True)

# ================= left bay: P3 violation report artifact =================
txt(2.0, 41.0, 'online output', fs=5.3, w='bold', c=ORANGE, ha='left')
folded_doc(1.5, 18.2, 21, 20.5,
           lines=['violating constraint: P3',
                  'affected parameter:',
                  '  router.weight',
                  'divergent ranks: 0 vs 1',
                  '(SwitchMLP router sync)',
                  'non-empty SQL result'],
           title='Violation Report', title_c=RED, ec=RED, fc='#FFF8F8',
           hl={0: RED, 2: RED})
txt(12, 15.3, 'empty result = holds', fs=4.0, style='italic', c=GRAY)
txt(12, 12.5, 'non-empty = violation set', fs=4.0, style='italic', c=RED)

# ================= legend (tight under outer) =================
ax.add_patch(FancyBboxPatch((25, 1.5), 117.5, 3.7,
             boxstyle='round,pad=0,rounding_size=0.55',
             fc='#F4F5F7', ec='#B8B8B8', lw=LW_HAIR, zorder=2))
txt(26.7, 3.35, 'Legend', fs=4.45, w='bold', ha='left')
ax.plot([36.2, 43.2], [3.35, 3.35], color=BLUE_SOFT, lw=LW_THICK)
txt(44.7, 3.35, 'offline mining', fs=4.15, ha='left')
ax.plot([62.2, 69.2], [3.35, 3.35], color=ORANGE, lw=LW_THICK)
txt(70.7, 3.35, 'online checking', fs=4.15, ha='left')
ax.plot([92.2, 99.2], [3.35, 3.35], color=RED_DEEP, lw=LW_MID, ls=(0, (2.4, 1.5)))
txt(100.7, 3.35, 'reject / violation', fs=4.15, ha='left')
txt(141.5, 3.35, 'LLM only inside offline Miner · online path: no LLM',
    fs=3.9, style='italic', c=GRAY, ha='right')

fig.savefig('fig_overview_v18.pdf')
fig.savefig('/tmp/fig_overview_v18.png', dpi=240)
print('done v18')
