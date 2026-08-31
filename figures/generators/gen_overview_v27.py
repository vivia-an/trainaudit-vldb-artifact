#!/usr/bin/env python3
"""v27 = v26 + online arrow polish (no label change).
Under-lane viol→output (no text cover); online bridges sized like offline.
Labels identical to v26/v22. No Agents / 0-4 / 43k / fake hex.
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
fig = plt.figure(figsize=(7.35, 4.62))  # denser print frame (QUITE packing)
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

def icon_funnel(cx, cy, s, c, lw=None):
    """Data Collector — lifecycle funnel (better than T-probe)."""
    if lw is None:
        lw = LW_THIN
    ec = 'white' if c == 'white' else c
    top = [(cx - 1.05 * s, cy + 0.85 * s), (cx + 1.05 * s, cy + 0.85 * s),
           (cx + 0.38 * s, cy + 0.0 * s), (cx - 0.38 * s, cy + 0.0 * s)]
    ax.add_patch(Polygon(top, closed=True,
                         fc=('none' if c == 'white' else '#F8EDE0'),
                         ec=ec, lw=lw, zorder=6))
    ax.add_patch(FancyBboxPatch((cx - 0.26 * s, cy - 0.9 * s), 0.52 * s, 0.95 * s,
                 boxstyle='round,pad=0,rounding_size=0.08',
                 fc=ec, ec='none', zorder=6))


def icon_code(cx, cy, s, c, lw=None):
    """Framework source — code brackets."""
    if lw is None:
        lw = LW_THIN
    ax.add_patch(FancyBboxPatch((cx - 0.95 * s, cy - 0.95 * s), 1.9 * s, 1.9 * s,
                 boxstyle='round,pad=0,rounding_size=0.25',
                 fc='#F4F8FC', ec=c, lw=lw, zorder=6))
    # < >
    ax.plot([cx - 0.35 * s, cx - 0.7 * s, cx - 0.35 * s],
            [cy + 0.45 * s, cy, cy - 0.45 * s], color=c, lw=lw * 1.1, zorder=7,
            solid_joinstyle='round', solid_capstyle='round')
    ax.plot([cx + 0.35 * s, cx + 0.7 * s, cx + 0.35 * s],
            [cy + 0.45 * s, cy, cy - 0.45 * s], color=c, lw=lw * 1.1, zorder=7,
            solid_joinstyle='round', solid_capstyle='round')


def icon_gap(cx, cy, s, c, lw=None):
    """Coverage gap — incomplete ring (missing slice in library P)."""
    if lw is None:
        lw = LW_THIN
    from matplotlib.patches import Wedge
    ax.add_patch(Wedge((cx, cy), 0.95 * s, 40, 320, width=0.38 * s,
                       fc='#EEF3F9', ec=c, lw=lw, zorder=6))
    ax.plot([cx + 0.55 * s, cx + 1.05 * s], [cy + 0.55 * s, cy + 0.95 * s],
            color=c, lw=lw * 1.1, zorder=7, solid_capstyle='round')
    ax.plot([cx + 0.55 * s, cx + 1.05 * s], [cy - 0.55 * s, cy - 0.95 * s],
            color=c, lw=lw * 1.1, zorder=7, solid_capstyle='round')


def icon_bolt(cx, cy, s, c, lw=None):
    """Construct / counterexample attack — lightning bolt."""
    if lw is None:
        lw = LW_THIN
    pts = [(cx + 0.15 * s, cy + 0.95 * s), (cx - 0.45 * s, cy + 0.1 * s),
           (cx + 0.05 * s, cy + 0.1 * s), (cx - 0.2 * s, cy - 0.95 * s),
           (cx + 0.55 * s, cy - 0.05 * s), (cx + 0.1 * s, cy - 0.05 * s)]
    fc = 'white' if c == 'white' else PEACH
    ec = c if c != 'white' else 'white'
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=lw, zorder=6))


def icon_anchor(cx, cy, s, c, lw=None):
    """Ground — evidence grounding."""
    if lw is None:
        lw = LW_THIN
    from matplotlib.patches import Arc as _Arc
    ax.add_patch(Circle((cx, cy + 0.55 * s), 0.28 * s, fc='#EEF3F9', ec=c, lw=lw, zorder=6))
    ax.plot([cx, cx], [cy + 0.27 * s, cy - 0.35 * s], color=c, lw=lw * 1.15, zorder=6)
    ax.add_patch(_Arc((cx, cy - 0.35 * s), 1.2 * s, 0.9 * s, angle=0,
                      theta1=200, theta2=340, ec=c, lw=lw, zorder=6))
    ax.plot([cx - 0.55 * s, cx + 0.55 * s], [cy - 0.35 * s] * 2, color=c, lw=lw, zorder=6)


def icon_verify(cx, cy, s, c, lw=None):
    """Verifier — magnifier + check (SQL checking)."""
    if lw is None:
        lw = LW_THIN
    ec = c if c != 'white' else 'white'
    ax.add_patch(Circle((cx - 0.2 * s, cy + 0.2 * s), 0.58 * s,
                        fc='#EAF1FA' if c != 'white' else 'none', ec=ec, lw=lw, zorder=6))
    ax.plot([cx + 0.25 * s, cx + 0.85 * s], [cy - 0.25 * s, cy - 0.85 * s],
            color=ec, lw=lw * 1.3, zorder=6, solid_capstyle='round')
    chk = 'white' if c == 'white' else GREEN
    ax.plot([cx - 0.42 * s, cx - 0.18 * s, cx + 0.22 * s],
            [cy + 0.18 * s, cy - 0.05 * s, cy + 0.45 * s],
            color=chk, lw=lw * 1.05, zorder=7, solid_capstyle='round', solid_joinstyle='round')


def icon_person(cx, cy, s, c):
    ax.add_patch(Circle((cx, cy + 0.5 * s), 0.42 * s, fc=c, ec='none', zorder=6))
    ax.add_patch(Ellipse((cx, cy - 0.45 * s), 1.5 * s, 1.05 * s, fc=c, ec='none', zorder=6))

# back-compat alias
icon_probe = icon_funnel


# ================= symmetry grid (left bay ↔ lanes ↔ right bay) =================
# Twin lanes share H; denser packing vs v24 (same inset symmetry).
BAND_X, BAND_W, BAND_H = 27.2, 78.6, 2.15
BAND_INSET = 0.32
ON_Y, ON_H = 15.6, 33.2            # compact equal lanes
BAND_ON_Y = ON_Y + ON_H + BAND_INSET
OFF_Y = BAND_ON_Y + BAND_H + BAND_INSET
OFF_H = ON_H
BAND_OFF_Y = OFF_Y + OFF_H + BAND_INSET
LX, LW = 0.8, 23.0
RX, RW = 108.5, 32.5
CX = LX + LW / 2
_OUTER_X, _OUTER_W = 25.0, 118.0
_TITLE_CX = _OUTER_X + _OUTER_W / 2   # true center of outer frame

# ================= outer container =================
_BOTTOM_PAD = 2.75   # room for under-lane output arrow (no text cover)
_TITLE_AIR = 2.05
_TOP_PAD = 1.75
_outer_y = ON_Y - _BOTTOM_PAD
_title_y = BAND_OFF_Y + BAND_H + _TITLE_AIR
_outer_top = _title_y + _TOP_PAD
_outer_h = _outer_top - _outer_y
assert _outer_top < 99.2, (_outer_top, 'title/frame overflow')
shadow(_OUTER_X, _outer_y, _OUTER_W, _outer_h, r=1.6)
box(_OUTER_X, _outer_y, _OUTER_W, _outer_h, fc='#FFFFFF', ec='#111111', lw=LW_THICK, r=1.4, z=1.8)
box(26.4, OFF_Y - 0.45, 80.2, OFF_H + BAND_H + BAND_INSET + 0.95, fc=BLUE_BG, ec='none', r=1.1, z=1.85)
box(26.4, ON_Y - 0.45, 80.2, ON_H + BAND_H + BAND_INSET + 0.95, fc=ORANGE_BG, ec='none', r=1.1, z=1.85)
box(RX - 0.3, ON_Y - 0.45, RW + 0.6, OFF_Y + OFF_H - ON_Y + 0.95, fc=HUB_BG, ec='none', r=1.1, z=1.85)
txt(_TITLE_CX, _title_y, 'TrainAudit', fs=8.6, w='bold', c=BLUE_DEEP)
ax.plot([_OUTER_X + 12.0, _TITLE_CX - 10.2], [_title_y, _title_y], color=BLUE_SOFT, lw=LW_HAIR, zorder=4)
ax.plot([_TITLE_CX + 10.2, _OUTER_X + _OUTER_W - 12.0], [_title_y, _title_y],
        color=BLUE_SOFT, lw=LW_HAIR, zorder=4)

ax.add_patch(FancyBboxPatch((BAND_X, BAND_OFF_Y), BAND_W, BAND_H,
             boxstyle='round,pad=0,rounding_size=0.5',
             fc='#C5D4E8', ec='none', zorder=2))
txt(BAND_X + 0.8, BAND_OFF_Y + BAND_H / 2, 'OFFLINE — mine once per framework',
    fs=5.95, w='bold', c=BLUE_DEEP, ha='left')
ax.add_patch(FancyBboxPatch((BAND_X, BAND_ON_Y), BAND_W, BAND_H,
             boxstyle='round,pad=0,rounding_size=0.5',
             fc='#E8C9A8', ec='none', zorder=2))
txt(BAND_X + 0.8, BAND_ON_Y + BAND_H / 2, 'ONLINE — every training job · fully deterministic · no LLM',
    fs=5.95, w='bold', c='#8A3A12', ha='left')


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

# ================= left bay: offline inputs (= OFF block y/h) =================
dashed_group(LX, OFF_Y, LW, OFF_H, '#E8F0F8', label=None, lc=BLUE_SOFT)
txt(LX + 1.2, OFF_Y + OFF_H - 1.55, 'offline inputs', fs=5.1, w='bold', c=BLUE, ha='left')
_doc_w = LW - 2.2
folded_doc(LX + 1.1, OFF_Y + 16.6, _doc_w, 14.4,
           lines=['template P3:',
                  '  pi_topo: TP > 1',
                  '  pi_precond: tpl=False',
                  '  pi_schema: share cksum',
                  '  (cross-rank replicas)',
                  'from 392 silent errors'],
           title='Pattern Catalog', title_c=BLUE, ec=BLUE, fc='#F7FAFF')
txt(CX, OFF_Y + 15.2, '16 templates · 13 fault classes', fs=3.25, c=GRAY)
folded_doc(LX + 1.1, OFF_Y + 1.35, _doc_w, 12.6,
           lines=['Megatron-LM',
                  'DeepSpeed',
                  'OLMo / OLMo-core',
                  'silent-vs-loud filter',
                  'no raise / assert'],
           title='Framework source', title_c=BLUE, ec='#5A7A9A', fc='#FAFAFA')
txt(CX, OFF_Y + 0.45, 'once per framework', fs=3.25, style='italic', c=GRAY)
arrow((LX + LW, OFF_Y + OFF_H * 0.72), (28.5, OFF_Y + OFF_H * 0.82),
      c=BLUE_SOFT, lw=LW_THICK, ms=10, halo=True)
arrow((LX + LW, OFF_Y + OFF_H * 0.28), (28.5, OFF_Y + OFF_H * 0.48),
      c=BLUE_SOFT, lw=LW_MID, ms=9, cs='arc3,rad=0.12', halo=True)

# ================= Invariant Miner (offline row) — denser stack =================
_my, _mh = OFF_Y, OFF_H
shadow(27, _my, 79, _mh)
box(27, _my, 79, _mh, fc=CARD, ec='#111111', lw=LW_MID + 0.15, r=1.15)
_mt = _my + _mh
badge(30.6, _mt - 2.25, '1')
txt(33.6, _mt - 2.25, 'Invariant Miner', fs=6.35, w='bold', c=BLUE_DEEP, ha='left')
chip(53.2, _mt - 2.25, 1.4, BLUE)
icon_gear(53.2, _mt - 2.25, 0.82, 'white')
txt(104.8, _mt - 2.25, 'LLM-instantiated candidates, untrusted until verified',
    fs=3.55, style='italic', c=GRAY, ha='right')

_comp_y = _mt - 8.55
dashed_group(28.3, _comp_y, 76.4, 4.85, '#F2F6FC', label=None, lc=BLUE_SOFT)
_plus_w, _plus_gap = 15.9, 3.2
_plus_x0 = 29.0
plus_cards = [
    (icon_book, 'Pattern Catalog\ntemplate'),
    (icon_code, 'framework\nsource & docs'),
    (icon_layers, 'evidence\n$\\mathcal{E}^{+}/\\mathcal{E}^{-}$'),
    (icon_gap, 'coverage gap\nin $\\mathcal{P}$'),
]
for i, (ic, lb) in enumerate(plus_cards):
    px = _plus_x0 + i * (_plus_w + _plus_gap)
    box(px, _comp_y + 0.3, _plus_w, 4.25, fc='#FFFFFF', ec=BLUE_SOFT, lw=LW_THIN, r=0.7)
    ic(px + 2.35, _comp_y + 2.4, 0.68, BLUE)
    txt(px + 4.9, _comp_y + 2.4, lb, fs=3.3, ls=1.28, ha='left')
for i in range(3):
    gx = _plus_x0 + (i + 1) * _plus_w + i * _plus_gap + _plus_gap / 2
    txt(gx, _comp_y + 2.4, '+', fs=5.9, w='bold', c=GRAY_LINE)
arrow((66.2, _comp_y + 0.15), (66.2, _comp_y - 0.7), c=BLUE_SOFT, lw=LW_THICK, ms=8, halo=True)

_step_y, _step_h = _my + 11.9, 8.55
dashed_group(28.2, _step_y - 0.3, 38.2, _step_h + 3.25, LAVENDER, label=None, lc=PURPLE)
dashed_group(66.8, _step_y - 0.3, 37.8, _step_h + 3.25, MINT, label=None, lc='#7A9E72')
box(29.0, _step_y + _step_h + 0.45, 12.8, 1.85, fc='#D9C8E4', ec=PURPLE, lw=LW_HAIR, r=0.4, ls=(0, (1.8, 1.2)))
txt(35.4, _step_y + _step_h + 1.35, 'instantiate', fs=3.75, w='bold', c='#6B4F7A')
box(67.6, _step_y + _step_h + 0.45, 16.8, 1.85, fc='#BFD9B8', ec='#5B8A5B', lw=LW_HAIR, r=0.4, ls=(0, (1.8, 1.2)))
txt(76.0, _step_y + _step_h + 1.35, 'verify (decisive)', fs=3.75, w='bold', c='#3D6B3D')
icon_brain(42.4, _step_y + _step_h + 1.35, 0.62, PURPLE)

steps = [
    (29.2, 'Scope', 'fix training phase\n+ parameter class', DARK, 0.7, '#FFFFFF'),
    (48.35, 'Ground', 'evidence $\\mathcal{E}^{+}/\\mathcal{E}^{-}$ from\nframework source & docs', DARK, 0.7, '#FFFFFF'),
    (67.6, 'Construct', 'counterexample attack\n$\\geq$2 CEs; one CE holds\n$\\Rightarrow$ reject outright', RED, 1.1, PEACH),
    (86.85, 'Accept', 'no CE holds $\\wedge$\nConf$(c) \\geq 0.8$\n$(\\theta = 0.8)$', DARK, 0.7, '#FFFFFF'),
]
_title_yy = _step_y + _step_h - 1.7
for x, name, sub, ec, lw, fc in steps:
    box(x, _step_y, 16.3, _step_h, fc=fc, ec=ec, lw=(LW_MID if ec is RED else LW_THIN), r=0.85)
    txt(x + 9.5, _title_yy, name, fs=5.25, w='bold', c=ec)
    ax.plot([x + 2.4, x + 14.0], [_title_yy - 1.05, _title_yy - 1.05], color=ec, lw=LW_HAIR, alpha=0.55, zorder=5)
    txt(x + 8.15, _step_y + 2.15, sub, fs=3.4, ls=1.22)
    if name == 'Scope':
        icon_target(x + 3.0, _title_yy, 0.75, DARK)
    elif name == 'Ground':
        icon_anchor(x + 3.0, _title_yy, 0.72, DARK)
    elif name == 'Accept':
        icon_checkc(x + 3.15, _title_yy, 0.75, DARK)
    elif name == 'Construct':
        chip(x + 2.7, _title_yy, 1.05, RED)
        icon_bolt(x + 2.7, _title_yy + 0.05, 0.68, 'white')
_mid_y = _step_y + _step_h / 2
for x0 in (45.7, 64.95, 84.25):
    arrow((x0, _mid_y), (x0 + 2.35, _mid_y), c=BLUE_SOFT, lw=LW_MID, ms=7)

_rej_y = _step_y - 1.0
arrow((95.0, _step_y - 0.12), (95.0, _rej_y), c=RED_DEEP, lw=LW_THIN, ls=(0, (2.4, 1.5)), style='-')
arrow((95.0, _rej_y), (37.3, _rej_y), c=RED_DEEP, lw=LW_THIN, ls=(0, (2.4, 1.5)), style='-')
arrow((37.3, _rej_y), (37.3, _step_y - 0.15), c=RED_DEEP, lw=LW_THIN, ls=(0, (2.4, 1.5)))
red_x(99.3, _rej_y, 0.45)
txt(66.0, _rej_y - 0.75, 'reject $\\rightarrow$ next iteration', fs=3.4, c=RED, style='italic')

_fsm_box_y, _fsm_box_h = _my + 0.2, 8.85
dashed_group(28.35, _fsm_box_y, 76.3, _fsm_box_h, '#F0F4FA', label=None, lc=BLUE_SOFT)
txt(28.9, _fsm_box_y + _fsm_box_h - 1.4, 'five-stage FSM', fs=3.6, c=GRAY, ha='left')
fsm = [('$S_1$', 'gap analysis', '#3F6BA8'), ('$S_2$', 'evidence retrieval', '#3F8F72'),
       ('$S_3$', 'synthesis+attack', '#B04A4A'), ('$S_4$', 'persistence', '#6E4F9A'),
       ('$S_5$', 'reporting', '#A87830')]
_fsm_r, _fsm_y, _fsm_dx = 1.65, _fsm_box_y + 5.35, 9.05
for k, (sn, lb, cc) in enumerate(fsm):
    cx = 40.6 + k * _fsm_dx
    ax.add_patch(Circle((cx, _fsm_y), _fsm_r, fc=cc, ec='#222222', lw=LW_HAIR, zorder=5))
    txt(cx, _fsm_y - 0.05, sn, fs=4.9, c='white', w='bold', z=6)
    txt(cx, _fsm_y - 2.45, lb, fs=2.95, c=DARK)
    if k < 4:
        arrow((cx + _fsm_r + 0.2, _fsm_y), (cx + _fsm_dx - _fsm_r - 0.2, _fsm_y),
              c='#555555', lw=LW_THIN, ms=6)

cred = [(29.5, 'decisive counterexample gate'),
        (54.0, 'healthy-run replay (verification only)'),
        (83.5, 'Conf$(c) \\geq 0.8$')]
for cx0, tt in cred:
    checkbox(cx0, _my + 0.75, 1.1, GREEN)
    txt(cx0 + 1.85, _my + 1.28, tt, fs=3.3, c='#0A6A68', ha='left')

arrow((103.2, _mid_y), (110.2, _mt - 10.2), c=BLUE_SOFT, lw=LW_THICK, ms=10, cs='arc3,rad=-0.16', halo=True)
txt(106.2, _mt - 6.6, 'accepted', fs=3.85, c=BLUE, style='italic', ha='right')

# ================= Verified Constraint Library (= OFF block y/h, mirrors left) =================
_rc = RX + RW / 2
shadow(RX, OFF_Y, RW, OFF_H)
box(RX, OFF_Y, RW, OFF_H, fc=CARD, ec='#111111', lw=LW_MID + 0.15, r=1.15)
cyl(_rc - 4.1, OFF_Y + OFF_H - 5.7, 8.2, 4.2, fc='#EDE6D8', ec=DARK)
txt(_rc, OFF_Y + OFF_H - 7.95, 'Verified Constraint Library $\\mathcal{P}$', fs=4.7, w='bold', c=BLUE_DEEP)
txt(_rc, OFF_Y + OFF_H - 9.8, 'guarded relational constraints', fs=3.55, style='italic', c=GRAY)
_p3y, _p3h = OFF_Y + 7.9, 15.6
shadow(RX + 2.0, _p3y, RW - 4.0, _p3h, r=0.95)
box(RX + 2.0, _p3y, RW - 4.0, _p3h, fc='#FFFFFF', ec='#222222', lw=LW_THIN, r=0.95)
box(RX + 2.0, _p3y + _p3h - 4.2, RW - 4.0, 4.2, fc='#1E242C', ec='none', r=0.95, z=3)
ax.add_patch(Rectangle((RX + 2.0, _p3y + _p3h - 4.2), RW - 4.0, 1.9, fc='#1E242C', ec='none', zorder=3))
txt(_rc, _p3y + _p3h - 1.4, 'P3 · cross-rank replication', fs=4.4, w='bold', c='white', z=5)
txt(_rc, _p3y + _p3h - 3.15, '(SwitchMLP router sync)', fs=3.4, w='bold', c='#B8C4D8', z=5)
txt(RX + 3.5, _p3y + 5.2, '$\\pi_{topo}$:  TP > 1\n$\\pi_{precond}$:  tpl = False\n   (not tensor-parallel sharded)\n$\\pi_{schema}$:  replicas share\n   checksum',
    fs=3.7, ha='left', ls=1.28)
txt(_rc, OFF_Y + 5.85, 'compiled to parameterized SQL at job start', fs=3.2, c=ORANGE, w='bold')
txt(_rc, OFF_Y + 4.35, 'accepted rules determine trace schema (S0–S6)',
    fs=3.1, c=ORANGE, style='italic')
pill(RX + 2.0, OFF_Y + 1.05, RW - 4.0, 2.6, '', GREEN_BG, GREEN)
txt(_rc, OFF_Y + 2.35, '$\\checkmark$  0 false positives over 764 clean evals', fs=3.35, c='#0A6A68', z=6)
arrow((RX, OFF_Y + 1.5), (55.0, ON_Y + ON_H - 0.5), c=ORANGE, lw=LW_MID, ms=10,
      cs='arc3,rad=0.10', halo=True)
arrow((_rc - 2.0, OFF_Y), (90.5, ON_Y + ON_H * 0.55), c=ORANGE, lw=LW_THICK, ms=11,
      cs='arc3,rad=0.16', halo=True)

# ================= Data Collector / DuckDB / Verifier — gutters = offline arrow span =====
_oy, _oh = ON_Y, ON_H
_ot = _oy + _oh
# gutters ≈ offline inputs→Miner span (~4); shafts visible, not head-only
_c0, _cw = 27.0, 33.0          # Collector → 60.0
_db0, _dbw = 64.0, 9.0         # DuckDB  → 73.0  (gap 4.0)
_v0, _vw = 77.0, 28.6          # Verifier → 105.6 (gap 4.0; →RX gap 2.9)
shadow(_c0, _oy, _cw, _oh)
box(_c0, _oy, _cw, _oh, fc=CARD, ec='#111111', lw=LW_MID + 0.15, r=1.15)
badge(_c0 + 3.6, _ot - 2.2, '2')
txt(_c0 + 6.6, _ot - 2.2, 'Data Collector', fs=6.2, w='bold', c=ORANGE, ha='left')
chip(_c0 + 27.8, _ot - 2.2, 1.35, ORANGE)
icon_funnel(_c0 + 27.8, _ot - 2.2, 0.76, 'white')
txt(_c0 + 1.4, _ot - 4.55, 'no user code changes · adapter 30–150 LoC', fs=3.65,
    style='italic', c=GRAY, ha='left')

anchors = ['before-\nforward', 'after-\nforward', 'main-grad-\nin-backward',
           'after-\nbackward', 'before-\noptimizer']
tints = ['#F0C89A', '#EBB886', '#E5A872', '#DF985E', '#D8884A']
_chev_mid = _ot - 8.85
for i, a in enumerate(anchors):
    x = _c0 + 1.2 + i * 6.45
    w, n = 7.0, 1.1
    pts = [(x, _chev_mid - 2.55), (x + w - n, _chev_mid - 2.55), (x + w, _chev_mid),
           (x + w - n, _chev_mid + 2.55), (x, _chev_mid + 2.55), (x + n, _chev_mid)]
    ax.add_patch(Polygon(pts, closed=True, fc=tints[i], ec=ORANGE, lw=LW_HAIR, zorder=3))
    txt(x + w / 2 + 0.15, _chev_mid, a, fs=3.25, family='monospace', w='bold', z=5)
txt(_c0 + _cw / 2, _chev_mid - 4.45, '+ auxiliary taps (ckpt · all_reduce · snapshot) = 8 hookpoints',
    fs=3.4, c=GRAY)
txt(_c0 + _cw / 2, _chev_mid - 6.55, 'GPU-side scalar reductions (checksums only when required)', fs=3.5)
_rec_y = _oy + 2.15
box(_c0 + 1.6, _rec_y, _cw - 3.2, 6.35, fc='#FBF6EF', ec=ORANGE, lw=LW_THIN, r=0.8)
txt(_c0 + _cw / 2, _rec_y + 4.85, 'captured record (S0):', fs=3.8, c=GRAY)
_pw = [('cksum', 4.6), ('param_name', 7.2), ('rank', 3.8), ('step', 3.8), ('stage', 4.6)]
_x = _c0 + 1.6 + ((_cw - 3.2) - (sum(w for _, w in _pw) + 0.55 * 4)) / 2
for _t, _w in _pw:
    pill(_x, _rec_y + 0.7, _w, 2.7, _t, 'white', ORANGE, fs=3.35)
    _x += _w + 0.55

# DuckDB
_db_y = _oy + 3.6
dashed_group(_db0, _db_y, _dbw, 19.4, '#FFF4E8', label=None, lc=ORANGE)
cyl(_db0 + 0.7, _db_y + 7.2, _dbw - 1.4, 10.6, fc='#F5E6D0', ec=ORANGE)
for _yy in (_db_y + 10.6, _db_y + 13.3):
    ax.plot([_db0 + 1.4, _db0 + _dbw - 1.4], [_yy, _yy], color=ORANGE, lw=LW_HAIR, alpha=0.45, zorder=4)
txt(_db0 + _dbw / 2, _db_y + 4.7, 'DuckDB\ntrace DB', fs=4.0, w='bold', c=ORANGE, ls=1.18)
txt(_db0 + _dbw / 2, _db_y + 1.05, 'two-step\nsliding window', fs=3.25, c=GRAY, ls=1.18)
# online bridges — same weight as offline input / Accept→Lib arrows
_bridge_y = _oy + _oh * 0.52
arrow((_c0 + _cw, _bridge_y), (_db0, _bridge_y), c=ORANGE, lw=LW_THICK, ms=11, halo=True)
arrow((_db0 + _dbw, _bridge_y), (_v0, _bridge_y), c=ORANGE, lw=LW_THICK, ms=11, halo=True)

# ================= Verifier =================
shadow(_v0, _oy, _vw, _oh)
box(_v0, _oy, _vw, _oh, fc=CARD, ec='#111111', lw=LW_MID + 0.15, r=1.15)
badge(_v0 + 3.6, _ot - 2.2, '3')
txt(_v0 + 6.6, _ot - 2.2, 'Verifier', fs=6.2, w='bold', c=ORANGE, ha='left')
chip(_v0 + 16.6, _ot - 2.2, 1.35, ORANGE)
icon_verify(_v0 + 16.6, _ot - 2.2, 0.76, 'white')
txt(_v0 + 1.2, _ot - 4.55, 'deterministic SQL checking — no LLM', fs=3.65,
    style='italic', c=GRAY, ha='left')
_vq_y, _vq_h = _oy + 10.2, 14.85
box(_v0 + 1.2, _vq_y, 12.8, _vq_h, fc=MINT, ec='#7A9E72', lw=LW_HAIR, r=0.85, ls=(0, (2.6, 1.8)))
box(_v0 + 1.7, _vq_y + 0.5, 11.8, _vq_h - 1.0, fc='#FFFFFF', ec='#B8C8B4', lw=LW_HAIR, r=0.65)
txt(_v0 + 7.6, _vq_y + _vq_h - 1.4, 'at job start', fs=4.15, w='bold')
ax.plot([_v0 + 2.6, _v0 + 12.6], [_vq_y + _vq_h - 2.6, _vq_y + _vq_h - 2.6],
        color=GRAY_LINE, lw=LW_HAIR, zorder=5)
txt(_v0 + 7.6, _vq_y + 5.85, 'extract topology $\\tau$\ntopology-aware\npruning · lineage:\nreplicated vs.\nsharded', fs=3.45, ls=1.32)
box(_v0 + 14.6, _vq_y, 13.4, _vq_h, fc='#1E242C', ec='#111111', lw=LW_THIN, r=0.85)
txt(_v0 + 21.3, _vq_y + _vq_h - 1.4, 'per step, per rule', fs=4.05, w='bold', c='#9DB8E8')
ax.plot([_v0 + 15.8, _v0 + 26.8], [_vq_y + _vq_h - 2.6, _vq_y + _vq_h - 2.6],
        color='#7A8A9A', lw=LW_HAIR, zorder=5)
_KW, _TX, _PI = '#61AFEF', '#E8EBF0', '#E5C07B'
_sql_y = _vq_y + _vq_h - 4.4
txt(_v0 + 15.8, _sql_y, 'WHERE', fs=3.45, ha='left', family='monospace', c=_KW, w='bold')
txt(_v0 + 19.4, _sql_y, '$\\pi_{topo}\\wedge\\pi_{precond}$', fs=3.3, ha='left', c=_PI, w='bold')
txt(_v0 + 15.8, _sql_y - 2.35, 'GROUP BY', fs=3.45, ha='left', family='monospace', c=_KW, w='bold')
txt(_v0 + 16.3, _sql_y - 4.35, 'param, step', fs=3.45, ha='left', family='monospace', c=_TX, w='bold')
txt(_v0 + 15.8, _sql_y - 6.55, 'HAVING', fs=3.45, ha='left', family='monospace', c=_KW, w='bold')
txt(_v0 + 19.5, _sql_y - 6.55, '$\\pi_{schema}$', fs=3.3, ha='left', c=_PI, w='bold')
txt(_v0 + 16.3, _sql_y - 8.45, 'violated', fs=3.35, ha='left', family='monospace', c=_TX, w='bold')
txt(_v0 + _vw / 2, _oy + 7.7, 'empty result = holds · non-empty IS the violation set',
    fs=3.35, style='italic')
txt(_v0 + _vw / 2, _oy + 4.7, 'coarse$\\rightarrow$fine: checksum$\\rightarrow$module$\\rightarrow$param$\\rightarrow$bitwise',
    fs=3.15, c=GRAY)
# Verifier → violation (same caliber as offline Accept→Lib)
arrow((_v0 + _vw, _bridge_y), (RX, _bridge_y), c=RED_DEEP, lw=LW_THICK, ms=11, halo=True)

# ================= Violation panel — top-down tight stack (kill mid-panel air) =================
shadow(RX, ON_Y, RW, ON_H)
box(RX, ON_Y, RW, ON_H, fc=VIOL_BG, ec=RED, lw=LW_MID, r=1.05, ls=(0, (3.0, 2.0)))
_vt = ON_Y + ON_H
_cols = [('param', 8.6), ('rank', 3.4), ('cksum', 7.2)]
_rows = [('router.weight', '0', 'cksum$_A$', False), ('router.weight', '1', 'cksum$_B$', True)]
_items = ['violating constraint', 'affected parameter',
          'divergent ranks', 'first violating step']
_tw = sum(w for _, w in _cols)
_tx, _rh = _rc - _tw / 2, 2.3
# title → rule → table → × → checks (fixed small gaps; bottom kept for exit arrow)
_vtitle_y = _vt - 1.5
txt(_rc, _vtitle_y, 'violation report', fs=5.05, w='bold', c=RED)
ax.plot([_rc - 8.2, _rc + 8.2], [_vtitle_y - 1.15, _vtitle_y - 1.15],
        color=RED, lw=LW_HAIR, alpha=0.7, zorder=5)
_ty = _vtitle_y - 2.0
_cx = _tx
for _h, _wd in _cols:
    ax.add_patch(Rectangle((_cx, _ty - _rh), _wd, _rh, fc='#F5F5F5', ec='#B3B3B3', lw=LW_HAIR, zorder=4))
    txt(_cx + _wd / 2, _ty - _rh / 2, _h, fs=3.35, c=GRAY, z=5)
    _cx += _wd
for _r, (_p, _rk, _ck, _bad) in enumerate(_rows):
    _cx = _tx
    _yy = _ty - _rh * (_r + 2)
    for _v, (_h, _wd) in zip((_p, _rk, _ck), _cols):
        ax.add_patch(Rectangle((_cx, _yy), _wd, _rh, fc='#F8CECC' if _bad else 'white',
                               ec='#B3B3B3', lw=LW_HAIR, zorder=4))
        txt(_cx + _wd / 2, _yy + _rh / 2, _v, fs=3.35, family='monospace',
            c=RED if _bad else DARK, z=5)
        _cx += _wd
_tbl_bot = _ty - 3 * _rh
_div_y = _tbl_bot - 1.1
txt(_rc, _div_y, '$\\mathbf{\\times}$ divergent checksums', fs=3.7, c=RED)
_chk0, _pitch = _div_y - 1.55, 2.45
for _k, _it in enumerate(_items):
    _yy = _chk0 - _k * _pitch
    checkbox(RX + 3.2, _yy - 0.5, 1.15, RED)
    txt(RX + 5.25, _yy, _it, fs=3.75, ha='left')
# viol → online output: under-lane (clear of ❷/❸/footer text), tip stops at bay edge
_uy = ON_Y - 1.05
_tip_y = ON_Y + 16.5                 # mid of folded Violation Report (not italic footer)
arrow((RX + 1.5, ON_Y + 1.2), (RX + 1.5, _uy), c=DARK, lw=LW_THICK, style='-',
      halo=True, z=3.4)
arrow((RX + 1.5, _uy), (LX + LW + 3.8, _uy), c=DARK, lw=LW_THICK, style='-',
      halo=True, z=3.4)
# soft rise into bay edge — head outside text (mirrors offline inputs→Miner approach)
arrow((LX + LW + 3.8, _uy), (LX + LW + 0.15, _tip_y), c=DARK, lw=LW_THICK, ms=11,
      cs='arc3,rad=-0.18', halo=True, z=3.4)

# ================= left bay: online output =================
dashed_group(LX, ON_Y, LW, ON_H, '#FBF0EC', label=None, lc=ORANGE_SOFT)
txt(LX + 1.2, ON_Y + ON_H - 1.55, 'online output', fs=5.1, w='bold', c=ORANGE, ha='left')
folded_doc(LX + 1.1, ON_Y + 5.1, _doc_w, ON_H - 9.2,
           lines=['violating constraint: P3',
                  'affected parameter:',
                  '  router.weight',
                  'divergent ranks: 0 vs 1',
                  '(SwitchMLP router sync)',
                  'non-empty SQL result'],
           title='Violation Report', title_c=RED, ec=RED, fc='#FFF8F8',
           hl={0: RED, 2: RED})
txt(CX, ON_Y + 3.3, 'empty result = holds', fs=3.7, style='italic', c=GRAY)
txt(CX, ON_Y + 1.4, 'non-empty = violation set', fs=3.7, style='italic', c=RED)

# ================= legend (tight under outer) =================
_leg_y = _outer_y - 4.2
ax.add_patch(FancyBboxPatch((25, _leg_y), 117.5, 3.45,
             boxstyle='round,pad=0,rounding_size=0.55',
             fc='#F4F5F7', ec='#B8B8B8', lw=LW_HAIR, zorder=2))
_leg_mid = _leg_y + 1.72
txt(26.7, _leg_mid, 'Legend', fs=4.45, w='bold', ha='left')
ax.plot([36.2, 43.2], [_leg_mid, _leg_mid], color=BLUE_SOFT, lw=LW_THICK)
txt(44.7, _leg_mid, 'offline mining', fs=4.15, ha='left')
ax.plot([62.2, 69.2], [_leg_mid, _leg_mid], color=ORANGE, lw=LW_THICK)
txt(70.7, _leg_mid, 'online checking', fs=4.15, ha='left')
ax.plot([92.2, 99.2], [_leg_mid, _leg_mid], color=RED_DEEP, lw=LW_MID, ls=(0, (2.4, 1.5)))
txt(100.7, _leg_mid, 'reject / violation', fs=4.15, ha='left')
txt(141.5, _leg_mid, 'LLM only inside offline Miner · online path: no LLM',
    fs=3.9, style='italic', c=GRAY, ha='right')

# crop dead canvas above title / below legend (keeps geometry, tight print frame)
ax.set_ylim(_leg_y - 0.45, min(100.0, _outer_top + 1.6))
fig.savefig('fig_overview_v27.pdf')
fig.savefig('/tmp/fig_overview_v27.png', dpi=240)
print('done v27')
print(f'  title_y={_title_y:.2f} outer=[{_outer_y:.2f},{_outer_top:.2f}] '
      f'bottom_pad={_BOTTOM_PAD} leg_y={_leg_y:.2f} ylim={ax.get_ylim()}')
