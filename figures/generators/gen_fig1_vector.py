#!/usr/bin/env python3
"""Fig.1 motivating example, polished vector version.

Card-based layout: (a) two id timelines with a highlighted first post-reset
slot; (b) two execution cards (Expected vs Observed) with operation chips
and a resulting buffer bar. Design language: blue = expected, red =
observed/buggy, orange = triggering event; soft card fills, solid-shaft
arrows, no text overflow (all labels measured against their containers).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import rcParams

rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,
})

BLUE = '#1f4e9c'
BLUEL = '#dce7f7'
BLUEF = '#f3f7fd'
RED = '#b3261e'
REDL = '#f7dcda'
REDF = '#fdf4f3'
ORANGE = '#d97706'
ORANGEL = '#fdeeda'
GRAY = '#6b6b6b'
DARK = '#1c1c1c'
SHADOW = '#d8d8d8'

W, H = 3.45, 2.58
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

FS = 5.6
FSS = 5.0
MONO = 5.0


def card(x, y, w, h, ec, fc, lw=0.9, r=1.8, shadow=True):
    if shadow:
        ax.add_patch(FancyBboxPatch((x + 0.7, y - 0.7), w, h,
                                    boxstyle=f'round,pad=0,rounding_size={r}',
                                    fc=SHADOW, ec='none', zorder=1))
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f'round,pad=0,rounding_size={r}',
                                fc=fc, ec=ec, lw=lw, zorder=2))


def chip(x, y, w, h, text, ec, fc, tc=None, bold=False, mono=True, fs=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle='round,pad=0,rounding_size=1.1',
                                fc=fc, ec=ec, lw=0.8, zorder=4))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs or MONO, family='monospace' if mono else 'serif',
            color=tc or ec, fontweight='bold' if bold else 'normal',
            zorder=5)


def arrow(x0, y0, x1, y1, color, lw=1.0, ls='-', z=6):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), zorder=z,
                arrowprops=dict(arrowstyle='-|>,head_width=0.22,'
                                           'head_length=0.36',
                                color=color, lw=lw, ls=ls,
                                shrinkA=0, shrinkB=0))


def panel_title(x, y, text):
    ax.text(x, y, text, fontsize=FS + 0.4, fontweight='bold', color=DARK,
            va='center', zorder=5)


# ============================ panel (a) ============================
panel_title(3, 96.5, '(a) Reset shifts the first post-reset slot')

AY_E, AY_O = 84.0, 71.5     # row centers
BW, BH = 12.0, 6.4          # id chip size
XS = [36, 54, 72]           # chip left edges

# highlight column for first post-reset slot
ax.add_patch(FancyBboxPatch((XS[0] - 2.0, AY_O - BH / 2 - 2.0),
                            BW + 4.0, (AY_E - AY_O) + BH + 4.0,
                            boxstyle='round,pad=0,rounding_size=1.6',
                            fc='#f0f0f0', ec='#cfcfcf', lw=0.6,
                            ls=(0, (2, 1.6)), zorder=1))
ax.text(XS[0] + BW / 2, AY_E + BH / 2 + 4.6, 'first post-reset slot',
        fontsize=FSS - 0.2, color=GRAY, ha='center', va='center', zorder=5)

# expected row
ax.text(3, AY_E, 'Expected\ncontinuation', fontsize=FSS, color=DARK,
        va='center', linespacing=1.25, zorder=5)
for x, s in zip(XS, ['id=1', 'id=2', 'id=3']):
    chip(x, AY_E - BH / 2, BW, BH, s, BLUE, BLUEL)
for x in XS[:-1]:
    arrow(x + BW, AY_E, x + BW + 6.0, AY_E, BLUE)
arrow(XS[-1] + BW, AY_E, XS[-1] + BW + 5.0, AY_E, BLUE)
ax.text(XS[-1] + BW + 7.5, AY_E, r'$\cdots$', fontsize=FS + 1, color=BLUE,
        va='center')

# observed row
ax.text(3, AY_O, 'Observed\nafter reset', fontsize=FSS, color=DARK,
        va='center', linespacing=1.25, zorder=5)
chip(17.5, AY_O - BH / 2, 12.5, BH, 'reset\nto -1', ORANGE, ORANGEL,
     mono=False, fs=FSS - 0.2)
arrow(30.0, AY_O, 34.0, AY_O, ORANGE)
for i, (x, s) in enumerate(zip(XS, ['id=0', 'id=1', 'id=2'])):
    chip(x, AY_O - BH / 2, BW, BH, s, RED, REDL if i == 0 else 'white',
         bold=(i == 0))
for x in XS[:-1]:
    arrow(x + BW, AY_O, x + BW + 6.0, AY_O, RED)
arrow(XS[-1] + BW, AY_O, XS[-1] + BW + 5.0, AY_O, RED)
ax.text(XS[-1] + BW + 7.5, AY_O, r'$\cdots$', fontsize=FS + 1, color=RED,
        va='center')

# mismatch callout under highlighted slot
ax.plot([XS[0] + BW / 2, XS[0] + BW / 2],
        [AY_O - BH / 2 - 2.0, AY_O - BH / 2 - 4.6],
        color=RED, lw=0.7, zorder=5)
ax.text(XS[0] + BW / 2, AY_O - BH / 2 - 7.4,
        'expected id=1, observed id=0',
        fontsize=FSS - 0.2, color=RED, ha='center', va='center', zorder=5)
ax.text(XS[0] + BW / 2, AY_O - BH / 2 - 11.2,
        'violates the sequencing invariant',
        fontsize=FSS - 0.2, color=RED, ha='center', va='center',
        style='italic', zorder=5)

# ============================ panel (b) ============================
ax.plot([3, 97], [53.5, 53.5], color='#d9d9d9', lw=0.6)
panel_title(3, 50.3, '(b) Expected vs. observed execution')

CY, CH = 8.0, 38.5           # card bottom & height
CW = 45.0
HDR = 5.8                    # header band height

# ---------- expected card ----------
card(3, CY, CW, CH, BLUE, BLUEF)
ax.add_patch(FancyBboxPatch((3, CY + CH - HDR), CW, HDR,
                            boxstyle='round,pad=0,rounding_size=1.8',
                            fc=BLUE, ec='none', zorder=3))
ax.text(3 + CW / 2, CY + CH - HDR / 2, 'Expected',
        fontsize=FSS, color='white', ha='center', va='center',
        fontweight='bold', zorder=5)

GW, GH = 15.0, 6.0
g1x, g2x = 7.5, 25.5
gy = CY + CH - HDR - 11.2         # G row bottom
oy = gy - 8.8                     # op row bottom
chip(g1x, gy, GW, GH, 'G1 (id=0)', BLUE, 'white')
chip(g2x, gy, GW, GH, 'G2 (id=0)', BLUE, 'white')
chip(g1x, oy, GW, 5.2, 'accumulate', BLUE, BLUEL, mono=False,
     fs=FSS - 0.8)
chip(g2x, oy, GW, 5.2, 'accumulate', BLUE, BLUEL, mono=False,
     fs=FSS - 0.8)
arrow(g1x + GW / 2, gy, g1x + GW / 2, oy + 5.2, BLUE, lw=0.8)
arrow(g2x + GW / 2, gy, g2x + GW / 2, oy + 5.2, BLUE, lw=0.8)
ax.text(3 + CW / 2, gy + GH + 3.2, 'one accumulation window',
        fontsize=FSS - 0.5, color=BLUE, ha='center', va='center',
        style='italic', zorder=5)
# buffer bar
chip(6.5, CY + 2.4, CW - 7, 5.8, 'buffer = state + G1 + G2', BLUE, 'white',
     bold=True)
arrow(3 + CW / 2, oy, 3 + CW / 2, CY + 8.6, BLUE, lw=0.8)

# ---------- observed card ----------
OX = 52
card(OX, CY, CW, CH, RED, REDF)
ax.add_patch(FancyBboxPatch((OX, CY + CH - HDR), CW, HDR,
                            boxstyle='round,pad=0,rounding_size=1.8',
                            fc=RED, ec='none', zorder=3))
ax.text(OX + CW / 2, CY + CH - HDR / 2, 'Observed after reset',
        fontsize=FSS, color='white', ha='center', va='center',
        fontweight='bold', zorder=5)

h1x, h2x = OX + 4.5, OX + 25.5
chip(h1x, gy, GW, GH, 'G1 (id=0)', RED, 'white')
chip(h2x, gy, GW, GH, 'G2 (id=1)', RED, REDL, bold=True)
chip(h1x, oy, GW, 5.2, 'accumulate', RED, 'white', mono=False,
     fs=FSS - 0.8)
chip(h2x, oy, GW, 5.2, 'copy', RED, REDL, mono=False,
     bold=True, fs=FSS - 0.8)
arrow(h1x + GW / 2, gy, h1x + GW / 2, oy + 5.2, RED, lw=0.8)
arrow(h2x + GW / 2, gy, h2x + GW / 2, oy + 5.2, RED, lw=0.8)
# spurious boundary between the two columns
bx = (h1x + GW + h2x) / 2
ax.plot([bx, bx], [oy - 1.2, gy + GH + 1.6], color=ORANGE, lw=1.1,
        ls=(0, (2.4, 1.6)), zorder=6)
ax.text(OX + CW / 2, gy + GH + 3.2, 'spurious window boundary',
        fontsize=FSS - 0.5, color=ORANGE, ha='center', va='center',
        style='italic', zorder=7,
        bbox=dict(boxstyle='round,pad=0.15', fc=REDF, ec='none'))
# buffer bar
chip(OX + 3.5, CY + 2.4, CW - 7, 5.8, 'buffer = G2', RED, REDL, bold=True)
arrow(h2x + GW / 2, oy, h2x + GW / 2, CY + 8.6, RED, lw=0.8)

# takeaway under observed card
ax.text(OX + CW / 2, CY - 4.2, "G1's gradient is silently discarded",
        fontsize=FSS, color=RED, style='italic', ha='center', va='center')

fig.savefig('figure1_vector.pdf', bbox_inches='tight', pad_inches=0.02)
print('done')
