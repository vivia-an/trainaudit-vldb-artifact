#!/usr/bin/env python3
"""Fig.1 motivating example — v5b (fix v5 overflow; keep P0+P1 cues).

Reserved vertical lanes (no overlap):
  (a) chips → callout lane → bridge lane → (b) title → cards → footer lane

Footer = one combined takeaway (avoids G1-line ∩ missing-check collision).
discarded sits in left-column air gap above buffer, never on buffer edge.

Output: figure1_vector_v5.pdf
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib import rcParams

rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,
})

BLUE = '#1a4a8a'
BLUE_FILL = '#e8f0fa'
BLUE_SOFT = '#f5f8fc'
RED = '#a61b1b'
RED_FILL = '#f8e8e8'
RED_SOFT = '#fdf6f6'
ORANGE = '#c45c06'
ORANGE_FILL = '#fcebd9'
INK = '#1a1a1a'
MUTED = '#5a5a5a'
RULE = '#c8c8c8'
HL = '#ececec'
SHADOW = '#ececec'

# taller canvas so lanes do not collide
W, H = 3.42, 3.05
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')
fig.patch.set_facecolor('white')

FS = 5.9
FSS = 5.35
MONO = 5.2


def rounded(x, y, w, h, fc, ec, lw=0.85, r=1.4, z=2, ls='-'):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f'round,pad=0,rounding_size={r}',
        fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=z)
    ax.add_patch(p)
    return p


def chip(x, y, w, h, text, ec, fc, bold=False, fs=None, family='monospace'):
    rounded(x, y, w, h, fc, ec, lw=0.9, r=1.1, z=4)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs or MONO, family=family, color=ec,
            fontweight='bold' if bold else 'normal', zorder=5)


def shaft_arrow(x0, y0, x1, y1, color, lw=1.0, z=6):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle='-|>,head_length=2.0,head_width=1.35',
        mutation_scale=1.0,
        lw=lw, color=color, zorder=z,
        shrinkA=0.5, shrinkB=0.7,
        connectionstyle='arc3,rad=0'))


def card_shell(x, y, w, h, soft, border):
    rounded(x + 0.3, y - 0.3, w, h, SHADOW, 'none', lw=0, r=1.8, z=1)
    rounded(x, y, w, h, soft, border, lw=1.05, r=1.8, z=2)


# ── lane Y coordinates (top → bottom) ──────────────────────────────
# title(a) 97 | chips ~86/74 | callout ends ≥62 | bridge 58 | (b)title 54
# cards 10–50 | footer 4
YA_TITLE = 97.2
YE, YO = 87.2, 75.8          # expected / observed chip centers
Y_CALL_1 = 67.2              # expected id=1 · observed id=0
Y_CALL_2 = 63.9              # violates…
Y_BRIDGE = 58.2
YB_TITLE = 54.4
CARD_Y, CARD_H = 11.5, 39.5
Y_FOOTER = 4.2

# ═══════════════════════════════════════════════════════════════════
# (a)
# ═══════════════════════════════════════════════════════════════════
ax.text(2.5, YA_TITLE, '(a)  Reset shifts the first post-reset slot',
        fontsize=FS, fontweight='bold', color=INK, va='center')

chip_w, chip_h = 11.6, 5.8
xs = [36.0, 53.2, 70.4]

rounded(xs[0] - 1.7, YO - chip_h / 2 - 1.4,
        chip_w + 3.4, (YE - YO) + chip_h + 2.8,
        HL, '#bdbdbd', lw=0.55, r=1.5, z=1, ls=(0, (2.2, 1.4)))
ax.text(xs[0] + chip_w / 2, YE + chip_h / 2 + 3.5,
        'first post-reset slot',
        fontsize=FSS - 0.15, color=MUTED, ha='center', va='center')

# left labels — keep clear of reset chip (reset starts x=16.8)
ax.text(2.2, YE, 'Expected\ncontinuation', fontsize=FSS - 0.1, color=INK,
        va='center', ha='left', linespacing=1.15)
ax.text(2.2, YO, 'Observed\nafter reset', fontsize=FSS - 0.1, color=INK,
        va='center', ha='left', linespacing=1.15)

for x, lab in zip(xs, ['id=1', 'id=2', 'id=3']):
    chip(x, YE - chip_h / 2, chip_w, chip_h, lab, BLUE, BLUE_FILL)
for i in range(2):
    shaft_arrow(xs[i] + chip_w + 0.2, YE, xs[i + 1] - 0.2, YE, BLUE)
shaft_arrow(xs[2] + chip_w + 0.2, YE, xs[2] + chip_w + 4.8, YE, BLUE)
ax.text(xs[2] + chip_w + 6.4, YE, r'$\cdots$', fontsize=FS + 0.3,
        color=BLUE, va='center', ha='left')

# reset chip — ends before first id column (xs[0]=36)
chip(17.0, YO - chip_h / 2, 14.2, chip_h, 'reset to -1', ORANGE, ORANGE_FILL,
     bold=True, fs=FSS - 0.15, family='serif')
shaft_arrow(31.4, YO, xs[0] - 0.25, YO, ORANGE)
for i, (x, lab) in enumerate(zip(xs, ['id=0', 'id=1', 'id=2'])):
    chip(x, YO - chip_h / 2, chip_w, chip_h, lab, RED,
         RED_FILL if i == 0 else 'white', bold=(i == 0))
for i in range(2):
    shaft_arrow(xs[i] + chip_w + 0.2, YO, xs[i + 1] - 0.2, YO, RED)
shaft_arrow(xs[2] + chip_w + 0.2, YO, xs[2] + chip_w + 4.8, YO, RED)
ax.text(xs[2] + chip_w + 6.4, YO, r'$\cdots$', fontsize=FS + 0.3,
        color=RED, va='center', ha='left')

cx = xs[0] + chip_w / 2
ax.add_line(Line2D([cx, cx], [YO - chip_h / 2 - 1.4, Y_CALL_1 + 1.6],
                   color=RED, lw=0.7, zorder=5))
ax.text(cx, Y_CALL_1, 'expected id=1  ·  observed id=0',
        fontsize=FSS - 0.1, color=RED, ha='center', va='center')
ax.text(cx, Y_CALL_2, 'violates the sequencing invariant',
        fontsize=FSS - 0.1, color=RED, ha='center', va='center',
        style='italic')

# ═══════════════════════════════════════════════════════════════════
# bridge lane (clear air above/below)
# ═══════════════════════════════════════════════════════════════════
ax.add_line(Line2D([2.5, 26], [Y_BRIDGE, Y_BRIDGE], color=RULE, lw=0.5, zorder=1))
ax.add_line(Line2D([74, 97.5], [Y_BRIDGE, Y_BRIDGE], color=RULE, lw=0.5, zorder=1))
ax.text(50, Y_BRIDGE, r'off-by-one  $\Rightarrow$  wrong buffer op',
        fontsize=FSS - 0.15, color=MUTED, ha='center', va='center',
        style='italic', zorder=5,
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=RULE, lw=0.45))

ax.text(2.5, YB_TITLE, '(b)  Expected vs. observed execution',
        fontsize=FS, fontweight='bold', color=INK, va='center')

# ═══════════════════════════════════════════════════════════════════
# (b) cards
# ═══════════════════════════════════════════════════════════════════
card_w = 45.5
gap = 100.0 - 2.5 - 2.5 - 2 * card_w
card_x = [2.5, 2.5 + card_w + gap]
hdr_h = 5.2
inset = 5.0
g_w, g_h = 15.4, 5.4
op_w, op_h = 15.4, 5.1  # wide enough for "accumulate" + pad


def exec_card(x, title, title_fc, border, soft, g1, g2, op1, op2,
              op1_fill, op2_fill, op2_bold, buffer, buffer_fill,
              top_note, top_note_c, discard_left=False):
    card_shell(x, CARD_Y, card_w, CARD_H, soft, border)
    rounded(x, CARD_Y + CARD_H - hdr_h, card_w, hdr_h, title_fc, title_fc,
            lw=0, r=1.8, z=3)
    ax.add_patch(Rectangle((x, CARD_Y + CARD_H - hdr_h), card_w, 2.0,
                           fc=title_fc, ec='none', zorder=3))
    ax.text(x + card_w / 2, CARD_Y + CARD_H - hdr_h / 2, title,
            fontsize=FSS - 0.05, color='white', ha='center', va='center',
            fontweight='bold', zorder=5)

    g1x = x + inset
    g2x = x + card_w - inset - g_w
    gy = CARD_Y + CARD_H - hdr_h - 10.4
    chip(g1x, gy, g_w, g_h, g1, border, 'white')
    chip(g2x, gy, g_w, g_h, g2, border,
         RED_FILL if op2_bold else 'white', bold=op2_bold)

    ax.text(x + card_w / 2, gy + g_h + 2.2, top_note,
            fontsize=FSS - 0.25, color=top_note_c, ha='center', va='center',
            style='italic', zorder=5)

    oy = gy - 9.2
    chip(g1x + (g_w - op_w) / 2, oy, op_w, op_h, op1, border, op1_fill,
         fs=FSS - 0.55, family='serif')
    chip(g2x + (g_w - op_w) / 2, oy, op_w, op_h, op2, border, op2_fill,
         bold=op2_bold, fs=FSS - 0.55, family='serif')

    c1 = g1x + g_w / 2
    c2 = g2x + g_w / 2
    shaft_arrow(c1, gy, c1, oy + op_h + 0.8, border, lw=0.95)
    shaft_arrow(c2, gy, c2, oy + op_h + 0.8, border, lw=0.95)

    by = CARD_Y + 2.8
    buf_h = 5.2
    chip(x + 3.6, by, card_w - 7.2, buf_h, buffer, border, buffer_fill,
         bold=True, fs=MONO - 0.2)

    if discard_left:
        bx = (g1x + g_w + g2x) / 2.0
        ax.add_line(Line2D([bx, bx], [oy - 0.2, gy + g_h + 0.4],
                           color=ORANGE, lw=1.1, linestyle=(0, (2.4, 1.4)),
                           zorder=6))
        # air gap between op bottom and buffer top — place ×+label mid-gap
        gap_mid = (oy + by + buf_h) / 2.0
        ax.text(c1, gap_mid + 0.9, r'$\times$', fontsize=FS + 2.2,
                color=RED, ha='center', va='center', fontweight='bold',
                zorder=8)
        ax.text(c1, gap_mid - 1.3, 'discarded',
                fontsize=FSS - 0.45, color=RED, ha='center', va='center',
                style='italic', zorder=8)
        shaft_arrow(c2, oy, c2, by + buf_h + 0.2, border, lw=0.95)
    else:
        mid = x + card_w / 2
        mid_y = (oy + by + buf_h) / 2.0
        for gx in (c1, c2):
            ax.add_line(Line2D([gx, gx], [oy, mid_y], color=border, lw=0.9,
                               zorder=6))
            ax.add_line(Line2D([gx, mid], [mid_y, mid_y], color=border, lw=0.9,
                               zorder=6))
        shaft_arrow(mid, mid_y, mid, by + buf_h + 0.2, border, lw=0.95)


exec_card(
    card_x[0], 'Expected', BLUE, BLUE, BLUE_SOFT,
    'G1 (id=0)', 'G2 (id=0)',
    'accumulate', 'accumulate',
    BLUE_FILL, BLUE_FILL, False,
    'buffer = state + G1 + G2', 'white',
    'one accumulation window', BLUE,
    discard_left=False)

exec_card(
    card_x[1], 'Observed after reset', RED, RED, RED_SOFT,
    'G1 (id=0)', 'G2 (id=1)',
    'accumulate', 'copy',
    'white', RED_FILL, True,
    'buffer = G2', RED_FILL,
    'spurious window boundary', ORANGE,
    discard_left=True)

# single footer lane — no second overlapping line
ax.text(50, Y_FOOTER,
        "G1 discarded  ·  missing check: buffer op must match accumulation phase",
        fontsize=FSS - 0.2, color=INK, ha='center', va='center',
        style='italic', zorder=5)

fig.savefig('figure1_vector_v5.pdf', bbox_inches='tight', pad_inches=0.04)
print('wrote figure1_vector_v5.pdf')
