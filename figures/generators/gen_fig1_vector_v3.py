#!/usr/bin/env python3
"""Fig.1 motivating example — v3 (symmetry + arrow polish, content-safe).

Keeps figure1_vector.pdf narrative DNA; fixes only geometry/arrows:
  - (b) Expected: orthogonal T-merge from both accumulate paths
  - (b) equal card insets; G1/G2 columns mirrored
  - (b) Observed: only copy→buffer; G1 path ends with discard mark
  - full shaft + triangular heads throughout (overview-v18 style)
  - labels/numbers/caption semantics unchanged

Output: figure1_vector_v3.pdf  (does not overwrite figure1_vector.pdf)
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

# palette — muted academic (aligned with figure1_vector + overview hairline feel)
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
DEAD = '#9a9a9a'

W, H = 3.42, 2.64
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')
fig.patch.set_facecolor('white')

FS = 5.7
FSS = 5.05
MONO = 5.0


def rounded(x, y, w, h, fc, ec, lw=0.85, r=1.4, z=2, ls='-'):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f'round,pad=0,rounding_size={r}',
        fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=z)
    ax.add_patch(p)
    return p


def chip(x, y, w, h, text, ec, fc, bold=False, fs=None, family='monospace'):
    rounded(x, y, w, h, fc, ec, lw=0.85, r=1.1, z=4)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs or MONO, family=family, color=ec,
            fontweight='bold' if bold else 'normal', zorder=5)


def shaft_arrow(x0, y0, x1, y1, color, lw=1.0, z=6):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle='-|>,head_length=2.0,head_width=1.35',
        mutation_scale=1.0,
        lw=lw, color=color, zorder=z,
        shrinkA=0.35, shrinkB=0.35,
        connectionstyle='arc3,rad=0'))


def hline(x0, x1, y, color=RULE, lw=0.55):
    ax.add_line(Line2D([x0, x1], [y, y], color=color, lw=lw, zorder=1))


# ═══════════════════════════════════════════════════════════════════
# (a) sequencing mismatch
# ═══════════════════════════════════════════════════════════════════
ax.text(2.5, 96.8, '(a)  Reset shifts the first post-reset slot',
        fontsize=FS, fontweight='bold', color=INK, va='center')

chip_w, chip_h = 11.8, 6.2
xs = [35.5, 53.0, 70.5]
ye, yo = 84.2, 71.0

# highlight band for first post-reset slot
rounded(xs[0] - 1.8, yo - chip_h / 2 - 1.6,
        chip_w + 3.6, (ye - yo) + chip_h + 3.2,
        HL, '#bdbdbd', lw=0.55, r=1.6, z=1, ls=(0, (2.2, 1.4)))
ax.text(xs[0] + chip_w / 2, ye + chip_h / 2 + 4.2,
        'first post-reset slot',
        fontsize=FSS - 0.3, color=MUTED, ha='center', va='center')

ax.text(2.5, ye, 'Expected\ncontinuation', fontsize=FSS, color=INK,
        va='center', ha='left', linespacing=1.2)
ax.text(2.5, yo, 'Observed\nafter reset', fontsize=FSS, color=INK,
        va='center', ha='left', linespacing=1.2)

for x, lab in zip(xs, ['id=1', 'id=2', 'id=3']):
    chip(x, ye - chip_h / 2, chip_w, chip_h, lab, BLUE, BLUE_FILL)
for i in range(2):
    shaft_arrow(xs[i] + chip_w + 0.25, ye, xs[i + 1] - 0.25, ye, BLUE)
shaft_arrow(xs[2] + chip_w + 0.25, ye, xs[2] + chip_w + 5.2, ye, BLUE)
ax.text(xs[2] + chip_w + 6.8, ye, r'$\cdots$', fontsize=FS + 0.5,
        color=BLUE, va='center', ha='left')

chip(16.5, yo - chip_h / 2, 15.0, chip_h, 'reset to -1', ORANGE, ORANGE_FILL,
     bold=True, fs=FSS - 0.1, family='serif')
shaft_arrow(31.7, yo, xs[0] - 0.25, yo, ORANGE)
for i, (x, lab) in enumerate(zip(xs, ['id=0', 'id=1', 'id=2'])):
    chip(x, yo - chip_h / 2, chip_w, chip_h, lab, RED,
         RED_FILL if i == 0 else 'white', bold=(i == 0))
for i in range(2):
    shaft_arrow(xs[i] + chip_w + 0.25, yo, xs[i + 1] - 0.25, yo, RED)
shaft_arrow(xs[2] + chip_w + 0.25, yo, xs[2] + chip_w + 5.2, yo, RED)
ax.text(xs[2] + chip_w + 6.8, yo, r'$\cdots$', fontsize=FS + 0.5,
        color=RED, va='center', ha='left')

cx = xs[0] + chip_w / 2
ax.add_line(Line2D([cx, cx], [yo - chip_h / 2 - 1.6, yo - chip_h / 2 - 4.0],
                   color=RED, lw=0.7, zorder=5))
ax.text(cx, yo - chip_h / 2 - 5.8,
        'expected id=1  ·  observed id=0',
        fontsize=FSS - 0.2, color=RED, ha='center', va='center')
ax.text(cx, yo - chip_h / 2 - 9.0,
        'violates the sequencing invariant',
        fontsize=FSS - 0.2, color=RED, ha='center', va='center',
        style='italic')

# ═══════════════════════════════════════════════════════════════════
# (b) execution cards — symmetric grid
# ═══════════════════════════════════════════════════════════════════
hline(2.5, 97.5, 54.0)
ax.text(2.5, 50.8, '(b)  Expected vs. observed execution',
        fontsize=FS, fontweight='bold', color=INK, va='center')

card_y, card_h, card_w = 7.2, 39.8, 45.5
gap = 100.0 - 2.5 - 2.5 - 2 * card_w  # ~4.0 between cards
card_x = [2.5, 2.5 + card_w + gap]
hdr_h = 5.5
inset = 5.0
g_w, g_h = 15.2, 5.8
op_w, op_h = 14.4, 5.0


def exec_card(x, title, title_fc, border, soft, g1, g2, op1, op2,
              op1_fill, op2_fill, op2_bold, buffer, buffer_fill,
              top_note, top_note_c, discard_left=False):
    rounded(x, card_y, card_w, card_h, soft, border, lw=1.0, r=1.8, z=2)
    rounded(x, card_y + card_h - hdr_h, card_w, hdr_h, title_fc, title_fc,
            lw=0, r=1.8, z=3)
    ax.add_patch(Rectangle((x, card_y + card_h - hdr_h), card_w, 2.0,
                           fc=title_fc, ec='none', zorder=3))
    ax.text(x + card_w / 2, card_y + card_h - hdr_h / 2, title,
            fontsize=FSS, color='white', ha='center', va='center',
            fontweight='bold', zorder=5)

    # mirrored columns
    g1x = x + inset
    g2x = x + card_w - inset - g_w
    gy = card_y + card_h - hdr_h - 11.5
    chip(g1x, gy, g_w, g_h, g1, border, 'white')
    chip(g2x, gy, g_w, g_h, g2, border,
         RED_FILL if op2_bold else 'white', bold=op2_bold)

    ax.text(x + card_w / 2, gy + g_h + 2.6, top_note,
            fontsize=FSS - 0.35, color=top_note_c, ha='center', va='center',
            style='italic', zorder=5)

    oy = gy - 9.2
    o1x = g1x + (g_w - op_w) / 2
    o2x = g2x + (g_w - op_w) / 2
    chip(o1x, oy, op_w, op_h, op1, border, op1_fill,
         fs=FSS - 0.45, family='serif')
    chip(o2x, oy, op_w, op_h, op2, border, op2_fill,
         bold=op2_bold, fs=FSS - 0.45, family='serif')

    c1 = g1x + g_w / 2
    c2 = g2x + g_w / 2
    shaft_arrow(c1, gy, c1, oy + op_h + 0.2, border, lw=0.95)
    shaft_arrow(c2, gy, c2, oy + op_h + 0.2, border, lw=0.95)

    if discard_left:
        # spurious window boundary — centered between columns
        bx = (g1x + g_w + g2x) / 2
        ax.add_line(Line2D([bx, bx], [oy - 0.6, gy + g_h + 0.6],
                           color=ORANGE, lw=1.1, linestyle=(0, (2.5, 1.5)),
                           zorder=6))
        # G1 dead-end: short stub + × (no path into buffer)
        stub_y = oy - 3.2
        ax.add_line(Line2D([c1, c1], [oy, stub_y + 0.4],
                           color=DEAD, lw=0.85, zorder=6))
        ax.text(c1, stub_y - 0.6, r'$\times$', fontsize=FS + 1.2,
                color=DEAD, ha='center', va='center', zorder=7)
        # only copy → buffer
        by = card_y + 3.0
        chip(x + 4.0, by, card_w - 8.0, 5.4, buffer, border, buffer_fill,
             bold=True, fs=MONO)
        shaft_arrow(c2, oy, c2, by + 5.4 + 0.15, border, lw=0.95)
        ax.text(x + card_w / 2, card_y - 3.4,
                "G1's gradient is silently discarded",
                fontsize=FSS - 0.05, color=RED, ha='center', va='center',
                style='italic')
    else:
        # orthogonal T-merge into buffer (symmetric)
        by = card_y + 3.0
        chip(x + 4.0, by, card_w - 8.0, 5.4, buffer, border, buffer_fill,
             bold=True, fs=MONO)
        mid = x + card_w / 2
        mid_y = (oy + by + 5.4) / 2
        for gx in (c1, c2):
            ax.add_line(Line2D([gx, gx], [oy, mid_y], color=border, lw=0.9,
                               zorder=6))
            ax.add_line(Line2D([gx, mid], [mid_y, mid_y], color=border, lw=0.9,
                               zorder=6))
        shaft_arrow(mid, mid_y, mid, by + 5.4 + 0.15, border, lw=0.95)


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

fig.savefig('figure1_vector_v3.pdf', bbox_inches='tight', pad_inches=0.03)
print('wrote figure1_vector_v3.pdf')
