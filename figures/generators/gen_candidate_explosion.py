#!/usr/bin/env python3
"""Candidate-space explosion example, Auto-Validate Fig-5 style.

Raw trace rows at the bottom with dashed boxes over the generalizable slots;
above, one block per slot listing its candidate generalizations, joined by
multiplication signs; punchline on the right gives the enumerated candidate
count and the verification survival rate. All numbers from the paper
(5,334 enumerated candidates -> 357 verified, x0.10).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib import rcParams

rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,
})

GREEN = '#2e7d32'
GREENF = '#eef6ee'
RED = '#c62828'
GRAY = '#666666'
DARK = '#2b2b2b'

W, H = 3.33, 2.35
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

FS = 5.6
FSS = 5.0
MONO = 4.9


def block(x, y, w, h, title, items):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle='round,pad=0,rounding_size=1.2',
                                fc=GREENF, ec=GREEN, lw=0.8))
    ax.text(x + w / 2, y + h - 4.2, title, ha='center', va='center',
            fontsize=FS, fontweight='bold', color=DARK)
    ax.plot([x + 1.5, x + w - 1.5], [y + h - 7.6, y + h - 7.6],
            color=GREEN, lw=0.4)
    n = len(items)
    for i, it in enumerate(items):
        yy = y + h - 11.5 - i * ((h - 14) / max(n - 1, 1))
        ax.text(x + w / 2, yy, it, ha='center', va='center', fontsize=FSS,
                color=DARK)


# ---------------- blocks (generalization choices per slot) ----------------
BY, BH = 42, 44
bw, gap = 21.0, 4.6
bx = [2, 2 + bw + gap, 2 + 2 * (bw + gap), 2 + 3 * (bw + gap)]

block(bx[0], BY, bw, BH, 'relation',
      ['cksum-equal', 'range / NaN', 'monotonic', 'invocation freq.',
       r'$\cdots$'])
block(bx[1], BY, bw, BH, 'scope',
      ['one param', 'param class', 'per module', 'all params',
       r'$\cdots$'])
block(bx[2], BY, bw, BH, r'$\pi_{\mathrm{topo}}$',
      ['none', 'TP>1', 'DP>1', 'EP>1 / PP>1', r'$\cdots$'])
block(bx[3], BY, bw, BH, r'$\pi_{\mathrm{precond}}$',
      ['none', 'step>0', 'tpl=False', 'after-optim.', r'$\cdots$'])

# multiplication signs between blocks
for k in range(3):
    xx = bx[k] + bw + gap / 2
    ax.text(xx, BY + BH / 2, r'$\times$', ha='center', va='center',
            fontsize=10, color=GREEN, fontweight='bold')

# ---------------- raw trace rows (bottom-left, AV Fig-5 layout) ----------
TY, TH = 3, 26
TX, TW = 2, 64
ax.add_patch(Rectangle((TX, TY), TW, TH, fc='white', ec=DARK, lw=0.7))
rows = [
    ('router.wt', 'r0', 'fwd', '5e11', 'TP2,tpl=F'),
    ('router.wt', 'r1', 'fwd', '0b4a', 'TP2,tpl=F'),
    ('qkv.wt',    'r0', 'fwd', '7c02', 'TP2,shard'),
]
colx = [TX + 2, TX + 18, TX + 25, TX + 34, TX + 46]
hdr = ['param', 'rank', 'stage', 'cksum', 'topology']
for cx, htxt in zip(colx, hdr):
    ax.text(cx, TY + TH - 3.2, htxt, fontsize=FSS - 0.2, color=GRAY,
            va='center', family='monospace')
for r, (p, rk, st, ck, tp) in enumerate(rows):
    yy = TY + TH - 9 - r * 6
    for cx, val in zip(colx, [p, rk, st, ck, tp]):
        ax.text(cx, yy, val, fontsize=MONO, family='monospace', va='center')

# dashed red boxes over generalizable slots + arrows up to blocks
slots = [(colx[3] - 1.0, 8.0, 0),    # cksum -> relation
         (colx[0] - 1.0, 14.0, 1),   # param -> scope
         (colx[4] - 1.0, 13.5, 2),   # topology -> pi_topo
         (colx[2] - 1.0, 6.5, 3)]    # stage -> pi_precond
for sx, sw, bi in slots:
    ax.add_patch(Rectangle((sx, TY + 1.4), sw, TH - 6.4, fill=False,
                           ec=RED, lw=0.6, ls=(0, (2, 1.4))))
    ax.annotate('',
                xy=(bx[bi] + bw / 2, BY - 0.6),
                xytext=(sx + sw / 2, TY + TH - 4.5),
                arrowprops=dict(arrowstyle='-|>', color=RED, lw=0.55,
                                ls=(0, (2, 1.4)), shrinkA=0, shrinkB=0))

# ---------------- punchline (bottom-right, AV Fig-5 layout) --------------
PX = 69
ax.text(PX, TY + TH - 4.5, r'$5{,}334$ enumerated',
        fontsize=FS + 1.6, color=DARK, va='center')
ax.text(PX, TY + TH - 11, 'candidate constraints',
        fontsize=FS + 1.6, color=DARK, va='center')
ax.text(PX, TY + TH - 17.5, 'verification admits 357',
        fontsize=FS + 0.6, color=RED, fontweight='bold', va='center')
ax.text(PX, TY + TH - 22.5, r'($\times$0.10); 45 deployed',
        fontsize=FS + 0.6, color=RED, fontweight='bold', va='center')

# ---------------- header ----------------
ax.text(2, 96.5, 'One trace relation generalizes along four slots:',
        fontsize=FS + 0.6, va='center', color=DARK)
ax.text(2, 91,
        'every combination is a candidate constraint '
        'the LLM could plausibly propose.',
        fontsize=FSS + 0.2, va='center', color=GRAY)

fig.savefig('fig_candidate_explosion.pdf', bbox_inches='tight',
            pad_inches=0.02)
print('done')
