#!/usr/bin/env python3
"""Three-predicate example, pixel-level replication of QUITE Fig.1 style.

Three rounded-dashed SQL panels (a)->(b)->(c): token-level keyword coloring
(blue bold keywords, black identifiers), one red rounded highlight per panel
around the problematic / newly-added block, red italic side annotations with
hand-drawn faces, captions as bold black + red bold numbers. SQL and numbers
are the paper's own (main.tex compiled-rule listing; predicate ablation
25.8 vs 83.3 FP/1M; 0/17 fixed-replay FP).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Arc
from matplotlib import rcParams

rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,
})

KW = '#1a3fbf'      # keyword blue (QUITE tone)
TX = '#111111'
RED = '#c62828'

W, H = 7.0, 2.15
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, 210)
ax.set_ylim(0, 64)
ax.axis('off')
fig.canvas.draw()

MONO = 4.5
ANN = 4.3
CAP = 5.6
LH = 3.6
CW = 1.14           # monospace char width in data units at MONO pt


def face(x, y, happy=True, r=1.7):
    """QUITE-style hand-drawn red face."""
    ax.add_patch(plt.Circle((x, y), r, fill=False, ec=RED, lw=0.7))
    ax.plot([x - r * 0.38], [y + r * 0.28], marker='.', ms=1.6, color=RED)
    ax.plot([x + r * 0.38], [y + r * 0.28], marker='.', ms=1.6, color=RED)
    if happy:
        ax.add_patch(Arc((x, y - r * 0.05), r * 1.1, r * 0.9,
                         theta1=200, theta2=340, ec=RED, lw=0.7))
    else:
        ax.add_patch(Arc((x, y - r * 0.62), r * 1.1, r * 0.9,
                         theta1=25, theta2=155, ec=RED, lw=0.7))


def code_line(x, y, segments):
    """Render one SQL line as (text, is_keyword) segments, measured."""
    cx = x
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    for txt, iskw in segments:
        t = ax.text(cx, y, txt, fontsize=MONO, family='monospace',
                    color=KW if iskw else TX,
                    fontweight='bold' if iskw else 'normal', va='center')
        bb = t.get_window_extent(renderer=renderer)
        cx = inv.transform((bb.x1, 0))[0]


def panel(x, w, lines, cap_black, cap_red, hl=None, anns=()):
    """Rounded dashed panel. hl=(row_start,row_end,char_w) highlight box."""
    top, bot = 58, 13
    ax.add_patch(FancyBboxPatch((x, bot), w, top - bot,
                                boxstyle='round,pad=0,rounding_size=2.2',
                                fill=False, ec=TX, lw=0.8,
                                ls=(0, (4, 2.2))))
    ys = []
    y = top - 3.6
    for seg in lines:
        code_line(x + 3.0, y, seg)
        ys.append(y)
        y -= LH
    if hl:
        r0, r1, hw = hl
        yy0 = ys[r1] - 1.5
        yy1 = ys[r0] + 1.5
        ax.add_patch(FancyBboxPatch((x + 2.2, yy0), hw, yy1 - yy0,
                                    boxstyle='round,pad=0,rounding_size=1.6',
                                    fill=False, ec=RED, lw=0.9))
    for atxt, ay, af, ahappy in anns:
        ax.text(x + w - 2.5, ay, atxt, fontsize=ANN, color=RED, ha='right',
                va='center', style='italic', linespacing=1.15)
        if af is not None:
            nl = atxt.count('\n') + 1
            face(x + w - af, ay - (nl * 2.1 + 2.6), happy=ahappy)
    # caption: bold black + red bold value
    cx = x + w / 2
    ax.text(cx, 8.2, cap_black, fontsize=CAP, ha='right', va='center',
            color=TX, fontweight='bold')
    ax.text(cx + 0.8, 8.2, cap_red, fontsize=CAP, ha='left', va='center',
            color=RED, fontweight='bold')


# ---------------- SQL content (ours) ----------------
a_lines = [
    [('SELECT', 1), (' param_name, step,', 0)],
    [('  COUNT(DISTINCT', 1), (' cksum)', 0), (' AS', 1), (' n', 0)],
    [('FROM', 1), (' trace', 0)],
    [('GROUP BY', 1), (' param_name, step', 0)],
    [('HAVING', 1), (' n > 1;', 0)],
]
b_lines = [
    [('SELECT', 1), (' param_name, step,', 0)],
    [('  COUNT(DISTINCT', 1), (' cksum)', 0), (' AS', 1), (' n', 0)],
    [('FROM', 1), (' trace', 0)],
    [('WHERE', 1), (' tensor_model_', 0)],
    [('  parallel = FALSE', 0)],
    [('GROUP BY', 1), (' param_name, step', 0)],
    [('HAVING', 1), (' n > 1;', 0)],
]
c_lines = [
    [('SELECT', 1), (' param_name, step,', 0)],
    [('  COUNT(DISTINCT', 1), (' cksum)', 0), (' AS', 1), (' n', 0)],
    [('FROM', 1), (' trace', 0)],
    [('WHERE', 1), (' tensor_model_', 0)],
    [('  parallel = FALSE', 0)],
    [('  AND', 1), (' stage =', 0)],
    [("  'after_optimizer'", 0)],
    [('GROUP BY', 1), (' param_name, step', 0)],
    [('HAVING', 1), (' n > 1;', 0)],
]

PW, GAP = 65, 4.5
X0 = 2

panel(X0, PW, a_lines,
      r'(a) $\pi_{\mathrm{schema}}$ only ', '(83.3 FP/1M)',
      hl=(2, 2, 15),
      anns=[('no topology filter:\nevery TP-sharded tensor\n'
             'becomes a violation', 30.5, 8.5, False)])

panel(X0 + PW + GAP, PW, b_lines,
      r'(b) $+\ \pi_{\mathrm{topo}}$ ', '(shards excluded)',
      hl=(3, 4, 30),
      anns=[('guard in the query:\nsharded tensors filtered\n'
             'before the check', 44.0, None, True),
            ('still fires at step 0,\nbefore weight sync', 22.5, 8.5,
             False)])

panel(X0 + 2 * (PW + GAP), PW, c_lines,
      '(c) verified rule ', '(25.8 FP/1M, 0/17 FP)',
      hl=(5, 6, 30),
      anns=[('valid only after\nthe optimizer step:\nclean run silent,\n'
             'bug DETECTED', 33.0, 9.0, True)])

# header
ax.text(2, 62.2,
        'The same compiled rule, growing its guards: '
        'a violation is a non-empty query result.',
        fontsize=CAP - 0.4, color='#666666', va='center')

fig.savefig('fig_three_predicate_sql.pdf', bbox_inches='tight',
            pad_inches=0.02)
print('done')
