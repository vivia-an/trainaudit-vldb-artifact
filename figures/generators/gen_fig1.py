#!/usr/bin/env python3
"""Generate Figure 1: micro_step_id mismatch in ZeRO-2."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

TEXT = "#303030"
MUTED = "#909090"
PANEL_FILL = "#ffffff"
PANEL_EDGE = "#d2d2d2"
HEADER_FILL = "#efefef"
HEADER_EDGE = "#c2c2c2"
NORM_FILL = "#f0f0f3"
NORM_EDGE = "#9595a5"
HI_FILL = "#fde5ea"
HI_EDGE = "#dc5070"
AMBER_FILL = "#fef5e3"
AMBER_EDGE = "#cc9825"
GRAY_FILL = "#f5f5f6"
GRAY_EDGE = "#b2b2b8"
ACCENT = "#dc5070"

MONO = "DejaVu Sans Mono"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8,
    "axes.unicode_minus": False,
    "mathtext.fontset": "dejavusans",
})

# Font-size hierarchy
FS_TITLE = 8.5
FS_HEADER = 8.0
FS_BODY = 7.8
FS_LABEL = 7.2
FS_NOTE = 6.8


def add_panel(ax, title, header_y=0.86, header_h=0.10):
    panel = FancyBboxPatch(
        (0.01, 0.02), 0.98, 0.96,
        transform=ax.transAxes,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=PANEL_FILL, edgecolor=PANEL_EDGE,
        linewidth=0.9, zorder=0,
    )
    header = FancyBboxPatch(
        (0.06, header_y), 0.88, header_h,
        transform=ax.transAxes,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        facecolor=HEADER_FILL, edgecolor=HEADER_EDGE,
        linewidth=0.8, zorder=1,
    )
    ax.add_patch(panel)
    ax.add_patch(header)
    ax.text(
        0.50, header_y + header_h / 2, title,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=FS_TITLE, fontweight="bold", color=TEXT,
    )


def add_box(ax, cx, cy, w, h, text, fc, ec, tc=TEXT,
            fontsize=FS_BODY, weight="normal", lw=1.0, family=None):
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        facecolor=fc, edgecolor=ec,
        linewidth=lw, zorder=2,
    )
    ax.add_patch(patch)
    kw = {}
    if family:
        kw["fontfamily"] = family
    ax.text(
        cx, cy, text,
        ha="center", va="center",
        fontsize=fontsize, fontweight=weight,
        color=tc, zorder=3, **kw,
    )


def add_arrow(ax, x0, y0, x1, y1, color=MUTED, lw=1.1, style="->", ls="-"):
    ax.annotate(
        "",
        xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle=style, color=color,
            lw=lw, linestyle=ls,
            mutation_scale=10, shrinkA=0, shrinkB=0,
        ),
        zorder=2.5,
    )


def add_diamond(ax, cx, cy, w, h, text, fc, ec,
                tc=TEXT, fontsize=FS_BODY, weight="bold", lw=1.0, family=None):
    diamond = Polygon(
        [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)],
        closed=True, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2,
    )
    ax.add_patch(diamond)
    kw = {}
    if family:
        kw["fontfamily"] = family
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=tc, zorder=3, **kw)


# ====================================================================
# Figure layout
# ====================================================================
fig = plt.figure(figsize=(7.0, 5.6), facecolor="white")
ax_a = fig.add_axes([0.02, 0.58, 0.96, 0.38])
ax_b = fig.add_axes([0.02, 0.02, 0.96, 0.52])


# ====================================================================
# Panel A: counter sequence mismatch
# ====================================================================
ax = ax_a
ax.set_xlim(0, 10)
ax.set_ylim(0, 3.2)
ax.axis("off")
add_panel(ax, "(a)  Inconsistent counter sequence across initialization and reset")

y1, y2 = 2.10, 0.90
SEQ_X = [2.90, 4.15, 5.40]
BOX_W, BOX_H = 0.92, 0.42

for y, label, vals, hi in [
    (y1, "Initial path\n(global step 1)", ["ID = 1", "ID = 2", "ID = 3"], 0),
    (y2, "Reset path\n(step n > 1)",      ["ID = 0", "ID = 1", "ID = 2"], 0),
]:
    ax.text(0.65, y, label, ha="center", va="center",
            fontsize=FS_LABEL, color=MUTED, style="italic")
    add_box(ax, 1.70, y, 0.55, BOX_H, "start",
            GRAY_FILL, GRAY_EDGE, tc=MUTED, fontsize=FS_LABEL)
    add_arrow(ax, 1.98, y, SEQ_X[0] - BOX_W / 2, y)
    for i, (x, v) in enumerate(zip(SEQ_X, vals)):
        fc = HI_FILL if i == hi else NORM_FILL
        ec = HI_EDGE if i == hi else NORM_EDGE
        wt = "bold" if i == hi else "normal"
        add_box(ax, x, y, BOX_W, BOX_H, v, fc, ec,
                fontsize=FS_BODY, weight=wt, family=MONO)
        if i < len(SEQ_X) - 1:
            add_arrow(ax, x + BOX_W / 2, y, SEQ_X[i + 1] - BOX_W / 2, y)
    ax.text(6.10, y, "\u2026", ha="center", va="center", fontsize=11, color=MUTED)
    add_arrow(ax, 6.30, y, 6.65, y)
    add_box(ax, 7.45, y, 1.30, BOX_H, "reset to \u22121",
            AMBER_FILL, AMBER_EDGE, fontsize=FS_LABEL)

add_arrow(ax, SEQ_X[0], y1 - BOX_H / 2 - 0.04, SEQ_X[0], y2 + BOX_H / 2 + 0.04,
          color=ACCENT, lw=1.5, style="<->")
mid_y = (y1 + y2) / 2
add_box(ax, 4.20, mid_y, 2.60, 0.50,
        "off-by-one after reset\nexpected ID = 1, got ID = 0",
        HI_FILL, HI_EDGE, tc=TEXT, fontsize=FS_LABEL, weight="bold", lw=1.1)


# ====================================================================
# Panel B: control-flow / state-machine view
# ====================================================================
ax = ax_b
ax.set_xlim(0, 10)
ax.set_ylim(0, 6.2)
ax.axis("off")
add_panel(ax, "(b)  Shifted branch selection corrupts gradient accumulation",
          header_y=0.90, header_h=0.08)

LX, RX = 2.5, 7.5
COL_W = 4.1

add_box(ax, LX, 5.20, COL_W, 0.48, "Micro-step 1",
        HEADER_FILL, HEADER_EDGE, fontsize=FS_HEADER, weight="bold")
add_box(ax, RX, 5.20, COL_W, 0.48, "Micro-step 2",
        HEADER_FILL, HEADER_EDGE, fontsize=FS_HEADER, weight="bold")
ax.text(LX, 4.82, "new accumulation window",
        ha="center", va="center", fontsize=FS_NOTE, color=MUTED, style="italic")
ax.text(RX, 4.82, "same window continues",
        ha="center", va="center", fontsize=FS_NOTE, color=MUTED, style="italic")
ax.plot([5.0, 5.0], [0.35, 4.90], color=PANEL_EDGE, lw=0.8, ls="--", zorder=1)

STATE_Y = 4.20
DIAM_Y = 3.10
ACTION_Y = 1.85
EXPECT_Y = 0.60

STATE_H = 0.68
DIAM_H = 0.80
ACTION_H = 0.62
EXPECT_H = 0.44

# Left column
add_box(ax, LX, STATE_Y, COL_W, STATE_H,
        "state s\u2080 :  input G\u2081 ,  micro_step_id = 0",
        NORM_FILL, NORM_EDGE, fontsize=FS_BODY, weight="bold")

add_diamond(ax, LX, DIAM_Y, 2.40, DIAM_H,
            "id == 1 ?", AMBER_FILL, AMBER_EDGE,
            fontsize=FS_BODY, family=MONO)

add_box(ax, LX, ACTION_Y, COL_W, ACTION_H,
        "action: accumulate\nbuffer \u2190 stale + G\u2081",
        GRAY_FILL, GRAY_EDGE, fontsize=FS_BODY)

add_arrow(ax, LX, STATE_Y - STATE_H / 2, LX, DIAM_Y + DIAM_H / 2)
ax.text(LX - 1.40, (DIAM_Y + ACTION_Y) / 2 + 0.08, "False",
        ha="center", va="center", fontsize=FS_LABEL, color=MUTED,
        fontweight="bold", style="italic")
add_arrow(ax, LX, DIAM_Y - DIAM_H / 2, LX, ACTION_Y + ACTION_H / 2)

# Right column
add_box(ax, RX, STATE_Y, COL_W, STATE_H,
        "state s\u2081 :  input G\u2082 ,  micro_step_id = 1",
        NORM_FILL, NORM_EDGE, fontsize=FS_BODY, weight="bold")

add_diamond(ax, RX, DIAM_Y, 2.40, DIAM_H,
            "id == 1 ?", AMBER_FILL, AMBER_EDGE,
            fontsize=FS_BODY, family=MONO)

add_box(ax, RX, ACTION_Y, COL_W, ACTION_H,
        "action: copy  (overwrites buffer)\nbuffer \u2190 G\u2082 only  \u2014  G\u2081 lost",
        HI_FILL, HI_EDGE, fontsize=FS_BODY, weight="bold", lw=1.2)

add_arrow(ax, RX, STATE_Y - STATE_H / 2, RX, DIAM_Y + DIAM_H / 2)
ax.text(RX + 1.40, (DIAM_Y + ACTION_Y) / 2 + 0.08, "True",
        ha="center", va="center", fontsize=FS_LABEL, color=ACCENT,
        fontweight="bold", style="italic")
add_arrow(ax, RX, DIAM_Y - DIAM_H / 2, RX, ACTION_Y + ACTION_H / 2)

# Horizontal transition arrow between columns (at action level)
add_arrow(ax, LX + COL_W / 2 + 0.05, ACTION_Y, RX - COL_W / 2 - 0.05, ACTION_Y,
          color=MUTED, lw=1.0)
ax.text(5.0, ACTION_Y + 0.18, "next",
        ha="center", va="bottom", fontsize=FS_NOTE, color=MUTED, style="italic")

# Expected-path dashed annotation
add_arrow(ax, RX - 2.40 / 2, DIAM_Y, 5.10, DIAM_Y, color=GRAY_EDGE, lw=0.9, style="-", ls="--")
add_arrow(ax, 5.10, DIAM_Y, 5.10, EXPECT_Y, color=GRAY_EDGE, lw=0.9, style="-", ls="--")
add_arrow(ax, 5.10, EXPECT_Y, RX - COL_W / 2, EXPECT_Y, color=GRAY_EDGE, lw=0.9, style="->", ls="--")

ax.text(5.10, (DIAM_Y + EXPECT_Y) / 2, "expected\npath",
        ha="right", va="center", fontsize=FS_NOTE, color=MUTED, style="italic",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))

add_box(ax, RX, EXPECT_Y, COL_W, EXPECT_H,
        "expected: accumulate G\u2082 \u2192 buffer = G\u2081 + G\u2082",
        GRAY_FILL, PANEL_EDGE, fontsize=FS_LABEL)


# ====================================================================
# Save
# ====================================================================
output_base = Path(__file__).resolve().parent / "fig1_clean"
plt.savefig(output_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print(f"Saved {output_base.with_suffix('.pdf')} and {output_base.with_suffix('.png')}")
