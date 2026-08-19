#!/usr/bin/env python3
"""Generate an academic-style Figure 1 for the micro_step_id bug."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

TEXT = "#1f2937"
MUTED = "#6b7280"
PANEL_FILL = "#fbfcfe"
PANEL_EDGE = "#cfd8e3"
HEADER_FILL = "#eef3f8"
HEADER_EDGE = "#c3cfdb"
BLUE_FILL = "#eef5ff"
BLUE_EDGE = "#4f83cc"
RED_FILL = "#fff1f1"
RED_EDGE = "#c65a5a"
AMBER_FILL = "#fff7e8"
AMBER_EDGE = "#d39a2f"
GRAY_FILL = "#f4f6f8"
GRAY_EDGE = "#98a2b3"
ACCENT = "#d9485f"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.unicode_minus": False,
    }
)


def add_panel(ax, title):
    panel = FancyBboxPatch(
        (0.01, 0.03),
        0.98,
        0.94,
        transform=ax.transAxes,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=PANEL_FILL,
        edgecolor=PANEL_EDGE,
        linewidth=0.9,
        zorder=0,
    )
    header = FancyBboxPatch(
        (0.06, 0.86),
        0.88,
        0.10,
        transform=ax.transAxes,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        facecolor=HEADER_FILL,
        edgecolor=HEADER_EDGE,
        linewidth=0.8,
        zorder=1,
    )
    ax.add_patch(panel)
    ax.add_patch(header)
    ax.text(
        0.50,
        0.91,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.4,
        fontweight="bold",
        color=TEXT,
    )


def add_box(ax, cx, cy, w, h, text, fc, ec, tc=TEXT, fontsize=8, weight="normal", lw=1.0):
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=tc,
        zorder=3,
    )


def add_arrow(ax, x0, y0, x1, y1, color=MUTED, lw=1.1, style="->", ls="-"):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle=style,
            color=color,
            lw=lw,
            linestyle=ls,
            mutation_scale=10,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=2.5,
    )


def draw_sequence_row(ax, y, label, values, highlight_index):
    ax.text(1.05, y, label, ha="center", va="center", fontsize=7.6, color=MUTED)

    xs = [2.25, 3.40, 4.55]
    box_w, box_h = 0.82, 0.42

    add_box(ax, 1.55, y, 0.60, 0.42, "start", GRAY_FILL, GRAY_EDGE, tc=MUTED, fontsize=7.2)
    add_arrow(ax, 1.86, y, 1.98, y)

    for idx, (x, value) in enumerate(zip(xs, values)):
        fc = RED_FILL if idx == highlight_index else BLUE_FILL
        ec = RED_EDGE if idx == highlight_index else BLUE_EDGE
        weight = "bold" if idx == highlight_index else "normal"
        add_box(ax, x, y, box_w, box_h, f"{value}", fc, ec, tc=TEXT, fontsize=8.1, weight=weight)
        if idx < len(xs) - 1:
            add_arrow(ax, x + box_w / 2, y, xs[idx + 1] - box_w / 2, y)

    ax.text(5.38, y, "...", ha="center", va="center", fontsize=10, color=MUTED)
    add_arrow(ax, 5.60, y, 6.00, y)
    add_box(ax, 6.80, y, 1.25, 0.42, "reset to -1", AMBER_FILL, AMBER_EDGE, fontsize=7.6)


fig = plt.figure(figsize=(6.6, 4.8), facecolor="white")
ax_a = fig.add_axes([0.03, 0.54, 0.94, 0.40])
ax_b = fig.add_axes([0.03, 0.07, 0.94, 0.40])

ax = ax_a
ax.set_xlim(0, 10)
ax.set_ylim(0, 3.1)
ax.axis("off")
add_panel(ax, "Panel A. Inconsistent counter sequence across initialization and reset")

y_top, y_bottom = 1.95, 0.88
draw_sequence_row(ax, y_top, "Initial path\n(global step 1)", ["ID = 1", "ID = 2", "ID = 3"], 0)
draw_sequence_row(ax, y_bottom, "Reset path\n(global step n > 1)", ["ID = 0", "ID = 1", "ID = 2"], 0)

add_arrow(ax, 2.25, y_top - 0.26, 2.25, y_bottom + 0.26, color=ACCENT, lw=1.5, style="<->")
add_box(
    ax,
    3.10,
    1.42,
    1.82,
    0.54,
    "offset after reset\nexpected first ID = 1, observed ID = 0",
    RED_FILL,
    RED_EDGE,
    tc=TEXT,
    fontsize=7.3,
    weight="bold",
    lw=1.1,
)

ax = ax_b
ax.set_xlim(0, 10)
ax.set_ylim(0, 5.4)
ax.axis("off")
add_panel(ax, "Panel B. Shifted branch selection corrupts gradient accumulation")

headers = [(2.7, "Micro-step 1"), (7.2, "Micro-step 2")]
for x, title in headers:
    add_box(ax, x, 4.55, 3.8, 0.52, title, HEADER_FILL, HEADER_EDGE, fontsize=8.3, weight="bold")

ax.text(2.7, 4.10, "new accumulation window", ha="center", va="center", fontsize=7.2, color=MUTED)
ax.text(7.2, 4.10, "continuation of the same window", ha="center", va="center", fontsize=7.2, color=MUTED)
ax.plot([4.95, 4.95], [0.60, 4.25], color=PANEL_EDGE, lw=0.9, ls="--", zorder=1)

ys = [3.45, 2.70, 1.95, 1.20]
left_text = [
    ("incoming gradient: G1", BLUE_FILL, BLUE_EDGE, "normal"),
    ("counter state: micro_step_id = 0", RED_FILL, RED_EDGE, "bold"),
    ("predicate micro_step_id == 1 is false", AMBER_FILL, AMBER_EDGE, "normal"),
    ("buffer after op: stale + G1", GRAY_FILL, GRAY_EDGE, "normal"),
]
right_text = [
    ("incoming gradient: G2", BLUE_FILL, BLUE_EDGE, "normal"),
    ("counter state: micro_step_id = 1", RED_FILL, RED_EDGE, "bold"),
    ("predicate micro_step_id == 1 is true\ncopy branch is taken", RED_FILL, RED_EDGE, "bold"),
    ("buffer after op: G2 only\nG1 is discarded", RED_FILL, RED_EDGE, "bold"),
]

for col_x, rows in [(2.7, left_text), (7.2, right_text)]:
    for idx, (text, fc, ec, weight) in enumerate(rows):
        lw = 1.2 if "discarded" in text or "copy branch" in text else 1.0
        add_box(ax, col_x, ys[idx], 3.8, 0.58, text, fc, ec, fontsize=7.7, weight=weight, lw=lw)
        if idx < len(rows) - 1:
            add_arrow(ax, col_x, ys[idx] - 0.31, col_x, ys[idx + 1] + 0.31)

add_arrow(ax, 4.60, 2.32, 5.30, 2.32, color=MUTED, lw=1.0)
add_box(
    ax,
    5.0,
    0.46,
    8.15,
    0.46,
    "Expected after two micro-steps: buffer = G1 + G2    |    Observed here: buffer = G2 only",
    GRAY_FILL,
    PANEL_EDGE,
    fontsize=7.5,
)

output_base = Path(__file__).resolve().parent / "fig1_academic"
plt.savefig(output_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
plt.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
print(f"Saved {output_base.with_suffix('.pdf')} and {output_base.with_suffix('.png')}")
