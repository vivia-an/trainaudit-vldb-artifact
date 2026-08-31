#!/usr/bin/env python3
"""
TrainAudit — System Overview Figure (ICML 2025, publication quality)

Three sections (left → right):
  1. Data Tracer    (top-left,    blue)   — online runtime monitoring
  2. Invariant Miner (bottom-left, green)  — offline LLM-driven discovery
  3. Error Analysis (right,       orange) — combined detection & diagnosis

Data flow:
  Training System → hook → states → processing → Trace DB ──────┐
                                                                   ▼
                                               SQL Synthesis (topology-aware)
  Docs/Code → Gap Analysis → Evidence Retrieval → Constraint Synthesis       │
                      ↑──────── Invalid (dashed) ─────────────────┘        ↓
                                                → Invariant Library ──→ Hierarchical Exec
                                                                        → Violation Retrieval
                                                                        → Silent Error Report
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9.0,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# ─── Palette ─────────────────────────────────────────────────────
BL, BL_LT, BL_MD = '#1A5FA8', '#E9F2FB', '#BAD4EE'   # Data Tracer   (blue)
GR, GR_LT, GR_MD = '#1D6A38', '#E8F5EE', '#A9D18E'   # Inv. Miner    (green)
OR, OR_LT, OR_MD = '#9E3D0C', '#FEF0E8', '#F4B183'   # Error Analysis (amber)
RD, RD_LT        = '#9B1111', '#FFF0EF'               # Report / reject (red)
DK, GY, WH       = '#1A202C', '#6B7280', '#FFFFFF'

FW, FH = 13.0, 5.5
fig = plt.figure(figsize=(FW, FH), facecolor=WH)
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis('off')


# ─── Primitives ──────────────────────────────────────────────────

def rbox(cx, cy, w, h, text='', *, fc=WH, ec=GY, tc=DK,
         fs=8.5, bold=False, lw=1.3, ls='-', z=3):
    p = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle='round,pad=0.10',
        fc=fc, ec=ec, lw=lw, ls=ls, zorder=z, clip_on=False,
    )
    ax.add_patch(p)
    if text:
        ax.text(cx, cy, text,
                ha='center', va='center',
                fontsize=fs, color=tc, zorder=z+1,
                fontweight='bold' if bold else 'normal',
                multialignment='center', linespacing=1.38)


def panel_box(x, y, w, h, title, *, fc, ec, lw=2.0, z=1):
    """Section panel with coloured title bar."""
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.08',
        fc=fc, ec=ec, lw=lw, zorder=z, clip_on=False,
    )
    ax.add_patch(p)
    th = 0.42
    bar = FancyBboxPatch(
        (x + 0.05, y + h - th), w - 0.10, th - 0.06,
        boxstyle='round,pad=0.04',
        fc=ec, ec=ec, lw=0, zorder=z+1, clip_on=False,
    )
    ax.add_patch(bar)
    ax.text(x + w/2, y + h - th/2 - 0.03, title,
            ha='center', va='center',
            fontsize=10.5, fontweight='bold', color=WH, zorder=z+2)


def arr(x0, y0, x1, y1, *, c=DK, lw=1.4, ls='-', z=8, cs=None):
    kw = dict(arrowstyle='->', color=c, lw=lw,
              mutation_scale=11, linestyle=ls)
    if cs:
        kw['connectionstyle'] = cs
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=kw, zorder=z)


def arrlbl(x0, y0, x1, y1, label, *, c=DK, lw=1.4, ls='-', z=8,
           lx=None, ly=None, lfs=7.2):
    """Arrow with floating label positioned explicitly."""
    arr(x0, y0, x1, y1, c=c, lw=lw, ls=ls, z=z)
    mx = lx if lx is not None else (x0 + x1) / 2
    my = ly if ly is not None else (y0 + y1) / 2
    ax.text(mx, my, label,
            ha='center', va='center',
            fontsize=lfs, color=c, zorder=z+1,
            bbox=dict(boxstyle='round,pad=0.18', fc=WH, ec='none', alpha=0.92))


def note(x, y, text, *, c=GY, fs=7.2, ha='center', style='italic'):
    ax.text(x, y, text, ha=ha, va='center', fontsize=fs,
            color=c, fontstyle=style, zorder=9)


# ══════════════════════════════════════════════════════════════════
#   SECTION PANELS
#   Data Tracer    : x 0.18–6.78,  y 3.08–5.42  (w=6.60, h=2.34)
#   Inv. Miner     : x 0.18–6.78,  y 0.18–2.92  (w=6.60, h=2.74)
#   Error Analysis : x 7.08–12.82, y 0.18–5.42  (w=5.74, h=5.24)
# ══════════════════════════════════════════════════════════════════

panel_box(0.18, 3.08, 6.60, 2.34, 'Data Tracer',      fc=BL_LT, ec=BL)
panel_box(0.18, 0.18, 6.60, 2.74, 'Invariant Miner',  fc=GR_LT, ec=GR)
panel_box(7.08, 0.18, 5.74, 5.24, 'Error Analysis',   fc=OR_LT, ec=OR)

note(3.48, 5.24, 'online — runs during training',              c=BL, fs=7.5)
note(3.48, 2.72, 'offline — LLM-driven invariant discovery',  c=GR, fs=7.5)


# ══════════════════════════════════════════════════════════════════
#   DATA TRACER  (horizontal flow, all at y = DY)
# ══════════════════════════════════════════════════════════════════

DY = 4.27   # y-center for all Data Tracer elements

# [1] Distributed Training System
rbox(1.25, DY, 1.68, 1.58,
     'Distributed\nTraining System\n\nMegatron-LM\nDeepSpeed / OLMo',
     fc=BL_MD, ec=BL, fs=8.0)

arr(2.09, DY, 2.32, DY, c=BL, lw=1.4)

# [2] RunTime Hook
rbox(2.82, DY, 0.98, 0.52, 'RunTime\nHook',
     fc=WH, ec=BL, tc=BL, bold=True, fs=8.5)

arr(3.31, DY, 3.52, DY, c=BL, lw=1.4)

# [3] Training States  (three sub-items listed)
rbox(4.15, DY, 1.26, 1.10,
     'Training States\n\nParams · Grads\nOptim. States',
     fc=WH, ec=BL, fs=8.0)

arr(4.78, DY, 4.98, DY, c=BL, lw=1.4)

# [4] Data Processing Pipeline
rbox(5.44, DY, 0.90, 0.58, 'Data\nProcessing',
     fc=WH, ec=BL, fs=8.0)

arr(5.89, DY, 6.05, DY, c=BL, lw=1.4)

# [5] Trace Database
rbox(6.44, DY, 0.80, 0.80, 'Trace\nDatabase',
     fc=BL_MD, ec=BL, bold=True, fs=8.5)

# Arrow: Trace DB ──────── → boundary
arrlbl(6.85, DY, 7.08, DY, 'traces',
       c=BL, lw=2.2, lx=6.97, ly=DY + 0.20, lfs=7.5)


# ══════════════════════════════════════════════════════════════════
#   INVARIANT MINER  (horizontal flow with rejection feedback)
#   [Docs & Code] → [Gap Analysis S1] → [Evidence Retrieval S2]
#                 → [Constraint Synthesis S3] → [Invariant Library]
#   Invalid: dashed curved arrow back to S1
# ══════════════════════════════════════════════════════════════════

IY = 1.52   # y-center for all Invariant Miner elements

# [A] Docs & Code (input, tall box on the left)
rbox(0.88, IY, 1.10, 1.95,
     'Framework\nSource Code\n&\nDocumentation',
     fc=GR_MD, ec=GR, fs=8.0)

arr(1.43, IY, 1.63, IY, c=GR, lw=1.4)

# [B] Gap Analysis  (S1)
rbox(2.17, IY, 1.05, 0.58, 'Gap\nAnalysis',
     fc=WH, ec=GR, tc=GR, bold=True, fs=8.5)
note(2.17, IY - 0.46, 'Stage 1', c=GR, fs=7.0, style='normal')

arr(2.70, IY, 2.88, IY, c=GR, lw=1.4)

# [C] Evidence Retrieval  (S2)
rbox(3.47, IY, 1.18, 0.58, 'Evidence\nRetrieval',
     fc=WH, ec=GR, tc=GR, bold=True, fs=8.5)
note(3.47, IY - 0.46, 'Stage 2', c=GR, fs=7.0, style='normal')

arr(4.06, IY, 4.22, IY, c=GR, lw=1.4)

# [D] Constraint Synthesis + Bidirectional Verification  (S3)
rbox(5.06, IY, 1.66, 0.72,
     'Constraint Synthesis\n& Bidirectional\nVerification',
     fc=WH, ec=GR, tc=GR, bold=True, fs=8.0)
note(5.06, IY - 0.54, 'Stage 3  (>=2 counterexamples)', c=GR, fs=7.0, style='normal')

# Valid → Invariant Library
arrlbl(5.89, IY, 6.05, IY, 'Valid',
       c=GR, lw=1.4, lx=5.97, ly=IY + 0.20, lfs=7.2)

# [E] Invariant Library
rbox(6.44, IY, 0.80, 0.80, 'Invariant\nLibrary',
     fc=GR_MD, ec=GR, bold=True, fs=8.5)

# Arrow: Invariant Library ──── → boundary
arrlbl(6.85, IY, 7.08, IY, 'invariants',
       c=GR, lw=2.2, lx=6.97, ly=IY + 0.20, lfs=7.5)

# Invalid feedback loop: dashed curved arrow from S3 bottom back to S1 bottom
ax.annotate('',
    xy     = (2.17, IY - 0.36),    # arrive at Gap Analysis base
    xytext = (5.06, IY - 0.46),    # leave from Constraint Synthesis base
    arrowprops=dict(
        arrowstyle='->', color=RD, lw=1.3,
        mutation_scale=11, linestyle='--',
        connectionstyle='arc3,rad=0.52',
    ),
    zorder=8,
)
note(3.60, 0.50, 'Invalid  →  reject & restart', c=RD, fs=7.5, style='normal')


# ══════════════════════════════════════════════════════════════════
#   ERROR ANALYSIS  (x: 7.08–12.82, y: 0.18–5.42)
#
#   Two inputs arrive at EA left boundary at different heights:
#     • traces      at y = DY = 4.27  (from Trace DB)
#     • invariants  at y = IY = 1.52  (from Invariant Library)
#
#   Both route to SQL Synthesis (top box, at y = DY = 4.27).
#   The invariants travel along the left margin of EA (x=7.35)
#   as a vertical connector, well left of all EA content boxes.
#
#   Four stages then flow downward:
#     SQL Synthesis → Hierarchical Execution → Violation Retrieval → Report
# ══════════════════════════════════════════════════════════════════

EX = 9.95   # center x of all EA boxes
EW = 4.40   # box width (left edge = EX - EW/2 = 7.75)

# ── SQL Synthesis is positioned at y = DY so the traces arrow is horizontal ──
Y_SQL = DY;   H_SQL = 0.72
Y_HEX = 3.10; H_HEX = 0.76
Y_VSR = 2.10; H_VSR = 0.72
Y_REP = 0.98; H_REP = 0.76

# ── Left-margin merge bus inside EA ──────────────────────────────
# A thin vertical line at x=7.38, from IY up to DY, collects both inputs
BUS_X = 7.38
ax.plot([BUS_X, BUS_X], [IY, DY], color=DK, lw=1.3, zorder=6)
# Small filled circle at the top junction (traces join here)
ax.plot(BUS_X, DY, 'o', color=DK, markersize=4.5, zorder=7)

# traces:     Trace DB boundary → bus top  (horizontal)
arr(7.08, DY, BUS_X, DY,  c=BL, lw=1.8)

# invariants: Inv. Library boundary → bus bottom (horizontal)
arr(7.08, IY, BUS_X, IY,  c=GR, lw=1.8)

# bus → SQL Synthesis  (horizontal from bus top to box left edge)
arr(BUS_X, DY, EX - EW/2, Y_SQL, c=DK, lw=1.5)

# labels on the horizontal entry stubs
note(7.22, DY + 0.20, 'traces',     c=BL, fs=7.5, style='normal')
note(7.22, IY + 0.20, 'invariants', c=GR, fs=7.5, style='normal')

# ── Stage 1: SQL Synthesis ──
rbox(EX, Y_SQL, EW, H_SQL,
     'SQL Synthesis\n(Topology-Aware Constraint Filtering & Query Generation)',
     fc=OR_MD, ec=OR, fs=8.5)

arr(EX, Y_SQL - H_SQL/2, EX, Y_HEX + H_HEX/2, c=OR, lw=1.5)

# ── Stage 2: Hierarchical Execution ──
rbox(EX, Y_HEX, EW, H_HEX,
     'Hierarchical Execution\n(Coarse-to-Fine Constraint Checking)',
     fc=OR_MD, ec=OR, fs=8.5)

arr(EX, Y_HEX - H_HEX/2, EX, Y_VSR + H_VSR/2, c=OR, lw=1.5)

# ── Stage 3: Violation-Selective Retrieval ──
rbox(EX, Y_VSR, EW, H_VSR,
     'Violation-Selective Retrieval\n(Causal Root-Cause Analysis)',
     fc=OR_MD, ec=OR, fs=8.5)

arr(EX, Y_VSR - H_VSR/2, EX, Y_REP + H_REP/2, c=OR, lw=1.5)

# ── Stage 4: Silent Error Report ──
rbox(EX, Y_REP, EW, H_REP,
     'Silent Error Report\n(Rank-Level Localization & Structured Diagnosis)',
     fc=RD_LT, ec=RD, tc=RD, bold=True, fs=8.5, lw=2.0)


# ══════════════════════════════════════════════════════════════════
#   SAVE
# ══════════════════════════════════════════════════════════════════

out_pdf = '/Users/miketang/Dev/sdc_llm_icml_2025/figures/system-framewoek.pdf'
out_png = '/Users/miketang/Dev/sdc_llm_icml_2025/figures/system-framewoek-preview.png'

fig.savefig(out_pdf, dpi=300, bbox_inches='tight', facecolor=WH)
fig.savefig(out_png, dpi=180, bbox_inches='tight', facecolor=WH)
print("Saved:", out_pdf)
