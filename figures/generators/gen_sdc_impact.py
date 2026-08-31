#!/usr/bin/env python3
"""§2.1 motivation figure: silent slow-burn under Megatron-LM #4641.

Three vertically-stacked panels share an x-axis (training step):
  (a) lm_loss             -- buggy and fixed visually overlap for 1500 steps
  (b) gradient norm        -- ditto, only 2x divergence late, never alarming
  (c) tied-embed drift     -- |ck_pp0 - ck_pp1|  buggy explodes to ~10^3,
                              fixed flatlines at exactly 0 (floored for log)
                              vertical marker at step 1 = "P3 fires"

All three panels use log y-axes so loss decay (10^1 -> 10^-3) and drift
explosion (0 -> 10^3) are both readable in compact column-width layout.

Data source: benchmark/bugs/M-NEW-MUON-MTP/runs/long_20260508_200120/
Real telemetry: 1500 steps, 2x H200, PP=2, MTP=1, hidden=1024, num_layers=12.

This file replaces the previous synthetic ZeRO-2 motivation figure.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Palette ------------------------------------------------------------------
BLUE = "#1f4e79"     # fixed / clean
RED = "#c0392b"      # buggy
TEAL = "#2a9d8f"     # TrainAudit detection
GRAY = "#6b7280"
LIGHT = "#dfe5ec"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
    "lines.linewidth": 1.2,
})

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / (
    "benchmark/bugs/M-NEW-MUON-MTP/runs/long_20260508_200120"
)


# --- data loading ---------------------------------------------------------
def _read_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def load_loss_curve(phase):
    rows = _read_jsonl(RUN_DIR / phase / "loss_curve.jsonl")
    rows = [r for r in rows if "step" in r]
    rows.sort(key=lambda r: r["step"])
    steps = np.array([r["step"] for r in rows])
    loss = np.array([r.get("lm_loss", np.nan) for r in rows])
    grad = np.array([r.get("grad_norm", np.nan) for r in rows])
    return steps, loss, grad


def load_drift(phase):
    """Return (steps, |ck_pp0 - ck_pp1|) for a phase."""
    r0 = _read_jsonl(RUN_DIR / phase / "steps.jsonl.rank0")
    r1 = _read_jsonl(RUN_DIR / phase / "steps.jsonl.rank1")
    r0 = {r["step"]: r for r in r0 if "step" in r}
    r1 = {r["step"]: r for r in r1 if "step" in r}
    common = sorted(set(r0) & set(r1))
    steps, drift = [], []
    for s in common:
        c0 = r0[s].get("embed_proj_checksum")
        c1 = r1[s].get("embed_proj_checksum")
        if c0 is None or c1 is None:
            continue
        steps.append(s)
        drift.append(abs(c0 - c1))
    return np.array(steps), np.array(drift)


# --- light visual smoothing (paper-fidelity, not data-altering) ----------
def _ema(y, alpha=0.10):
    """Exponential smoothing: visual de-jitter only. Preserves trajectory
    and asymptotes; the user requested slight visual idealization."""
    out = np.empty_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        if np.isnan(y[i]):
            out[i] = out[i - 1]
        else:
            out[i] = alpha * y[i] + (1 - alpha) * out[i - 1]
    return out


# --- panels ---------------------------------------------------------------
def _decorate_panel(ax, ylabel, panel_label, ymin, ymax):
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-15, 1530)
    ax.tick_params(labelsize=6.8, colors=GRAY)
    ax.set_ylabel(ylabel, fontsize=7.5, color="black")
    ax.text(0.985, 0.92, panel_label,
            transform=ax.transAxes, fontsize=8, color=GRAY,
            fontweight="bold", va="top", ha="right")
    ax.yaxis.grid(True, which="major", linestyle=":", linewidth=0.3,
                  color=LIGHT, dashes=(1, 1))
    ax.set_axisbelow(True)


def panel_loss(ax, b_steps, b_loss, f_steps, f_loss):
    ax.plot(f_steps, _ema(np.maximum(f_loss, 1e-5)),
            color=BLUE, linestyle=(0, (4, 2)), lw=1.0,
            label="fixed", zorder=4)
    ax.plot(b_steps, _ema(np.maximum(b_loss, 1e-5)),
            color=RED, lw=1.3, label="buggy", zorder=5)
    _decorate_panel(ax, "lm_loss", "(a)", 1e-5, 5e1)
    ax.text(1490, 0.012, "buggy", color=RED, ha="right", va="bottom",
            fontsize=6.8, fontweight="bold")
    ax.text(1490, 5e-4, "fixed", color=BLUE, ha="right", va="top",
            fontsize=6.8)


def panel_grad(ax, b_steps, b_grad, f_steps, f_grad):
    ax.plot(f_steps, _ema(np.maximum(f_grad, 1e-5)),
            color=BLUE, linestyle=(0, (4, 2)), lw=1.0,
            zorder=4)
    ax.plot(b_steps, _ema(np.maximum(b_grad, 1e-5)),
            color=RED, lw=1.3, zorder=5)
    _decorate_panel(ax, "grad_norm", "(b)", 5e-3, 2e1)


def panel_drift(ax, b_steps, b_drift, f_steps, f_drift):
    # Floor drift to 1e-4 for log-scale rendering of zero values.
    b_y = np.maximum(b_drift, 1e-4)
    f_y = np.maximum(f_drift, 1e-4)
    ax.plot(f_steps, f_y, color=BLUE, linestyle=(0, (4, 2)), lw=1.0,
            zorder=4)
    ax.plot(b_steps, b_y, color=RED, lw=1.3, zorder=5)
    # P3 detection marker at step 1
    ax.axvline(1, color=TEAL, linestyle=(0, (2, 2)), lw=0.7, zorder=3)
    # Compact teal annotation: type-C above, P3 below, both right of vline
    ax.text(45, 1.5e3, "Type-C @ init  /  P3 @ step 1",
            color=TEAL, ha="left", va="top",
            fontsize=7, fontweight="bold")
    _decorate_panel(ax, r"$|\Delta\, \mathrm{embed\_ck}|$", "(c)", 1e-4, 5e3)
    ax.text(1490, 4.5e2, "buggy", color=RED, ha="right", va="bottom",
            fontsize=6.8, fontweight="bold")
    ax.text(1490, 4e-4, "fixed (=0)", color=BLUE, ha="right", va="bottom",
            fontsize=6.8)
    ax.set_xlabel("training step", fontsize=7.5, color="black")


# --- driver ---------------------------------------------------------------
def main():
    out_dir = Path(__file__).parent
    b_steps, b_loss, b_grad = load_loss_curve("buggy")
    f_steps, f_loss, f_grad = load_loss_curve("fixed")
    bd_steps, b_drift = load_drift("buggy")
    fd_steps, f_drift = load_drift("fixed")

    print(f"loaded buggy: {len(b_steps)} loss/grad, {len(bd_steps)} drift")
    print(f"loaded fixed: {len(f_steps)} loss/grad, {len(fd_steps)} drift")
    print(f"buggy drift: max={b_drift.max():.2f} @ step "
          f"{bd_steps[b_drift.argmax()]}")

    fig, axes = plt.subplots(
        3, 1, figsize=(3.4, 2.5),
        sharex=True,
        gridspec_kw=dict(hspace=0.20, left=0.16, right=0.97,
                         top=0.97, bottom=0.13),
    )
    panel_loss(axes[0], b_steps, b_loss, f_steps, f_loss)
    panel_grad(axes[1], b_steps, b_grad, f_steps, f_grad)
    panel_drift(axes[2], bd_steps, b_drift, fd_steps, f_drift)

    # x-tick labels only on bottom panel
    for ax in axes[:2]:
        ax.tick_params(axis="x", labelbottom=False)
    axes[2].set_xticks([0, 300, 600, 900, 1200, 1500])

    # (Experimental config moves into the LaTeX figure caption.)

    pdf = out_dir / "fig_sdc_impact.pdf"
    png = out_dir / "fig_sdc_impact.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png, dpi=200, bbox_inches="tight", pad_inches=0.02)
    print(f"saved: {pdf}")
    print(f"saved: {png}")
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    print(f"size: {bbox.width:.2f} in x {bbox.height:.2f} in")


if __name__ == "__main__":
    main()
