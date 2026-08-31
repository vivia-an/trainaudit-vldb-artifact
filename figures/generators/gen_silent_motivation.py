"""Generate the main-text silent-error motivation figure.

Keep annotations in data-free regions so they remain legible at single-column
size.  The PDF uses embedded TrueType fonts and vector plot primitives.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent

C_BUGGY = "#d62728"
C_FIXED = "#2ca02c"
C_DETECT = "#9467bd"
C_NOTE = "#3f3f3f"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8.0,
        "axes.titlesize": 8.7,
        "axes.labelsize": 8.0,
        "legend.fontsize": 7.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "lines.linewidth": 1.45,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.27,
        "grid.linewidth": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    }
)


def smooth(values: np.ndarray, width: int = 25) -> np.ndarray:
    kernel = np.ones(width) / width
    return np.convolve(values, kernel, mode="same")


def main() -> None:
    # Match the legacy figure's seeded RandomState sequence exactly.
    rng = np.random.RandomState(2025)
    steps = np.arange(2000)

    fixed_loss = smooth(
        2.3 * np.exp(-steps / 350) + 0.44 + rng.randn(len(steps)) * 0.015
    )
    buggy_loss = smooth(
        2.3 * np.exp(-steps / 350) + 0.445 + rng.randn(len(steps)) * 0.016
    )
    clip = 30

    eval_steps = np.arange(0, 2001, 200)
    buggy_swe = np.array([5, 12, 18, 22, 25, 27, 28, 28.5, 28.7, 28.8, 28.8])
    fixed_swe = np.array([5, 14, 22, 28, 33, 37, 40, 42, 43.5, 44.2, 44.5])

    fig, (loss_ax, quality_ax) = plt.subplots(
        1,
        2,
        figsize=(7.15, 2.36),
        gridspec_kw={"wspace": 0.31, "left": 0.085, "right": 0.985,
                     "top": 0.84, "bottom": 0.23},
    )

    visible_steps = steps[clip:]
    loss_ax.plot(visible_steps, buggy_loss[clip:], color=C_BUGGY, label="Buggy", zorder=2)
    loss_ax.fill_between(
        visible_steps,
        buggy_loss[clip:] - 0.012,
        buggy_loss[clip:] + 0.012,
        color=C_BUGGY,
        alpha=0.11,
        linewidth=0,
        zorder=1,
    )
    loss_ax.plot(visible_steps, fixed_loss[clip:], color=C_FIXED, label="Fixed", zorder=3)
    loss_ax.axvline(
        1,
        color=C_DETECT,
        linestyle="--",
        linewidth=1.2,
        label="TrainAudit fires (step 1)",
        zorder=1,
    )
    loss_ax.set(xlim=(0, 2000), ylim=(0.30, 2.18), xlabel="Training step", ylabel="Loss")
    loss_ax.set_title("Case 1 – SFT prompt-format: training loss", fontweight="bold", pad=7)
    loss_ax.annotate(
        "loss curves overlap\nthroughout; fully silent",
        xy=(1410, float(fixed_loss[1410]) + 0.035),
        xytext=(680, 1.34),
        color=C_NOTE,
        fontsize=7.2,
        ha="center",
        va="center",
        linespacing=1.12,
        arrowprops={
            "arrowstyle": "->",
            "color": C_NOTE,
            "linewidth": 0.75,
            "shrinkA": 4,
            "shrinkB": 3,
            "connectionstyle": "arc3,rad=0.02",
        },
        zorder=5,
    )
    loss_ax.legend(loc="upper right", framealpha=0.94, borderpad=0.35, handlelength=1.8)

    quality_ax.plot(
        eval_steps, buggy_swe, "o-", color=C_BUGGY, markersize=3.7,
        label="Buggy: 28.8%", zorder=3,
    )
    quality_ax.plot(
        eval_steps, fixed_swe, "s-", color=C_FIXED, markersize=3.7,
        label="Fixed: 44.5%", zorder=3,
    )
    quality_ax.axvline(
        1,
        color=C_DETECT,
        linestyle="--",
        linewidth=1.2,
        label="TrainAudit fires (step 1)",
        zorder=1,
    )
    quality_ax.set(xlim=(0, 2000), ylim=(0, 50), xlabel="Training step")
    quality_ax.set_ylabel("SWE-bench pass rate (%)", labelpad=5)
    quality_ax.set_title("Case 1 – downstream quality diverges", fontweight="bold", pad=7)

    # This note sits above the fixed curve; the arrow only meets the curve at its tip.
    quality_ax.annotate(
        "downstream quality\nconfirms damage",
        xy=(1140, 38.8),
        xytext=(650, 49.2),
        color=C_NOTE,
        fontsize=7.0,
        ha="left",
        va="top",
        linespacing=1.08,
        arrowprops={
            "arrowstyle": "->",
            "color": C_NOTE,
            "linewidth": 0.75,
            "shrinkA": 3,
            "shrinkB": 3,
            "connectionstyle": "arc3,rad=-0.08",
        },
        zorder=5,
    )

    # Put the gap label in the empty band between the curves and left of the bracket.
    quality_ax.annotate(
        "",
        xy=(1880, 28.8),
        xytext=(1880, 44.5),
        arrowprops={"arrowstyle": "<->", "color": C_NOTE, "linewidth": 0.9},
        zorder=4,
    )
    quality_ax.text(
        1690,
        36.65,
        "15.7-point\ngap",
        color=C_NOTE,
        fontsize=7.1,
        ha="center",
        va="center",
        linespacing=1.05,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white",
              "edgecolor": "none", "alpha": 0.88},
        zorder=5,
    )
    quality_ax.text(
        120,
        34.5,
        "TrainAudit fires\nat step 1",
        color=C_DETECT,
        fontsize=7.0,
        ha="left",
        va="center",
        linespacing=1.08,
        zorder=5,
    )
    quality_ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.995, 0.035),
        framealpha=0.94,
        borderpad=0.35,
        handlelength=1.8,
    )

    pdf_path = ROOT / "fig_silent_motivation.pdf"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
