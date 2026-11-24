"""Figures of the shared-basis refractive-index sweep for the theory note and the README.

Reads shared_basis_limits.csv and shared_basis_times.csv in this directory and writes
shared_basis_limits.{pdf,png} and shared_basis_times.{pdf,png}.

Run from the repository root:
    uv run --with matplotlib python docs/figs/make_shared_basis_figures.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent

# Ink and chrome
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Categorical slots 1 to 4 in fixed order, each with its own marker shape
SERIES = {
    "bc": ("#2a78d6", "o", "Black carbon, 1.45–2.25 + 0.35–1.25i"),
    "bcw": ("#eb6834", "s", "Black carbon, wider, 1.45–2.55 + 0.35–1.55i"),
    "nonabs": ("#1baf7a", "^", "Non-absorbing, 1.30–1.70"),
    "nonabshi": ("#eda100", "D", "Non-absorbing, 2.00–3.00"),
}
SHORT = {"bc": "BC", "bcw": "BC wider", "nonabs": "low index", "nonabshi": "high index"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8.5,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "pdf.fonttype": 42,
})


def read_csv(name):
    with open(HERE / name, newline="") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(which="both", length=0)


def figure_limits():
    rows = read_csv("shared_basis_limits.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.5), sharey=True)
    # direct labels: (class, panel) -> (index of the point, offset in points, alignment)
    labels = {
        ("bc", 0): (7, (8, -3), "left"), ("bcw", 0): (1, (-6, 7), "right"),
        ("nonabs", 0): (5, (6, -9), "left"), ("nonabshi", 0): (5, (-7, 4), "right"),
        ("bc", 1): (6, (6, -9), "left"), ("bcw", 1): (1, (-6, 7), "right"),
        ("nonabs", 1): (4, (7, -8), "left"), ("nonabshi", 1): (5, (-7, 4), "right"),
    }
    for p, (ax, col, title) in enumerate(zip(axes, ("err_forward", "err_backward"),
                                             ("Forward $S_1$, $S_2$", "Backward $S_1$, $S_2$"))):
        style_axes(ax)
        for level in (1e-3, 1e-2):
            ax.axhline(level, color=MUTED, linewidth=0.8, zorder=1)
        for cls, (color, marker, _) in SERIES.items():
            x = [float(r["x_mean"]) for r in rows if r["class"] == cls]
            y = [float(r[col]) for r in rows if r["class"] == cls]
            ax.plot(x, y, color=color, linewidth=2, marker=marker, markersize=6.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, solid_joinstyle="round",
                    solid_capstyle="round", zorder=3)
            i, off, ha = labels[(cls, p)]
            ax.annotate(SHORT[cls], (x[i], y[i]), xytext=off, textcoords="offset points",
                        ha=ha, va="center", color=INK2, fontsize=7.5, zorder=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.17, 3.0)
        ax.set_ylim(1e-7, 1.0)
        ax.set_xticks([0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.5])
        ax.set_xticklabels(["0.2", "0.3", "0.5", "0.7", "1", "1.5", "2.5"])
        ax.minorticks_off()
        ax.set_xlabel("Mean monomer size parameter")
        ax.set_title(title, loc="left", color=INK, fontsize=9, pad=6)
    axes[0].set_ylabel("Relative error of the shared basis")
    handles = [Line2D([0], [0], color=c, linewidth=2, marker=m, markersize=6.5,
                      markeredgecolor=SURFACE, markeredgewidth=1.2, label=lab)
               for c, m, lab in SERIES.values()]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8,
               labelcolor=INK2, handlelength=2.6, columnspacing=2.5, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.92, bottom=0.27, wspace=0.08)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"shared_basis_limits.{ext}", dpi=220)
    plt.close(fig)


def figure_times():
    rows = read_csv("shared_basis_times.csv")
    names = [f"$N$ = {r['n_monomers']}, $D_f$ = {r['Df']}, {r['wavelength_nm']} nm, order {r['order']}"
             for r in rows]
    fig, ax = plt.subplots(figsize=(6.9, 2.9))
    style_axes(ax)
    ax.grid(False, axis="y")
    blue, gray = "#2a78d6", MUTED
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        tc, ts = float(r["t_cbicg_s"]), float(r["t_shared_s"])
        ax.plot([ts, tc], [y, y], color=AXIS, linewidth=1.2, zorder=2, solid_capstyle="round")
        ax.plot(tc, y, "o", color=gray, markersize=7.5, markeredgecolor=SURFACE,
                markeredgewidth=1.2, zorder=3)
        ax.plot(ts, y, "o", color=blue, markersize=7.5, markeredgecolor=SURFACE,
                markeredgewidth=1.2, zorder=3)
        ax.annotate(f"{tc / ts:.0f} times faster", (tc, y), xytext=(8, 0),
                    textcoords="offset points", ha="left", va="center", color=INK2, fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlim(4, 4e5)
    ax.set_xticks([10, 100, 1000, 10000])
    ax.set_xticklabels(["10 s", "100 s", "1000 s", "10000 s"])
    ax.minorticks_off()
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(names[::-1], color=INK2)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Time for 195 or 196 refractive indices of one aggregate, one thread")
    ax.spines["left"].set_visible(False)
    handles = [Line2D([0], [0], linestyle="none", marker="o", color=blue, markersize=7.5,
                      markeredgecolor=SURFACE, markeredgewidth=1.2, label="Shared basis"),
               Line2D([0], [0], linestyle="none", marker="o", color=gray, markersize=7.5,
                      markeredgecolor=SURFACE, markeredgewidth=1.2,
                      label="CBICG, one solve per refractive index")]
    fig.legend(handles=handles, loc="upper left", ncol=2, frameon=False, fontsize=8,
               labelcolor=INK2, bbox_to_anchor=(0.30, 1.0), handletextpad=0.3, columnspacing=2.0)
    fig.subplots_adjust(left=0.36, right=0.985, top=0.90, bottom=0.17)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"shared_basis_times.{ext}", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    figure_limits()
    figure_times()
