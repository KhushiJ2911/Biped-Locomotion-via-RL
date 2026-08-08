#!/usr/bin/env python3
"""Figure 6: what is held fixed, what varies, and what is measured.

A measurement paper lives or dies on whether the reader believes the comparison
is controlled. Prose makes that hard to check; a diagram makes it inspectable at
a glance. The single most important thing this figure asserts is that all three
engines are built from the *same* compiled mj_model and driven from the *same*
initial state with the *same* control sequence -- so any divergence is
attributable to the implementation and nothing else.

    python3 tools/plot_schematic.py
"""

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIGS = pathlib.Path(__file__).resolve().parent.parent / "figures"
C_COL, JAX, WARP = "#4a3aa7", "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdbd6", "#ffffff"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "pdf.fonttype": 42, "ps.fonttype": 42})


def box(ax, x, y, w, h, text, fc, ec, fs=9, weight="normal", tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                facecolor=fc, edgecolor=ec, linewidth=1.3, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=3, fontweight=weight, linespacing=1.4)


def arrow(ax, p0, p1, color=MUTED, lw=1.2, style="-|>"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=11,
                                 color=color, lw=lw, zorder=1,
                                 shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    # Headroom above 1.0 rather than shifting every box down: the title needs
    # its own band, and overlapping the model box was the first defect here.
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.09); ax.axis("off")

    # --- source of truth ----------------------------------------------------
    box(ax, 0.30, 0.845, 0.40, 0.115,
        "one compiled MJCF model\n(identical qpos₀, qvel₀, control sequence)",
        "#f2f1ec", MUTED, fs=8.5, weight="bold")

    # --- what is deliberately varied ---------------------------------------
    box(ax, 0.015, 0.845, 0.245, 0.115,
        "varied, one at a time:\ndt · solver iters · solref\nfriction · armature · cone",
        SURFACE, GRID, fs=7.5, tc=MUTED)
    arrow(ax, (0.262, 0.902), (0.298, 0.902), color=GRID)

    # --- the three engines --------------------------------------------------
    xs = [0.075, 0.395, 0.715]
    names = [("MuJoCo-C\nreference · float64", C_COL),
             ("MJX-JAX\nGPU · float32", JAX),
             ("MuJoCo-Warp\nGPU · float32", WARP)]
    for x, (nm, col) in zip(xs, names):
        arrow(ax, (0.50, 0.845), (x + 0.105, 0.665), color=GRID)
        box(ax, x, 0.545, 0.21, 0.115, nm, SURFACE, col, fs=8.5, weight="bold", tc=col)

    # --- pairwise measurement ----------------------------------------------
    box(ax, 0.175, 0.315, 0.65, 0.115,
        "pairwise state divergence after 1 s of physics\n"
        "base position · base orientation · joint RMSE",
        "#f2f1ec", MUTED, fs=8.5)
    for x, (_, col) in zip(xs, names):
        arrow(ax, (x + 0.105, 0.545), (min(max(x + 0.105, 0.22), 0.78), 0.432), color=col)

    # --- what the controls rule out ----------------------------------------
    box(ax, 0.015, 0.055, 0.30, 0.175,
        "control 1 — precision\nrun MJX in float64\n\n→ contact-free gap vanishes\n"
        "→ contact gap unchanged",
        SURFACE, GRID, fs=7.5, tc=MUTED)
    box(ax, 0.345, 0.055, 0.30, 0.175,
        "control 2 — discretisation\nrefine dt and solver iters\n\n→ Warp converges to C\n"
        "→ MJX-JAX hits a floor",
        SURFACE, GRID, fs=7.5, tc=MUTED)
    box(ax, 0.675, 0.055, 0.31, 0.175,
        "closed loop\ntrained policy in each engine\n\n→ 2.7% worse in MuJoCo-C\n"
        "→ 0.53× seed spread (null)",
        SURFACE, GRID, fs=7.5, tc=MUTED)
    # Arrows must leave from *inside* the measurement box (x 0.175-0.825);
    # starting at the child centres put the outer two in empty space.
    for x0, x1 in ((0.26, 0.165), (0.50, 0.495), (0.74, 0.83)):
        arrow(ax, (x0, 0.315), (x1, 0.232), color=GRID)

    ax.text(0.0, 1.075, "Everything upstream is identical; only the engine differs",
            fontsize=10, color=INK, fontweight="bold", va="top")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = FIGS / f"fig6_schematic.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=SURFACE)
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
