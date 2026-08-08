#!/usr/bin/env python3
"""Figure 5: every engine scored against closed-form solutions.

Most simulator comparisons rank engines against one another, which cannot say
which is *wrong*. These four systems have exact analytic answers, so each engine
gets an absolute error -- and the ordering of the four cases is the argument:
contact-free the three agree to float32, contact-dominated they separate, and
every engine creeps under a block that should not move at all.

Horizontal layout with a log axis because the errors span twelve orders of
magnitude; a bar chart would be unreadable and a linear axis would collapse
three of the four cases onto zero.

    python3 tools/plot_groundtruth.py
"""

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS, FIGS = ROOT / "results", ROOT / "figures"
C_COL, JAX, WARP = "#4a3aa7", "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdbd6", "#ffffff"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42})


def main():
    g = json.loads((RESULTS / "groundtruth.json").read_text())
    cases = [
        ("free fall\n(no contact)", "error vs exact discrete solution",
         [g["free_fall"][e]["err_vs_discrete_euler"] for e in ("c", "jax", "warp")]),
        ("pendulum\n(no contact)", "relative energy drift",
         [g["pendulum"][e]["rel_drift"] for e in ("c", "jax", "warp")]),
        ("incline, sliding\n(contact)", "error vs analytic distance",
         [g["incline_sliding"][e]["abs_err_m"] for e in ("c", "jax", "warp")]),
        ("incline, static\n(contact)", "creep — should be exactly zero",
         [abs(g["incline_static"][e]["x_final"]) for e in ("c", "jax", "warp")]),
    ]

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)

    engines = [("MuJoCo-C", C_COL, "D"), ("MJX-JAX", JAX, "o"), ("MuJoCo-Warp", WARP, "s")]
    # MJX-JAX and MuJoCo-Warp agree to all printed digits on the contact-free
    # cases -- which is the result -- so their markers land on top of each other.
    # A small vertical offset per engine keeps all three visible; without it the
    # figure silently shows two engines where there are three.
    offsets = [+0.20, 0.0, -0.20]

    for i, (_, _, vals) in enumerate(cases):
        y = len(cases) - 1 - i
        ax.plot([min(vals), max(vals)], [y, y], color=GRID, lw=1.2, zorder=1)
        for v, (name, col, mk), off in zip(vals, engines, offsets):
            ax.scatter([v], [y + off], s=52, color=col, marker=mk, zorder=3,
                       edgecolor=SURFACE, linewidth=0.9,
                       label=name if i == 0 else None)

    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels(
        [f"{c[0]}\n{c[1]}" for c in reversed(cases)], fontsize=8)
    ax.set_ylim(-0.6, len(cases) - 0.4)
    ax.set_xscale("log")
    ax.set_xlim(1e-16, 1e0)
    ax.set_xlabel("absolute error against the closed-form answer", color=INK)
    ax.set_title("All three engines scored against exact solutions",
                 loc="left", fontsize=10, color=INK, pad=8, fontweight="bold")
    ax.grid(True, axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02), ncol=1,
              fontsize=8, frameon=False, labelcolor=INK, handletextpad=0.4,
              borderpad=0.2, labelspacing=0.35)

    # Name the coincidence rather than leaving the reader to wonder.
    ax.annotate("JAX and Warp coincide", xy=(6.97e-06, 3.0), xytext=(14, 14),
                textcoords="offset points", fontsize=7.5, color=MUTED,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))

    fig.text(0.0, -0.10,
             "Contact-free, MJX-JAX and MuJoCo-Warp are numerically identical and "
             "both sit at float32 distance from MuJoCo-C.\nUnder contact they "
             "separate. Every engine creeps ~1.1 mm/s under a block that should be "
             "perfectly static — a\nproperty of MuJoCo's soft-contact model, not of "
             "any one implementation.",
             fontsize=7.5, color=MUTED, ha="left", va="top")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = FIGS / f"fig5_groundtruth.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=SURFACE)
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
