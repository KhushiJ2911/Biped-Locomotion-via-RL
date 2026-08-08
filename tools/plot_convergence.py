#!/usr/bin/env python3
"""Figure 1: the MJX-JAX discrepancy is algorithmic, shown two independent ways.

Panel (a) refines the timestep. On log-log axes a discretisation difference is a
straight line of slope 1 heading to zero; an algorithmic difference flattens.
MuJoCo-Warp does the former, MJX-JAX the latter -- so the claim is carried by
the *shape* of the curves, not by a number in the caption.

Panel (b) is the precision control. The contact-free case is included precisely
because it is the positive control: float64 collapses it by nine orders of
magnitude, proving the manipulation works, while leaving the contact gap
untouched. Without that pairing, "float64 didn't help" would be unfalsifiable.

Design follows the validated categorical palette (CVD-checked, worst-pair deutan
dE 9.2), with series identified by marker shape as well as hue so identity never
rests on colour alone.

    python3 tools/plot_convergence.py
"""

import glob
import json
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"

JAX, WARP = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdbd6", "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "pdf.fonttype": 42,   # embed TrueType so the PDF is editable/searchable
    "ps.fonttype": 42,
})


def load_grid():
    grid = {}
    for f in sorted(glob.glob(str(RESULTS / "divergence_*conv_dt*.json"))):
        m = re.search(r"conv_dt([0-9.]+)_it(\d+)", f)
        d = json.loads(pathlib.Path(f).read_text())
        conds = {c["label"]: c for c in d["conditions"]}
        p = conds["baseline"]["pairs"]
        grid[(float(m.group(1)), int(m.group(2)))] = {
            k: float(np.median(p[k]["seed_finals"]["base_pos_err_m"])) for k in p}
    return grid


def panel_convergence(ax, grid, iters=50):
    dts = sorted({k[0] for k in grid if k[1] == iters})
    xj = np.array(dts)
    yj = np.array([grid[(d, iters)]["c|jax"] for d in dts])
    yw = np.array([grid[(d, iters)]["c|warp"] for d in dts])

    # Slope-1 guide: what a pure discretisation difference must look like here.
    guide_x = np.array([dts[0], dts[-1]])
    guide_y = yw[0] * (guide_x / dts[0])
    ax.plot(guide_x, guide_y, color=MUTED, lw=1.0, ls=(0, (2, 2)), zorder=1)
    mid = int(len(dts) // 2)
    ax.annotate("slope 1 = pure discretisation",
                xy=(dts[mid] * 0.85, yw[0] * (dts[mid] / dts[0]) * 0.42),
                fontsize=7.5, color=MUTED, ha="center", va="top")

    ax.plot(xj, yj, color=JAX, marker="o", ms=6, lw=1.8, zorder=3,
            markeredgecolor=SURFACE, markeredgewidth=0.9)
    ax.plot(xj, yw, color=WARP, marker="s", ms=6, lw=1.8, zorder=3,
            markeredgecolor=SURFACE, markeredgewidth=0.9)

    # Direct labels beat a legend box: each curve is named where the reader
    # is already looking.
    ax.annotate("MJX-JAX", xy=(xj[0], yj[0]), xytext=(6, 2),
                textcoords="offset points", color=JAX, fontsize=9,
                fontweight="bold", va="center")
    ax.annotate("MuJoCo-Warp", xy=(xj[0], yw[0]), xytext=(6, -2),
                textcoords="offset points", color=WARP, fontsize=9,
                fontweight="bold", va="center")

    floor = yj.min()
    ax.axhline(floor, color=JAX, lw=0.8, ls=":", alpha=0.55, zorder=1)
    ax.annotate(f"algorithmic floor  {floor:.1e} m",
                xy=(dts[1], floor), xytext=(0, -13), textcoords="offset points",
                fontsize=7.5, color=JAX, ha="center", va="top")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.invert_xaxis()   # refinement reads left-to-right
    # Explicit ticks: matplotlib's log minor ticks collide badly over this range.
    ax.set_xticks(dts)
    ax.set_xticklabels([f"{d:g}" for d in dts])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("integration timestep  dt  (s)   →  finer", color=INK)
    ax.set_ylabel("divergence from MuJoCo-C  (m)", color=INK)
    ax.set_title("(a)  Refining the solver", loc="left", fontsize=10,
                 color=INK, pad=8, fontweight="bold")


def panel_precision(ax):
    """float64 / float32 ratio. Plotting absolutes spans twelve orders of
    magnitude and crushes the contact points; the ratio puts every condition on
    one interpretable scale where 1.0 means "precision changed nothing"."""
    f32 = json.loads((RESULTS / "precision_G1JoystickFlatTerrain_f32.json").read_text())
    f64 = json.loads((RESULTS / "precision_G1JoystickFlatTerrain_f64.json").read_text())
    # Contact-free control from Experiment 0's free-fall case.
    items = [("contact-free\n(free fall)", 4.441e-15 / 6.97e-06, True),
             ("contact\n(baseline)",
              f64["conditions"]["baseline"]["median"] /
              f32["conditions"]["baseline"]["median"], False),
             ("contact\n(soft contacts)",
              f64["conditions"]["solref_scale=2.0"]["median"] /
              f32["conditions"]["solref_scale=2.0"]["median"], False)]

    x = np.arange(len(items))
    for i, (_, r, is_ctrl) in enumerate(items):
        col = WARP if is_ctrl else JAX
        ax.plot([i, i], [1.0, r], color=col, lw=6, solid_capstyle="butt",
                alpha=0.85, zorder=2)
        # A ratio of 1.05 is a sub-pixel bar on a log axis and reads as missing
        # data. The marker makes "measured, and unchanged" explicit.
        ax.scatter([i], [r], s=46, color=col, zorder=4,
                   edgecolor=SURFACE, linewidth=1.0,
                   marker="D" if is_ctrl else "o")
        ax.annotate(f"{r:.3g}" if r > 1e-3 else f"{r:.1e}".replace("e-", "e−"),
                    xy=(i, r), xytext=(0, 10 if not is_ctrl else 12),
                    textcoords="offset points", fontsize=7.5, color=col,
                    ha="center", fontweight="bold")
    ax.axhline(1.0, color=INK, lw=1.0, zorder=3)
    ax.annotate("no change", xy=(-0.5, 1.0), xytext=(2, -11),
                textcoords="offset points", fontsize=7.5, color=INK, ha="left")

    ax.annotate("precision fixes it\n(9 orders)", xy=(0, items[0][1] ** 0.5),
                xytext=(12, 0), textcoords="offset points", fontsize=7.5,
                color=WARP, va="center")
    ax.annotate("precision does\nnothing", xy=(1.5, 1.0), xytext=(0, -34),
                textcoords="offset points", fontsize=7.5, color=JAX,
                ha="center", va="top")

    ax.set_yscale("log")
    ax.set_ylim(1e-10, 1e2)
    ax.set_xticks(x); ax.set_xticklabels([a for a, _, _ in items], fontsize=8)
    ax.set_xlim(-0.6, len(items) - 0.35)
    ax.set_ylabel("divergence ratio   float64 / float32", color=INK)
    ax.set_title("(b)  Doubling the precision", loc="left", fontsize=10,
                 color=INK, pad=8, fontweight="bold")


def main():
    grid = load_grid()
    if not grid:
        raise SystemExit("no convergence grid found")

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3))
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, which="major", color=GRID, lw=0.6, alpha=0.9, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)

    panel_convergence(axes[0], grid)
    panel_precision(axes[1])

    fig.text(0.0, -0.06,
             "Neither refinement nor double precision removes the MJX-JAX "
             "disagreement with MuJoCo-C, while both behave as expected for "
             "MuJoCo-Warp.\nMedians over 20 initial conditions; solver "
             "iterations = 50 in (a).",
             fontsize=7.5, color=MUTED, ha="left", va="top")

    fig.tight_layout(w_pad=3.0)
    FIGS.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        out = FIGS / f"fig1_algorithmic.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=SURFACE)
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
