#!/usr/bin/env python3
"""Figure 2: the discrepancy across four bipeds and fourteen physics conditions.

Comparing engines at each robot's shipped baseline -- one cell of this grid --
is what produced our initial and wrong conclusion that only G1 differs. The
whole grid is the result, so the whole grid is the figure.

Colour is diverging on log(ratio) with a neutral midpoint at 1.0, because the
quantity has a meaningful centre: above 1 MuJoCo-Warp is closer to the C
reference, below 1 MJX-JAX is. Hue therefore encodes *which* backend is closer
and saturation how strongly, matching Figure 1's assignment (blue = MJX-JAX,
orange = MuJoCo-Warp) so the two figures can be read together.

Significance is marked with an explicit glyph rather than by colour alone:
34 of 56 cells clear Benjamini-Hochberg at 5%, and a reader must be able to
tell a large-but-noisy cell from a large-and-resolved one.

    python3 tools/plot_crossembodiment.py
"""

import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGS = ROOT / "figures"

JAX, WARP = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdbd6", "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
})

# Conditions grouped by what they perturb, so the contact block reads together.
ORDER = [
    "baseline",
    "solref_scale=0.5", "solref_scale=2.0",
    "friction_scale=0.5", "friction_scale=2.0",
    "cone=1",
    "armature_scale=0.5", "armature_scale=2.0",
    "timestep=0.001", "timestep=0.004",
    "iterations=10", "iterations=50", "iterations=100",
    "ls_iterations=20",
]
PRETTY = {
    "baseline": "baseline",
    "solref_scale=0.5": "solref ×0.5  (stiff)",
    "solref_scale=2.0": "solref ×2.0  (soft)",
    "friction_scale=0.5": "friction ×0.5",
    "friction_scale=2.0": "friction ×2.0",
    "cone=1": "elliptic cone",
    "armature_scale=0.5": "armature ×0.5",
    "armature_scale=2.0": "armature ×2.0",
    "timestep=0.001": "dt = 0.001",
    "timestep=0.004": "dt = 0.004",
    "iterations=10": "iters = 10",
    "iterations=50": "iters = 50",
    "iterations=100": "iters = 100",
    "ls_iterations=20": "ls iters = 20",
}
ROBOTS = ["G1", "Berkeley", "T1", "Op3"]


def main():
    path = FIGS / "condition_level.csv"
    if not path.exists():
        raise SystemExit("run tools/analyze_conditions.py --csv figures/condition_level.csv")
    rows = list(csv.DictReader(open(path)))

    M = np.full((len(ORDER), len(ROBOTS)), np.nan)
    S = np.zeros_like(M, dtype=bool)
    for r in rows:
        if r["condition"] in ORDER and r["robot"] in ROBOTS:
            i, j = ORDER.index(r["condition"]), ROBOTS.index(r["robot"])
            M[i, j] = float(r["ratio"])
            S[i, j] = r["significant"] == "True"

    L = np.log10(M)
    cmap = LinearSegmentedColormap.from_list("jax_warp", [JAX, "#f2f1ec", WARP])
    # Centre exactly on ratio = 1 so hue flips where the winner changes.
    norm = TwoSlopeNorm(vmin=np.nanmin(L), vcenter=0.0, vmax=np.nanmax(L))

    fig, ax = plt.subplots(figsize=(5.2, 6.4))
    fig.patch.set_facecolor(SURFACE)
    im = ax.imshow(L, cmap=cmap, norm=norm, aspect="auto")

    for i in range(len(ORDER)):
        for j in range(len(ROBOTS)):
            if np.isnan(M[i, j]):
                continue
            val = M[i, j]
            txt = f"{val:.1f}" if val >= 10 else f"{val:.2f}"
            # Ink stays neutral; the cell colour carries identity. White only
            # where the fill is too dark for black to read.
            dark = abs(L[i, j]) > 0.75 * max(abs(norm.vmin), abs(norm.vmax))
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="#ffffff" if dark else INK,
                    fontweight="bold" if S[i, j] else "normal")
            if S[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor=INK, lw=1.6, zorder=3))

    ax.set_xticks(range(len(ROBOTS)))
    ax.set_xticklabels(ROBOTS, fontsize=9)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([PRETTY[c] for c in ORDER], fontsize=8)
    ax.set_xticks(np.arange(-0.5, len(ROBOTS)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ORDER)), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=2)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(colors=MUTED)
    for s in ax.spines.values():
        s.set_visible(False)

    n_sig = int(S.sum())
    ax.set_title(f"MJX-JAX vs MuJoCo-Warp: which is closer to MuJoCo-C?\n"
                 f"{n_sig} of {S.size} cells significant "
                 f"(Benjamini-Hochberg, 5%)",
                 fontsize=10, color=INK, pad=12, loc="left", fontweight="bold")

    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    # Tick at round ratios rather than round logs: "3x" reads, "3.16228x" does not.
    nice = [0.33, 0.5, 1, 2, 5, 10, 30, 80]
    ticks = [np.log10(v) for v in nice if norm.vmin <= np.log10(v) <= norm.vmax]
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{10**t:.3g}×" for t in ticks])
    cb.set_label("divergence ratio   (c|jax) / (c|warp)", fontsize=8, color=INK)
    cb.ax.tick_params(labelsize=8, colors=MUTED)
    cb.outline.set_visible(False)

    fig.text(0.0, -0.015,
             "Above 1× MuJoCo-Warp is closer to the C reference; below 1× MJX-JAX is.\n"
             "Bold text and a black border mark cells that survive multiple-comparison "
             "correction.\nMedians over 10–30 initial conditions per cell.",
             fontsize=7.5, color=MUTED, ha="left", va="top")

    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        out = FIGS / f"fig2_crossembodiment.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=SURFACE)
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
