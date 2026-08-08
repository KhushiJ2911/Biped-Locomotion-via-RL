#!/usr/bin/env python3
"""Figure 3: the policy-level effect is real, and smaller than seed noise.

Both facts must appear together or the figure lies. Showing only the paired
difference and its confidence interval would imply an important effect
(pooled t = 2.88, p < 0.01). Showing only the null-model comparison would imply
nothing was found. The result is precisely that the effect is *detectable and
irrelevant*, so the seed-noise band is drawn behind the effects: anything inside
it is smaller than the consequence of picking a different training seed.

    python3 tools/plot_policy_null.py
"""

import glob
import json
import pathlib
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS, FIGS = ROOT / "results", ROOT / "figures"
JAX, WARP = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdbd6", "#ffffff"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42})


def main():
    files = sorted(glob.glob(str(RESULTS / "ceval_*_jax.json")))
    seeds, diffs, sems, mjx_means = [], [], [], []
    for f in files:
        e = json.loads(pathlib.Path(f).read_text())["episodes"]
        mt = np.array([x["mjx_track_err"] for x in e])
        ct = np.array([x["c_track_err"] for x in e])
        d = ct - mt
        seeds.append(pathlib.Path(f).name.split("seed")[1][0])
        diffs.append(d.mean())
        sems.append(d.std(ddof=1) / np.sqrt(len(d)))
        mjx_means.append(mt.mean())

    pooled = np.concatenate([
        np.array([x["c_track_err"] for x in json.loads(pathlib.Path(f).read_text())["episodes"]])
        - np.array([x["mjx_track_err"] for x in json.loads(pathlib.Path(f).read_text())["episodes"]])
        for f in files])
    p_mean = pooled.mean()
    p_sem = pooled.std(ddof=1) / np.sqrt(len(pooled))
    seed_sd = stdev(mjx_means)

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)

    # The null model, drawn first so effects sit on top of it.
    ax.axhspan(-seed_sd, seed_sd, color=MUTED, alpha=0.13, zorder=0)
    # Sits in the empty lower half of the band, clear of the error bars.
    ax.annotate("seed-to-seed spread:\nchanging the training seed\n"
                "moves the result this much",
                xy=(-0.35, -seed_sd * 0.45), fontsize=7.5, color=MUTED,
                va="center", ha="left")
    ax.axhline(0, color=INK, lw=1.0, zorder=2)

    x = np.arange(len(seeds) + 1)
    vals = diffs + [p_mean]
    errs = sems + [p_sem]
    cols = [JAX] * len(seeds) + [WARP]
    for i, (v, e, c) in enumerate(zip(vals, errs, cols)):
        ax.errorbar(i, v, yerr=1.96 * e, color=c, marker="o", ms=7, lw=0,
                    elinewidth=1.6, capsize=4, zorder=4,
                    markeredgecolor=SURFACE, markeredgewidth=0.9)
    ax.annotate(f"{p_mean:+.4f}", xy=(x[-1], p_mean), xytext=(11, 0),
                textcoords="offset points", fontsize=8, color=WARP,
                fontweight="bold", va="center")

    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds] + ["pooled\n(192 eps)"], fontsize=8)
    ax.set_xlim(-0.5, len(x) - 0.35)
    ax.set_ylabel("tracking error:  MuJoCo-C − MJX-JAX", color=INK)
    ax.set_title("Policy degrades in MuJoCo-C — by less than seed noise",
                 loc="left", fontsize=10, color=INK, pad=8, fontweight="bold")
    ax.grid(True, axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)

    fig.text(0.0, -0.10,
             f"Error bars are 95% CI. Pooled effect is significant "
             f"(t(191) = 2.88, p < 0.01) yet only {abs(p_mean)/seed_sd:.2f}× the "
             f"seed-to-seed spread —\nbelow the pre-registered threshold of 2. "
             f"Positive means worse tracking in MuJoCo-C.",
             fontsize=7.5, color=MUTED, ha="left", va="top")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = FIGS / f"fig3_policy_null.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=SURFACE)
        print(f"[saved] {out}")
    print(f"  pooled {p_mean:+.4f}  seed_sd {seed_sd:.4f}  ratio {abs(p_mean)/seed_sd:.2f}")


if __name__ == "__main__":
    main()
