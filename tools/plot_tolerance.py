#!/usr/bin/env python3
"""The paper's centrepiece figure: the simulator-divergence tolerance curve.

Plots closed-loop performance loss against the *measured* open-loop divergence
that produced it, for several different perturbed physics parameters, and marks
where the real MJX backend gaps fall on the same axis.

Two design decisions carry the argument:

  x is measured divergence, not a parameter value. Nobody else's simulator pair
  is characterised by our ``solref_scale``; every pair can be located by how far
  apart it is. That is what makes the curve transferable.

  the engine gaps are drawn on the same axis. The reader can see directly
  whether a real backend difference lands in the flat region or past the knee --
  which is the question the whole paper is asking.

Colours are the validated categorical slots 1-3 (CVD-checked, worst-pair
deutan dE 9.2). Series are additionally distinguished by marker shape, so
identity never rests on colour alone; the aqua slot sits below 3:1 against the
surface, which obliges direct labels rather than a colour-only legend.

    python3 tools/plot_tolerance.py
"""

import glob
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"

# Validated categorical slots 1-3 (light mode).
SERIES = {
    "solref_scale":   ("#2a78d6", "o", "contact stiffness (solref)"),
    "friction_scale": ("#eb6834", "s", "friction"),
    "armature_scale": ("#1baf7a", "^", "armature"),
}
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


def load_points():
    """Collect dose/response points, pooled across policy seeds.

    Extreme perturbations can destabilise the simulation outright -- stiffening
    contacts 4x (solref x0.25) drives divergence to ~1 m and returns to NaN.
    Those points are separated rather than plotted: a NaN averaged into a series
    silently removes the point from the figure while leaving it in the data, and
    "the simulator broke" is a different statement from "the policy degraded".
    """
    pts, unstable = {}, {}
    files = sorted(glob.glob(str(RESULTS / "doseresponse_*.json")))
    if not files:
        raise SystemExit("no doseresponse_*.json in results/ -- run sim2sim.doseresponse")
    for f in files:
        d = json.loads(pathlib.Path(f).read_text())
        for p in d["points"]:
            key = (p["param"], p["scale"])
            dose = p["dose"]["base_pos_err_m"]
            drop = 100 * p["return_drop_frac"]
            term = p["performance"]["termination_rate"]
            target = pts if (np.isfinite(dose) and np.isfinite(drop)) else unstable
            target.setdefault(key, {"dose": [], "drop": [], "term": []})
            target[key]["dose"].append(dose)
            target[key]["drop"].append(drop)
            target[key]["term"].append(term)
    if unstable:
        print("EXCLUDED (simulation unstable / non-finite):")
        for (p, s), v in sorted(unstable.items()):
            print(f"  {p}={s:g}  dose={np.nanmean(v['dose']):.3e} m  "
                  f"return_loss={np.nanmean(v['drop'])}  term={np.mean(v['term']):.3f}")
    return pts, len(files), unstable


def engine_gaps():
    """Measured baseline engine gaps on G1, to mark on the same axis."""
    f = RESULTS / "divergence_G1JoystickFlatTerrain.json"
    if not f.exists():
        return {}
    d = json.loads(f.read_text())
    base = next(c for c in d["conditions"] if c["label"] == "baseline")
    return {k: v["mean"]["base_pos_err_m"] for k, v in base["pairs"].items()}


def main():
    pts, n_seeds, unstable = load_points()
    gaps = engine_gaps()

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), sharex=True)
    fig.patch.set_facecolor("#fcfcfb")

    # Termination rate leads: it is bounded [0,1] and monotonic, whereas
    # return-loss-% diverges once returns cross zero and lets one extreme
    # perturbation compress the low-dose region that actually matters.
    panels = [
        (axes[0], "term", "episodes terminated early  (fraction)", (-0.03, 1.05),
         "Contact mismatch >> inertial mismatch, at equal divergence"),
        (axes[1], "drop", "policy return loss  (%)", (-15, 130),
         "Return loss (one point clipped)"),
    ]

    for ax, field, ylabel, ylim, title in panels:
        ax.set_facecolor("#fcfcfb")
        for name, x in sorted(gaps.items(), key=lambda kv: kv[1]):
            if name == "jax|warp":
                continue  # near-duplicate of c|jax; adds clutter, not information
            ax.axvline(x, color=INK_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
            ax.annotate(f"{name}\n{x:.1e} m", xy=(x, ylim[1]), xytext=(x * 1.15, ylim[1]),
                        fontsize=7.5, color=INK_MUTED, va="top", ha="left")

        # Scatter, not connected lines: scales below and above 1.0 both map to
        # positive divergence, so joining them would trace a path that doubles
        # back and implies a progression the data does not contain.
        for param, (color, marker, label) in SERIES.items():
            xs, ys, es = [], [], []
            for (p, _), v in pts.items():
                if p != param:
                    continue
                xs.append(np.mean(v["dose"]))
                ys.append(np.mean(v[field]))
                es.append(np.std(v[field], ddof=1) / np.sqrt(len(v[field]))
                          if len(v[field]) > 1 else 0.0)
            if not xs:
                continue
            ax.errorbar(xs, ys, yerr=es, color=color, marker=marker, ms=7, lw=0,
                        capsize=2.5, elinewidth=1.0, label=label, zorder=3,
                        markeredgecolor="#fcfcfb", markeredgewidth=0.8)

        # Two trends, not one. Pooling them would assert that divergence
        # magnitude alone predicts loss, which the data refutes: within the
        # contact parameters dose predicts loss almost perfectly (Spearman
        # rho = +0.96 solref, +0.94 friction), while for armature it does not
        # (rho = +0.14) and the loss never exceeds ~6% at any divergence. A
        # single pooled curve would hide exactly that distinction.
        families = {
            "contact params (solref, friction)": (
                ["solref_scale", "friction_scale"], "#7a2f10"),
            "inertial param (armature)": (["armature_scale"], "#0d6b4a"),
        }
        for fam_label, (params, fam_color) in families.items():
            fx = np.array([np.mean(v["dose"]) for k, v in pts.items() if k[0] in params])
            fy = np.array([np.mean(v[field]) for k, v in pts.items() if k[0] in params])
            if len(fx) < 4:
                continue
            o = np.argsort(fx)
            fx, fy = fx[o], fy[o]
            k = max(2, len(fx) // 3)
            sm = np.convolve(fy, np.ones(k) / k, mode="valid")
            ax.plot(fx[k - 1:], sm, color=fam_color, lw=1.6, alpha=0.75, ls="--",
                    zorder=2, label=fam_label if field == "term" else None)

        ax.set_xscale("log")
        ax.set_ylim(*ylim)
        ax.set_xlabel("open-loop divergence between the two physics configs  (m)",
                      fontsize=9, color=INK)
        ax.set_ylabel(ylabel, fontsize=9, color=INK)
        ax.set_title(title, fontsize=9.5, color=INK, pad=10, loc="left")
        ax.axhline(0, color=GRID, lw=1.0, zorder=0)
        ax.grid(True, which="major", color=GRID, lw=0.6, alpha=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=8.5)

    axes[0].legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK)

    # Name the clipped point rather than letting it silently vanish.
    worst = max(pts.items(), key=lambda kv: np.mean(kv[1]["drop"]))
    if np.mean(worst[1]["drop"]) > 130:
        axes[1].annotate(
            f"{worst[0][0].replace('_scale','')}={worst[0][1]:g} clipped\n"
            f"({np.mean(worst[1]['drop']):.0f}% loss)",
            xy=(np.mean(worst[1]["dose"]), 118), xytext=(np.mean(worst[1]["dose"]) * 1.3, 70),
            fontsize=7.5, color="#eb6834", ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color="#eb6834", lw=0.8, alpha=0.7))

    note = (f"Pooled over {n_seeds} policy seed(s); error bars are s.e.m. across seeds. "
            "Dashed lines mark the measured MJX backend gaps on G1.")
    if unstable:
        names = ", ".join(f"{p.replace('_scale','')}={s:g}" for p, s in sorted(unstable))
        note += f"  Excluded (simulation unstable, non-finite returns): {names}."
    fig.text(0.01, -0.04, note, fontsize=7.5, color=INK_MUTED)

    FIGS.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        out = FIGS / f"tolerance_curve.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[saved] {out}")

    # A table view accompanies the figure: the aqua series sits below 3:1
    # against the surface, so the numbers must be readable without colour.
    csv = FIGS / "tolerance_curve.csv"
    with open(csv, "w") as fh:
        fh.write("param,scale,dose_m,return_loss_pct,termination_rate,n_seeds\n")
        for (p, s), v in sorted(pts.items(), key=lambda kv: np.mean(kv[1]["dose"])):
            fh.write(f"{p},{s},{np.mean(v['dose']):.6e},{np.mean(v['drop']):.3f},"
                     f"{np.mean(v['term']):.3f},{len(v['drop'])}\n")
    print(f"[saved] {csv}")


if __name__ == "__main__":
    main()
