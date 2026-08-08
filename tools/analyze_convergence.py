#!/usr/bin/env python3
"""Does the engine gap vanish as the solver converges?

Experiment 6 showed the MuJoCo-C vs MJX-JAX contact discrepancy is algorithmic
rather than arithmetic: double precision eliminates the contact-free gap by nine
orders of magnitude and leaves the contact gap untouched. This is the
independent check. If the gap were a discretisation artefact it should shrink
toward zero as the timestep falls and the solver iterates to convergence. If it
plateaus at a nonzero floor, two implementations are converging to *different*
answers -- which is what "algorithmic" means, arrived at a second way.

Reads the grid written by tools/run_convergence.sh.

    python3 tools/analyze_convergence.py
"""

import glob
import json
import pathlib
import re

import numpy as np

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"


def load_grid():
    rows = {}
    for f in sorted(glob.glob(str(RESULTS / "divergence_*conv_dt*.json"))):
        m = re.search(r"conv_dt([0-9.]+)_it(\d+)", f)
        if not m:
            continue
        d = json.loads(pathlib.Path(f).read_text())
        conds = {c["label"]: c for c in d["conditions"]}
        pairs = conds["baseline"]["pairs"]
        rows[(float(m.group(1)), int(m.group(2)))] = {
            k: float(np.median(pairs[k]["seed_finals"]["base_pos_err_m"]))
            for k in pairs
        }
    return rows


def main():
    grid = load_grid()
    if not grid:
        raise SystemExit("no convergence grid found -- run tools/run_convergence.sh")

    dts = sorted({k[0] for k in grid}, reverse=True)
    its = sorted({k[1] for k in grid})
    n_expected = len(dts) * len(its)
    print(f"convergence grid: {len(grid)}/{n_expected} cells\n")

    for pair in ("c|jax", "c|warp"):
        print(f"=== {pair} (median base-position divergence, m) ===")
        print("      dt \\ iters " + "".join(f"{i:>12d}" for i in its))
        for dt in dts:
            cells = "".join(
                f"{grid[(dt, i)][pair]:12.3e}" if (dt, i) in grid else f"{'--':>12s}"
                for i in its
            )
            print(f"  {dt:12.4f} {cells}")
        print()

    # Convergence in iterations, at each timestep: does refining the solver help?
    print("=== does the gap shrink with MORE ITERATIONS? (ratio last/first) ===")
    for dt in dts:
        have = [i for i in its if (dt, i) in grid]
        if len(have) < 2:
            continue
        for pair in ("c|jax", "c|warp"):
            a, b = grid[(dt, have[0])][pair], grid[(dt, have[-1])][pair]
            print(f"  dt={dt:<7.4f} {pair:8s} iters {have[0]:>2d}->{have[-1]:<3d}  "
                  f"{a:.3e} -> {b:.3e}   ratio {b/a:6.3f}")

    # Convergence in timestep, at the most-converged iteration count available.
    print("\n=== does the gap shrink with SMALLER TIMESTEP? (at highest iters) ===")
    it_max = max(its)
    have = [dt for dt in sorted(dts) if (dt, it_max) in grid]
    for pair in ("c|jax", "c|warp"):
        if len(have) < 2:
            continue
        vals = [grid[(dt, it_max)][pair] for dt in have]
        print(f"  {pair:8s} iters={it_max}: " +
              "  ".join(f"dt={d:g}:{v:.2e}" for d, v in zip(have, vals)))
        print(f"           {'':8s} finest/coarsest ratio = {vals[0]/vals[-1]:.3f}")

    print("\n  A discretisation artefact would trend toward zero under refinement.")
    print("  A ratio that flattens well above zero means the two implementations")
    print("  converge to different answers -- an algorithmic difference.")


if __name__ == "__main__":
    main()
