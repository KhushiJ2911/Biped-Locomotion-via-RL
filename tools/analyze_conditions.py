#!/usr/bin/env python3
"""Condition-level engine comparison across robots, with FDR correction.

Comparing engines at a single physics configuration -- typically each robot's
shipped baseline -- answers a much narrower question than it appears to. Every
robot here is swept over 14 configurations, and the engine discrepancy is
strongly condition-dependent: T1 shows no advantage at baseline and a
significant one in six other conditions. Reading robot-dependence off the
baseline alone produced a conclusion ("only G1 differs") that the full grid
refutes.

So every robot x condition cell is tested, and because that is dozens of
simultaneous tests, p-values are corrected with Benjamini-Hochberg. At 56 tests
and alpha = 0.05, roughly three cells would clear an uncorrected threshold by
chance alone.

Ratios use medians with a bootstrap CI: divergence distributions are
heavy-tailed on several robots (mean/median up to 10x), so a mean ratio is
dominated by single seeds.

    python3 tools/analyze_conditions.py
    python3 tools/analyze_conditions.py --metric base_rot_err_rad --csv out.csv
"""

import argparse
import glob
import json
import pathlib

import numpy as np

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"

# Files to include, and the display name for each. Anything under
# results/INVALID_* is excluded by construction.
DEFAULT_ENVS = {
    "G1": "divergence_G1JoystickFlatTerrain.json",
    "Berkeley": "divergence_BerkeleyHumanoidJoystickFlatTerrain.json",
    "T1": "divergence_T1JoystickFlatTerrain.json",
    "Op3": "divergence_Op3Joystick_fixed_asshipped.json",
}


def boot_ratio(a, b, n_boot=20000, seed=0):
    """Median ratio, 95% CI, and a two-sided bootstrap p against ratio == 1."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        ma = np.median(a[rng.integers(0, len(a), len(a))])
        mb = np.median(b[rng.integers(0, len(b), len(b))])
        if mb > 0:
            out.append(ma / mb)
    o = np.array(out)
    p = 2 * min((o <= 1).mean(), (o >= 1).mean())
    return float(np.median(o)), float(np.percentile(o, 2.5)), \
        float(np.percentile(o, 97.5)), float(max(p, 1.0 / n_boot))


def benjamini_hochberg(pvals, alpha=0.05):
    """Return the largest p-value that passes BH, or 0.0 if none do."""
    p = np.asarray(pvals)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    passing = ranked <= alpha * np.arange(1, m + 1) / m
    if not passing.any():
        return 0.0
    return float(ranked[np.max(np.where(passing)[0])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="base_pos_err_m")
    ap.add_argument("--pair", default="c|jax", help="numerator engine pair")
    ap.add_argument("--vs", default="c|warp", help="denominator engine pair")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    rows = []
    for name, fname in DEFAULT_ENVS.items():
        path = RESULTS / fname
        if not path.exists():
            print(f"[skip] {name}: {fname} not found")
            continue
        d = json.loads(path.read_text())
        conds = {c["label"]: c for c in d["conditions"]}
        for label, c in conds.items():
            if args.pair not in c["pairs"] or args.vs not in c["pairs"]:
                continue
            a = np.array(c["pairs"][args.pair]["seed_finals"][args.metric])
            b = np.array(c["pairs"][args.vs]["seed_finals"][args.metric])
            r, lo, hi, p = boot_ratio(a, b)
            rows.append({
                "robot": name, "condition": label, "n": d["seeds"],
                "ratio": r, "ci_low": lo, "ci_high": hi, "p": p,
                "median_num": float(np.median(a)), "median_den": float(np.median(b)),
            })

    if not rows:
        raise SystemExit("no data found")

    crit = benjamini_hochberg([r["p"] for r in rows], args.alpha)
    for r in rows:
        r["significant"] = r["p"] <= crit

    print(f"metric      : {args.metric}")
    print(f"comparison  : {args.pair} / {args.vs}   (>1 means {args.vs.split('|')[1]} "
          f"is closer to {args.vs.split('|')[0]})")
    print(f"tests       : {len(rows)}   Benjamini-Hochberg FDR {args.alpha:.0%} "
          f"-> p <= {crit:.4f}\n")

    by_robot = {}
    for r in rows:
        by_robot.setdefault(r["robot"], []).append(r)

    for name, rs in by_robot.items():
        n_sig = sum(x["significant"] for x in rs)
        n_rev = sum(x["significant"] and x["ratio"] < 1 for x in rs)
        med = np.median([x["ratio"] for x in rs if x["significant"]]) if n_sig else float("nan")
        print(f"{name:<10s} {n_sig:>2d}/{len(rs)} significant   "
              f"median significant ratio {med:6.2f}"
              + (f"   ({n_rev} REVERSED, ratio < 1)" if n_rev else ""))

    total = sum(x["significant"] for x in rows)
    rev = [x for x in rows if x["significant"] and x["ratio"] < 1]
    print(f"\noverall     : {total}/{len(rows)} conditions significant")
    if rev:
        print("reversals (the other backend is closer) -- the effect is not "
              "universal in direction:")
        for x in rev:
            print(f"  {x['robot']:<10s} {x['condition']:22s} "
                  f"{x['ratio']:.2f} [{x['ci_low']:.2f}, {x['ci_high']:.2f}]")

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[saved] {args.csv}")


if __name__ == "__main__":
    main()
