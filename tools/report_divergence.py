#!/usr/bin/env python3
"""Turn a three-way divergence sweep into a publication-ready table.

Reads ``results/divergence_<env>.json`` (written by ``sim2sim.divergence``) and
reports, for each engine pair, how much each physics factor changes the gap
relative to the baseline configuration.

The ratio column is the headline number: >1 means the factor *widens* the
engine gap, <1 means it narrows it. A factor whose ratio is within its own
error bar of 1.0 is a null result and is marked as such -- that distinction is
the whole point, so it is computed rather than eyeballed.

    python3 tools/report_divergence.py results/divergence_G1JoystickFlatTerrain.json
    python3 tools/report_divergence.py ... --metric base_pos_err_m --csv out.csv
"""

import argparse
import json
import math
import pathlib


def ratio_with_error(mean, sem, base_mean, base_sem):
    """Ratio mean/base_mean and its 1-sigma error via standard propagation.

    Both numerator and denominator carry uncertainty, so the relative errors add
    in quadrature. Returns (ratio, sigma) and (nan, nan) on a zero baseline.
    """
    if base_mean == 0:
        return float("nan"), float("nan")
    r = mean / base_mean
    rel_a = (sem / mean) if mean else 0.0
    rel_b = (base_sem / base_mean) if base_mean else 0.0
    return r, abs(r) * math.sqrt(rel_a**2 + rel_b**2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", help="divergence_<env>.json")
    ap.add_argument("--metric", default="base_rot_err_rad")
    ap.add_argument("--csv", default=None, help="also write a CSV here")
    args = ap.parse_args()

    data = json.loads(pathlib.Path(args.json).read_text())
    conds = data["conditions"]
    seeds = data.get("seeds", 1)

    pair_names = list(conds[0].get("pairs", {}).keys())
    if not pair_names:
        raise SystemExit(
            "No 'pairs' key -- this file predates the three-way sweep. Re-run "
            "sim2sim.divergence to regenerate it."
        )

    print(f"env      : {data['env']}")
    print(f"duration : {data.get('duration_s', '?')} s   seeds: {seeds}")
    print(f"metric   : {args.metric}")
    if seeds < 3:
        print("WARNING  : fewer than 3 seeds -- error bars are not meaningful.")
    print()

    rows = []
    for pair in pair_names:
        base = next(c for c in conds if c["label"] == "baseline")["pairs"][pair]
        b_mean = base["mean"][args.metric]
        b_sem = base.get("sem", {}).get(args.metric, 0.0)

        print(f"=== engine pair: {pair} ===")
        print(f"{'condition':22s} {'mean':>11s} {'sem':>10s} {'ratio':>8s} "
              f"{'+-':>7s}  verdict")

        entries = []
        for c in conds:
            p = c["pairs"][pair]
            m = p["mean"][args.metric]
            s = p.get("sem", {}).get(args.metric, 0.0)
            r, rs = ratio_with_error(m, s, b_mean, b_sem)
            entries.append((c["label"], m, s, r, rs, c.get("perturbation_effect_on_mjc")))

        entries.sort(key=lambda e: (-e[3] if e[3] == e[3] else 0))

        for label, m, s, r, rs, effect in entries:
            if label == "baseline":
                verdict = "(reference)"
            elif effect == 0.0:
                verdict = "INVALID - perturbation was a no-op"
            elif r != r:
                verdict = "undefined"
            elif abs(r - 1.0) <= 2 * rs:
                verdict = "null (within 2 sigma of 1)"
            elif r > 1:
                verdict = "WIDENS gap"
            else:
                verdict = "narrows gap"
            print(f"{label:22s} {m:11.3e} {s:10.2e} {r:8.2f} {rs:7.2f}  {verdict}")
            rows.append([pair, label, m, s, r, rs, verdict])
        print()

    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["pair", "condition", "mean", "sem", "ratio", "ratio_sem", "verdict"])
            w.writerows(rows)
        print(f"[saved] {args.csv}")


if __name__ == "__main__":
    main()
