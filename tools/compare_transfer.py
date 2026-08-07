#!/usr/bin/env python3
"""Decide whether the cross-backend gap exceeds ordinary training noise.

This is the test the policy-level claim lives or dies by. Two policies trained
with different seeds in the *same* backend already differ from each other. If
swapping the physics backend moves performance by less than that, there is no
finding -- only seed variance with a narrative attached.

Two quantities are computed from the same set of evaluation runs:

  effect  the paired cross-backend gap, measured within each policy
  null    the spread of performance across seeds within a single backend

and the verdict is simply whether the effect clears the null.

    python3 tools/compare_transfer.py results/eval_*.json
"""

import argparse
import glob
import json
import pathlib
import statistics as st


def load(paths):
    runs = []
    for p in paths:
        d = json.loads(pathlib.Path(p).read_text())
        if "backends" not in d:
            print(f"[skip] {p}: not an evaluation file")
            continue
        d["_path"] = p
        runs.append(d)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="results/eval_*.json (globs ok)")
    ap.add_argument("--metric", default="return_mean")
    args = ap.parse_args()

    paths = []
    for f in args.files:
        paths.extend(glob.glob(f))
    runs = load(sorted(set(paths)))
    if not runs:
        raise SystemExit("no evaluation files found")

    backends = sorted({b for r in runs for b in r["backends"]})
    print(f"evaluation runs : {len(runs)}")
    print(f"backends        : {backends}")
    print(f"metric          : {args.metric}\n")

    # Within-backend episode-to-episode spread, pooled over policies. This is
    # the matched null for per-episode perturbation (Q2 below).
    episode_sd = {}
    for b in backends:
        eps = []
        for r in runs:
            raw = r.get("raw", {}).get(b, {}).get("return")
            if raw:
                eps.extend(raw)
        if len(eps) > 1:
            episode_sd[b] = st.stdev(eps)

    # --- null model: seed-to-seed spread within one backend ------------------
    print("=== NULL MODEL: seed-to-seed spread within a single backend ===")
    null_spread = {}
    for b in backends:
        vals = [r["backends"][b][args.metric] for r in runs if b in r["backends"]]
        if len(vals) < 2:
            print(f"  {b:6s} n={len(vals)} -- need >=2 seeds to estimate noise")
            continue
        sd = st.stdev(vals)
        null_spread[b] = sd
        rng = max(vals) - min(vals)
        print(f"  {b:6s} n={len(vals)}  mean={st.mean(vals):9.3f}  "
              f"sd={sd:8.4f}  range={rng:8.4f}")
    print()

    # --- effect: paired cross-backend gap within each policy -----------------
    print("=== EFFECT: cross-backend gap, paired within each policy ===")
    ref = backends[0]
    per_policy = {}
    for r in runs:
        key = f"gap_vs_{ref}"
        if key not in r:
            continue
        for impl, rec in r[key].items():
            per_policy.setdefault(impl, []).append(rec)
            print(f"  {pathlib.Path(r['_path']).name}")
            print(f"    {impl} vs {ref}: mean={rec['mean_gap']:+.4f}  "
                  f"mean|d|={rec['mean_abs_gap']:.4f}  "
                  f"max|d|={rec['max_abs_gap']:.4f}  "
                  f"effect_size={rec['paired_effect_size']:+.3f}")
    print()

    # --- verdict -------------------------------------------------------------
    print("=== VERDICT ===")
    if not per_policy:
        print("  no paired gap records found -- run evaluate.py with >=2 backends")
        return
    for impl, recs in per_policy.items():
        signed = [x["mean_gap"] for x in recs]
        absol = [x["mean_abs_gap"] for x in recs]
        m_signed = st.mean(signed)
        m_abs = st.mean(absol)
        noise = null_spread.get(ref)
        print(f"\n  {impl} vs {ref}   (n_policies={len(recs)})")

        # The verdict rests on comparing like with like. An earlier version of
        # this script divided mean|gap| -- a *per-episode* quantity -- by the
        # seed-to-seed spread of *means*. Those have different units and the
        # ratio is inflated by roughly sqrt(n_episodes), which turned a null
        # into an apparent 3.2x effect. Both questions are now reported
        # separately, against their own matched null.
        print("    [Q1] systematic shift: difference of means vs seed spread")
        print(f"      difference of means : {m_signed:+.4f}")
        if noise is None:
            print("      seed spread (sd)    : unknown (need >=2 seeds)")
            print("      VERDICT             : INCONCLUSIVE -- no null model")
        else:
            print(f"      seed spread (sd)    : {noise:.4f}")
            r = abs(m_signed) / noise if noise else float("inf")
            print(f"      ratio               : {r:.2f}")
            if r >= 2.0:
                print("      VERDICT             : systematic shift EXCEEDS seed noise")
            elif r >= 1.0:
                print("      VERDICT             : comparable to seed noise -- WEAK")
            else:
                print("      VERDICT             : below seed noise -- NO systematic effect")

        # Per-episode perturbation is a genuinely different claim: the backend
        # can reshuffle individual episodes without moving the mean. Its matched
        # null is the within-backend episode-to-episode spread.
        ep_sd = episode_sd.get(ref)
        print("    [Q2] per-episode perturbation vs natural episode variation")
        print(f"      mean|per-episode d| : {m_abs:.4f}")
        if ep_sd:
            print(f"      within-backend ep sd: {ep_sd:.4f}")
            r2 = m_abs / ep_sd
            print(f"      ratio               : {r2:.3f}")
            if r2 >= 1.0:
                print("      VERDICT             : perturbation exceeds natural variation")
            else:
                print("      VERDICT             : perturbation SMALLER than natural "
                      "episode variation")
        else:
            print("      within-backend ep sd: unavailable (no raw episode returns)")


if __name__ == "__main__":
    main()
