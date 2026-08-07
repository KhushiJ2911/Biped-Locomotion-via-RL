#!/usr/bin/env python3
"""Check that every number in the paper draft traces back to a result file.

Transcription errors are the class of mistake that survives careful proofreading:
the prose is coherent, the argument is sound, and one digit is wrong. This
recomputes the full set of values derivable from the committed result JSONs --
raw means, standard errors, ratios against baseline with propagated errors, and
cross-engine advantage ratios -- then checks each numeric literal in the draft
against that set.

Unmatched numbers are reported, not silently ignored. Many will be legitimate
(sample sizes, version strings, section numbers), so the output is a review list
rather than a pass/fail gate. The failure it is built to catch is a number that
*looks* like a measurement but matches nothing that was measured.

    python3 tools/verify_paper_numbers.py paper/draft.md
"""

import argparse
import json
import math
import pathlib
import re

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"

# Numbers that are structural rather than measured.
IGNORE_EXACT = {
    # sample sizes, counts, dimensions, indices
    0, 1, 2, 3, 4, 5, 6, 10, 13, 14, 20, 29, 32, 35, 36, 50, 100, 128, 512,
    1000, 1024, 8192,
    # version-ish and config numbers
    0.002, 0.001, 0.004, 0.3, 0.5, 0.8, 2.0, 15.0, 30.0,
}

NUM_RE = re.compile(r"(?<![\w.])[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?(?![\w])")


def close(a, b, rel=0.02):
    """Match with tolerance, since the draft rounds to 2-4 significant figures."""
    if a == 0 or b == 0:
        return abs(a - b) < 1e-12
    return abs(a - b) / max(abs(a), abs(b)) <= rel


def ratio_err(mean, sem, bmean, bsem):
    if bmean == 0:
        return float("nan"), float("nan")
    r = mean / bmean
    ra = (sem / mean) if mean else 0.0
    rb = (bsem / bmean) if bmean else 0.0
    return r, abs(r) * math.sqrt(ra**2 + rb**2)


def derivable_values():
    """Every value the draft is entitled to quote, with a human-readable source."""
    vals = []  # (value, description)

    gt = RESULTS / "groundtruth.json"
    if gt.exists():
        g = json.loads(gt.read_text())
        for case in ("free_fall", "pendulum", "incline_sliding", "incline_static"):
            if case not in g:
                continue
            for eng, rec in g[case].items():
                if not isinstance(rec, dict):
                    continue
                for k, v in rec.items():
                    if isinstance(v, (int, float)):
                        vals.append((float(v), f"groundtruth.{case}.{eng}.{k}"))
        ref = g.get("free_fall_reference", {})
        for k, v in ref.items():
            if isinstance(v, (int, float)):
                vals.append((float(v), f"groundtruth.free_fall_reference.{k}"))

    for path in sorted(RESULTS.glob("divergence_*.json")):
        d = json.loads(path.read_text())
        conds = {c["label"]: c for c in d.get("conditions", [])}
        if "baseline" not in conds:
            continue
        pairs = list(conds["baseline"].get("pairs", {}))
        for pair in pairs:
            base = conds["baseline"]["pairs"][pair]
            for label, c in conds.items():
                p = c["pairs"][pair]
                for metric in p.get("mean", {}):
                    m = p["mean"][metric]
                    s = p.get("sem", {}).get(metric, 0.0)
                    vals.append((m, f"{label}|{pair}.mean.{metric}"))
                    vals.append((s, f"{label}|{pair}.sem.{metric}"))
                    r, rs = ratio_err(m, s, base["mean"][metric],
                                      base.get("sem", {}).get(metric, 0.0))
                    vals.append((r, f"{label}|{pair}.ratio.{metric}"))
                    vals.append((rs, f"{label}|{pair}.ratio_sem.{metric}"))
        # cross-engine advantage ratios, e.g. (c|jax) / (c|warp)
        for label, c in conds.items():
            ps = c.get("pairs", {})
            if "c|jax" in ps and "c|warp" in ps:
                for metric in ps["c|jax"].get("mean", {}):
                    a = ps["c|jax"]["mean"][metric]
                    b = ps["c|warp"]["mean"][metric]
                    if b:
                        vals.append((a / b, f"{label}.warp_advantage.{metric}"))

    for path in sorted(RESULTS.glob("eval_*.json")):
        d = json.loads(path.read_text())
        for b, rec in d.get("backends", {}).items():
            for k, v in rec.items():
                if isinstance(v, (int, float)):
                    vals.append((float(v), f"{path.name}:{b}.{k}"))
        for key, block in d.items():
            if key.startswith("gap_vs_"):
                for impl, rec in block.items():
                    for k, v in rec.items():
                        if isinstance(v, (int, float)):
                            vals.append((float(v), f"{path.name}:{key}.{impl}.{k}"))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("draft", nargs="?", default="paper/draft.md")
    ap.add_argument("--rel", type=float, default=0.02, help="match tolerance")
    args = ap.parse_args()

    text = pathlib.Path(args.draft).read_text()
    vals = derivable_values()
    if not vals:
        raise SystemExit("no result files found -- nothing to verify against")

    print(f"draft            : {args.draft}")
    print(f"derivable values : {len(vals)} from results/*.json")
    print(f"tolerance        : {args.rel:.1%}\n")

    matched, unmatched = [], []
    seen = set()
    for tok in NUM_RE.findall(text):
        try:
            x = float(tok)
        except ValueError:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        if x in IGNORE_EXACT:
            continue
        hits = [d for v, d in vals if close(x, v, args.rel)]
        (matched if hits else unmatched).append((tok, hits))

    # A value matching many unrelated sources tells us little: with hundreds of
    # derivable numbers, a 2% window will catch coincidences. Showing every hit
    # makes that visible instead of hiding it behind the first one found.
    print(f"=== MATCHED ({len(matched)}) ===")
    for tok, hits in matched:
        uniq = sorted({h.split(".")[0] for h in hits})
        flag = "  [AMBIGUOUS]" if len(uniq) > 3 else ""
        print(f"  {tok:>14s}  <- {hits[0]}{flag}")
        if flag:
            print(f"  {'':>14s}     ...and {len(hits)-1} other sources; "
                  f"confirm by hand which one this is")

    print(f"\n=== UNMATCHED ({len(unmatched)}) -- review each ===")
    if not unmatched:
        print("  none")
    for tok, _ in unmatched:
        print(f"  {tok:>14s}  (no result value within {args.rel:.1%})")

    print("\nUnmatched numbers are not necessarily wrong -- counts, versions and")
    print("config values live here too. But any number that reads as a measurement")
    print("and appears above needs to be traced by hand before submission.")


if __name__ == "__main__":
    main()
