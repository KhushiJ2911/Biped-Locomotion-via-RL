#!/usr/bin/env python3
"""Check the hand-typed LaTeX tables against the source JSON.

Transcribing numbers from a result file into a table by hand is where
errors enter a paper, and nothing else in the pipeline would catch one.
Each claim below names the file it must match.

    python3 paper/ieee/verify_tex_numbers.py
"""
import json
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
R = ROOT / "results"
tex = (ROOT / "paper/ieee/main.tex").read_text()

fails, checks = [], 0


def claim(desc, actual, want, tol=0.02):
    """want is what the paper prints; actual is what the data says."""
    global checks
    checks += 1
    if want == 0:
        ok = abs(actual) < 1e-12
    else:
        ok = abs(actual - want) / abs(want) <= tol
    if not ok:
        fails.append(f"{desc}: paper says {want:g}, data says {actual:g}")


def in_tex(s):
    global checks
    checks += 1
    if s not in tex:
        fails.append(f"string missing from main.tex: {s!r}")


# ---- Experiment 0: closed-form accuracy -------------------------------
gt = json.loads((R / "groundtruth.json").read_text())
ff = gt["free_fall"]
claim("freefall C", ff["c"]["err_vs_discrete_euler"], 4.44e-15, tol=0.05)
claim("freefall JAX", ff["jax"]["err_vs_discrete_euler"], 6.97e-06, tol=0.02)
claim("freefall Warp", ff["warp"]["err_vs_discrete_euler"], 6.97e-06, tol=0.02)

pend = gt["pendulum"]
claim("pendulum C", abs(pend["c"]["rel_drift"]), 8.884e-04)
claim("pendulum JAX", abs(pend["jax"]["rel_drift"]), 8.891e-04)
claim("pendulum Warp", abs(pend["warp"]["rel_drift"]), 8.891e-04)

sl = gt["incline_sliding"]
claim("incline slide C", sl["c"]["abs_err_m"], 2.03e-03)
claim("incline slide JAX", sl["jax"]["abs_err_m"], 2.12e-03)
claim("incline slide Warp", sl["warp"]["abs_err_m"], 2.03e-03)
# The 23.9x agreement claim, recomputed from final positions.
gap_w = abs(sl["warp"]["x_final"] - sl["c"]["x_final"])
gap_j = abs(sl["jax"]["x_final"] - sl["c"]["x_final"])
claim("warp-vs-C slide gap", gap_w, 3.99e-06, tol=0.03)
claim("jax-vs-C slide gap", gap_j, 9.51e-05, tol=0.03)
claim("slide agreement ratio", gap_j / gap_w, 23.9, tol=0.03)

st = gt["incline_static"]
claim("incline static C", st["c"]["abs_err_m"], 1.148e-03)
claim("incline static JAX", st["jax"]["abs_err_m"], 1.128e-03)
claim("incline static Warp", st["warp"]["abs_err_m"], 1.148e-03)

# ---- Experiment 6: precision ------------------------------------------
f32 = json.loads((R / "precision_G1JoystickFlatTerrain_f32.json").read_text())
f64 = json.loads((R / "precision_G1JoystickFlatTerrain_f64.json").read_text())
for cond, v32, v64, ratio in [
    ("baseline", 1.921e-03, 2.013e-03, 1.048),
    ("solref_scale=2.0", 3.101e-02, 3.310e-02, 1.067),
    ("solref_scale=0.5", 2.928e-03, 3.120e-03, 1.065),
    ("friction_scale=2.0", 2.236e-03, 2.261e-03, 1.011),
    ("timestep=0.001", 1.955e-03, 2.062e-03, 1.055),
]:
    a = f32["conditions"][cond]["median"]
    b = f64["conditions"][cond]["median"]
    claim(f"precision {cond} f32", a, v32)
    claim(f"precision {cond} f64", b, v64)
    claim(f"precision {cond} ratio", b / a, ratio)

# ---- Experiment 7: convergence ----------------------------------------
def conv(dt, it, pair):
    d = json.loads((R / f"divergence_G1JoystickFlatTerrain_conv_dt{dt}_it{it}.json").read_text())
    c = next(x for x in d["conditions"] if x["label"] == "baseline")
    return float(np.median(c["pairs"][pair]["seed_finals"]["base_pos_err_m"]))

for dt, cj, cw in [("0.004", 3.668e-03, 6.359e-04),
                   ("0.002", 2.216e-03, 2.536e-04),
                   ("0.001", 1.915e-03, 1.511e-04),
                   ("0.0005", 1.956e-03, 7.459e-05)]:
    claim(f"conv dt={dt} c|jax", conv(dt, 50, "c|jax"), cj)
    claim(f"conv dt={dt} c|warp", conv(dt, 50, "c|warp"), cw)

# Warp's step ratios and the 26x claim, recomputed rather than trusted.
w = [conv(dt, 50, "c|warp") for dt in ("0.004", "0.002", "0.001", "0.0005")]
j = [conv(dt, 50, "c|jax") for dt in ("0.004", "0.002", "0.001", "0.0005")]
for i, want in zip(range(1, 4), (0.40, 0.60, 0.49)):
    claim(f"warp step ratio {i}", w[i] / w[i - 1], want, tol=0.03)
claim("warp 8x refinement reduction", w[0] / w[-1], 8.5, tol=0.05)
claim("jax/warp at finest dt", j[-1] / w[-1], 26, tol=0.05)

# ---- Experiment 1: baseline -------------------------------------------
d = json.loads((R / "divergence_G1JoystickFlatTerrain.json").read_text())
base = next(c for c in d["conditions"] if c["label"] == "baseline")
for pair, want in [("c|jax", 3.354e-03), ("c|warp", 4.120e-04), ("jax|warp", 3.385e-03)]:
    claim(f"baseline {pair}", base["pairs"][pair]["mean"]["base_pos_err_m"], want)

# ---- Experiment 5: validation ------------------------------------------
v = json.loads((R / "ceval_validation_G1JoystickFlatTerrain.json").read_text())
claim("ceval reset agreement", v["reset"]["max_abs_diff"], 2.38e-07)
claim("ceval trajectory agreement", v["trajectory"]["max_bookkeeping_diff"], 5.96e-08)
claim("ceval phase exact", v["trajectory"]["max_phase_diff"], 0)

# ---- Cross-embodiment counts ------------------------------------------
import csv
rows = list(csv.DictReader(open(ROOT / "figures/condition_level.csv")))
nsig = sum(1 for r in rows if r["significant"] == "True")
claim("cross-embodiment significant cells", nsig, 34, tol=0.0)
claim("cross-embodiment total cells", len(rows), 56, tol=0.0)
for robot, want in [("G1", 13), ("Berkeley", 13), ("T1", 6), ("Op3", 2)]:
    n = sum(1 for r in rows if r["robot"] == robot and r["significant"] == "True")
    claim(f"{robot} significant", n, want, tol=0.0)

# ---- Claims that must appear verbatim ---------------------------------
in_tex("34 of 56")
in_tex("$t(191)=2.877$")
in_tex("1.86\\%")

print(f"{checks} checks run")
if fails:
    print(f"\n{len(fails)} MISMATCH(ES):")
    for f in fails:
        print("  " + f)
    sys.exit(1)
print("all numbers in main.tex match their source files")
