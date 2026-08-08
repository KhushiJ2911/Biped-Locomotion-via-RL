#!/usr/bin/env python3
"""Check the co-author explainer's numbers against the source data.

The explainer restates the paper's results in plain language for people who
will not read the LaTeX. A number that drifts here is just as wrong as one in
the paper, and this document is the one that actually gets shared.

    python3 tools/verify_explainer.py
"""
import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
R = ROOT / "results"
html = (ROOT / "docs/explainer.html").read_text()

fails, checks = [], 0


def must_contain(s, why=""):
    global checks
    checks += 1
    if s not in html:
        fails.append(f"missing {s!r} {why}")


# Cross-embodiment counts, recomputed.
rows = list(csv.DictReader(open(ROOT / "figures/condition_level.csv")))
nsig = sum(1 for r in rows if r["significant"] == "True")
checks += 1
if not (nsig == 34 and len(rows) == 56):
    fails.append(f"grid is {nsig}/{len(rows)}, explainer says 34 of 56")
must_contain("34 of 56")

# Robot specs, straight from the compiled models.
rs = json.loads((R / "robot_specs.json").read_text())
for robot, nu, mass, height in [("Unitree G1", 29, 33.34, 0.785),
                                ("Booster T1", 23, 31.61, 0.665),
                                ("ROBOTIS Op3", 20, 3.15, 0.244),
                                ("Berkeley Humanoid", 12, 16.06, 0.515)]:
    s = rs[robot]
    checks += 1
    if s["nu"] != nu or abs(s["mass_kg"] - mass) > 0.01 or abs(s["standing_height_m"] - height) > 0.001:
        fails.append(f"{robot}: models say nu={s['nu']} mass={s['mass_kg']} h={s['standing_height_m']}")
    must_contain(f"{mass} kg", f"({robot} mass)")

# Precision ratio.
a = json.loads((R / "precision_G1JoystichFlatTerrain_f32.json").read_text()) \
    if False else json.loads((R / "precision_G1JoystickFlatTerrain_f32.json").read_text())
b = json.loads((R / "precision_G1JoystickFlatTerrain_f64.json").read_text())
ratio = b["conditions"]["baseline"]["median"] / a["conditions"]["baseline"]["median"]
checks += 1
if abs(ratio - 1.048) > 0.01:
    fails.append(f"float64/float32 ratio is {ratio:.3f}, explainer says 1.05")
must_contain("1.05")

# Statements the explainer makes that must match the paper exactly.
for s in ["2.7%", "t = 2.877", "t = 1.400", "0.53", "0.56",
          "51.7%", "1.3%", "40&times;", "1.86%", "63&times;",
          "2.1&times;", "8.1&times;", "28.2&times;", "1 of 14", "13 of 14",
          "109&times;", "3.32", "2.91", "3.17&times;", "1.1 mm"]:
    must_contain(s)

print(f"{checks} checks run")
if fails:
    print(f"\n{len(fails)} PROBLEM(S):")
    for f in fails:
        print("  " + f)
    sys.exit(1)
print("explainer numbers match the source data and the paper")
