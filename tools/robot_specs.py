#!/usr/bin/env python3
"""Extract the four bipeds' specifications from their compiled models.

The paper compares four robots but never says what they are, which leaves a
reader unable to tell a 12-DoF robot from a 29-DoF one. Every number here is
read from the model MuJoCo actually compiles, not from a datasheet, so the
table describes the thing that was simulated.

    python3 tools/robot_specs.py
"""

import json
import pathlib

import mujoco
import numpy as np
from mujoco_playground import registry

ENVS = [
    ("Unitree G1", "G1JoystickFlatTerrain"),
    ("Berkeley Humanoid", "BerkeleyHumanoidJoystickFlatTerrain"),
    ("Booster T1", "T1JoystickFlatTerrain"),
    ("ROBOTIS Op3", "Op3Joystick"),
]

OUT = pathlib.Path(__file__).resolve().parent.parent / "results" / "robot_specs.json"


def specs(env_name):
    env = registry.load(env_name)
    m = env.mj_model
    total_mass = float(np.sum(m.body_mass))

    # Standing height: the free joint's initial z from the keyframe the task
    # resets to, which is the pose these comparisons actually start from.
    z0 = float(m.key_qpos[0][2]) if m.nkey else float("nan")

    # How foot contact is declared decides which solver path is exercised,
    # and was a hypothesis we tested explicitly (Section on contact
    # representation), so it belongs in the table.
    n_pairs = int(m.npair)

    return {
        "nq": int(m.nq), "nv": int(m.nv), "nu": int(m.nu),
        "n_bodies": int(m.nbody), "n_geoms": int(m.ngeom),
        "mass_kg": round(total_mass, 2),
        "standing_height_m": round(z0, 3),
        "explicit_pairs": n_pairs,
        "shipped_dt": float(m.opt.timestep),
        "shipped_iterations": int(m.opt.iterations),
        "shipped_ls_iterations": int(m.opt.ls_iterations),
    }


def main():
    out = {}
    for label, env_name in ENVS:
        try:
            out[label] = specs(env_name)
            out[label]["env"] = env_name
            print(f"[ok] {label}")
        except Exception as e:
            print(f"[FAIL] {label}: {type(e).__name__}: {e}")
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {OUT}\n")

    hdr = f"{'robot':20s}{'DoF(nu)':>9s}{'nq':>5s}{'nv':>5s}{'mass':>8s}{'height':>8s}{'pairs':>7s}{'dt':>8s}{'iters':>7s}"
    print(hdr); print("-" * len(hdr))
    for k, v in out.items():
        print(f"{k:20s}{v['nu']:9d}{v['nq']:5d}{v['nv']:5d}{v['mass_kg']:8.2f}"
              f"{v['standing_height_m']:8.3f}{v['explicit_pairs']:7d}"
              f"{v['shipped_dt']:8.4f}{v['shipped_iterations']:7d}")


if __name__ == "__main__":
    main()
