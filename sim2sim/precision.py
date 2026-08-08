"""Experiment 6: is the MJX-JAX vs MuJoCo-C gap arithmetic, or algorithmic?

Experiment 0 established that *without contact* the two agree to float32
round-off (~7e-06 on free fall), which says the contact-free difference is
purely precision: MJX computes in float32, MuJoCo-C in float64. Experiment 1
then found gaps three orders of magnitude larger once contact is involved,
across 34/56 robot x condition tests -- but never checked whether those are the
same phenomenon scaled up.

That is the obvious question, and the answer changes the paper. If running MJX
in double precision collapses the gap, the finding is "MJX is float32 and that
matters under contact" -- narrower, and arguably a known property. If the gap
survives at float64, the two implementations genuinely disagree about contact
dynamics, which is a substantive claim about the solver.

JAX supports float64 via ``jax_enable_x64``; MuJoCo Warp does not (it rejects
float64 arrays outright), so this compares MuJoCo-C against MJX-JAX only. That
is the comparison that matters here, since JAX is the backend that departs from
the reference.

``jax_enable_x64`` must be set before any JAX array exists, so precision is
selected from argv at import time rather than inside main().

    python -m sim2sim.precision --seeds 20            # float32 (default)
    python -m sim2sim.precision --seeds 20 --x64      # float64
"""
from __future__ import annotations

import sys

# Must happen before jax is used for anything.
_X64 = "--x64" in sys.argv
import jax  # noqa: E402

jax.config.update("jax_enable_x64", _X64)

import argparse  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402

import numpy as np  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"

# Conditions spanning the range where Experiment 1 found the largest and
# smallest gaps, so the precision question is asked where it matters most.
CONDITIONS = ["baseline", "solref_scale=2.0", "solref_scale=0.5",
              "friction_scale=2.0", "timestep=0.001"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="G1JoystickFlatTerrain")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--duration", type=float, default=1.0)
    ap.add_argument("--njmax", type=int, default=512)
    ap.add_argument("--x64", action="store_true", help="run MJX in float64")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    from mujoco_playground import registry

    from sim2sim import _wandb
    from sim2sim.divergence import hold_pose_ctrl, home_state, perturb_state
    from sim2sim.engines import PhysicsOpts, divergence_metrics, rollout_mjc, rollout_mjx

    dtype = jax.numpy.zeros(1).dtype
    print(f"[precision] jax_enable_x64={args.x64}  jnp default dtype={dtype}")
    if args.x64 and dtype != np.float64:
        raise SystemExit("x64 requested but JAX is still float32 -- aborting rather "
                         "than reporting a float32 run as float64")

    env = registry.load(args.env)
    base = env.mj_model
    free_base = base.nq > base.nu + 6

    wb = _wandb.init(enabled=not args.no_wandb,
                     name=f"precision_{args.env}_{'f64' if args.x64 else 'f32'}",
                     group=args.env, job_type="precision",
                     config={"env": args.env, "x64": args.x64,
                             "seeds": args.seeds, "duration_s": args.duration})

    out = {"env": args.env, "x64": args.x64, "seeds": args.seeds,
           "duration_s": args.duration, "conditions": {}}

    for label in CONDITIONS:
        opts = PhysicsOpts()
        if label != "baseline":
            key, val = label.split("=")
            setattr(opts, key, float(val))
        model = opts.apply(base)
        steps = int(round(args.duration / model.opt.timestep))

        vals = []
        for s in range(args.seeds):
            q0, v0 = home_state(model)
            q0 = perturb_state(model, q0, s)
            ctrl = hold_pose_ctrl(model, home_state(model)[0], steps)
            qc, vc = rollout_mjc(model, q0, v0, ctrl)
            qj, vj = rollout_mjx(model, q0, v0, ctrl, impl="jax", njmax=args.njmax)
            m = divergence_metrics(qc, qj, vc, vj, free_base=free_base)
            vals.append(float(m["base_pos_err_m"][-1]))

        arr = np.array(vals)
        rec = {"median": float(np.median(arr)), "mean": float(arr.mean()),
               "sem": float(arr.std(ddof=1) / np.sqrt(len(arr))), "values": vals}
        out["conditions"][label] = rec
        print(f"  {label:20s} c|jax median={rec['median']:.4e}  mean={rec['mean']:.4e}",
              flush=True)
        _wandb.summary(wb, {f"{label}/median": rec["median"],
                            f"{label}/mean": rec["mean"]})

    RESULTS.mkdir(exist_ok=True)
    tag = "f64" if args.x64 else "f32"
    path = RESULTS / f"precision_{args.env}_{tag}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"[saved] {path}")
    _wandb.finish(wb)


if __name__ == "__main__":
    main()
